"""Worker / 主图节点与 LLM 协议。"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from dl_agent.config import Settings
from dl_agent.domain.models import AskTurn, Evidence, ExternalRef, LibraryHit
from dl_agent.observability.langfuse import clip_value, observe_span
from dl_agent.understand.agent.prompts import (
    CHAT_IDENTITY_REPLY,
    CLARIFY_REPLY,
    NO_EVIDENCE_ANSWER,
    aggregation_prompt,
    compress_prompt,
    dialogue_memory_compress_prompt,
    dialogue_prompt,
    fallback_prompt,
    orchestrator_prompt,
)
from dl_agent.understand.agent.schemas import AskTask, AskTaskKind, DialogueAct
from dl_agent.understand.agent.state import (
    AskState,
    WorkerState,
    _merge_evidence,
    _merge_external_refs,
    _merge_library_hits,
)
from dl_agent.understand.agent.tools import BOOTSTRAP_SEARCH_CALL_ID, RetrievalTools
from dl_agent.understand.citations import rank_evidence_for_prompt, resolve_citations, sanitize_answer_zh
from dl_agent.understand.progress import emit_phase, emit_tool, outcome_note
from dl_agent.understand.route import is_obvious_chat, primary_ask_mode, suggest_task_kinds

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class WorkerReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class WorkerModel(Protocol):
    def orchestrate(
        self,
        messages: list[dict[str, Any]],
        question: str,
        context_summary: str,
    ) -> WorkerReply: ...

    def compress(self, conversation_text: str) -> str: ...

    def fallback(self, question: str, context_text: str) -> str: ...


class AskModel(WorkerModel, Protocol):
    def dialogue(self, query: str, recent_turns: str) -> DialogueAct: ...

    def compress_dialogue(self, older_turns: str, query: str) -> str: ...

    def aggregate(
        self,
        original: str,
        answers: list[dict[str, Any]],
        evidence: list[Evidence],
    ) -> str: ...


def estimate_tokens(state: WorkerState) -> int:
    texts = [str(item.get("content") or "") for item in state.get("messages") or []]
    texts.append(state.get("context_summary") or "")
    chars = sum(len(text) for text in texts)
    return max(1, chars // 4)


def bootstrap_search(state: WorkerState, tools: RetrievalTools) -> dict[str, Any]:
    question = (state.get("question") or "").strip()
    emit_tool(
        name="search_child_chunks",
        status="start",
        call_id=BOOTSTRAP_SEARCH_CALL_ID,
        args={"query": question},
    )
    with observe_span(
        "search_child_chunks",
        as_type="tool",
        input={"query": question, "bootstrap": True},
    ) as span:
        text = tools.search_child_chunks(question)
        span.update(output=clip_value(text))
        logger.info("ask tool name=search_child_chunks bootstrap=true")
    hits = list(tools.last_child_hits)
    emit_tool(
        name="search_child_chunks",
        status="done",
        call_id=BOOTSTRAP_SEARCH_CALL_ID,
        args={"query": question},
        note=outcome_note(text, len(hits)),
    )
    return {
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": BOOTSTRAP_SEARCH_CALL_ID,
                        "name": "search_child_chunks",
                        "args": {"query": question},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_child_chunks",
                "tool_call_id": BOOTSTRAP_SEARCH_CALL_ID,
                "content": text,
            },
        ],
        "retrieval_keys": [f"search::{question}"],
        "evidence_bag": list(tools.last_child_hits),
        "tool_call_count": 1,
    }


def orchestrator(state: WorkerState, model: WorkerModel) -> dict[str, Any]:
    emit_phase("orchestrator", "正在决定下一步")
    reply = model.orchestrate(
        list(state.get("messages") or []),
        state.get("question") or "",
        state.get("context_summary") or "",
    )
    if not reply.tool_calls:
        forced = _forced_parent_call(state)
        if forced is not None:
            reply = WorkerReply(tool_calls=[forced])
    message: dict[str, Any] = {
        "role": "assistant",
        "content": reply.content or "",
    }
    if reply.tool_calls:
        message["tool_calls"] = [
            {
                "id": item.id or _new_call_id(item.name, index),
                "name": item.name,
                "args": dict(item.args or {}),
            }
            for index, item in enumerate(reply.tool_calls)
        ]
    return {"messages": [message], "iteration_count": 1}


def run_tools(state: WorkerState, tools: RetrievalTools) -> dict[str, Any]:
    messages = list(state.get("messages") or [])
    last = messages[-1] if messages else {}
    known = set(state.get("retrieval_keys") or [])
    new_keys: list[str] = []
    evidence = []
    refs: list[ExternalRef] = []
    executed = 0
    out: list[dict[str, Any]] = []
    for call in last.get("tool_calls") or []:
        name = str(call.get("name") or "")
        args = call.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        call_id = str(call.get("id") or call.get("tool_call_id") or "")
        key, skipped = _tool_dedup_key(name, args)
        if skipped:
            content = skipped
            logger.info("ask tool name=%s skipped=invalid", name or "tool")
            emit_tool(
                name=name,
                status="skip",
                call_id=call_id,
                args=args,
                note=outcome_note(content, skipped="invalid"),
            )
        elif key and key in known:
            content = f"(skipped duplicate {name})"
            logger.info("ask tool name=%s skipped=duplicate", name or "tool")
            emit_tool(
                name=name,
                status="skip",
                call_id=call_id,
                args=args,
                note=outcome_note(content, skipped="duplicate"),
            )
        else:
            emit_tool(name=name, status="start", call_id=call_id, args=args)
            with observe_span(name or "tool", as_type="tool", input=args) as span:
                content, hits, extra_refs, unknown = _invoke_tool(tools, name, args)
                if unknown:
                    content = f"UNKNOWN_TOOL: {name}"
                    span.update(output=content, level="WARNING")
                    logger.info("ask tool name=%s unknown=true", name or "tool")
                else:
                    if key:
                        known.add(key)
                        new_keys.append(key)
                    evidence.extend(hits)
                    refs.extend(extra_refs)
                    executed += 1
                    span.update(output=clip_value(content))
                    logger.info("ask tool name=%s executed=true", name or "tool")
            emit_tool(
                name=name,
                status="done",
                call_id=call_id,
                args=args,
                note=outcome_note(content, len(hits)),
            )
        out.append(
            {
                "role": "tool",
                "name": name or "tool",
                "tool_call_id": str(call.get("id") or call.get("tool_call_id") or ""),
                "content": content,
            }
        )
    return {
        "messages": out,
        "retrieval_keys": new_keys,
        "evidence_bag": evidence,
        "external_refs": refs,
        "tool_call_count": executed,
    }


def compress_context(state: WorkerState, model: WorkerModel) -> dict[str, Any]:
    question = state.get("question") or ""
    existing = (state.get("context_summary") or "").strip()
    lines = [f"USER QUESTION:\n{question}\n"]
    if existing:
        lines.append(f"[PRIOR COMPRESSED CONTEXT]\n{existing}\n")
    for item in state.get("messages") or []:
        role = item.get("role")
        content = str(item.get("content") or "")
        if role == "assistant":
            calls = item.get("tool_calls") or []
            extra = ""
            if calls:
                extra = " | " + ", ".join(f"{c.get('name')}({c.get('args')})" for c in calls)
            lines.append(f"[ASSISTANT{extra}]\n{content or '(tool call only)'}\n")
        elif role == "tool":
            lines.append(f"[TOOL RESULT — {item.get('name', 'tool')}]\n{content}\n")
    summary = model.compress("\n".join(lines)).strip()
    keys = list(state.get("retrieval_keys") or [])
    if keys:
        summary += "\n\n---\n**Already executed (do NOT repeat):**\n" + "\n".join(f"- {key}" for key in keys)
    return {
        "context_summary": summary,
        "messages": [
            {"__reset__": True},
            {"role": "user", "content": question},
            {"role": "system", "content": f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{summary}"},
        ],
    }


def fallback_response(state: WorkerState, model: WorkerModel) -> dict[str, Any]:
    emit_phase("fallback", "正在根据已有证据作答")
    seen: set[str] = set()
    chunks: list[str] = []
    for item in state.get("messages") or []:
        if item.get("role") != "tool":
            continue
        content = str(item.get("content") or "")
        if content in seen:
            continue
        seen.add(content)
        chunks.append(content)
    summary = (state.get("context_summary") or "").strip()
    parts: list[str] = []
    if summary:
        parts.append(f"## Compressed Research Context\n\n{summary}")
    if chunks:
        parts.append("## Retrieved Data\n\n" + "\n\n".join(chunks))
    context = "\n\n".join(parts) if parts else "No data was retrieved from the documents."
    answer = model.fallback(state.get("question") or "", context)
    return {"messages": [{"role": "assistant", "content": answer}]}


def collect_answer(state: WorkerState) -> dict[str, Any]:
    answer = "Unable to generate an answer."
    for item in reversed(state.get("messages") or []):
        if item.get("role") != "assistant":
            continue
        if item.get("tool_calls"):
            continue
        text = str(item.get("content") or "").strip()
        if text:
            answer = text
            break
    return {"final_answer": answer}


_ALLOWED_KINDS = {"close_read", "library", "arxiv"}
DIALOGUE_COMPRESS_AFTER = 8
DIALOGUE_KEEP_RECENT = 4
EARLIER_COMPRESSED_MARKER = "[Earlier, compressed]"


def _history_lines(history: list[Any] | None) -> list[str]:
    lines: list[str] = []
    for item in list(history or []):
        role = _history_role(item)
        content = _history_content(item)
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return lines


def format_recent_history(history: list[Any] | None) -> str:
    """短对话：全部原文，不截断。"""
    lines = _history_lines(history)
    if not lines:
        return ""
    return "[Recent turns]\n" + "\n".join(lines)


def pack_dialogue_memory(
    history: list[Any] | None,
    query: str,
    *,
    compress_fn: Callable[[str, str], str] | None = None,
    compress_after: int = DIALOGUE_COMPRESS_AFTER,
    keep_recent: int = DIALOGUE_KEEP_RECENT,
) -> str:
    """窗口化记忆：未满阈值全原文；满了则压缩更早轮次并留下笔记，近处原文不截断。"""
    lines = _history_lines(history)
    if not lines:
        return ""
    threshold = max(1, compress_after)
    keep = max(1, keep_recent)
    if len(lines) < threshold or compress_fn is None or len(lines) <= keep:
        return "[Recent turns]\n" + "\n".join(lines)
    recent = lines[-keep:]
    older = lines[:-keep]
    older_text = "\n".join(older)
    summary = ""
    try:
        summary = (compress_fn(older_text, query) or "").strip()
    except Exception:
        logger.exception("dialogue memory compression failed; keeping earlier turns raw")
    parts: list[str] = []
    if summary:
        parts.append(f"{EARLIER_COMPRESSED_MARKER}\n{summary}")
    else:
        parts.append("[Earlier turns]\n" + older_text)
    parts.append("[Recent turns]\n" + "\n".join(recent))
    return "\n\n".join(parts)


def dialogue_query(state: AskState, model: AskModel, settings: Settings) -> dict[str, Any]:
    emit_phase("dialogue", "正在理解问题")
    original = (state.get("originalQuery") or "").strip()
    history = list(state.get("history") or [])
    if is_obvious_chat(original):
        packed = pack_dialogue_memory(history, original, compress_fn=None)
        result = _direct_dialogue_state("chat", CHAT_IDENTITY_REPLY)
        result["conversation_summary"] = packed
        return result
    packed = pack_dialogue_memory(
        history,
        original,
        compress_fn=model.compress_dialogue,
    )
    act = model.dialogue(original, packed)
    if act.intent == "chat":
        result = _direct_dialogue_state("chat", (act.reply or "").strip() or CHAT_IDENTITY_REPLY)
        result["conversation_summary"] = packed
        return result
    if act.intent == "clarify":
        result = _direct_dialogue_state("clarify", (act.reply or "").strip() or CLARIFY_REPLY)
        result["conversation_summary"] = packed
        return result
    tasks = _plan_tasks_from_act(act, original, settings, state.get("index_status"))
    if not tasks:
        result = _direct_dialogue_state("retrieve", NO_EVIDENCE_ANSWER)
        result["no_evidence"] = True
        result["conversation_summary"] = packed
        return result
    close_questions = [item["question"] for item in tasks if item["kind"] == "close_read"]
    return {
        "conversation_summary": packed,
        "dialogue_intent": "retrieve",
        "skip_workers": False,
        "rewrittenQuestions": close_questions or [item["question"] for item in tasks],
        "plan_tasks": tasks,
        "mode": primary_ask_mode(kinds=[item["kind"] for item in tasks]),
        "answer_zh": "",
        "no_evidence": False,
    }


def rewrite_query(state: AskState, model: AskModel, settings: Settings) -> dict[str, Any]:
    return dialogue_query(state, model, settings)


def direct_reply(state: AskState) -> dict[str, Any]:
    emit_phase("direct_reply", "正在回答")
    return {
        "answer_zh": sanitize_answer_zh(state.get("answer_zh") or CHAT_IDENTITY_REPLY),
        "citations": list(state.get("citations") or []),
        "external_refs": list(state.get("external_refs") or []),
        "library_hits": list(state.get("library_hits") or []),
        "no_evidence": bool(state.get("no_evidence")),
        "partial": False,
        "mode": state.get("mode") or "close_read",
    }


def _direct_dialogue_state(intent: str, reply: str) -> dict[str, Any]:
    return {
        "conversation_summary": "",
        "dialogue_intent": intent,
        "skip_workers": True,
        "rewrittenQuestions": [],
        "plan_tasks": [],
        "mode": "close_read",
        "answer_zh": sanitize_answer_zh(reply),
        "citations": [],
        "external_refs": [],
        "library_hits": [],
        "no_evidence": False,
        "partial": False,
    }


def _plan_tasks_from_act(
    act: DialogueAct,
    original: str,
    settings: Settings,
    index_status: str | None,
) -> list[dict[str, Any]]:
    hint = suggest_task_kinds(original)
    raw_tasks = [
        {"kind": item.kind, "question": item.question.strip()}
        for item in act.tasks
        if item.kind in _ALLOWED_KINDS and item.question.strip()
    ]
    explicit = [kind for kind in act.kinds if kind in _ALLOWED_KINDS]
    kinds: list[str] = list(dict.fromkeys(explicit or hint))
    if "close_read" not in hint and any(kind in hint for kind in ("library", "arxiv")):
        kinds = [kind for kind in kinds if kind != "close_read"]
        raw_tasks = [item for item in raw_tasks if item["kind"] != "close_read"]
        if not kinds:
            kinds = list(hint)
    if not kinds:
        kinds = list(hint) or ["close_read"]
    if "close_read" in kinds and not _index_allows_close_read(index_status):
        kinds = [kind for kind in kinds if kind != "close_read"]
    aux_kinds = [kind for kind in kinds if kind in {"library", "arxiv"}]
    tasks: list[dict[str, Any]] = []
    if "close_read" in kinds:
        close_questions = [item["question"] for item in raw_tasks if item["kind"] == "close_read"]
        budget = max(1, settings.ask_max_subquestions - len(aux_kinds))
        questions = _cap_questions(close_questions, budget)
        if not questions and original:
            questions = [original]
        tasks.extend({"kind": "close_read", "question": item} for item in questions)
    for kind in aux_kinds:
        question = next((item["question"] for item in raw_tasks if item["kind"] == kind), original)
        if question:
            tasks.append({"kind": kind, "question": question})
    if not tasks and original and _index_allows_close_read(index_status):
        tasks = [{"kind": "close_read", "question": original}]
    return tasks


def aggregate_answers(state: AskState, model: AskModel) -> dict[str, Any]:
    emit_phase("aggregate", "正在整理回答")
    answers = sorted(list(state.get("agent_answers") or []), key=lambda item: int(item.get("index") or 0))
    evidence: list[Evidence] = []
    refs: list[ExternalRef] = []
    hits: list[LibraryHit] = []
    for item in answers:
        evidence = _merge_evidence(evidence, list(item.get("evidence") or []))
        refs = _merge_external_refs(refs, list(item.get("external_refs") or []))
        hits = _merge_library_hits(hits, list(item.get("library_hits") or []))
    executed = [str(item.get("kind") or "close_read") for item in answers]
    mode = primary_ask_mode(kinds=executed) if executed else str(state.get("mode") or "close_read")
    kinds = set(executed) or {"close_read"}
    if not evidence:
        aux = [item for item in answers if (item.get("kind") or "close_read") in {"library", "arxiv"}]
        if aux and any(str(item.get("answer") or "").strip() for item in aux):
            if kinds <= {"library"} or kinds <= {"arxiv"}:
                chosen = aux[0]
                aux_mode = str(chosen.get("kind") or mode)
                return {
                    "answer_zh": sanitize_answer_zh(str(chosen.get("answer") or "").strip() or NO_EVIDENCE_ANSWER),
                    "citations": [],
                    "external_refs": refs,
                    "library_hits": hits,
                    "mode": aux_mode,
                    "no_evidence": bool(chosen.get("no_evidence")),
                    "partial": False,
                }
            stitched = "\n\n".join(
                str(item.get("answer") or "").strip() for item in answers if str(item.get("answer") or "").strip()
            )
            return {
                "answer_zh": sanitize_answer_zh(stitched or NO_EVIDENCE_ANSWER),
                "citations": [],
                "external_refs": refs,
                "library_hits": hits,
                "mode": mode,
                "no_evidence": False,
                "partial": True,
            }
        return {
            "answer_zh": NO_EVIDENCE_ANSWER,
            "citations": [],
            "external_refs": refs,
            "library_hits": hits,
            "mode": mode,
            "no_evidence": True,
            "partial": False,
        }
    context = "\n".join(
        [state.get("originalQuery") or ""]
        + [str(item.get("answer") or "") for item in answers]
    )
    evidence = rank_evidence_for_prompt(evidence, context)
    raw = model.aggregate(state.get("originalQuery") or "", answers, evidence)
    payload = _parse_json_object(raw)
    answer = sanitize_answer_zh(
        str(payload.get("answer_zh") or "").strip() or "模型没有给出可用回答，请换个问法重试。"
    )
    used = _used_indices(payload.get("used_evidence"), len(evidence))
    citations = resolve_citations(answer, evidence, used)
    return {
        "answer_zh": answer,
        "citations": citations,
        "external_refs": refs,
        "library_hits": hits,
        "mode": mode,
        "no_evidence": False,
        "partial": bool(payload.get("partial")),
    }


def _index_allows_close_read(status: str | None) -> bool:
    if not status:
        return True
    return status not in {"pending", "failed", "skipped"}


def build_orchestrator_messages(
    question: str,
    context_summary: str,
    messages: list[dict[str, Any]],
    *,
    native: bool = False,
    enable_external: bool = False,
) -> list[dict[str, Any]]:
    packed = [{"role": "system", "content": orchestrator_prompt(native=native, enable_external=enable_external)}]
    if context_summary.strip():
        packed.append(
            {
                "role": "user",
                "content": f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}",
            }
        )
    packed.extend(messages_for_chat_api(messages, native=native))
    if not messages:
        packed.append({"role": "user", "content": question})
    return packed


def build_fallback_messages(question: str, context_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": fallback_prompt()},
        {
            "role": "user",
            "content": f"USER QUERY: {question}\n\n{context_text}\n\nINSTRUCTION:\nProvide the best possible answer using only the data above.",
        },
    ]


def build_compress_messages(conversation_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": compress_prompt()},
        {"role": "user", "content": conversation_text},
    ]


def build_dialogue_messages(query: str, recent_turns: str) -> list[dict[str, Any]]:
    packed = (recent_turns or "").strip()
    context = ""
    if packed:
        if packed.startswith("[Earlier") or packed.startswith("[Recent turns]"):
            context += packed + "\n\n"
        else:
            context += f"[Recent turns]\n{packed}\n\n"
    context += f"[Current question]\n{query}\n"
    return [
        {"role": "system", "content": dialogue_prompt()},
        {"role": "user", "content": context},
    ]


def build_dialogue_compress_messages(older_turns: str, query: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": dialogue_memory_compress_prompt()},
        {
            "role": "user",
            "content": f"Current question:\n{query}\n\nEarlier turns:\n{older_turns}\n",
        },
    ]


def build_rewrite_messages(query: str, summary: str) -> list[dict[str, Any]]:
    return build_dialogue_messages(query, summary)


def parse_dialogue_act(raw: str) -> DialogueAct:
    data = _parse_json_object_loose(raw)
    intent = str(data.get("intent") or "").strip().lower()
    if intent not in {"chat", "clarify", "retrieve"}:
        if data.get("is_clear") is False:
            intent = "clarify"
        else:
            intent = "retrieve"
    reply = str(data.get("reply") or data.get("clarification_needed") or "").strip()
    kinds = _parse_task_kinds(data.get("kinds"))
    tasks = _parse_act_tasks(data.get("tasks"))
    if not tasks:
        questions = data.get("questions") if isinstance(data.get("questions"), list) else []
        tasks = [
            AskTask(kind="close_read", question=str(item).strip())
            for item in questions
            if str(item).strip()
        ]
    return DialogueAct(intent=intent, kinds=kinds, tasks=tasks, reply=reply)  # type: ignore[arg-type]


def _parse_json_object_loose(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_task_kinds(raw: Any) -> list[AskTaskKind]:
    if not isinstance(raw, list):
        return []
    out: list[AskTaskKind] = []
    for item in raw:
        kind = str(item).strip()
        if kind in _ALLOWED_KINDS and kind not in out:
            out.append(kind)  # type: ignore[arg-type]
    return out


def _parse_act_tasks(raw: Any) -> list[AskTask]:
    if not isinstance(raw, list):
        return []
    out: list[AskTask] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "close_read").strip()
        question = str(item.get("question") or "").strip()
        if kind not in _ALLOWED_KINDS or not question:
            continue
        out.append(AskTask(kind=kind, question=question))  # type: ignore[arg-type]
    return out


def build_aggregate_messages(
    original: str,
    answers: list[dict[str, Any]],
    evidence: list[Evidence],
) -> list[dict[str, Any]]:
    lines = [
        f"Original user question: {original}",
        "",
        "[Sub-answers]",
    ]
    for item in answers:
        lines.append(f"Q{int(item.get('index') or 0) + 1}: {item.get('question')}")
        lines.append(str(item.get("answer") or ""))
        lines.append("")
    lines.append("[Pinned Evidence]")
    for index, item in enumerate(evidence, start=1):
        title = item.section_title or "Untitled"
        lines.append(f"[{index}] section={title} page={item.page} id={item.section_id or ''}")
        lines.append(item.quote)
        lines.append("")
    return [
        {"role": "system", "content": aggregation_prompt()},
        {"role": "user", "content": "\n".join(lines)},
    ]


def tokens_over_budget(state: WorkerState, settings: Settings) -> bool:
    return estimate_tokens(state) > settings.ask_compress_tokens


def messages_for_chat_api(messages: list[dict[str, Any]], *, native: bool = False) -> list[dict[str, Any]]:
    """把 Worker 消息打成发给 OpenAI 兼容接口的 messages。

    native=True 时保留 assistant.tool_calls 与 role=tool + tool_call_id。
    json 协议仍把工具结果折成 user 文本，避免缺 id 时 400。
    """
    out: list[dict[str, Any]] = []
    for item in messages:
        if item.get("__reset__"):
            continue
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "")
        if role == "tool":
            if native:
                packed = _native_tool_message(item, content)
                if packed is not None:
                    out.append(packed)
                    continue
            name = str(item.get("name") or "tool")
            out.append({"role": "user", "content": f"[TOOL RESULT — {name}]\n{content}"})
            continue
        if role == "assistant":
            calls = item.get("tool_calls") or []
            if native and calls:
                openai_calls = [_to_openai_tool_call(call, index) for index, call in enumerate(calls)]
                out.append(
                    {
                        "role": "assistant",
                        "content": content.strip() or None,
                        "tool_calls": openai_calls,
                    }
                )
                continue
            if calls:
                requested = "; ".join(
                    f"{call.get('name')}({call.get('args')})" for call in calls
                )
                note = f"Requested tools: {requested}"
                content = f"{content}\n{note}".strip() if content.strip() else note
            out.append({"role": "assistant", "content": content})
            continue
        if role in {"system", "user"}:
            out.append({"role": role, "content": content})
            continue
        out.append({"role": "user", "content": content})
    return out


def _native_tool_message(item: dict[str, Any], content: str) -> dict[str, Any] | None:
    tool_call_id = str(item.get("tool_call_id") or "").strip()
    if not tool_call_id:
        return None
    packed: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    name = str(item.get("name") or "").strip()
    if name:
        packed["name"] = name
    return packed


def _to_openai_tool_call(call: Any, index: int) -> dict[str, Any]:
    if not isinstance(call, dict):
        return {
            "id": _new_call_id("tool", index),
            "type": "function",
            "function": {"name": "tool", "arguments": "{}"},
        }
    if isinstance(call.get("function"), dict) and call.get("id"):
        return {
            "id": str(call["id"]),
            "type": "function",
            "function": {
                "name": str(call["function"].get("name") or ""),
                "arguments": _dump_arguments(call["function"].get("arguments")),
            },
        }
    name = str(call.get("name") or "").strip() or "tool"
    args = call.get("args") if call.get("args") is not None else call.get("arguments") or {}
    return {
        "id": str(call.get("id") or "").strip() or _new_call_id(name, index),
        "type": "function",
        "function": {
            "name": name,
            "arguments": _dump_arguments(args),
        },
    }


def _dump_arguments(raw: Any) -> str:
    if isinstance(raw, str):
        return raw if raw.strip() else "{}"
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    return "{}"


def _new_call_id(name: str, index: int = 0) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in (name or "tool"))[:24] or "tool"
    return f"call_{slug}_{index}_{uuid.uuid4().hex[:8]}"


def _cap_questions(raw: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max(1, limit):
            break
    return out


def _history_role(item: Any) -> str:
    if isinstance(item, AskTurn):
        return "User" if item.role == "user" else "Assistant"
    if isinstance(item, dict):
        return "User" if item.get("role") == "user" else "Assistant"
    return "User"


def _history_content(item: Any) -> str:
    if isinstance(item, AskTurn):
        return (item.content or "").strip()
    if isinstance(item, dict):
        return str(item.get("content") or "").strip()
    return str(item).strip()


def _used_indices(raw: Any, total: int) -> list[int]:
    if not isinstance(raw, list):
        return []
    seen: set[int] = set()
    out: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > total or index in seen:
            continue
        seen.add(index)
        out.append(index)
    return out


def parse_orchestrator_reply(text: str) -> WorkerReply:
    data = _first_json_object(text or "")
    tool = str(data.get("tool") or data.get("name") or "").strip()
    parsed = _json_tool_call(tool, data)
    if parsed is not None:
        return WorkerReply(tool_calls=[parsed])
    if data.get("answer_zh"):
        return WorkerReply(content=str(data["answer_zh"]).strip())
    return WorkerReply(content=(text or "").strip())


def _json_tool_call(tool: str, data: dict[str, Any]) -> ToolCall | None:
    if tool == "retrieve_parent_chunks":
        section_id = str(data.get("section_id") or data.get("parent_id") or "").strip()
        return ToolCall(tool, {"section_id": section_id})
    if tool == "search_child_chunks":
        args: dict[str, Any] = {"query": str(data.get("query") or "").strip()}
        if data.get("limit") is not None:
            try:
                args["limit"] = int(data["limit"])
            except (TypeError, ValueError):
                pass
        return ToolCall(tool, args)
    if tool == "list_sections":
        args = {}
        kind = str(data.get("kind") or "").strip()
        if kind:
            args["kind"] = kind
        return ToolCall(tool, args)
    if tool == "list_figures":
        args = {}
        section_id = str(data.get("section_id") or "").strip()
        if section_id:
            args["section_id"] = section_id
        return ToolCall(tool, args)
    if tool == "get_figure_caption":
        return ToolCall(tool, {"figure_id": str(data.get("figure_id") or "").strip()})
    if tool == "search_captions":
        return ToolCall(tool, {"query": str(data.get("query") or "").strip()})
    if tool == "get_translation":
        return ToolCall(tool, {"section_id": str(data.get("section_id") or "").strip()})
    if tool == "arxiv_search":
        args: dict[str, Any] = {}
        query = str(data.get("query") or "").strip()
        if query:
            args["query"] = query
        arxiv_id = str(data.get("arxiv_id") or data.get("id") or "").strip()
        if arxiv_id:
            args["arxiv_id"] = arxiv_id
        if data.get("max_results") is not None:
            try:
                args["max_results"] = int(data["max_results"])
            except (TypeError, ValueError):
                pass
        return ToolCall(tool, args)
    return None


def _tool_dedup_key(name: str, args: dict[str, Any]) -> tuple[str, str]:
    """返回 (去重键, 跳过原因)。跳过原因非空则不执行。"""
    if name == "search_child_chunks":
        query = str(args.get("query") or "").strip()
        if not query:
            return "", "(skipped duplicate search)"
        return f"search::{query}", ""
    if name == "retrieve_parent_chunks":
        section_id = str(args.get("section_id") or args.get("parent_id") or "").strip()
        if not section_id:
            return "", "(skipped duplicate parent)"
        return f"parent::{section_id}", ""
    if name == "list_sections":
        kind = str(args.get("kind") or "").strip().lower() or "*"
        return f"outline::{kind}", ""
    if name == "list_figures":
        section_id = str(args.get("section_id") or "").strip() or "*"
        return f"figures::{section_id}", ""
    if name == "get_figure_caption":
        figure_id = str(args.get("figure_id") or "").strip()
        if not figure_id:
            return "", "(skipped duplicate figure)"
        return f"caption::{figure_id}", ""
    if name == "search_captions":
        query = str(args.get("query") or "").strip()
        if not query:
            return "", "(skipped duplicate captions)"
        return f"captions::{query}", ""
    if name == "get_translation":
        section_id = str(args.get("section_id") or "").strip()
        if not section_id:
            return "", "(skipped duplicate translation)"
        return f"zh::{section_id}", ""
    if name == "arxiv_search":
        query = str(args.get("query") or "").strip()
        arxiv_id = str(args.get("arxiv_id") or args.get("id") or "").strip()
        if not query and not arxiv_id:
            return "", "(skipped duplicate arxiv)"
        return f"arxiv::{arxiv_id or query}", ""
    return "", ""


def _invoke_tool(tools: RetrievalTools, name: str, args: dict[str, Any]) -> tuple[str, list[Evidence], list[ExternalRef], bool]:
    if name == "search_child_chunks":
        query = str(args.get("query") or "").strip()
        limit = args.get("limit")
        content = tools.search_child_chunks(query, None if limit is None else int(limit))
        return content, list(getattr(tools, "last_hits", None) or tools.last_child_hits), [], False
    if name == "retrieve_parent_chunks":
        section_id = str(args.get("section_id") or args.get("parent_id") or "").strip()
        content = tools.retrieve_parent_chunks(section_id)
        return content, list(getattr(tools, "last_hits", None) or getattr(tools, "last_parent_hits", None) or []), [], False
    if name == "arxiv_search":
        fn = getattr(tools, "arxiv_search", None)
        if not callable(fn):
            return "", [], [], True
        query = str(args.get("query") or "").strip()
        arxiv_id = str(args.get("arxiv_id") or args.get("id") or "").strip()
        max_results = args.get("max_results")
        if max_results is None:
            content = fn(query, arxiv_id)
        else:
            content = fn(query, arxiv_id, int(max_results))
        refs = list(getattr(tools, "last_external_refs", None) or [])
        return content, [], refs, False
    fn = getattr(tools, name, None)
    if name not in {
        "list_sections",
        "list_figures",
        "get_figure_caption",
        "search_captions",
        "get_translation",
    } or not callable(fn):
        return "", [], [], True
    if name == "list_sections":
        content = fn(str(args.get("kind") or "").strip() or None)
    elif name == "list_figures":
        content = fn(str(args.get("section_id") or "").strip() or None)
    elif name == "get_figure_caption":
        content = fn(str(args.get("figure_id") or "").strip())
    elif name == "search_captions":
        content = fn(str(args.get("query") or "").strip())
    else:
        content = fn(str(args.get("section_id") or "").strip())
    return content, list(getattr(tools, "last_hits", None) or []), [], False


def _forced_parent_call(state: WorkerState) -> ToolCall | None:
    keys = list(state.get("retrieval_keys") or [])
    if any(str(key).startswith("parent::") for key in keys):
        return None
    section_id = _best_section_id(list(state.get("evidence_bag") or []))
    if not section_id:
        return None
    return ToolCall("retrieve_parent_chunks", {"section_id": section_id}, id="call_forced_parent")


def _best_section_id(evidence: list[Evidence]) -> str:
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for item in evidence:
        section_id = (item.section_id or "").strip()
        if not section_id or section_id in seen:
            continue
        seen.add(section_id)
        title = (item.section_title or "").lower()
        if "reference" in title:
            continue
        score = 0
        if any(word in title for word in ("method", "approach", "model", "architecture")):
            score = 3
        elif any(word in title for word in ("experiment", "result", "ablation")):
            score = 2
        elif "intro" in title:
            score = 1
        ranked.append((score, section_id))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def _first_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
