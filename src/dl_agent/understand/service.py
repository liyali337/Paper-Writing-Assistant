"""论文内问答：检索当前篇的证据，再让模型只根据证据作答。"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
from collections.abc import Callable, Iterator
from typing import Any

from dl_agent.config import Settings, get_settings
from dl_agent.domain.models import AskTurn, Evidence, MethodExplain, PaperAnswer, PaperIntro
from dl_agent.harness.complete import LlmNotConfiguredError, LlmRequestError, chat, chat_turn
from dl_agent.knowledge.service import IndexNotReadyError, KnowledgeService, PaperNotFoundError
from dl_agent.knowledge.service import PaperNotReadyError as IngestNotReady
from dl_agent.observability.langfuse import mark_error, observe_span
from dl_agent.understand.citations import resolve_citations, sanitize_answer_zh
from dl_agent.understand.arxiv import ask_arxiv as run_arxiv_ask
from dl_agent.understand.library import ask_library as run_library_ask
from dl_agent.understand.progress import bind_progress, emit_phase, reset_progress
from dl_agent.understand.route import is_obvious_chat, suggest_task_kinds

logger = logging.getLogger(__name__)

ChatFn = Callable[..., str]
ChatTurnFn = Callable[..., Any]

ASK_SYSTEM = """你是论文精读助手，只根据用户消息里的编号证据回答当前这一篇论文。

硬约束：
- 只用证据里出现的事实；证据没有的数字、模块名、数据集、超参不要用训练记忆补。
- 不知道就写「原文未给出」，可建议去哪类章节看。
- 关键论断用 [1]、[2] 标注证据编号，编号必须存在；不要用 Front Matter / 标题页支撑实验数字。
- 不把与本篇无关的常识讲成「本文证明了…」。
- 中文讲解；术语、模型名、数据集名保留英文原文。
- 公式保持证据里的 LaTeX：行内 $...$，独立公式 $$...$$。不要改成 Unicode 或纯文本。
- 不要抄表格角标（◦ † $^\\circ$）。
- answer_zh 用 Markdown 分段，小节标题用 **…**，段与段空行。
- 缺信息一两句带过，不要列「尚未覆盖」清单。
- 只输出一个 JSON 对象，不要 Markdown 围栏。"""

_NO_EVIDENCE = "当前论文索引中没有找到直接证据，请换个问法或打开对应章节阅读。"


class UnderstandService:
    def __init__(
        self,
        knowledge: KnowledgeService,
        settings: Settings | None = None,
        chat_fn: ChatFn | None = None,
        chat_turn_fn: ChatTurnFn | None = None,
        ask_model: Any = None,
        worker_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        arxiv_gateway: Any = None,
    ):
        self.knowledge = knowledge
        self.settings = settings or get_settings()
        self.chat_fn = chat_fn or chat
        if chat_turn_fn is not None:
            self.chat_turn_fn = chat_turn_fn
        elif chat_fn is None:
            self.chat_turn_fn = chat_turn
        else:
            self.chat_turn_fn = None
        self.ask_model = ask_model
        self.worker_fn = worker_fn
        self.arxiv_gateway = arxiv_gateway

    def ask(
        self,
        paper_id: str,
        question: str,
        history: list[AskTurn] | None = None,
    ) -> PaperAnswer:
        paper = self.knowledge.get_paper(paper_id)
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if paper.status != "ready":
            raise PaperNotReadyError(paper.status)
        kinds = suggest_task_kinds(question)
        if kinds == ["library"]:
            return self.ask_library(question, history, exclude_paper_id=paper_id, current_paper_id=paper_id)
        if kinds == ["arxiv"]:
            return self.ask_arxiv(question, history, current_paper_id=paper_id)
        if "library" in kinds or "arxiv" in kinds:
            return self._ask_simple_mixed(paper_id, question, history, kinds)
        return self._close_read_simple(paper_id, question, history)

    def _close_read_simple(
        self,
        paper_id: str,
        question: str,
        history: list[AskTurn] | None = None,
    ) -> PaperAnswer:
        version = self.settings.ask_prompt_version
        model = self.settings.model_name
        emit_phase("retrieve", "正在检索本篇")
        with observe_span(
            "ask",
            input={"paper_id": paper_id, "question": question},
            metadata={"prompt_version": version, "mode": "simple"},
            session_id=paper_id,
            tags=["ask", "simple"],
        ) as span:
            try:
                evidence = self.knowledge.query(paper_id, question)
                if not evidence:
                    result = PaperAnswer(
                        paper_id=paper_id,
                        question=question,
                        answer_zh=_NO_EVIDENCE,
                        no_evidence=True,
                        model=model,
                        prompt_version=version,
                    )
                    span.update(output=_answer_trace(result))
                    return result

                raw = self.chat_fn(
                    [
                        {"role": "system", "content": ASK_SYSTEM},
                        {
                            "role": "user",
                            "content": _user_prompt(question, evidence, history, self.settings.ask_history_turns),
                        },
                    ],
                    settings=self.settings,
                )
                payload = _parse_json_object(raw)
                answer = sanitize_answer_zh(
                    str(payload.get("answer_zh") or "").strip() or "模型没有给出可用回答，请换个问法重试。"
                )
                used = _used_indices(payload.get("used_evidence"), len(evidence))
                citations = resolve_citations(answer, evidence, used)
                citations = [_verify_quote(item, paper_id, self.knowledge) for item in citations]
                figure_ids: list[str] = []
                for item in citations:
                    for figure_id in item.figure_ids:
                        if figure_id not in figure_ids:
                            figure_ids.append(figure_id)
                result = PaperAnswer(
                    paper_id=paper_id,
                    question=question,
                    answer_zh=answer,
                    citations=citations,
                    figure_ids=figure_ids[:8],
                    partial=bool(payload.get("partial")),
                    model=model,
                    prompt_version=version,
                )
                span.update(output=_answer_trace(result))
                return result
            except Exception as exc:
                mark_error(span, exc)
                raise

    def _ask_simple_mixed(
        self,
        paper_id: str,
        question: str,
        history: list[AskTurn] | None,
        kinds: list[str],
    ) -> PaperAnswer:
        paper = self.knowledge.get_paper(paper_id)
        close: PaperAnswer | None = None
        if "close_read" in kinds and paper.index_status == "ready":
            try:
                close = self._close_read_simple(paper_id, question, history)
            except IndexNotReadyError:
                close = None
        library = (
            self.ask_library(question, history, exclude_paper_id=paper_id, current_paper_id=paper_id)
            if "library" in kinds
            else None
        )
        arxiv = self.ask_arxiv(question, history, current_paper_id=paper_id) if "arxiv" in kinds else None
        hits = list(library.library_hits) if library else []
        refs = list(arxiv.external_refs) if arxiv else []
        extras = [
            item.answer_zh
            for item in (library, arxiv)
            if item is not None and item.answer_zh.strip() and not item.no_evidence
        ]
        if close is not None and close.citations:
            answer = close.answer_zh
            if extras:
                answer = answer.rstrip() + "\n\n" + "\n\n".join(extras)
            return close.model_copy(
                update={
                    "answer_zh": sanitize_answer_zh(answer),
                    "library_hits": hits,
                    "external_refs": refs or list(close.external_refs),
                    "partial": bool(extras) or close.partial,
                    "mode": "close_read",
                }
            )
        if library is not None and arxiv is None:
            return library
        if arxiv is not None and library is None:
            return arxiv
        if library is not None and arxiv is not None:
            if library.no_evidence and not arxiv.no_evidence:
                return arxiv.model_copy(update={"library_hits": hits})
            if arxiv.no_evidence and not library.no_evidence:
                return library.model_copy(update={"external_refs": refs})
            stitched = "\n\n".join(
                item.answer_zh for item in (library, arxiv) if item.answer_zh.strip()
            )
            return PaperAnswer(
                paper_id=paper_id,
                question=question,
                answer_zh=sanitize_answer_zh(stitched or library.answer_zh),
                library_hits=hits,
                external_refs=refs,
                mode="library",
                no_evidence=library.no_evidence and arxiv.no_evidence,
                partial=not (library.no_evidence and arxiv.no_evidence),
                model=self.settings.model_name,
                prompt_version=self.settings.ask_prompt_version,
            )
        return PaperAnswer(
            paper_id=paper_id,
            question=question,
            answer_zh=_NO_EVIDENCE,
            no_evidence=True,
            model=self.settings.model_name,
            prompt_version=self.settings.ask_prompt_version,
        )

    def ask_library(
        self,
        question: str,
        history: list[AskTurn] | None = None,
        *,
        exclude_paper_id: str | None = None,
        current_paper_id: str | None = None,
    ) -> PaperAnswer:
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        version = self.settings.library_prompt_version
        with observe_span(
            "ask_library",
            input={"question": question, "exclude_paper_id": exclude_paper_id},
            metadata={"prompt_version": version, "mode": "library"},
            tags=["ask", "library"],
        ) as span:
            try:
                result = run_library_ask(
                    self.knowledge,
                    question,
                    history,
                    settings=self.settings,
                    chat_fn=self.chat_fn,
                    exclude_paper_id=exclude_paper_id,
                    current_paper_id=current_paper_id,
                )
                span.update(output=_answer_trace(result))
                return result
            except Exception as exc:
                mark_error(span, exc)
                raise

    def ask_arxiv(
        self,
        question: str,
        history: list[AskTurn] | None = None,
        *,
        current_paper_id: str | None = None,
    ) -> PaperAnswer:
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        version = self.settings.arxiv_prompt_version
        with observe_span(
            "ask_arxiv",
            input={"question": question, "current_paper_id": current_paper_id},
            metadata={"prompt_version": version, "mode": "arxiv"},
            tags=["ask", "arxiv"],
        ) as span:
            try:
                result = run_arxiv_ask(
                    self.knowledge,
                    question,
                    history,
                    settings=self.settings,
                    chat_fn=self.chat_fn,
                    gateway=self.arxiv_gateway,
                    current_paper_id=current_paper_id,
                )
                span.update(output=_answer_trace(result))
                return result
            except Exception as exc:
                mark_error(span, exc)
                raise

    def ask_agent(
        self,
        paper_id: str,
        question: str,
        history: list[AskTurn] | None = None,
    ) -> PaperAnswer:
        from dl_agent.understand.agent.graph import paper_answer_from_state, run_ask
        from dl_agent.understand.agent.model import ChatAskModel

        paper = self.knowledge.get_paper(paper_id)
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if paper.status != "ready":
            raise PaperNotReadyError(paper.status)
        kinds = suggest_task_kinds(question)
        has_close = "close_read" in kinds
        has_aux = "library" in kinds or "arxiv" in kinds
        chat_only = is_obvious_chat(question)
        if has_close and not has_aux and not chat_only:
            if paper.index_status == "pending":
                raise IndexNotReadyError("pending")
            if paper.index_status == "failed":
                raise IndexNotReadyError("failed", paper.index_error)
        version = self.settings.ask_agent_prompt_version
        model_name = self.settings.model_name
        embedding_version = paper.embedding_version or "lexical-v1"
        with observe_span(
            "ask_agent",
            input={"paper_id": paper_id, "question": question},
            metadata={
                "prompt_version": version,
                "mode": "agent",
                "index_status": paper.index_status,
                "embedding_version": embedding_version,
            },
            session_id=paper_id,
            tags=["ask", "agent"],
        ) as span:
            try:
                if paper.index_status == "skipped" and has_close and not has_aux and not chat_only:
                    result = PaperAnswer(
                        paper_id=paper_id,
                        question=question,
                        answer_zh=_NO_EVIDENCE,
                        no_evidence=True,
                        model=model_name,
                        prompt_version=version,
                        embedding_version=embedding_version,
                    )
                    span.update(output=_answer_trace(result))
                    return result

                model = self.ask_model or ChatAskModel(
                    self.chat_fn,
                    self.settings,
                    chat_turn_fn=self.chat_turn_fn,
                )
                state = run_ask(
                    model,
                    self.settings,
                    paper_id,
                    question,
                    history,
                    knowledge=self.knowledge,
                    worker_fn=self.worker_fn,
                    chat_fn=self.chat_fn,
                    gateway=self.arxiv_gateway,
                    index_status=paper.index_status or "",
                )
                citations = [
                    _verify_quote(item, paper_id, self.knowledge) for item in (state.get("citations") or [])
                ]
                result = paper_answer_from_state(
                    state,
                    paper_id=paper_id,
                    question=question,
                    model_name=model_name,
                    prompt_version=version,
                    embedding_version=embedding_version,
                    citations=citations,
                )
                span.update(output=_answer_trace(result))
                return result
            except Exception as exc:
                mark_error(span, exc)
                raise

    def guard_ask(self, paper_id: str, question: str, *, agent: bool) -> None:
        """流式开始前抛出与 ask / ask_agent 相同的前置错误。"""
        paper = self.knowledge.get_paper(paper_id)
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")
        if paper.status != "ready":
            raise PaperNotReadyError(paper.status)
        kinds = suggest_task_kinds(question)
        has_close = "close_read" in kinds
        has_aux = "library" in kinds or "arxiv" in kinds
        if not has_close or has_aux:
            return
        if agent and is_obvious_chat(question):
            return
        if paper.index_status == "pending":
            raise IndexNotReadyError("pending")
        if paper.index_status == "failed":
            raise IndexNotReadyError("failed", paper.index_error)

    def iter_ask_events(
        self,
        paper_id: str,
        question: str,
        history: list[AskTurn] | None = None,
        *,
        agent: bool,
    ) -> Iterator[dict[str, Any]]:
        events: queue.Queue[dict[str, Any] | object] = queue.Queue()
        done = object()

        def runner() -> None:
            token = bind_progress(events.put)
            try:
                if agent:
                    answer = self.ask_agent(paper_id, question, history)
                else:
                    answer = self.ask(paper_id, question, history)
                events.put({"type": "answer", "answer": answer.model_dump(mode="json")})
            except Exception as exc:
                events.put(_ask_failure_event(exc))
            finally:
                reset_progress(token)
                events.put(done)

        thread = threading.Thread(target=runner, name="ask-stream", daemon=True)
        thread.start()
        while True:
            item = events.get()
            if item is done:
                break
            if isinstance(item, dict):
                yield item
        thread.join(timeout=5)


def _ask_failure_event(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, PaperNotFoundError):
        return _ask_error(404, "not_found", "论文不存在")
    if isinstance(exc, (PaperNotReadyError, IngestNotReady)):
        return _ask_error(409, "not_ready", "论文尚未解析完成")
    if isinstance(exc, IndexNotReadyError):
        if exc.status == "pending":
            return _ask_error(409, "index_pending", "索引尚未完成")
        return _ask_error(503, "index_failed", exc.error or "索引失败")
    if isinstance(exc, LlmNotConfiguredError):
        return _ask_error(503, "llm_not_configured", str(exc))
    if isinstance(exc, LlmRequestError):
        return _ask_error(503, "llm_request_failed", str(exc))
    if isinstance(exc, ValueError):
        return _ask_error(400, "invalid_request", str(exc))
    logger.exception("ask stream failed")
    return _ask_error(500, "ask_failed", "问答失败")


def _ask_error(status: int, code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "status": status, "code": code, "stage": "ask", "message": message}


def _answer_trace(answer: PaperAnswer) -> dict[str, Any]:
    return {
        "answer_zh": answer.answer_zh,
        "no_evidence": answer.no_evidence,
        "partial": answer.partial,
        "citations": len(answer.citations),
        "external_refs": len(answer.external_refs),
        "library_hits": len(answer.library_hits),
        "figure_ids": list(answer.figure_ids),
    }


class PaperNotReadyError(RuntimeError):
    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


def _user_prompt(
    question: str,
    evidence: list[Evidence],
    history: list[AskTurn] | None,
    history_turns: int,
) -> str:
    lines = [
        "只根据下列编号证据回答。输出 JSON：",
        '{"answer_zh":"中文回答，可用[1][2]标注","used_evidence":[1,2],"partial":false}',
        "",
        "[Pinned Evidence]",
    ]
    for index, item in enumerate(evidence, start=1):
        title = item.section_title or "Untitled"
        lines.append(f"[{index}] section={title} page={item.page}")
        lines.append(item.quote)
        lines.append("")
    recent = _recent_history(history, history_turns)
    if recent:
        lines.append("[Recent history]")
        for turn in recent:
            prefix = "Q" if turn.role == "user" else "A"
            content = re.sub(r"\s+", " ", turn.content).strip()[:400]
            lines.append(f"{prefix}: {content}")
        lines.append("")
    lines.append("[User question]")
    lines.append(question)
    return "\n".join(lines)


def _recent_history(history: list[AskTurn] | None, turns: int) -> list[AskTurn]:
    if not history or turns <= 0:
        return []
    cleaned = [item for item in history if item.content.strip()]
    return cleaned[-max(turns, 1) * 2 :]


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


def _verify_quote(item: Evidence, paper_id: str, knowledge: KnowledgeService) -> Evidence:
    if not item.section_id:
        return item
    try:
        sections = knowledge.get_sections(paper_id)
    except PaperNotFoundError:
        return item.model_copy(update={"sourced": False})
    match = next((section for section in sections if section.section_id == item.section_id), None)
    if match is None:
        return item.model_copy(update={"sourced": False})
    sourced = _normalized(item.quote.rstrip("…")) in _normalized(match.text)
    if sourced == item.sourced:
        return item
    return item.model_copy(update={"sourced": sourced})


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LlmRequestError("问答结果不是合法 JSON") from exc
    if not isinstance(data, dict):
        raise LlmRequestError("问答结果不是 JSON 对象")
    return data


def build_intro(paper_id: str) -> PaperIntro:
    _ = paper_id
    raise NotImplementedError("understand.build_intro is scheduled for M2")


def build_method(paper_id: str) -> MethodExplain:
    _ = paper_id
    raise NotImplementedError("understand.build_method is scheduled for M2")
