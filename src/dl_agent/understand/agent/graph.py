"""Worker 子图 + 主图。"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from dl_agent.config import Settings
from dl_agent.domain.models import AskTurn, Evidence, ExternalRef, LibraryHit, PaperAnswer
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.mcp_gateway import build_gateway
from dl_agent.observability.langfuse import observe_span
from dl_agent.understand.agent.edges import (
    route_after_dialogue,
    route_after_orchestrator_call,
    route_after_rewrite,
    route_after_tools,
)
from dl_agent.understand.agent.nodes import (
    AskModel,
    WorkerModel,
    aggregate_answers,
    bootstrap_search,
    collect_answer,
    compress_context,
    dialogue_query,
    direct_reply,
    fallback_response,
    orchestrator,
    run_tools,
)
from dl_agent.understand.agent.state import AskState, WorkerState
from dl_agent.understand.arxiv import ask_arxiv as run_arxiv_ask
from dl_agent.understand.library import ask_library as run_library_ask
from dl_agent.understand.agent.tools import RetrievalTools
from dl_agent.understand.citations import sanitize_answer_zh

WorkerFn = Callable[[dict[str, Any]], dict[str, Any]]
ChatFn = Callable[..., str]


def compile_worker_graph(tools: RetrievalTools, model: WorkerModel, settings: Settings):
    builder = StateGraph(WorkerState)
    builder.add_node("bootstrap_search", partial(bootstrap_search, tools=tools))
    builder.add_node("orchestrator", partial(orchestrator, model=model))
    builder.add_node("tools", partial(run_tools, tools=tools))
    builder.add_node("compress_context", partial(compress_context, model=model))
    builder.add_node("fallback_response", partial(fallback_response, model=model))
    builder.add_node("collect_answer", collect_answer)

    builder.add_edge(START, "bootstrap_search")
    builder.add_edge("bootstrap_search", "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        lambda state: route_after_orchestrator_call(state, settings),
        {
            "tools": "tools",
            "fallback_response": "fallback_response",
            "collect_answer": "collect_answer",
        },
    )
    builder.add_conditional_edges(
        "tools",
        lambda state: route_after_tools(state, settings),
        {"compress_context": "compress_context", "orchestrator": "orchestrator"},
    )
    builder.add_edge("compress_context", "orchestrator")
    builder.add_edge("fallback_response", "collect_answer")
    builder.add_edge("collect_answer", END)
    return builder.compile()


def empty_worker_state(question: str, question_index: int = 0, paper_id: str = "") -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "question": question,
        "question_index": question_index,
        "messages": [],
        "context_summary": "",
        "retrieval_keys": [],
        "evidence_bag": [],
        "external_refs": [],
        "final_answer": "",
        "tool_call_count": 0,
        "iteration_count": 0,
    }


def run_worker(
    tools: RetrievalTools,
    model: WorkerModel,
    settings: Settings,
    question: str,
    *,
    question_index: int = 0,
    paper_id: str = "",
) -> WorkerState:
    graph = compile_worker_graph(tools, model, settings)
    return graph.invoke(
        empty_worker_state(question, question_index, paper_id),
        {"recursion_limit": 50},
    )


def default_worker_node(
    state: dict[str, Any],
    knowledge: KnowledgeService,
    model: WorkerModel,
    settings: Settings,
) -> dict[str, Any]:
    paper_id = state.get("paper_id") or ""
    question = state.get("question") or ""
    index = int(state.get("question_index") or 0)
    tools = RetrievalTools(
        knowledge,
        paper_id,
        gateway=build_gateway(settings) if settings.ask_enable_external else None,
        enable_external=settings.ask_enable_external,
    )
    with observe_span(
        "worker",
        input={"paper_id": paper_id, "question": question, "index": index},
    ):
        result = run_worker(tools, model, settings, question, question_index=index, paper_id=paper_id)
    return {
        "agent_answers": [
            {
                "index": index,
                "question": question,
                "kind": "close_read",
                "answer": result.get("final_answer") or "",
                "evidence": list(result.get("evidence_bag") or []),
                "external_refs": list(result.get("external_refs") or []),
            }
        ]
    }


def library_worker_node(
    state: dict[str, Any],
    knowledge: KnowledgeService,
    settings: Settings,
    chat_fn: ChatFn,
) -> dict[str, Any]:
    question = state.get("question") or ""
    index = int(state.get("question_index") or 0)
    paper_id = state.get("paper_id") or ""
    result = run_library_ask(
        knowledge,
        question,
        [],
        settings=settings,
        chat_fn=chat_fn,
        exclude_paper_id=paper_id or None,
        current_paper_id=paper_id or None,
    )
    return {"agent_answers": [_aux_answer(index, question, "library", result)]}


def arxiv_worker_node(
    state: dict[str, Any],
    knowledge: KnowledgeService,
    settings: Settings,
    chat_fn: ChatFn,
    gateway: Any = None,
) -> dict[str, Any]:
    question = state.get("question") or ""
    index = int(state.get("question_index") or 0)
    paper_id = state.get("paper_id") or ""
    result = run_arxiv_ask(
        knowledge,
        question,
        [],
        settings=settings,
        chat_fn=chat_fn,
        gateway=gateway,
        current_paper_id=paper_id or None,
    )
    return {"agent_answers": [_aux_answer(index, question, "arxiv", result)]}


def _aux_answer(index: int, question: str, kind: str, result: PaperAnswer) -> dict[str, Any]:
    return {
        "index": index,
        "question": question,
        "kind": kind,
        "answer": result.answer_zh,
        "evidence": [],
        "external_refs": list(result.external_refs or []),
        "library_hits": list(result.library_hits or []),
        "no_evidence": bool(result.no_evidence),
    }


def dispatch_worker(
    state: dict[str, Any],
    *,
    close_read_fn: WorkerFn,
    knowledge: KnowledgeService | None,
    settings: Settings,
    chat_fn: ChatFn | None,
    gateway: Any = None,
) -> dict[str, Any]:
    kind = str(state.get("kind") or "close_read")
    if kind in {"library", "arxiv"}:
        if knowledge is not None and chat_fn is not None:
            if kind == "library":
                return library_worker_node(state, knowledge, settings, chat_fn)
            return arxiv_worker_node(state, knowledge, settings, chat_fn, gateway)
        return {
            "agent_answers": [
                {
                    "index": int(state.get("question_index") or 0),
                    "question": state.get("question") or "",
                    "kind": kind,
                    "answer": "",
                    "evidence": [],
                    "external_refs": [],
                    "library_hits": [],
                    "no_evidence": True,
                }
            ]
        }
    return close_read_fn(state)


def compile_ask_graph(
    model: AskModel,
    settings: Settings,
    *,
    knowledge: KnowledgeService | None = None,
    worker_fn: WorkerFn | None = None,
    chat_fn: ChatFn | None = None,
    gateway: Any = None,
):
    resolved = worker_fn
    if resolved is None:
        if knowledge is None:
            raise ValueError("compile_ask_graph 需要 knowledge 或 worker_fn")

        def resolved(state: dict[str, Any], _k=knowledge, _m=model, _s=settings) -> dict[str, Any]:
            return default_worker_node(state, _k, _m, _s)

    def routed(state: dict[str, Any], _close=resolved, _k=knowledge, _s=settings, _c=chat_fn, _g=gateway) -> dict[str, Any]:
        return dispatch_worker(
            state,
            close_read_fn=_close,
            knowledge=_k,
            settings=_s,
            chat_fn=_c,
            gateway=_g,
        )

    builder = StateGraph(AskState)
    builder.add_node("dialogue", partial(dialogue_query, model=model, settings=settings))
    builder.add_node("direct_reply", direct_reply)
    builder.add_node("dispatch", lambda _state: {})
    builder.add_node("worker", routed)
    builder.add_node("aggregate_answers", partial(aggregate_answers, model=model))

    builder.add_edge(START, "dialogue")
    builder.add_conditional_edges(
        "dialogue",
        route_after_dialogue,
        {"direct_reply": "direct_reply", "dispatch": "dispatch"},
    )
    builder.add_conditional_edges("dispatch", route_after_rewrite)
    builder.add_edge("direct_reply", END)
    builder.add_edge(["worker"], "aggregate_answers")
    builder.add_edge("aggregate_answers", END)
    return builder.compile()


def run_ask(
    model: AskModel,
    settings: Settings,
    paper_id: str,
    question: str,
    history: list[AskTurn] | None = None,
    *,
    knowledge: KnowledgeService | None = None,
    worker_fn: WorkerFn | None = None,
    chat_fn: ChatFn | None = None,
    gateway: Any = None,
    index_status: str = "",
) -> AskState:
    graph = compile_ask_graph(
        model,
        settings,
        knowledge=knowledge,
        worker_fn=worker_fn,
        chat_fn=chat_fn,
        gateway=gateway,
    )
    return graph.invoke(
        {
            "paper_id": paper_id,
            "originalQuery": question,
            "history": list(history or []),
            "conversation_summary": "",
            "dialogue_intent": "",
            "skip_workers": False,
            "rewrittenQuestions": [],
            "plan_tasks": [],
            "index_status": index_status,
            "agent_answers": [],
            "answer_zh": "",
            "citations": [],
            "external_refs": [],
            "library_hits": [],
            "mode": "close_read",
            "no_evidence": False,
            "partial": False,
        },
        {"recursion_limit": 50},
    )


def paper_answer_from_state(
    state: AskState,
    *,
    paper_id: str,
    question: str,
    model_name: str,
    prompt_version: str,
    embedding_version: str,
    citations: list[Evidence] | None = None,
    external_refs: list[ExternalRef] | None = None,
    library_hits: list[LibraryHit] | None = None,
) -> PaperAnswer:
    used = citations if citations is not None else list(state.get("citations") or [])
    refs = external_refs if external_refs is not None else list(state.get("external_refs") or [])
    hits = library_hits if library_hits is not None else list(state.get("library_hits") or [])
    figure_ids: list[str] = []
    for item in used:
        for figure_id in item.figure_ids:
            if figure_id not in figure_ids:
                figure_ids.append(figure_id)
    mode = state.get("mode") or "close_read"
    if mode not in {"close_read", "library", "arxiv"}:
        mode = "close_read"
    return PaperAnswer(
        paper_id=paper_id,
        question=question,
        answer_zh=sanitize_answer_zh(state.get("answer_zh") or ""),
        citations=used,
        figure_ids=figure_ids[:8],
        external_refs=refs,
        library_hits=hits,
        mode=mode,  # type: ignore[arg-type]
        no_evidence=bool(state.get("no_evidence")),
        partial=bool(state.get("partial")),
        model=model_name,
        prompt_version=prompt_version,
        embedding_version=embedding_version,
    )
