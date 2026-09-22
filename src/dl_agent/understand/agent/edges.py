"""Worker 条件边与主图改写扇出。"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Send

from dl_agent.config import Settings
from dl_agent.understand.agent.nodes import tokens_over_budget
from dl_agent.understand.agent.state import AskState, WorkerState

AfterOrch = Literal["tools", "fallback_response", "collect_answer"]
AfterTools = Literal["compress_context", "orchestrator"]
AfterDialogue = Literal["direct_reply", "dispatch"]


def route_after_orchestrator_call(state: WorkerState, settings: Settings) -> AfterOrch:
    iteration = int(state.get("iteration_count") or 0)
    tool_count = int(state.get("tool_call_count") or 0)
    if iteration >= settings.ask_max_iterations or tool_count >= settings.ask_max_tool_calls:
        return "fallback_response"
    messages = list(state.get("messages") or [])
    last = messages[-1] if messages else {}
    calls = last.get("tool_calls") or []
    if calls:
        return "tools"
    return "collect_answer"


def route_after_tools(state: WorkerState, settings: Settings) -> AfterTools:
    if tokens_over_budget(state, settings):
        return "compress_context"
    return "orchestrator"


def route_after_dialogue(state: AskState) -> AfterDialogue:
    if state.get("skip_workers"):
        return "direct_reply"
    return "dispatch"


def route_after_rewrite(state: AskState) -> list[Send]:
    original = (state.get("originalQuery") or "").strip()
    paper_id = state.get("paper_id") or ""
    tasks = [
        item
        for item in (state.get("plan_tasks") or [])
        if isinstance(item, dict) and str(item.get("question") or "").strip()
    ]
    if not tasks:
        questions = [item.strip() for item in (state.get("rewrittenQuestions") or []) if str(item).strip()]
        if not questions and original:
            questions = [original]
        tasks = [{"kind": "close_read", "question": question} for question in questions]
    return [
        Send(
            "worker",
            {
                "paper_id": paper_id,
                "question": str(task.get("question") or "").strip(),
                "question_index": index,
                "kind": str(task.get("kind") or "close_read"),
            },
        )
        for index, task in enumerate(tasks)
    ]
