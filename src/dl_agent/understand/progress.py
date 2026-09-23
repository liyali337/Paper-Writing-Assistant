"""问答过程事件。未绑定接收端时静默丢弃，非流式 ask 行为不变。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

ProgressHandler = Callable[[dict[str, Any]], None]
_handler: ContextVar[ProgressHandler | None] = ContextVar("ask_progress", default=None)

TOOL_LABELS: dict[str, str] = {
    "search_child_chunks": "检索文内片段",
    "retrieve_parent_chunks": "读取章节全文",
    "list_sections": "查看目录",
    "list_figures": "查看图表",
    "get_figure_caption": "读取图注",
    "search_captions": "检索图注",
    "get_translation": "读取译文",
    "arxiv_search": "检索 arXiv",
}


def bind_progress(handler: ProgressHandler | None) -> Token:
    return _handler.set(handler)


def reset_progress(token: Token) -> None:
    _handler.reset(token)


def emit_progress(event: dict[str, Any]) -> None:
    handler = _handler.get()
    if handler is None:
        return
    try:
        handler(event)
    except Exception:
        logger.exception("ask progress handler failed")


def emit_phase(phase: str, label: str) -> None:
    emit_progress({"type": "phase", "phase": phase, "label": label})


def emit_tool(
    *,
    name: str,
    status: str,
    call_id: str,
    args: dict[str, Any] | None = None,
    note: str = "",
) -> None:
    tool_name = name or "tool"
    emit_progress(
        {
            "type": "tool",
            "name": tool_name,
            "status": status,
            "label": TOOL_LABELS.get(tool_name, tool_name),
            "call_id": call_id,
            "detail": tool_detail(tool_name, args or {}),
            "note": note,
        }
    )


def tool_detail(name: str, args: dict[str, Any]) -> str:
    if name in {"search_child_chunks", "search_captions"}:
        raw = args.get("query")
    elif name == "retrieve_parent_chunks":
        raw = args.get("section_id") or args.get("parent_id")
    elif name == "list_sections":
        raw = args.get("kind")
    elif name == "list_figures":
        raw = args.get("section_id")
    elif name == "get_figure_caption":
        raw = args.get("figure_id")
    elif name == "get_translation":
        raw = args.get("section_id")
    elif name == "arxiv_search":
        raw = args.get("arxiv_id") or args.get("id") or args.get("query")
    else:
        raw = ""
    return _clip(str(raw or ""))


def outcome_note(content: str, hit_count: int = 0, *, skipped: str = "") -> str:
    if skipped == "duplicate":
        return "重复，已跳过"
    if skipped == "invalid":
        return "参数无效，已跳过"
    text = (content or "").strip()
    if text.startswith("UNKNOWN_TOOL"):
        return "未知工具"
    if text.startswith("NO_") or text.startswith("ARXIV_DISABLED"):
        return "没有命中"
    if hit_count > 0:
        return f"命中 {hit_count} 条"
    return "已完成"


def format_sse(event: dict[str, Any]) -> str:
    kind = str(event.get("type") or "message")
    if kind not in {"phase", "tool", "answer", "error"}:
        kind = "message"
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("\r", " ").replace("\n", " ")
    return f"event: {kind}\ndata: {payload}\n\n"


def _clip(text: str, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
