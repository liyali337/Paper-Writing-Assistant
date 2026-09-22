"""测试用内存 Gateway：按标题/摘要/id 子串过滤，不打外网。"""

from __future__ import annotations

from typing import Any

from dl_agent.domain.models import ExternalRef
from dl_agent.mcp_gateway.adapters.arxiv import (
    ARXIV_TOOL_SPEC,
    DEFAULT_MAX_RESULTS,
    MAX_RESULTS_CAP,
    NO_ARXIV_HITS,
    format_arxiv_hits,
    normalize_arxiv_id,
)
from dl_agent.mcp_gateway.base import ToolResult, ToolSpec


class MemoryMcpGateway:
    def __init__(self, catalog: list[ExternalRef] | None = None):
        self.catalog = list(catalog or [])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[ToolSpec]:
        return [ARXIV_TOOL_SPEC]

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        payload = dict(args or {})
        self.calls.append((name, payload))
        if name != "arxiv_search":
            return ToolResult(ok=False, sentinel=f"UNKNOWN_MCP_TOOL: {name}")
        query = str(payload.get("query") or "").strip().lower()
        arxiv_id = normalize_arxiv_id(str(payload.get("arxiv_id") or payload.get("id") or ""))
        try:
            limit = int(payload.get("max_results") or DEFAULT_MAX_RESULTS)
        except (TypeError, ValueError):
            limit = DEFAULT_MAX_RESULTS
        limit = max(1, min(limit, MAX_RESULTS_CAP))
        if arxiv_id:
            hits = [
                item
                for item in self.catalog
                if normalize_arxiv_id(item.identifier or "") == arxiv_id
            ]
        elif query:
            hits = [item for item in self.catalog if query in _haystack(item)]
        else:
            return ToolResult(ok=True, sentinel=NO_ARXIV_HITS)
        hits = hits[:limit]
        if not hits:
            return ToolResult(ok=True, sentinel=NO_ARXIV_HITS)
        return ToolResult(ok=True, text=format_arxiv_hits(hits), refs=hits)


def _haystack(item: ExternalRef) -> str:
    return f"{item.title} {item.snippet} {item.identifier or ''}".lower()
