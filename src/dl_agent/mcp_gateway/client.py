"""把各适配器收成一把 Gateway；业务只 call(name, args)。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from dl_agent.config import Settings
from dl_agent.mcp_gateway.adapters.arxiv import ARXIV_TOOL_SPEC, search_arxiv
from dl_agent.mcp_gateway.base import ToolResult, ToolSpec

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], ToolResult]


class RouterGateway:
    def __init__(
        self,
        handlers: dict[str, ToolHandler],
        specs: list[ToolSpec] | None = None,
    ):
        self._handlers = dict(handlers)
        self._specs = list(specs or [])

    def list_tools(self) -> list[ToolSpec]:
        return list(self._specs)

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(ok=False, sentinel=f"UNKNOWN_MCP_TOOL: {name}")
        try:
            return handler(dict(args or {}))
        except Exception:
            logger.exception("mcp tool %s failed", name)
            return ToolResult(ok=False, sentinel="MCP_ERROR")


def build_gateway(
    settings: Settings,
    *,
    arxiv_call: ToolHandler | None = None,
) -> RouterGateway:
    def _arxiv(args: dict[str, Any]) -> ToolResult:
        if arxiv_call is not None:
            return arxiv_call(args)
        return search_arxiv(
            args,
            timeout=settings.mcp_timeout_s,
            base_url=settings.arxiv_api_url,
        )

    return RouterGateway(
        {"arxiv_search": _arxiv},
        specs=[ARXIV_TOOL_SPEC],
    )
