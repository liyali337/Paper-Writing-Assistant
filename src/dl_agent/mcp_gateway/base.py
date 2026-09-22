"""MCP / 外部工具的内部契约。业务节点只认 name + ToolResult，不写死第三方 SDK。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from dl_agent.domain.models import ExternalRef


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    source: str = ""


@dataclass
class ToolResult:
    ok: bool
    text: str = ""
    refs: list[ExternalRef] = field(default_factory=list)
    sentinel: str | None = None

    def observation(self) -> str:
        if self.sentinel:
            return self.sentinel
        return (self.text or "").strip()


class McpGateway(Protocol):
    def list_tools(self) -> list[ToolSpec]: ...

    def call(self, name: str, args: dict[str, Any]) -> ToolResult: ...
