from __future__ import annotations

from dataclasses import dataclass, field

from dl_agent.domain.models import ParserName
from dl_agent.knowledge.layout import LayoutItem


@dataclass
class ParseResult:
    parser: ParserName
    page_count: int
    items: list[LayoutItem]
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    language: str = "eng"

    @property
    def char_count(self) -> int:
        return sum(len(item.text) for item in self.items if item.kind in {"text", "table", "caption", "heading"})
