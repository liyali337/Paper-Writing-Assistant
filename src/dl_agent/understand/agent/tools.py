"""论文内感知工具：检索 Child/Parent，以及目录、图注、已有译文。

对照教程仓 ToolFactory，但 paper_id 在构造时绑死，检索走 KnowledgeService.query，
不包 LangChain VectorStore，也不设 score_threshold=0.7。
"""

from __future__ import annotations

import re
from typing import Any

from dl_agent.domain.models import Evidence, ExternalRef, Figure, Section
from dl_agent.knowledge.retrieve import tokenize
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.mcp_gateway.base import McpGateway
from dl_agent.mcp_gateway.adapters.arxiv import ARXIV_TOOL_SPEC

_LATIN = re.compile(r"[a-z0-9]")

NO_RELEVANT_CHUNKS = "NO_RELEVANT_CHUNKS"
NO_PARENT_DOCUMENT = "NO_PARENT_DOCUMENT"
NO_SECTIONS = "NO_SECTIONS"
NO_FIGURES = "NO_FIGURES"
NO_FIGURE = "NO_FIGURE"
NO_CAPTION_HITS = "NO_CAPTION_HITS"
NO_TRANSLATION = "NO_TRANSLATION"
ARXIV_DISABLED = "ARXIV_DISABLED"
PARENT_MAX_CHARS = 4000
NEARBY_MAX_CHARS = 400
DEFAULT_SEARCH_LIMIT = 5
BOOTSTRAP_SEARCH_CALL_ID = "call_bootstrap_search"
_SECTION_KINDS = {
    "abstract",
    "intro",
    "related",
    "method",
    "experiment",
    "conclusion",
    "references",
    "other",
}

RETRIEVAL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_child_chunks",
            "description": "在当前这篇论文的 Child 切片里混合检索。query 用自包含英文或术语；不要指定其它论文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "自包含检索句"},
                    "limit": {
                        "type": "integer",
                        "description": "返回条数，默认 5",
                        "minimum": 1,
                        "maximum": 32,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_parent_chunks",
            "description": "按 section_id 取出该节 Parent 全文。section_id 必须来自已有工具结果。一次只取一节。",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "章节 id，例如 sec-012",
                    },
                },
                "required": ["section_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sections",
            "description": "列出当前论文章节目录（标题、kind、页码、是否有图），不含正文。问「方法在哪一节 / 先看什么」时用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "description": "可选过滤：abstract/intro/related/method/experiment/conclusion/references/other",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_figures",
            "description": "列出本篇或某节的图/表目录（figure_id、页、题注）。不要假设能看见 PNG 像素。",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_id": {
                        "type": "string",
                        "description": "可选，只列出该节的图",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_figure_caption",
            "description": "取单图题注及所在节邻近摘录。figure_id 必须来自 list_figures / search_captions。",
            "parameters": {
                "type": "object",
                "properties": {
                    "figure_id": {"type": "string", "description": "如图 fig-001"},
                },
                "required": ["figure_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_captions",
            "description": "在图题注、表注、label 里检索，适合 Figure 3 / Table 1 / architecture 对不上正文检索时。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "图号、表号或题注关键词"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_translation",
            "description": "读取某节已经生成的中译。未翻译时返回 NO_TRANSLATION，不要当作已开翻。",
            "parameters": {
                "type": "object",
                "properties": {
                    "section_id": {"type": "string", "description": "章节 id"},
                },
                "required": ["section_id"],
            },
        },
    },
]

ARXIV_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": ARXIV_TOOL_SPEC.name,
        "description": ARXIV_TOOL_SPEC.description,
        "parameters": ARXIV_TOOL_SPEC.parameters,
    },
}


def worker_tool_schemas(*, enable_external: bool = False) -> list[dict[str, Any]]:
    if enable_external:
        return [*RETRIEVAL_TOOL_SCHEMAS, ARXIV_SEARCH_SCHEMA]
    return list(RETRIEVAL_TOOL_SCHEMAS)


class RetrievalTools:
    """模型只能传 query / section_id / figure_id / kind；禁止指定搜全部论文。"""

    def __init__(
        self,
        knowledge: KnowledgeService,
        paper_id: str,
        *,
        parent_max_chars: int = PARENT_MAX_CHARS,
        default_limit: int = DEFAULT_SEARCH_LIMIT,
        gateway: McpGateway | None = None,
        enable_external: bool = False,
    ):
        self.knowledge = knowledge
        self.paper_id = paper_id
        self.parent_max_chars = max(1, parent_max_chars)
        self.default_limit = max(1, default_limit)
        self.gateway = gateway
        self.enable_external = enable_external
        self.last_child_hits: list[Evidence] = []
        self.last_parent_hits: list[Evidence] = []
        self.last_hits: list[Evidence] = []
        self.last_external_refs: list[ExternalRef] = []

    def search_child_chunks(self, query: str, limit: int | None = None) -> str:
        question = (query or "").strip()
        if not question:
            self.last_child_hits = []
            self.last_hits = []
            return NO_RELEVANT_CHUNKS
        k = limit if limit is not None else self.default_limit
        k = max(1, min(int(k), 32))
        try:
            hits = self.knowledge.query(self.paper_id, question, k=k)
        except Exception as exc:
            self.last_child_hits = []
            self.last_hits = []
            return f"RETRIEVAL_ERROR: {exc}"
        # Hybrid / RRF 总会 Top-k。英文问句仍要求词面落在摘录/节名里，避免乱入。
        # 纯中文问句对英文论文做不到词面重合，交给 query 的排序（含章节回退）。
        relevant = [item for item in hits if _grounded(question, item)]
        if not relevant:
            self.last_child_hits = []
            self.last_hits = []
            return NO_RELEVANT_CHUNKS
        self.last_child_hits = relevant
        self.last_hits = list(relevant)
        return "\n\n".join(_format_child(index, item) for index, item in enumerate(relevant, start=1))

    def retrieve_parent_chunks(self, section_id: str) -> str:
        sid = (section_id or "").strip()
        if not sid:
            self.last_parent_hits = []
            self.last_hits = []
            return NO_PARENT_DOCUMENT
        try:
            sections = self.knowledge.get_sections(self.paper_id)
        except Exception as exc:
            self.last_parent_hits = []
            self.last_hits = []
            return f"PARENT_RETRIEVAL_ERROR: {exc}"
        match = next((item for item in sections if item.section_id == sid), None)
        if match is None or not (match.text or "").strip():
            self.last_parent_hits = []
            self.last_hits = []
            return NO_PARENT_DOCUMENT
        self.last_parent_hits = [_parent_evidence(match)]
        self.last_hits = list(self.last_parent_hits)
        return _format_parent(match, self.parent_max_chars)

    def list_sections(self, kind: str | None = None) -> str:
        self.last_hits = []
        try:
            sections = self.knowledge.get_sections(self.paper_id)
        except Exception as exc:
            return f"SECTIONS_ERROR: {exc}"
        wanted = (kind or "").strip().lower()
        if wanted:
            if wanted not in _SECTION_KINDS:
                return NO_SECTIONS
            sections = [item for item in sections if item.kind == wanted]
        if not sections:
            return NO_SECTIONS
        return "\n".join(_format_section_row(index, item) for index, item in enumerate(sections, start=1))

    def list_figures(self, section_id: str | None = None) -> str:
        self.last_hits = []
        sid = (section_id or "").strip() or None
        try:
            figures = self.knowledge.get_figures(self.paper_id, sid)
        except Exception as exc:
            return f"FIGURES_ERROR: {exc}"
        if not figures:
            return NO_FIGURES
        return "\n\n".join(_format_figure_row(index, item) for index, item in enumerate(figures, start=1))

    def get_figure_caption(self, figure_id: str) -> str:
        fid = (figure_id or "").strip()
        if not fid:
            self.last_hits = []
            return NO_FIGURE
        try:
            figures = self.knowledge.get_figures(self.paper_id)
            sections = self.knowledge.get_sections(self.paper_id)
        except Exception as exc:
            self.last_hits = []
            return f"FIGURE_ERROR: {exc}"
        match = next((item for item in figures if item.figure_id == fid), None)
        if match is None:
            self.last_hits = []
            return NO_FIGURE
        section = next((item for item in sections if item.section_id == match.section_id), None)
        self.last_hits = [_caption_evidence(match, section)]
        return _format_figure_caption(match, section)

    def search_captions(self, query: str) -> str:
        question = (query or "").strip()
        if not question:
            self.last_hits = []
            return NO_CAPTION_HITS
        try:
            figures = self.knowledge.get_figures(self.paper_id)
            sections = {item.section_id: item for item in self.knowledge.get_sections(self.paper_id)}
        except Exception as exc:
            self.last_hits = []
            return f"CAPTION_SEARCH_ERROR: {exc}"
        hits = [item for item in figures if _caption_matches(question, item)]
        if not hits:
            self.last_hits = []
            return NO_CAPTION_HITS
        hits = hits[:8]
        self.last_hits = [
            _caption_evidence(item, sections.get(item.section_id) if item.section_id else None)
            for item in hits
        ]
        return "\n\n".join(_format_figure_row(index, item) for index, item in enumerate(hits, start=1))

    def get_translation(self, section_id: str) -> str:
        self.last_hits = []
        sid = (section_id or "").strip()
        if not sid:
            return NO_TRANSLATION
        try:
            self.knowledge.get_paper(self.paper_id)
            payload = self.knowledge.store.get_translation(self.paper_id)
        except Exception as exc:
            return f"TRANSLATION_ERROR: {exc}"
        if payload is None:
            return NO_TRANSLATION
        match = next((item for item in payload.sections if item.section_id == sid), None)
        text = (match.text_zh or "").strip() if match is not None else ""
        if not text:
            return NO_TRANSLATION
        if len(text) > self.parent_max_chars:
            text = text[: self.parent_max_chars].rstrip() + "…"
        title = (match.title_zh or "").strip() if match is not None else ""
        status = payload.status
        header = f"section_id={sid} status={status}"
        if title:
            header += f" title_zh={title}"
        return f"{header}\n{text}"

    def arxiv_search(
        self,
        query: str = "",
        arxiv_id: str = "",
        max_results: int | None = None,
    ) -> str:
        self.last_hits = []
        self.last_external_refs = []
        if not self.enable_external or self.gateway is None:
            return ARXIV_DISABLED
        args: dict[str, Any] = {}
        question = (query or "").strip()
        ident = (arxiv_id or "").strip()
        if question:
            args["query"] = question
        if ident:
            args["arxiv_id"] = ident
        if max_results is not None:
            args["max_results"] = int(max_results)
        result = self.gateway.call("arxiv_search", args)
        self.last_external_refs = list(result.refs)
        return result.observation() or ARXIV_DISABLED


def _grounded(query: str, item: Evidence) -> bool:
    latin = {token for token in tokenize(query) if _LATIN.search(token)}
    if not latin:
        return True
    haystack = set(tokenize(f"{item.section_title or ''} {item.quote}"))
    return bool(latin & haystack)


def _caption_matches(query: str, figure: Figure) -> bool:
    hay = f"{figure.label or ''} {figure.caption or ''} {figure.kind} {figure.figure_id}"
    lowered = hay.lower()
    needle = query.lower().strip()
    if needle and needle in lowered:
        return True
    tokens = tokenize(query)
    if not tokens:
        return False
    return bool(set(tokens) & set(tokenize(hay)))


def _format_child(index: int, item: Evidence) -> str:
    title = item.section_title or "Untitled"
    section_id = item.section_id or ""
    return f"[{index}] section_id={section_id} title={title} page={item.page}\n{item.quote}"


def _format_section_row(index: int, section: Section) -> str:
    figures = len(section.figure_ids or [])
    return (
        f"[{index}] section_id={section.section_id} kind={section.kind} "
        f"level={section.level} page={section.page_start}-{section.page_end} "
        f"figures={figures} title={section.title}"
    )


def _format_figure_row(index: int, figure: Figure) -> str:
    caption = (figure.caption or "").strip() or "(no caption)"
    label = (figure.label or "").strip()
    extra = f" label={label}" if label else ""
    section_id = figure.section_id or ""
    return (
        f"[{index}] figure_id={figure.figure_id} kind={figure.kind} "
        f"page={figure.page} section_id={section_id}{extra}\n{caption}"
    )


def _format_figure_caption(figure: Figure, section: Section | None) -> str:
    caption = (figure.caption or "").strip() or "(no caption)"
    title = section.title if section is not None else ""
    section_id = figure.section_id or ""
    nearby = _nearby_excerpt(section, caption)
    lines = [
        f"figure_id={figure.figure_id} kind={figure.kind} page={figure.page} "
        f"section_id={section_id} title={title}",
        f"caption: {caption}",
    ]
    if nearby:
        lines.append(f"nearby: {nearby}")
    return "\n".join(lines)


def _nearby_excerpt(section: Section | None, caption: str) -> str:
    if section is None:
        return ""
    text = (section.text or "").strip()
    if not text:
        return ""
    needle = (caption or "").strip()
    if needle and needle != "(no caption)":
        index = text.lower().find(needle.lower()[:80])
        if index >= 0:
            start = max(0, index - 80)
            end = min(len(text), index + NEARBY_MAX_CHARS)
            excerpt = text[start:end].strip()
            if start > 0:
                excerpt = "…" + excerpt
            if end < len(text):
                excerpt = excerpt.rstrip() + "…"
            return excerpt
    if len(text) > NEARBY_MAX_CHARS:
        return text[:NEARBY_MAX_CHARS].rstrip() + "…"
    return text


def _parent_evidence(section: Section) -> Evidence:
    text = (section.text or "").strip()
    quote = text if len(text) <= 420 else text[:420].rstrip() + "…"
    return Evidence(
        page=section.page_start,
        section_title=section.title,
        quote=quote,
        sourced=True,
        section_id=section.section_id,
        figure_ids=list(section.figure_ids or []),
        chunk_id=f"{section.section_id}:parent",
    )


def _caption_evidence(figure: Figure, section: Section | None) -> Evidence:
    quote = (figure.caption or "").strip() or (figure.label or figure.figure_id)
    title = section.title if section is not None else None
    return Evidence(
        page=figure.page,
        section_title=title,
        quote=quote[:420],
        sourced=bool((figure.caption or "").strip()),
        section_id=figure.section_id,
        figure_ids=[figure.figure_id],
        chunk_id=f"{figure.figure_id}:caption",
    )


def _format_parent(section: Section, max_chars: int) -> str:
    text = (section.text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return (
        f"section_id={section.section_id} title={section.title} "
        f"page={section.page_start}-{section.page_end}\n{text}"
    )
