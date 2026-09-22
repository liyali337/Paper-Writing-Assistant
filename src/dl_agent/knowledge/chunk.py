"""把现有 Section 切成 Child。Parent 仍是章节，不再另切大块。"""

from __future__ import annotations

import hashlib

from dl_agent.domain.models import ChildChunk, Section
from dl_agent.knowledge.classify import is_references_heading, is_skip_heading

_SKIP_KINDS = {"references"}
_DROP_ORDER = ("other", "related", "conclusion", "abstract")


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    blob = (text or "").strip()
    if not blob:
        return []
    if len(blob) <= size:
        return [blob]
    parts: list[str] = []
    start = 0
    step_overlap = min(overlap, max(size - 1, 0))
    while start < len(blob):
        end = min(len(blob), start + size)
        if end < len(blob):
            end = _prefer_break(blob, start, end, size)
        piece = blob[start:end].strip()
        if piece:
            parts.append(piece)
        if end >= len(blob):
            break
        start = max(end - step_overlap, start + 1)
    return parts


def _prefer_break(blob: str, start: int, end: int, size: int) -> int:
    lo = start + max(size // 2, 1)
    para = blob.rfind("\n\n", lo, end)
    if para > start:
        return para
    space = blob.rfind(" ", lo, end)
    if space > start:
        return space
    return end


def build_chunks(
    sections: list[Section],
    *,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[ChildChunk]:
    chunks: list[ChildChunk] = []
    for section in sections:
        if _skip_section(section):
            continue
        pieces = chunk_text(section.text, chunk_size, overlap)
        for index, piece in enumerate(pieces):
            chunks.append(
                ChildChunk(
                    chunk_id=f"{section.section_id}:{index}",
                    paper_id=section.paper_id,
                    section_id=section.section_id,
                    section_title=section.title,
                    section_kind=section.kind,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    text=piece,
                    figure_ids=list(section.figure_ids),
                    order=index,
                    source_hash=_piece_hash(section.section_id, index, piece),
                )
            )
    return chunks


def cap_chunks(chunks: list[ChildChunk], limit: int) -> list[ChildChunk]:
    if limit <= 0 or len(chunks) <= limit:
        return chunks
    keep = list(chunks)
    for kind in _DROP_ORDER:
        extra = len(keep) - limit
        if extra <= 0:
            break
        victims = [item for item in keep if item.section_kind == kind][-extra:]
        drop = {item.chunk_id for item in victims}
        keep = [item for item in keep if item.chunk_id not in drop]
    return keep[:limit]


def chunks_source_hash(chunks: list[ChildChunk], embedding_version: str) -> str:
    payload = embedding_version + "\n" + "\n".join(
        f"{item.chunk_id}\t{item.text}" for item in chunks
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _skip_section(section: Section) -> bool:
    if section.kind in _SKIP_KINDS:
        return True
    if is_skip_heading(section.title) or is_references_heading(section.title):
        return True
    return not (section.text or "").strip()


def _piece_hash(section_id: str, order: int, text: str) -> str:
    blob = f"{section_id}:{order}:{text}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
