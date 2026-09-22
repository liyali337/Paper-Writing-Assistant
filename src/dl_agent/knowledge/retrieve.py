"""按问题从 Child 块里选证据。无向量库时走词法 BM25。"""

from __future__ import annotations

import math
import re
from collections import Counter

from dl_agent.domain.models import ChildChunk, Evidence, Section
from dl_agent.knowledge.chunk import build_chunks

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{1,2}")
_LATIN = re.compile(r"[a-z0-9]")
_FALLBACK_KINDS = ("abstract", "intro", "method", "experiment", "conclusion")
_QUOTE_MAX = 420


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text or "")]


def retrieve_evidence(
    sections: list[Section],
    question: str,
    *,
    k: int = 8,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Evidence]:
    chunks = build_chunks(sections, chunk_size=chunk_size, overlap=overlap)
    return retrieve_from_chunks(chunks, question, k=k)


def retrieve_from_chunks(
    chunks: list[ChildChunk],
    question: str,
    *,
    k: int = 8,
    kinds: list[str] | None = None,
) -> list[Evidence]:
    if kinds:
        allowed = set(kinds)
        chunks = [item for item in chunks if item.section_kind in allowed]
    if not chunks:
        return []
    limit = max(1, k)
    query_tokens = tokenize(question)
    latin = [token for token in query_tokens if _LATIN.search(token)]
    scored = _score_chunks(chunks, latin or query_tokens)
    selected = _pick_chunks(chunks, scored, limit=limit, require_score=bool(latin))
    return [to_evidence(chunk, score) for chunk, score in selected]


def retrieve_corpus_from_chunks(
    chunks: list[ChildChunk],
    question: str,
    *,
    paper_limit: int = 6,
    chunks_per_paper: int = 2,
) -> list[tuple[str, list[tuple[ChildChunk, float]]]]:
    """跨篇词法检索：先给块打分，再按 paper_id 聚成 Top 论文。"""
    if not chunks:
        return []
    query_tokens = tokenize(question)
    latin = [token for token in query_tokens if _LATIN.search(token)]
    scored = _score_chunks(chunks, latin or query_tokens)
    require_score = bool(latin)
    per_paper: dict[str, list[tuple[ChildChunk, float]]] = {}
    for score, chunk in scored:
        if require_score and score <= 0:
            continue
        bucket = per_paper.setdefault(chunk.paper_id, [])
        if len(bucket) >= max(1, chunks_per_paper):
            continue
        bucket.append((chunk, score))
    ranked = sorted(
        per_paper.items(),
        key=lambda item: item[1][0][1] if item[1] else 0.0,
        reverse=True,
    )
    return ranked[: max(1, paper_limit)]


def lexical_ranks(chunks: list[ChildChunk], question: str) -> list[str]:
    query_tokens = tokenize(question)
    latin = [token for token in query_tokens if _LATIN.search(token)]
    scored = _score_chunks(chunks, latin or query_tokens)
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk.chunk_id for score, chunk in scored if score > 0 or not latin]


def _score_chunks(chunks: list[ChildChunk], query_tokens: list[str]) -> list[tuple[float, ChildChunk]]:
    if not query_tokens:
        return [(0.0, chunk) for chunk in chunks]
    tokenized = [tokenize(f"{chunk.section_title} {chunk.text}") for chunk in chunks]
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    n_docs = len(chunks)
    avgdl = max(sum(len(tokens) for tokens in tokenized) / n_docs, 1.0)
    idf = {
        term: math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
        for term in set(query_tokens)
    }
    ranked: list[tuple[float, ChildChunk]] = []
    k1, b = 1.5, 0.75
    for chunk, tokens in zip(chunks, tokenized, strict=True):
        tf = Counter(tokens)
        length = max(len(tokens), 1)
        score = 0.0
        for term in set(query_tokens):
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + k1 * (1 - b + b * length / avgdl)
            score += idf.get(term, 0.0) * (freq * (k1 + 1)) / denom
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _pick_chunks(
    chunks: list[ChildChunk],
    scored: list[tuple[float, ChildChunk]],
    *,
    limit: int,
    require_score: bool,
) -> list[tuple[ChildChunk, float]]:
    hits = [(chunk, score) for score, chunk in scored if (score > 0 or not require_score)]
    if require_score:
        hits = [(chunk, score) for chunk, score in hits if score > 0]
    picked: list[tuple[ChildChunk, float]] = []
    seen: set[str] = set()
    for chunk, score in hits:
        if chunk.chunk_id in seen:
            continue
        picked.append((chunk, score))
        seen.add(chunk.chunk_id)
        if len(picked) >= limit:
            return picked
    score_map = {chunk.chunk_id: score for score, chunk in scored}
    for kind in _FALLBACK_KINDS:
        for chunk in chunks:
            if chunk.section_kind != kind or chunk.chunk_id in seen or chunk.order != 0:
                continue
            picked.append((chunk, score_map.get(chunk.chunk_id, 0.0)))
            seen.add(chunk.chunk_id)
            if len(picked) >= limit:
                return picked
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        picked.append((chunk, score_map.get(chunk.chunk_id, 0.0)))
        seen.add(chunk.chunk_id)
        if len(picked) >= limit:
            break
    return picked


def to_evidence(chunk: ChildChunk, score: float) -> Evidence:
    quote = re.sub(r"\s+", " ", chunk.text).strip()
    if len(quote) > _QUOTE_MAX:
        quote = quote[:_QUOTE_MAX].rstrip() + "…"
    return Evidence(
        page=chunk.page_start,
        section_title=chunk.section_title,
        quote=quote,
        sourced=True,
        section_id=chunk.section_id,
        score=round(score, 4),
        figure_ids=list(chunk.figure_ids),
        chunk_id=chunk.chunk_id,
    )


def rrf_merge(*rank_lists: list[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, chunk_id in enumerate(ranks, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)]
