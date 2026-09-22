from __future__ import annotations

import math
from collections import defaultdict

from dl_agent.domain.models import ChildChunk
from dl_agent.knowledge.adapters.vector.base import SparseEmbedding, VectorHit


class MemoryVectorIndex:
    """进程内稠密 + 稀疏表，单测用；融合方式与 Qdrant RRF 相同。"""

    def __init__(self) -> None:
        self._dense: dict[str, dict[str, list[float]]] = {}
        self._sparse: dict[str, dict[str, SparseEmbedding]] = {}
        self._kinds: dict[str, dict[str, str]] = {}

    def upsert(
        self,
        paper_id: str,
        chunks: list[ChildChunk],
        dense: list[list[float]],
        sparse: list[SparseEmbedding] | None = None,
    ) -> None:
        d_bucket = self._dense.setdefault(paper_id, {})
        s_bucket = self._sparse.setdefault(paper_id, {})
        k_bucket = self._kinds.setdefault(paper_id, {})
        sparse = sparse or [SparseEmbedding([], []) for _ in chunks]
        for chunk, vector, sparse_vec in zip(chunks, dense, sparse, strict=True):
            d_bucket[chunk.chunk_id] = vector
            s_bucket[chunk.chunk_id] = sparse_vec
            k_bucket[chunk.chunk_id] = chunk.section_kind

    def delete_paper(self, paper_id: str) -> None:
        self._dense.pop(paper_id, None)
        self._sparse.pop(paper_id, None)
        self._kinds.pop(paper_id, None)

    def search_hybrid(
        self,
        paper_id: str | None,
        query_dense: list[float],
        query_sparse: SparseEmbedding | None,
        k: int,
        kinds: list[str] | None = None,
    ) -> list[VectorHit]:
        papers = [paper_id] if paper_id else list(self._dense.keys())
        allowed = set(kinds) if kinds else None
        keys: list[tuple[str, str]] = []
        for pid in papers:
            d_bucket = self._dense.get(pid) or {}
            k_bucket = self._kinds.get(pid) or {}
            for chunk_id in d_bucket:
                if allowed is None or k_bucket.get(chunk_id) in allowed:
                    keys.append((pid, chunk_id))
        dense_ranked = sorted(
            (
                (pid, chunk_id, _cosine(query_dense, self._dense[pid][chunk_id]))
                for pid, chunk_id in keys
            ),
            key=lambda item: item[2],
            reverse=True,
        )
        sparse_ranked = sorted(
            (
                (
                    pid,
                    chunk_id,
                    _sparse_dot(query_sparse, self._sparse.get(pid, {}).get(chunk_id)),
                )
                for pid, chunk_id in keys
            ),
            key=lambda item: item[2],
            reverse=True,
        )
        scores: dict[tuple[str, str], float] = defaultdict(float)
        for rank, (pid, chunk_id, _) in enumerate(dense_ranked, start=1):
            scores[(pid, chunk_id)] += 1.0 / (60 + rank)
        for rank, (pid, chunk_id, score) in enumerate(sparse_ranked, start=1):
            if score <= 0:
                continue
            scores[(pid, chunk_id)] += 1.0 / (60 + rank)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            VectorHit(paper_id=pid, chunk_id=chunk_id, score=score)
            for (pid, chunk_id), score in ordered[: max(1, k)]
        ]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    na = math.sqrt(sum(a * a for a in left))
    nb = math.sqrt(sum(b * b for b in right))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _sparse_dot(query: SparseEmbedding | None, doc: SparseEmbedding | None) -> float:
    if query is None or doc is None or not query.indices or not doc.indices:
        return 0.0
    right = dict(zip(doc.indices, doc.values, strict=True))
    return sum(value * right[index] for index, value in zip(query.indices, query.values, strict=True) if index in right)
