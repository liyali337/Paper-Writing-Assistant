from __future__ import annotations

from dataclasses import dataclass

from dl_agent.domain.models import ChildChunk


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class VectorHit:
    paper_id: str
    chunk_id: str
    score: float


class VectorIndex:
    def upsert(
        self,
        paper_id: str,
        chunks: list[ChildChunk],
        dense: list[list[float]],
        sparse: list[SparseEmbedding] | None = None,
    ) -> None:
        raise NotImplementedError

    def delete_paper(self, paper_id: str) -> None:
        raise NotImplementedError

    def search_hybrid(
        self,
        paper_id: str | None,
        query_dense: list[float],
        query_sparse: SparseEmbedding | None,
        k: int,
        kinds: list[str] | None = None,
    ) -> list[VectorHit]:
        raise NotImplementedError
