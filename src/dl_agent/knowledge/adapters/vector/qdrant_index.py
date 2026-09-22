from __future__ import annotations

import logging
import uuid
from pathlib import Path

from dl_agent.domain.models import ChildChunk
from dl_agent.knowledge.adapters.vector.base import SparseEmbedding, VectorHit

logger = logging.getLogger(__name__)

DENSE_NAME = "dense"
SPARSE_NAME = "sparse"


def _point_id(paper_id: str, chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{paper_id}:{chunk_id}"))


class QdrantVectorIndex:
    """本地 Qdrant：unnamed 语义上的 named dense + BM25 sparse，query 侧 RRF。"""

    def __init__(self, path: Path, collection: str, dense_name: str = DENSE_NAME, sparse_name: str = SPARSE_NAME):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.dense_name = dense_name
        self.sparse_name = sparse_name
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))

    def upsert(
        self,
        paper_id: str,
        chunks: list[ChildChunk],
        dense: list[list[float]],
        sparse: list[SparseEmbedding] | None = None,
    ) -> None:
        if not chunks:
            return
        sparse = sparse or [SparseEmbedding([], []) for _ in chunks]
        self._ensure_collection(len(dense[0]))
        from qdrant_client.http import models as rest

        points = []
        for chunk, vector, sparse_vec in zip(chunks, dense, sparse, strict=True):
            points.append(
                rest.PointStruct(
                    id=_point_id(paper_id, chunk.chunk_id),
                    vector={
                        self.dense_name: vector,
                        self.sparse_name: rest.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                    },
                    payload={
                        "paper_id": paper_id,
                        "chunk_id": chunk.chunk_id,
                        "section_id": chunk.section_id,
                        "section_kind": chunk.section_kind,
                    },
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def delete_paper(self, paper_id: str) -> None:
        from qdrant_client.http import models as rest

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=rest.FilterSelector(
                    filter=_paper_filter(paper_id, rest),
                ),
            )
        except Exception:
            logger.info("qdrant delete skipped collection=%s paper_id=%s", self.collection, paper_id)

    def search_hybrid(
        self,
        paper_id: str | None,
        query_dense: list[float],
        query_sparse: SparseEmbedding | None,
        k: int,
        kinds: list[str] | None = None,
    ) -> list[VectorHit]:
        from qdrant_client.http import models as rest

        self._ensure_collection(len(query_dense))
        limit = max(1, k)
        flt = _hit_filter(paper_id, kinds, rest)
        sparse_query = rest.SparseVector(
            indices=(query_sparse.indices if query_sparse else []),
            values=(query_sparse.values if query_sparse else []),
        )
        prefetch_kw: dict = {
            "using": self.dense_name,
            "query": query_dense,
            "limit": limit,
        }
        sparse_kw: dict = {
            "using": self.sparse_name,
            "query": sparse_query,
            "limit": limit,
        }
        if flt is not None:
            prefetch_kw["filter"] = flt
            sparse_kw["filter"] = flt
        result = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                rest.Prefetch(**prefetch_kw),
                rest.Prefetch(**sparse_kw),
            ],
            query=rest.FusionQuery(fusion=rest.Fusion.RRF),
            limit=limit,
        )
        out: list[VectorHit] = []
        for hit in result.points:
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id")
            hit_paper = str(payload.get("paper_id") or paper_id or "")
            if not chunk_id or not hit_paper:
                continue
            out.append(
                VectorHit(
                    paper_id=hit_paper,
                    chunk_id=str(chunk_id),
                    score=float(hit.score),
                )
            )
        return out

    def _ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.http import models as rest

        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.dense_name: rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            },
            sparse_vectors_config={
                self.sparse_name: rest.SparseVectorParams(),
            },
        )
        for field in ("paper_id", "section_id", "section_kind"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )


def _paper_filter(paper_id: str, rest):
    return rest.Filter(
        must=[
            rest.FieldCondition(key="paper_id", match=rest.MatchValue(value=paper_id)),
        ]
    )


def _hit_filter(paper_id: str | None, kinds: list[str] | None, rest):
    must = []
    if paper_id:
        must.append(rest.FieldCondition(key="paper_id", match=rest.MatchValue(value=paper_id)))
    if kinds:
        must.append(
            rest.FieldCondition(key="section_kind", match=rest.MatchAny(any=kinds))
        )
    if not must:
        return None
    return rest.Filter(must=must)
