from pathlib import Path

from dl_agent.knowledge.adapters.vector.base import SparseEmbedding, VectorHit, VectorIndex
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex

__all__ = [
    "SparseEmbedding",
    "VectorHit",
    "VectorIndex",
    "MemoryVectorIndex",
    "try_qdrant_index",
]


def try_qdrant_index(path, collection: str):
    try:
        from dl_agent.knowledge.adapters.vector.qdrant_index import QdrantVectorIndex
    except ImportError:
        return None
    try:
        return QdrantVectorIndex(Path(path), collection)
    except Exception:
        return None
