"""本地稠密 + BM25 稀疏编码器，对齐 tutorial-agentic-rag 的 hybrid 形态。"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from dl_agent.config import Settings
from dl_agent.knowledge.adapters.vector.base import SparseEmbedding
from dl_agent.knowledge.adapters.vector.embed import embed_texts

logger = logging.getLogger(__name__)

DenseFn = Callable[[list[str]], list[list[float]]]
SparseFn = Callable[[list[str]], list[SparseEmbedding]]


class HybridEncoder:
    def __init__(
        self,
        dense_documents: DenseFn,
        dense_query: Callable[[str], list[float]],
        sparse_documents: SparseFn,
        sparse_query: Callable[[str], SparseEmbedding],
        version: str,
    ):
        self._dense_documents = dense_documents
        self._dense_query = dense_query
        self._sparse_documents = sparse_documents
        self._sparse_query = sparse_query
        self.version = version

    def embed_dense_documents(self, texts: list[str]) -> list[list[float]]:
        return self._dense_documents(texts)

    def embed_dense_query(self, text: str) -> list[float]:
        return self._dense_query(text)

    def embed_sparse_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        return self._sparse_documents(texts)

    def embed_sparse_query(self, text: str) -> SparseEmbedding:
        return self._sparse_query(text)


def try_hybrid_encoder(settings: Settings) -> HybridEncoder | None:
    sparse_docs, sparse_query = _try_fastembed_sparse(settings.sparse_model)
    if sparse_docs is None or sparse_query is None:
        return None
    dense_docs, dense_query = _try_dense(settings)
    if dense_docs is None or dense_query is None:
        return None
    return HybridEncoder(
        dense_docs,
        dense_query,
        sparse_docs,
        sparse_query,
        settings.embedding_version(),
    )


def _try_dense(settings: Settings) -> tuple[DenseFn, Callable[[str], list[float]]] | tuple[None, None]:
    if settings.embedding_configured():
        def documents(texts: list[str]) -> list[list[float]]:
            return embed_texts(texts, settings)

        def query(text: str) -> list[float]:
            return embed_texts([text], settings)[0]

        return documents, query
    _apply_hf_endpoint(settings)
    st = _try_sentence_transformer(settings.dense_model)
    if st is not None:
        return st
    found = _try_fastembed_dense(settings.dense_model)
    if found is None:
        return None, None
    return found


def _apply_hf_endpoint(settings: Settings) -> None:
    endpoint = settings.hf_endpoint.strip()
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint


def _try_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    model = _load_sentence_transformer(model_name, offline=True)
    if model is None:
        logger.info("dense model %s not in local cache, downloading", model_name)
        model = _load_sentence_transformer(model_name, offline=False)
    if model is None:
        logger.warning("failed to load dense model %s", model_name)
        return None
    logger.info("loaded dense model %s", model_name)

    def documents(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    def query(text: str) -> list[float]:
        return model.encode([text], normalize_embeddings=True)[0].tolist()

    return documents, query


def _load_sentence_transformer(model_name: str, *, offline: bool):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            return SentenceTransformer(model_name, local_files_only=True)
        return SentenceTransformer(model_name, local_files_only=False)
    except Exception:
        if not offline:
            logger.debug("sentence-transformers download failed for %s", model_name, exc_info=True)
        return None
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _try_fastembed_dense(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    try:
        model = TextEmbedding(model_name=model_name)
    except Exception:
        logger.warning("fastembed dense model unavailable: %s", model_name, exc_info=True)
        return None

    def documents(texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in model.embed(texts)]

    def query(text: str) -> list[float]:
        return next(model.embed([text])).tolist()

    return documents, query


def _try_fastembed_sparse(model_name: str):
    try:
        from fastembed import SparseTextEmbedding
    except ImportError:
        return None, None
    try:
        model = SparseTextEmbedding(model_name=model_name)
    except Exception:
        logger.warning("fastembed sparse model unavailable: %s", model_name, exc_info=True)
        return None, None

    def documents(texts: list[str]) -> list[SparseEmbedding]:
        return [_as_sparse(item) for item in model.embed(texts)]

    def query(text: str) -> SparseEmbedding:
        return _as_sparse(next(model.query_embed([text])))

    return documents, query


def _as_sparse(item) -> SparseEmbedding:
    indices = getattr(item, "indices", None)
    values = getattr(item, "values", None)
    if indices is None or values is None:
        raise RuntimeError("sparse embedding missing indices/values")
    return SparseEmbedding(
        indices=[int(index) for index in list(indices)],
        values=[float(value) for value in list(values)],
    )
