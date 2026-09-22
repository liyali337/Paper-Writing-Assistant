from __future__ import annotations

import logging
from collections.abc import Callable

import httpx

from dl_agent.config import Settings

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]


class EmbeddingNotConfiguredError(RuntimeError):
    def __init__(self, message: str = "embedding_not_configured"):
        super().__init__(message)


def embed_texts(texts: list[str], settings: Settings) -> list[list[float]]:
    if not texts:
        return []
    if not settings.embedding_configured():
        raise EmbeddingNotConfiguredError()
    key = (settings.embedding_api_key or settings.openai_api_key).strip()
    base = settings.embedding_base_url.rstrip("/")
    model = settings.embedding_model.strip()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": model, "input": texts}
    url = f"{base}/embeddings"
    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=settings.llm_timeout_s)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"embedding 请求失败: {exc}") from exc
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or len(items) != len(texts):
        raise RuntimeError("embedding 返回条数与输入不一致")
    ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("embedding 缺少向量")
        vectors.append([float(value) for value in vector])
    return vectors
