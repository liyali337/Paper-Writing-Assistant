"""OpenAI 兼容 LLM 调用。"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from dl_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LlmNotConfiguredError(RuntimeError):
    pass


class LlmRequestError(RuntimeError):
    pass


def chat(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    timeout_s: float = 120.0,
) -> str:
    settings = settings or get_settings()
    if not settings.openai_api_key.strip():
        raise LlmNotConfiguredError("未配置 OPENAI_API_KEY，无法调用翻译模型")
    base = settings.openai_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise LlmRequestError(f"LLM 请求失败: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text[:400]
        raise LlmRequestError(f"LLM 返回 {response.status_code}: {detail}")
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmRequestError("LLM 响应格式异常") from exc
    if not isinstance(content, str) or not content.strip():
        raise LlmRequestError("LLM 返回空内容")
    return content.strip()


def complete(schema: type, messages: list[dict[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError("harness.complete is scheduled for M2")
