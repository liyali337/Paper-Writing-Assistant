"""LLM 调用：默认阿里云 DashScope（OpenAI 兼容）；可选 Gemini 原生。"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from dl_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LlmNotConfiguredError(RuntimeError):
    pass


class LlmRequestError(RuntimeError):
    """LLM 请求失败。attrs: status_code, fatal。"""

    def __init__(self, message: str, *, status_code: int | None = None, fatal: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.fatal = fatal


def chat(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    timeout_s: float | None = None,
) -> str:
    settings = settings or get_settings()
    key = settings.openai_api_key.strip()
    if not key:
        raise LlmNotConfiguredError("未配置 OPENAI_API_KEY，无法调用翻译模型")
    timeout = timeout_s if timeout_s is not None else settings.llm_timeout_s
    attempts = max(1, settings.llm_max_retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            if _use_gemini_native(settings, key):
                content = _chat_gemini_native(messages, settings=settings, key=key, timeout_s=timeout)
            else:
                content = _chat_openai_compatible(messages, settings=settings, key=key, timeout_s=timeout)
        except LlmRequestError as exc:
            last_error = exc
            logger.error(
                "llm failed model=%s attempt=%s/%s fatal=%s status=%s elapsed_ms=%s error=%s",
                settings.model_name,
                attempt,
                attempts,
                getattr(exc, "fatal", False),
                getattr(exc, "status_code", None),
                int((time.perf_counter() - started) * 1000),
                exc,
            )
            if not _should_retry(exc) or attempt >= attempts:
                raise
            sleep_s = _retry_sleep_s(exc, attempt)
            logger.warning("llm retry sleep_s=%.1f model=%s", sleep_s, settings.model_name)
            time.sleep(sleep_s)
            continue
        except Exception:
            logger.exception(
                "llm unexpected error model=%s attempt=%s/%s elapsed_ms=%s",
                settings.model_name,
                attempt,
                attempts,
                int((time.perf_counter() - started) * 1000),
            )
            raise
        logger.info(
            "llm ok model=%s chars=%s elapsed_ms=%s",
            settings.model_name,
            len(content),
            int((time.perf_counter() - started) * 1000),
        )
        return content
    assert last_error is not None
    raise last_error


def _should_retry(exc: LlmRequestError) -> bool:
    if exc.status_code == 429:
        return True
    if exc.status_code is not None and exc.status_code >= 500:
        return True
    msg = str(exc).lower()
    if "timeout" in msg or "超时" in msg:
        return True
    if "ssl" in msg or "eof" in msg or "connection" in msg:
        return True
    return False


def _retry_sleep_s(exc: LlmRequestError, attempt: int) -> float:
    if exc.status_code == 429:
        return min(60.0, 8.0 * attempt)
    return min(20.0, 3.0 * attempt)


def complete(schema: type, messages: list[dict[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError("harness.complete is scheduled for M2")


def _use_gemini_native(settings: Settings, key: str) -> bool:
    """仅当明确配置 Gemini 端点时走原生接口；DashScope 等一律 OpenAI 兼容。"""
    base = settings.openai_base_url.lower()
    if "dashscope" in base or "aliyuncs.com" in base:
        return False
    if "generativelanguage.googleapis.com" in base:
        return True
    # 仅在未改 base_url 且仍是 Google Auth Key 时兜底
    return key.startswith("AQ.") and "openai.com" not in base


def _chat_gemini_native(
    messages: list[dict[str, Any]],
    *,
    settings: Settings,
    key: str,
    timeout_s: float,
) -> str:
    model = settings.model_name.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        text = str(message.get("content") or "")
        if not text.strip():
            continue
        if role == "system":
            system_parts.append(text)
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})
    if not contents:
        raise LlmRequestError("LLM 请求为空", fatal=True)

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": 0.2},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
    }
    logger.info("llm request native model=%s timeout_s=%s", model, timeout_s)
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LlmRequestError(f"LLM 超时（{timeout_s:.0f}s）: {exc}", fatal=False) from exc
    except httpx.HTTPError as exc:
        raise LlmRequestError(f"LLM 请求失败: {exc}", fatal=False) from exc

    if response.status_code >= 400:
        detail = response.text[:500]
        raise LlmRequestError(
            f"LLM 返回 {response.status_code}: {detail}",
            status_code=response.status_code,
            fatal=_is_fatal_status(response.status_code, detail),
        )

    data = response.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        texts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
        content = "".join(texts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmRequestError(f"LLM 响应格式异常: {str(data)[:300]}", fatal=True) from exc
    if not content:
        raise LlmRequestError("LLM 返回空内容", fatal=True)
    return content


def _chat_openai_compatible(
    messages: list[dict[str, Any]],
    *,
    settings: Settings,
    key: str,
    timeout_s: float,
) -> str:
    base = settings.openai_base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": settings.model_name,
        "messages": messages,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Google AQ. Key 走兼容层时额外带 x-goog-api-key
    if key.startswith("AQ.") or "generativelanguage.googleapis.com" in base:
        headers["x-goog-api-key"] = key
    logger.info("llm request openai-compat model=%s timeout_s=%s url=%s", settings.model_name, timeout_s, url)
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LlmRequestError(f"LLM 超时（{timeout_s:.0f}s）: {exc}", fatal=False) from exc
    except httpx.HTTPError as exc:
        raise LlmRequestError(f"LLM 请求失败: {exc}", fatal=False) from exc

    if response.status_code >= 400:
        detail = response.text[:500]
        raise LlmRequestError(
            f"LLM 返回 {response.status_code}: {detail}",
            status_code=response.status_code,
            fatal=_is_fatal_status(response.status_code, detail),
        )
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmRequestError("LLM 响应格式异常", fatal=True) from exc
    if not isinstance(content, str) or not content.strip():
        raise LlmRequestError("LLM 返回空内容", fatal=True)
    return content.strip()


def _is_fatal_status(status_code: int, detail: str) -> bool:
    if status_code == 429:
        return False
    if status_code in {401, 403, 404}:
        return True
    blob = detail.lower()
    if "access_token_type_unsupported" in blob:
        return True
    if "api key not valid" in blob or "invalid api key" in blob:
        return True
    if "permission" in blob and "denied" in blob:
        return True
    return False
