"""LLM 调用：默认阿里云 DashScope（OpenAI 兼容）；可选 Gemini 原生。"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from dl_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LlmNotConfiguredError(RuntimeError):
    pass


class LlmRequestError(RuntimeError):
    """LLM 请求失败。attrs: status_code, fatal。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        fatal: bool = False,
        aborted: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.fatal = fatal
        self.aborted = aborted


@dataclass
class ChatToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatTurn:
    content: str = ""
    tool_calls: list[ChatToolCall] = field(default_factory=list)


def chat(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    timeout_s: float | None = None,
    cancel_event: threading.Event | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    turn = chat_turn(
        messages,
        settings=settings,
        timeout_s=timeout_s,
        cancel_event=cancel_event,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    if not turn.content.strip():
        raise LlmRequestError("LLM 返回空内容", fatal=True)
    return turn.content


def chat_turn(
    messages: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    timeout_s: float | None = None,
    cancel_event: threading.Event | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> ChatTurn:
    settings = settings or get_settings()
    key = (api_key if api_key is not None else settings.openai_api_key).strip()
    if not key:
        raise LlmNotConfiguredError("未配置 OPENAI_API_KEY，无法调用翻译模型")
    timeout = timeout_s if timeout_s is not None else settings.llm_timeout_s
    model_name = (model or settings.model_name).strip()
    openai_base_url = (base_url or settings.openai_base_url).strip()
    gemini = _use_gemini_native(openai_base_url, key)
    send_tools = list(tools or []) if tools else []
    if gemini and send_tools:
        logger.warning("gemini native 不发送 OpenAI tools，orchestrator 应回落 json 协议")
        send_tools = []
    from dl_agent.observability.langfuse import clip_value, mark_error, observe_span, safe_messages

    with observe_span(
        "chat_turn",
        as_type="generation",
        input=safe_messages(messages),
        model=model_name,
        metadata={
            "base_url": openai_base_url,
            "tools": len(send_tools),
            "gemini_native": gemini,
        },
    ) as generation:
        try:
            turn = _chat_turn_request(
                messages,
                settings=settings,
                timeout=timeout,
                cancel_event=cancel_event,
                model_name=model_name,
                openai_base_url=openai_base_url,
                key=key,
                gemini=gemini,
                send_tools=send_tools,
                tool_choice=tool_choice,
            )
            generation.update(
                output=clip_value(
                    {
                        "content": turn.content,
                        "tool_calls": [
                            {"id": item.id, "name": item.name, "arguments": item.arguments}
                            for item in turn.tool_calls
                        ],
                    }
                )
            )
            return turn
        except Exception as exc:
            mark_error(generation, exc)
            raise


def _chat_turn_request(
    messages: list[dict[str, Any]],
    *,
    settings: Settings,
    timeout: float,
    cancel_event: threading.Event | None,
    model_name: str,
    openai_base_url: str,
    key: str,
    gemini: bool,
    send_tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
) -> ChatTurn:
    attempts = max(1, settings.llm_max_retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        _raise_if_cancelled(cancel_event)
        started = time.perf_counter()
        try:
            if gemini:
                content = _chat_gemini_native(
                    messages,
                    model_name=model_name,
                    key=key,
                    timeout_s=timeout,
                    cancel_event=cancel_event,
                )
                turn = ChatTurn(content=content)
            else:
                turn = _chat_openai_compatible(
                    messages,
                    model_name=model_name,
                    base_url=openai_base_url,
                    key=key,
                    timeout_s=timeout,
                    cancel_event=cancel_event,
                    tools=send_tools,
                    tool_choice=tool_choice,
                )
        except LlmRequestError as exc:
            last_error = exc
            logger.error(
                "llm failed model=%s attempt=%s/%s fatal=%s status=%s elapsed_ms=%s error=%s",
                model_name,
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
            logger.warning("llm retry sleep_s=%.1f model=%s", sleep_s, model_name)
            _wait_or_cancel(cancel_event, sleep_s)
            continue
        except Exception:
            logger.exception(
                "llm unexpected error model=%s attempt=%s/%s elapsed_ms=%s",
                model_name,
                attempt,
                attempts,
                int((time.perf_counter() - started) * 1000),
            )
            raise
        logger.info(
            "llm ok model=%s chars=%s tool_calls=%s elapsed_ms=%s",
            model_name,
            len(turn.content),
            len(turn.tool_calls),
            int((time.perf_counter() - started) * 1000),
        )
        return turn
    assert last_error is not None
    raise last_error


def uses_gemini_native(base_url: str, key: str) -> bool:
    return _use_gemini_native(base_url, key)


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise LlmRequestError("翻译已中止", aborted=True)


def _wait_or_cancel(cancel_event: threading.Event | None, seconds: float) -> None:
    if seconds <= 0:
        return
    if cancel_event is None:
        time.sleep(seconds)
        return
    if cancel_event.wait(timeout=seconds):
        raise LlmRequestError("翻译已中止", aborted=True)


def _http_post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
    cancel_event: threading.Event | None = None,
) -> httpx.Response:
    _raise_if_cancelled(cancel_event)
    with httpx.Client(timeout=timeout_s) as client:
        stop_watch = threading.Event()
        watcher: threading.Thread | None = None
        if cancel_event is not None:
            def _watch() -> None:
                while not stop_watch.is_set():
                    if cancel_event.wait(0.15):
                        try:
                            client.close()
                        except Exception:
                            pass
                        return

            watcher = threading.Thread(target=_watch, daemon=True, name="llm-cancel-watch")
            watcher.start()
        try:
            return client.post(url, json=payload, headers=headers)
        except (httpx.HTTPError, RuntimeError) as exc:
            _raise_if_cancelled(cancel_event)
            raise
        finally:
            stop_watch.set()


def _should_retry(exc: LlmRequestError) -> bool:
    if getattr(exc, "aborted", False):
        return False
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


def _use_gemini_native(base_url: str, key: str) -> bool:
    """仅当明确配置 Gemini 端点时走原生接口；DashScope 等一律 OpenAI 兼容。"""
    base = base_url.lower()
    if "dashscope" in base or "aliyuncs.com" in base:
        return False
    if "generativelanguage.googleapis.com" in base:
        return True
    # 仅在未改 base_url 且仍是 Google Auth Key 时兜底
    return key.startswith("AQ.") and "openai.com" not in base


def _chat_gemini_native(
    messages: list[dict[str, Any]],
    *,
    model_name: str,
    key: str,
    timeout_s: float,
    cancel_event: threading.Event | None = None,
) -> str:
    model = model_name.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        parts = _gemini_parts(message.get("content"))
        if role == "system":
            system_parts.extend(part["text"] for part in parts if "text" in part)
            continue
        if not parts:
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": parts})
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
        response = _http_post_json(
            url,
            payload=payload,
            headers=headers,
            timeout_s=timeout_s,
            cancel_event=cancel_event,
        )
    except LlmRequestError:
        raise
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


def _gemini_parts(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content.strip() else []
    if not isinstance(content, list):
        blob = str(content)
        return [{"text": blob}] if blob.strip() else []
    parts: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            if item.strip():
                parts.append({"text": item})
            continue
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if kind == "text" or (not kind and item.get("text")):
            text = str(item.get("text") or "")
            if text:
                parts.append({"text": text})
            continue
        if kind != "image_url":
            continue
        image = item.get("image_url")
        url = ""
        if isinstance(image, dict):
            url = str(image.get("url") or "")
        elif isinstance(image, str):
            url = image
        inline = _data_url_to_inline(url)
        if inline:
            parts.append(inline)
    return parts


def _data_url_to_inline(url: str) -> dict[str, Any] | None:
    blob = url.strip()
    if not blob.startswith("data:") or ";base64," not in blob:
        return None
    header, _, data = blob.partition(";base64,")
    mime = header[5:] or "image/png"
    if not data:
        return None
    return {"inlineData": {"mimeType": mime, "data": data}}


def _chat_openai_compatible(
    messages: list[dict[str, Any]],
    *,
    model_name: str,
    base_url: str,
    key: str,
    timeout_s: float,
    cancel_event: threading.Event | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> ChatTurn:
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Google AQ. Key 走兼容层时额外带 x-goog-api-key
    if key.startswith("AQ.") or "generativelanguage.googleapis.com" in base:
        headers["x-goog-api-key"] = key
    logger.info(
        "llm request openai-compat model=%s timeout_s=%s url=%s tools=%s",
        model_name,
        timeout_s,
        url,
        len(tools or []),
    )
    try:
        response = _http_post_json(
            url,
            payload=payload,
            headers=headers,
            timeout_s=timeout_s,
            cancel_event=cancel_event,
        )
    except LlmRequestError:
        raise
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
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmRequestError("LLM 响应格式异常", fatal=True) from exc
    if not isinstance(message, dict):
        raise LlmRequestError("LLM 响应格式异常", fatal=True)
    raw_content = message.get("content")
    content = raw_content.strip() if isinstance(raw_content, str) else ""
    tool_calls = _parse_openai_tool_calls(message.get("tool_calls"))
    if not content and not tool_calls:
        raise LlmRequestError("LLM 返回空内容", fatal=True)
    return ChatTurn(content=content, tool_calls=tool_calls)


def _parse_openai_tool_calls(raw: Any) -> list[ChatToolCall]:
    if not isinstance(raw, list):
        return []
    out: list[ChatToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        name = str(fn.get("name") or item.get("name") or "").strip()
        if not name:
            continue
        arguments: Any = fn.get("arguments")
        if arguments is None:
            arguments = item.get("arguments") if item.get("arguments") is not None else item.get("args")
        parsed: dict[str, Any]
        if isinstance(arguments, str):
            try:
                loaded = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                loaded = {}
            parsed = loaded if isinstance(loaded, dict) else {}
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            parsed = {}
        out.append(
            ChatToolCall(
                id=str(item.get("id") or "").strip(),
                name=name,
                arguments=parsed,
            )
        )
    return out


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
