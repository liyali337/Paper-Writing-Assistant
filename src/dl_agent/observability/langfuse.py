"""Optional Langfuse tracing. Missing SDK or keys => no-op."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from dl_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)

_MAX_CHARS = 8000
_client: Any | None = None
_client_failed = False


class _NoopObs:
    def update(self, **kwargs: Any) -> "_NoopObs":
        return self

    def update_trace(self, **kwargs: Any) -> "_NoopObs":
        return self

    def end(self, **kwargs: Any) -> None:
        return None


def tracing_enabled(settings: Settings | None = None) -> bool:
    forced = os.getenv("LANGFUSE_TRACING", "").strip().lower()
    if forced in {"0", "false", "off", "no"}:
        return False
    if os.getenv("PYTEST_CURRENT_TEST") and forced not in {"1", "true", "on", "yes"}:
        return False
    settings = settings or get_settings()
    if not settings.langfuse_tracing:
        return False
    return bool(settings.langfuse_public_key.strip() and settings.langfuse_secret_key.strip())


def tracing_status(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if not tracing_enabled(settings):
        if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("LANGFUSE_TRACING", "").strip().lower() not in {
            "1",
            "true",
            "on",
            "yes",
        }:
            return "off-pytest"
        if not settings.langfuse_tracing:
            return "off"
        if not (settings.langfuse_public_key.strip() and settings.langfuse_secret_key.strip()):
            return "off-no-keys"
        return "off"
    if _get_client(settings) is None:
        return "off-no-sdk"
    return "on"


def _get_client(settings: Settings | None = None) -> Any | None:
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None
    settings = settings or get_settings()
    if not tracing_enabled(settings):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("已配置 Langfuse 密钥但未安装 SDK，运行 pip install -e \".[obs]\"")
        _client_failed = True
        return None
    public = settings.langfuse_public_key.strip()
    secret = settings.langfuse_secret_key.strip()
    host = settings.langfuse_base_url.strip() or "https://cloud.langfuse.com"
    client = None
    last_error: Exception | None = None
    for kwargs in (
        {"public_key": public, "secret_key": secret, "base_url": host},
        {"public_key": public, "secret_key": secret, "host": host},
    ):
        try:
            client = Langfuse(**kwargs)
            break
        except TypeError as exc:
            last_error = exc
    if client is None:
        logger.warning("Langfuse 客户端初始化失败: %s", last_error)
        _client_failed = True
        return None
    _client = client
    return _client


def reset_client() -> None:
    global _client, _client_failed
    _client = None
    _client_failed = False


def flush_tracing() -> None:
    client = _client
    if client is None:
        return
    try:
        if hasattr(client, "flush"):
            client.flush()
        elif hasattr(client, "shutdown"):
            client.shutdown()
    except Exception:
        logger.debug("langfuse flush failed", exc_info=True)


def mark_error(obs: Any, exc: BaseException) -> None:
    try:
        obs.update(level="ERROR", status_message=str(exc)[:500])
    except Exception:
        logger.debug("langfuse mark_error failed", exc_info=True)


def clip_value(value: Any, limit: int = _MAX_CHARS) -> Any:
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"…[+{len(value) - limit} chars]"
    if isinstance(value, list):
        return [clip_value(item, limit) for item in value[:50]]
    if isinstance(value, dict):
        return {str(key): clip_value(item, limit) for key, item in list(value.items())[:40]}
    return value


def safe_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [clip_value(_redact_message(item)) for item in messages]


def _redact_message(message: dict[str, Any]) -> dict[str, Any]:
    out = dict(message)
    content = out.get("content")
    if isinstance(content, list):
        out["content"] = [_redact_part(part) for part in content]
    return out


def _redact_part(part: Any) -> Any:
    if not isinstance(part, dict):
        return part
    kind = str(part.get("type") or "")
    if kind == "image_url" or "image_url" in part:
        copied = dict(part)
        copied["image_url"] = {"url": "[omitted]"}
        return copied
    if kind == "image" or "inlineData" in part or "inline_data" in part:
        copied = dict(part)
        copied.pop("inlineData", None)
        copied.pop("inline_data", None)
        copied["omitted"] = "image"
        return copied
    return part


@contextmanager
def observe_span(
    name: str,
    *,
    as_type: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any]:
    client = _get_client()
    start = getattr(client, "start_as_current_observation", None) if client is not None else None
    if start is None:
        yield _NoopObs()
        return
    kwargs: dict[str, Any] = {"as_type": as_type, "name": name}
    if input is not None:
        kwargs["input"] = clip_value(input)
    if metadata:
        kwargs["metadata"] = clip_value(metadata)
    if model:
        kwargs["model"] = model
    attr_cm: Any = nullcontext()
    if session_id or tags:
        try:
            from langfuse import propagate_attributes

            attr_kwargs: dict[str, Any] = {}
            if session_id:
                attr_kwargs["session_id"] = session_id
            if tags:
                attr_kwargs["tags"] = tags
            attr_cm = propagate_attributes(**attr_kwargs)
        except Exception:
            attr_cm = nullcontext()
    with attr_cm:
        with _SafeObservation(start, kwargs) as obs:
            yield obs


class _SafeObservation:
    def __init__(self, start: Any, kwargs: dict[str, Any]):
        self._start = start
        self._kwargs = kwargs
        self._inner: Any | None = None

    def __enter__(self) -> Any:
        try:
            self._inner = self._start(**self._kwargs)
            return self._inner.__enter__()
        except TypeError:
            if self._kwargs.get("as_type") != "tool":
                logger.debug("langfuse observation start failed", exc_info=True)
                self._inner = None
                return _NoopObs()
            fallback = dict(self._kwargs)
            fallback["as_type"] = "span"
            try:
                self._inner = self._start(**fallback)
                return self._inner.__enter__()
            except Exception:
                logger.debug("langfuse observation start failed", exc_info=True)
                self._inner = None
                return _NoopObs()
        except Exception:
            logger.debug("langfuse observation start failed", exc_info=True)
            self._inner = None
            return _NoopObs()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._inner is None:
            return False
        return bool(self._inner.__exit__(exc_type, exc, tb))
