"""LLM 调用内核。M0 空实现，M2 接 OpenAI 兼容端点。"""

from typing import Any


def complete(schema: type, messages: list[dict[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError("harness.complete is scheduled for M2")
