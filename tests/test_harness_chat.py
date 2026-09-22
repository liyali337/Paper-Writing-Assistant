"""harness.chat 鉴权与错误路径。"""

from __future__ import annotations

import json

import httpx
import pytest

from dl_agent.config import Settings
from dl_agent.harness.complete import LlmRequestError, chat, chat_turn


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | str):
        self.status_code = status_code
        if isinstance(payload, str):
            self._text = payload
            self._json = None
        else:
            self._json = payload
            self._text = json.dumps(payload)

    @property
    def text(self) -> str:
        return self._text

    def json(self) -> dict:
        if self._json is None:
            raise ValueError("no json")
        return self._json


class _FakeClient:
    def __init__(self, response: _FakeResponse, capture: dict):
        self._response = response
        self._capture = capture

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, json=None, headers=None):
        self._capture["url"] = url
        self._capture["json"] = json
        self._capture["headers"] = headers
        return self._response


def test_aq_key_uses_gemini_native(monkeypatch) -> None:
    capture: dict = {}
    response = _FakeResponse(
        200,
        {"candidates": [{"content": {"parts": [{"text": '{"title_zh":"摘要","text_zh":"你好"}'}]}}]},
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(response, capture))
    settings = Settings(
        openai_api_key="AQ.test-key",
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_name="gemini-3.6-flash",
        llm_timeout_s=10,
    )
    out = chat(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        settings=settings,
    )
    assert "你好" in out or "title_zh" in out
    assert ":generateContent" in capture["url"]
    assert capture["headers"]["x-goog-api-key"] == "AQ.test-key"
    assert "systemInstruction" in capture["json"]


def test_dashscope_uses_openai_compat(monkeypatch) -> None:
    capture: dict = {}
    response = _FakeResponse(
        200,
        {"choices": [{"message": {"content": '{"title_zh":"摘要","text_zh":"你好"}'}}]},
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(response, capture))
    settings = Settings(
        openai_api_key="sk-test-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen3.8-max",
        llm_timeout_s=10,
    )
    out = chat([{"role": "user", "content": "hi"}], settings=settings)
    assert "你好" in out or "title_zh" in out
    assert capture["url"].endswith("/chat/completions")
    assert "qwen3.8-max" in capture["json"]["model"]
    assert capture["headers"]["Authorization"] == "Bearer sk-test-key"
    assert "x-goog-api-key" not in capture["headers"]


def test_openai_compat_sends_image_url(monkeypatch) -> None:
    capture: dict = {}
    response = _FakeResponse(
        200,
        {"choices": [{"message": {"content": '{"latex":"a=b"}'}}]},
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(response, capture))
    settings = Settings(
        openai_api_key="sk-test-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen3.8-max",
        llm_timeout_s=10,
    )
    out = chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aaaa"},
                    },
                ],
            }
        ],
        settings=settings,
        model="qwen-vl-plus",
    )
    assert "latex" in out
    assert capture["json"]["model"] == "qwen-vl-plus"
    content = capture["json"]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"


def test_gemini_native_sends_inline_image(monkeypatch) -> None:
    capture: dict = {}
    response = _FakeResponse(
        200,
        {"candidates": [{"content": {"parts": [{"text": '{"latex":"a=b"}'}]}}]},
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(response, capture))
    settings = Settings(
        openai_api_key="AQ.test-key",
        openai_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model_name="gemini-3.6-flash",
        llm_timeout_s=10,
    )
    chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aaaa"},
                    },
                ],
            }
        ],
        settings=settings,
    )
    parts = capture["json"]["contents"][0]["parts"]
    assert parts[0] == {"text": "transcribe"}
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert parts[1]["inlineData"]["data"] == "aaaa"


def test_timeout_raises_fatal(monkeypatch) -> None:
    class _TimeoutClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "Client", _TimeoutClient)
    settings = Settings(
        openai_api_key="sk-test-key",
        openai_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen3.8-max",
        llm_timeout_s=5,
        llm_max_retries=1,
    )
    with pytest.raises(LlmRequestError) as exc:
        chat([{"role": "user", "content": "hi"}], settings=settings)
    assert "超时" in str(exc.value)


def test_chat_turn_sends_tools_and_parses_calls(monkeypatch) -> None:
    capture: dict = {}
    response = _FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "retrieve_parent_chunks",
                                    "arguments": '{"section_id":"s-method"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(response, capture))
    settings = Settings(
        openai_api_key="sk-test-key",
        openai_base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
        llm_timeout_s=10,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_parent_chunks",
                "parameters": {"type": "object", "properties": {"section_id": {"type": "string"}}},
            },
        }
    ]
    turn = chat_turn(
        [{"role": "user", "content": "hi"}],
        settings=settings,
        tools=tools,
    )
    assert capture["json"]["tools"] == tools
    assert capture["json"]["tool_choice"] == "auto"
    assert turn.content == ""
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "retrieve_parent_chunks"
    assert turn.tool_calls[0].arguments["section_id"] == "s-method"


def test_chat_rejects_tool_only_turn(monkeypatch) -> None:
    capture: dict = {}
    response = _FakeResponse(
        200,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search_child_chunks", "arguments": '{"query":"q"}'},
                            }
                        ],
                    }
                }
            ]
        },
    )
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient(response, capture))
    settings = Settings(
        openai_api_key="sk-test-key",
        openai_base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
        llm_timeout_s=10,
        llm_max_retries=1,
    )
    with pytest.raises(LlmRequestError) as exc:
        chat([{"role": "user", "content": "hi"}], settings=settings)
    assert "空内容" in str(exc.value)
