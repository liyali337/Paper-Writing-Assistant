from dl_agent.observability.langfuse import (
    clip_value,
    observe_span,
    reset_client,
    safe_messages,
    tracing_enabled,
    tracing_status,
)


def test_tracing_disabled_during_pytest() -> None:
    assert tracing_enabled() is False
    assert tracing_status().startswith("off")


def test_observe_span_is_noop_without_client() -> None:
    reset_client()
    with observe_span("ask_agent", input={"q": "hi"}) as span:
        span.update(output="ok")
        span.update_trace(name="x")


def test_clip_value_truncates_long_text() -> None:
    blob = "a" * 9000
    clipped = clip_value(blob, limit=100)
    assert clipped.startswith("a" * 100)
    assert "+8900 chars" in clipped


def test_safe_messages_redact_images() -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    out = safe_messages(messages)
    assert out[0]["content"][1]["image_url"]["url"] == "[omitted]"
    assert out[0]["content"][0]["text"] == "看图"
