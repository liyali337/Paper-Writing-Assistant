import json

from dl_agent.understand.agent.nodes import bootstrap_search, run_tools
from dl_agent.understand.agent.tools import BOOTSTRAP_SEARCH_CALL_ID
from dl_agent.understand.progress import bind_progress, emit_tool, format_sse, reset_progress


class _Tools:
    def __init__(self) -> None:
        self.last_child_hits: list[object] = []
        self.last_hits: list[object] = []

    def search_child_chunks(self, query: str, limit: int | None = None) -> str:
        self.last_child_hits = [object()]
        self.last_hits = list(self.last_child_hits)
        return f"FULL CHUNK {query}"

    def retrieve_parent_chunks(self, section_id: str) -> str:
        return "NO_PARENT_DOCUMENT"


def _capture(fn):
    seen: list[dict] = []
    token = bind_progress(seen.append)
    try:
        fn()
    finally:
        reset_progress(token)
    return seen


def test_progress_is_silent_without_handler() -> None:
    emit_tool(name="search_child_chunks", status="start", call_id="c1", args={"query": "lambda"})


def test_bootstrap_streams_tool_without_chunk_text() -> None:
    tools = _Tools()
    seen = _capture(lambda: bootstrap_search({"question": "lambda loss"}, tools))
    assert [item["status"] for item in seen] == ["start", "done"]
    assert seen[0]["name"] == "search_child_chunks"
    assert seen[0]["label"] == "检索文内片段"
    assert seen[0]["call_id"] == BOOTSTRAP_SEARCH_CALL_ID
    assert seen[0]["detail"] == "lambda loss"
    assert seen[1]["note"] == "命中 1 条"
    blob = json.dumps(seen, ensure_ascii=False)
    assert "FULL CHUNK" not in blob


def test_run_tools_streams_start_then_done_or_skip() -> None:
    tools = _Tools()

    def run() -> None:
        run_tools(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-parent",
                                "name": "retrieve_parent_chunks",
                                "args": {"section_id": "s-method"},
                            },
                            {
                                "id": "call-dup",
                                "name": "search_child_chunks",
                                "args": {"query": "lambda"},
                            },
                        ],
                    }
                ],
                "retrieval_keys": ["search::lambda"],
            },
            tools,
        )

    seen = _capture(run)
    assert [(item["call_id"], item["status"]) for item in seen] == [
        ("call-parent", "start"),
        ("call-parent", "done"),
        ("call-dup", "skip"),
    ]
    assert seen[1]["note"] == "没有命中"
    assert seen[2]["note"] == "重复，已跳过"
    assert "NO_PARENT_DOCUMENT" not in json.dumps(seen, ensure_ascii=False)


def test_format_sse_is_one_data_line() -> None:
    raw = format_sse({"type": "phase", "phase": "dialogue", "label": "正在理解问题"})
    assert raw.startswith("event: phase\n")
    assert raw.endswith("\n\n")
    data = raw.split("data: ", 1)[1].strip()
    assert "\n" not in data
    assert json.loads(data)["label"] == "正在理解问题"
