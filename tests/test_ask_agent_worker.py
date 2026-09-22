from pathlib import Path

from dl_agent.config import Settings
from dl_agent.domain.models import Evidence, Paper, Section
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.understand.agent.graph import run_worker
from dl_agent.understand.agent.nodes import ToolCall, WorkerReply, run_tools
from dl_agent.understand.agent.tools import BOOTSTRAP_SEARCH_CALL_ID, RetrievalTools


class ScriptedModel:
    def __init__(
        self,
        replies: list[WorkerReply] | None = None,
        *,
        compress_text: str = "SUMMARY",
        fallback_text: str = "FALLBACK_ANSWER",
    ):
        self.replies = list(replies or [])
        self.compress_text = compress_text
        self.fallback_text = fallback_text
        self.orch_calls = 0
        self.compress_calls = 0
        self.fallback_calls = 0

    def orchestrate(self, messages, question, context_summary) -> WorkerReply:
        self.orch_calls += 1
        if self.replies:
            return self.replies.pop(0)
        return WorkerReply(content="done")

    def compress(self, conversation_text: str) -> str:
        self.compress_calls += 1
        return self.compress_text

    def fallback(self, question: str, context_text: str) -> str:
        self.fallback_calls += 1
        return self.fallback_text


class CountingTools:
    def __init__(self, payload: str = "[1] section_id=s-method title=Approach page=6\nlambda = 0.5"):
        self.searches: list[str] = []
        self.parents: list[str] = []
        self.payload = payload
        self.last_child_hits = [
            Evidence(
                page=6,
                section_title="Approach",
                quote="lambda = 0.5",
                sourced=True,
                section_id="s-method",
                chunk_id="s-method:0",
            )
        ]
        self.last_parent_hits: list[Evidence] = []

    def search_child_chunks(self, query: str, limit: int | None = None) -> str:
        self.searches.append(query)
        return self.payload

    def retrieve_parent_chunks(self, section_id: str) -> str:
        self.parents.append(section_id)
        self.last_parent_hits = [
            Evidence(
                page=6,
                section_title="Approach",
                quote="lambda = 0.5 in the full section.",
                sourced=True,
                section_id=section_id,
                chunk_id=f"{section_id}:parent",
            )
        ]
        return f"section_id={section_id} title=Approach page=6-8\nlambda = 0.5 in the full section."


def _settings(tmp_path: Path, **kwargs) -> Settings:
    data = {
        "data_dir": tmp_path,
        "formula_vision_enabled": False,
        "ask_max_tool_calls": 6,
        "ask_max_iterations": 8,
        "ask_compress_tokens": 2000,
        "ask_enable_external": False,
    }
    data.update(kwargs)
    return Settings(**data)


def test_worker_always_searches_once_before_answer(tmp_path: Path) -> None:
    tools = CountingTools()
    model = ScriptedModel(
        [
            WorkerReply(content="先看摘要。"),
            WorkerReply(content="λ 设为 0.5。"),
        ]
    )
    state = run_worker(tools, model, _settings(tmp_path), "lambda")
    assert tools.searches == ["lambda"]
    assert tools.parents == ["s-method"]
    assert model.orch_calls == 2
    assert model.fallback_calls == 0
    assert state["final_answer"] == "λ 设为 0.5。"
    assert any(item.chunk_id == "s-method:parent" for item in state["evidence_bag"])


def test_worker_hits_tool_limit_then_fallback(tmp_path: Path) -> None:
    tools = CountingTools()
    model = ScriptedModel(
        [
            WorkerReply(tool_calls=[ToolCall("search_child_chunks", {"query": "lambda extra"})]),
            WorkerReply(tool_calls=[ToolCall("search_child_chunks", {"query": "lambda again"})]),
            WorkerReply(tool_calls=[ToolCall("search_child_chunks", {"query": "lambda more"})]),
        ]
    )
    state = run_worker(
        tools,
        model,
        _settings(tmp_path, ask_max_tool_calls=2, ask_max_iterations=8),
        "lambda",
    )
    assert len(tools.searches) == 2
    assert tools.searches[0] == "lambda"
    assert model.fallback_calls == 1
    assert state["final_answer"] == "FALLBACK_ANSWER"
    assert state["tool_call_count"] == 2


def test_worker_skips_duplicate_search(tmp_path: Path) -> None:
    tools = CountingTools()
    model = ScriptedModel(
        [
            WorkerReply(tool_calls=[ToolCall("search_child_chunks", {"query": "lambda"})]),
            WorkerReply(content="用第一次结果作答。"),
            WorkerReply(content="用第一次结果作答。"),
        ]
    )
    state = run_worker(tools, model, _settings(tmp_path), "lambda")
    assert tools.searches == ["lambda"]
    assert tools.parents == ["s-method"]
    assert "search::lambda" in state["retrieval_keys"]
    assert "parent::s-method" in state["retrieval_keys"]
    assert state["final_answer"] == "用第一次结果作答。"


def test_worker_compresses_when_over_token_budget(tmp_path: Path) -> None:
    tools = CountingTools(payload="[1] section_id=s-method title=Approach page=6\n" + ("lambda body " * 400))
    model = ScriptedModel(
        [
            WorkerReply(tool_calls=[ToolCall("retrieve_parent_chunks", {"section_id": "s-method"})]),
            WorkerReply(content="压缩后的回答"),
        ]
    )
    state = run_worker(
        tools,
        model,
        _settings(tmp_path, ask_compress_tokens=20, ask_max_tool_calls=6),
        "lambda",
    )
    assert model.compress_calls == 1
    assert "SUMMARY" in (state.get("context_summary") or "")
    assert state["final_answer"] == "压缩后的回答"
    assert tools.parents == ["s-method"]


def test_worker_collects_evidence_from_real_index(tmp_path: Path) -> None:
    store = FilePaperStore(tmp_path)
    svc = KnowledgeService(
        store,
        Settings(data_dir=tmp_path, formula_vision_enabled=False),
        embed_fn=lambda texts: [
            [1.0 if "lambda" in text.lower() else 0.0, 0.0, 0.0] for text in texts
        ],
        vector_index=MemoryVectorIndex(),
    )
    store.save_paper(
        Paper(paper_id="paper-a", sha256="paper-a", filename="paper-a.pdf", status="ready")
    )
    store.save_sections(
        "paper-a",
        [
            Section(
                section_id="s-method",
                paper_id="paper-a",
                title="4. Approach",
                kind="method",
                level=1,
                page_start=6,
                page_end=8,
                text="The total loss is L = lambda * L_con + L_ce.",
                figure_ids=["fig-1"],
            )
        ],
    )
    svc.index_paper("paper-a", force=True)
    tools = RetrievalTools(svc, "paper-a")
    model = ScriptedModel(
        [
            WorkerReply(content="先看片段。"),
            WorkerReply(content="损失里有 lambda。"),
        ]
    )
    state = run_worker(tools, model, _settings(tmp_path), "lambda")
    assert state["final_answer"] == "损失里有 lambda。"
    assert any(item.section_id == "s-method" for item in state["evidence_bag"])
    assert any((item.chunk_id or "").endswith(":parent") for item in state["evidence_bag"])
    assert "parent::s-method" in state["retrieval_keys"]


def test_orchestrator_chat_payload_omits_tool_role() -> None:
    from dl_agent.understand.agent.nodes import build_orchestrator_messages

    packed = build_orchestrator_messages(
        "lambda",
        "",
        [
            {"role": "user", "content": "lambda"},
            {"role": "tool", "name": "search_child_chunks", "content": "[1] lambda = 0.5"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "retrieve_parent_chunks", "args": {"section_id": "s-method"}}],
            },
            {"role": "tool", "name": "retrieve_parent_chunks", "content": "full section"},
        ],
    )
    assert [item["role"] for item in packed] == ["system", "user", "user", "assistant", "user"]
    assert all("tool_call_id" not in item for item in packed)
    assert "[TOOL RESULT — search_child_chunks]" in packed[2]["content"]
    assert "retrieve_parent_chunks" in packed[3]["content"]
    assert "[TOOL RESULT — retrieve_parent_chunks]" in packed[4]["content"]


def test_orchestrator_chat_payload_native_keeps_tool_call_id() -> None:
    from dl_agent.understand.agent.nodes import build_orchestrator_messages

    packed = build_orchestrator_messages(
        "lambda",
        "",
        [
            {"role": "user", "content": "lambda"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": BOOTSTRAP_SEARCH_CALL_ID,
                        "name": "search_child_chunks",
                        "args": {"query": "lambda"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_child_chunks",
                "tool_call_id": BOOTSTRAP_SEARCH_CALL_ID,
                "content": "[1] lambda = 0.5",
            },
        ],
        native=True,
    )
    assert [item["role"] for item in packed] == ["system", "user", "assistant", "tool"]
    assert packed[2]["tool_calls"][0]["id"] == BOOTSTRAP_SEARCH_CALL_ID
    assert packed[2]["tool_calls"][0]["type"] == "function"
    assert packed[2]["tool_calls"][0]["function"]["name"] == "search_child_chunks"
    assert packed[3]["tool_call_id"] == BOOTSTRAP_SEARCH_CALL_ID
    assert packed[3]["name"] == "search_child_chunks"
    assert "需要更多证据时调用提供的工具" in packed[0]["content"]


def test_run_tools_echoes_tool_call_id(tmp_path: Path) -> None:
    tools = CountingTools()
    out = run_tools(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "name": "search_child_chunks",
                            "args": {"query": "extra"},
                        }
                    ],
                }
            ],
            "retrieval_keys": [],
        },
        tools,
    )
    assert tools.searches == ["extra"]
    assert out["messages"][0]["role"] == "tool"
    assert out["messages"][0]["tool_call_id"] == "call_abc"


def test_parse_orchestrator_reply_reads_parent_tool_json() -> None:
    from dl_agent.understand.agent.model import ChatAskModel
    from dl_agent.understand.agent.nodes import parse_orchestrator_reply

    reply = parse_orchestrator_reply(
        '先看整节。\n{"tool":"retrieve_parent_chunks","section_id":"s-method"}'
    )
    assert reply.tool_calls
    assert reply.tool_calls[0].name == "retrieve_parent_chunks"
    assert reply.tool_calls[0].args["section_id"] == "s-method"

    settings = Settings(data_dir=".", formula_vision_enabled=False)
    captured: list[list[dict]] = []

    def chat_fn(messages, **_kwargs) -> str:
        captured.append(messages)
        return '{"tool":"retrieve_parent_chunks","section_id":"s-exp"}'

    model = ChatAskModel(chat_fn, settings)
    parsed = model.orchestrate(
        [
            {"role": "user", "content": "方法"},
            {"role": "tool", "name": "search_child_chunks", "content": "[1] section_id=s-exp"},
        ],
        "方法",
        "",
    )
    assert parsed.tool_calls[0].args["section_id"] == "s-exp"
    assert all(item["role"] != "tool" for item in captured[0])


def test_chat_ask_model_uses_native_tool_calls(tmp_path: Path) -> None:
    from dl_agent.harness.complete import ChatToolCall, ChatTurn
    from dl_agent.understand.agent.model import ChatAskModel
    from dl_agent.understand.agent.tools import RETRIEVAL_TOOL_SCHEMAS

    settings = _settings(tmp_path, ask_tool_protocol="native")
    captured: list[dict] = []

    def chat_fn(messages, **_kwargs) -> str:
        raise AssertionError("native 路径不应走 chat()")

    def chat_turn_fn(messages, **kwargs) -> ChatTurn:
        captured.append({"messages": messages, "tools": kwargs.get("tools")})
        return ChatTurn(
            content="",
            tool_calls=[
                ChatToolCall(
                    id="call_native",
                    name="retrieve_parent_chunks",
                    arguments={"section_id": "s-exp"},
                )
            ],
        )

    model = ChatAskModel(chat_fn, settings, chat_turn_fn=chat_turn_fn)
    parsed = model.orchestrate(
        [
            {"role": "user", "content": "方法"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": BOOTSTRAP_SEARCH_CALL_ID,
                        "name": "search_child_chunks",
                        "args": {"query": "方法"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "search_child_chunks",
                "tool_call_id": BOOTSTRAP_SEARCH_CALL_ID,
                "content": "[1] section_id=s-exp",
            },
        ],
        "方法",
        "",
    )
    assert parsed.tool_calls[0].name == "retrieve_parent_chunks"
    assert parsed.tool_calls[0].args["section_id"] == "s-exp"
    assert parsed.tool_calls[0].id == "call_native"
    assert captured[0]["tools"] == RETRIEVAL_TOOL_SCHEMAS
    roles = [item["role"] for item in captured[0]["messages"]]
    assert "tool" in roles
    assert any(item.get("tool_call_id") == BOOTSTRAP_SEARCH_CALL_ID for item in captured[0]["messages"])


def test_chat_ask_model_adds_arxiv_schema_when_enabled(tmp_path: Path) -> None:
    from dl_agent.harness.complete import ChatToolCall, ChatTurn
    from dl_agent.understand.agent.model import ChatAskModel
    from dl_agent.understand.agent.tools import ARXIV_SEARCH_SCHEMA, RETRIEVAL_TOOL_SCHEMAS

    settings = _settings(tmp_path, ask_tool_protocol="native", ask_enable_external=True)
    captured: list[dict] = []

    def chat_fn(messages, **_kwargs) -> str:
        raise AssertionError("native 路径不应走 chat()")

    def chat_turn_fn(messages, **kwargs) -> ChatTurn:
        captured.append({"messages": messages, "tools": kwargs.get("tools")})
        return ChatTurn(content="enough", tool_calls=[])

    ChatAskModel(chat_fn, settings, chat_turn_fn=chat_turn_fn).orchestrate(
        [{"role": "user", "content": "相关工作"}],
        "相关工作",
        "",
    )
    assert captured[0]["tools"] == [*RETRIEVAL_TOOL_SCHEMAS, ARXIV_SEARCH_SCHEMA]
    assert "arxiv_search" in captured[0]["messages"][0]["content"]


def test_chat_ask_model_native_400_falls_back_to_json(tmp_path: Path) -> None:
    from dl_agent.harness.complete import LlmRequestError
    from dl_agent.understand.agent.model import ChatAskModel

    settings = _settings(tmp_path, ask_tool_protocol="native")

    def chat_fn(messages, **_kwargs) -> str:
        return '{"tool":"search_child_chunks","query":"architecture"}'

    def chat_turn_fn(messages, **_kwargs):
        raise LlmRequestError("bad tools", status_code=400, fatal=False)

    model = ChatAskModel(chat_fn, settings, chat_turn_fn=chat_turn_fn)
    parsed = model.orchestrate(
        [{"role": "user", "content": "方法"}],
        "方法",
        "",
    )
    assert parsed.tool_calls[0].name == "search_child_chunks"
    assert parsed.tool_calls[0].args["query"] == "architecture"


def test_parse_orchestrator_reply_reads_perception_tools() -> None:
    from dl_agent.understand.agent.nodes import parse_orchestrator_reply

    outline = parse_orchestrator_reply('{"tool":"list_sections","kind":"method"}')
    assert outline.tool_calls[0].name == "list_sections"
    assert outline.tool_calls[0].args["kind"] == "method"

    caption = parse_orchestrator_reply('{"tool":"get_figure_caption","figure_id":"fig-1"}')
    assert caption.tool_calls[0].args["figure_id"] == "fig-1"

    zh = parse_orchestrator_reply('{"tool":"get_translation","section_id":"s-method"}')
    assert zh.tool_calls[0].name == "get_translation"
