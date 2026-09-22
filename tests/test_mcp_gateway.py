from __future__ import annotations

import httpx

from dl_agent.config import Settings
from dl_agent.domain.models import ExternalRef, Paper, Section
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.mcp_gateway import ARXIV_UNAVAILABLE, NO_ARXIV_HITS, MemoryMcpGateway, search_arxiv
from dl_agent.mcp_gateway.adapters.arxiv import (
    build_arxiv_search_query,
    format_arxiv_hits,
    is_withdrawn_text,
    parse_atom,
    parse_openalex,
)
from dl_agent.understand.agent.nodes import parse_orchestrator_reply, run_tools
from dl_agent.understand.agent.tools import ARXIV_DISABLED, ARXIV_SEARCH_SCHEMA, RetrievalTools, worker_tool_schemas

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2303.08774v1</id>
    <title>GPT-4 Technical Report</title>
    <published>2023-03-15T17:00:00Z</published>
    <summary>We report the development of GPT-4, a large-scale multimodal model.</summary>
    <author><name>OpenAI</name></author>
    <link href="http://arxiv.org/abs/2303.08774v1" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""

MIXED_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1304.1836v2</id>
    <title>A Simulation and Modeling of Access Points with Definition Language</title>
    <published>2013-04-06T00:18:59Z</published>
    <summary>This submission has been withdrawn by arXiv administrators because it contains fictitious content.</summary>
    <arxiv:comment>Withdrawn by arXiv admins</arxiv:comment>
    <author><name>Tairen Sun</name></author>
    <link href="http://arxiv.org/abs/1304.1836v2" rel="alternate" type="text/html"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2303.08774v1</id>
    <title>GPT-4 Technical Report</title>
    <published>2023-03-15T17:00:00Z</published>
    <summary>We report the development of GPT-4, a large-scale multimodal model.</summary>
    <author><name>OpenAI</name></author>
    <link href="http://arxiv.org/abs/2303.08774v1" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""


def _ref() -> ExternalRef:
    return ExternalRef(
        source="arxiv",
        title="GPT-4 Technical Report",
        url="https://arxiv.org/abs/2303.08774",
        identifier="arxiv:2303.08774",
        snippet="We report the development of GPT-4.",
        year=2023,
        authors=["OpenAI"],
    )


def test_memory_gateway_filters_by_query() -> None:
    gateway = MemoryMcpGateway([_ref()])
    miss = gateway.call("arxiv_search", {"query": "diffusion policy"})
    assert miss.sentinel == NO_ARXIV_HITS
    assert miss.refs == []
    hit = gateway.call("arxiv_search", {"query": "gpt-4"})
    assert hit.ok
    assert hit.refs[0].identifier == "arxiv:2303.08774"
    assert "[ext-1]" in hit.text
    assert "sourced" not in hit.text.lower()


def test_memory_gateway_filters_by_id() -> None:
    gateway = MemoryMcpGateway([_ref()])
    hit = gateway.call("arxiv_search", {"arxiv_id": "2303.08774"})
    assert [item.identifier for item in hit.refs] == ["arxiv:2303.08774"]
    empty = gateway.call("arxiv_search", {"arxiv_id": "0000.00000"})
    assert empty.sentinel == NO_ARXIV_HITS


def test_memory_gateway_unknown_tool() -> None:
    memory = MemoryMcpGateway([_ref()])
    result = memory.call("web_search", {"query": "x"})
    assert result.sentinel == "UNKNOWN_MCP_TOOL: web_search"


def test_parse_atom_and_http_search() -> None:
    refs = parse_atom(ATOM)
    assert refs[0].identifier == "arxiv:2303.08774"
    assert refs[0].year == 2023
    assert refs[0].source == "arxiv"
    assert refs[0].title.startswith("GPT-4")
    assert "https://arxiv.org/abs/2303.08774" in (refs[0].url or "")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "search_query" in str(request.url)
        return httpx.Response(200, text=ATOM)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = search_arxiv({"query": "gpt-4"}, client=client)
    assert result.ok
    assert result.refs[0].identifier == "arxiv:2303.08774"
    assert "GPT-4 Technical Report" in format_arxiv_hits(result.refs)


def test_http_unavailable_and_empty() -> None:
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    client = httpx.Client(transport=httpx.MockTransport(timeout_handler))
    failed = search_arxiv({"query": "gpt"}, client=client)
    assert failed.sentinel == ARXIV_UNAVAILABLE
    assert failed.refs == []

    def empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text='<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>',
        )

    empty_client = httpx.Client(transport=httpx.MockTransport(empty_handler))
    empty = search_arxiv({"query": "zzz"}, client=empty_client)
    assert empty.sentinel == NO_ARXIV_HITS

    blank = search_arxiv({})
    assert blank.sentinel == NO_ARXIV_HITS

    generic = search_arxiv({"query": "在 arXiv 上检索相关论文"})
    assert generic.sentinel == NO_ARXIV_HITS


def test_build_query_uses_and_and_drops_arxiv_token() -> None:
    assert build_arxiv_search_query("在 arXiv 上检索相关论文") == ""
    built = build_arxiv_search_query("diffusion policy")
    assert built == 'all:"diffusion policy"'
    hyphen = build_arxiv_search_query("person re-id")
    assert hyphen == 'all:"person re id"'


def test_parse_openalex_prefers_arxiv_id() -> None:
    refs = parse_openalex(
        {
            "results": [
                {
                    "display_name": "A Re-ID Paper",
                    "publication_year": 2024,
                    "ids": {"arxiv": "https://arxiv.org/abs/2401.12345"},
                    "authorships": [{"author": {"display_name": "Li"}}],
                    "abstract_inverted_index": {"Hello": [0], "world": [1]},
                },
                {
                    "display_name": "Venue-only Paper",
                    "publication_year": 2023,
                    "ids": {"doi": "https://doi.org/10.1000/xyz"},
                    "primary_location": {"landing_page_url": "https://doi.org/10.1000/xyz"},
                    "authorships": [],
                },
            ]
        },
        5,
    )
    assert refs[0].identifier == "arxiv:2401.12345"
    assert refs[0].source == "arxiv"
    assert "Hello world" in (refs[0].snippet or "")
    assert refs[1].source == "web"


def test_search_retries_after_unavailable() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, text=ATOM)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = search_arxiv({"query": "gpt-4"}, client=client)
    assert calls["n"] == 2
    assert result.ok
    assert result.refs[0].identifier == "arxiv:2303.08774"


def test_keyword_search_skips_withdrawn_but_id_lookup_keeps_it() -> None:
    kept = parse_atom(MIXED_ATOM, include_withdrawn=False)
    assert [item.identifier for item in kept] == ["arxiv:2303.08774"]
    assert is_withdrawn_text("Withdrawn by arXiv admins", "This paper has been withdrawn")

    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, text=MIXED_ATOM)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    searched = search_arxiv({"query": "gpt-4", "max_results": 5}, client=client)
    assert [item.identifier for item in searched.refs] == ["arxiv:2303.08774"]
    assert "1304.1836" not in (searched.text or "")
    assert "sortBy=relevance" in captured["url"]

    by_id = search_arxiv({"arxiv_id": "1304.1836"}, client=client)
    assert by_id.refs[0].identifier == "arxiv:1304.1836"
    assert "withdrawn" in (by_id.refs[0].snippet or "").lower()


def _tools(tmp_path, gateway=None, enable_external=False) -> RetrievalTools:
    svc = KnowledgeService(
        FilePaperStore(tmp_path),
        Settings(data_dir=tmp_path, formula_vision_enabled=False),
        embed_fn=lambda texts: [[0.0, 0.0, 0.0] for _ in texts],
        vector_index=MemoryVectorIndex(),
    )
    svc.store.save_paper(
        Paper(paper_id="paper-a", sha256="paper-a", filename="a.pdf", status="ready")
    )
    svc.store.save_sections(
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
            )
        ],
    )
    return RetrievalTools(
        svc,
        "paper-a",
        gateway=gateway,
        enable_external=enable_external,
    )


def test_arxiv_search_disabled_without_flag(tmp_path) -> None:
    tools = _tools(tmp_path, gateway=MemoryMcpGateway([_ref()]), enable_external=False)
    assert tools.arxiv_search("gpt-4") == ARXIV_DISABLED
    assert tools.last_hits == []
    assert tools.last_external_refs == []
    assert ARXIV_SEARCH_SCHEMA not in worker_tool_schemas(enable_external=False)
    assert ARXIV_SEARCH_SCHEMA in worker_tool_schemas(enable_external=True)


def test_arxiv_search_does_not_enter_evidence(tmp_path) -> None:
    tools = _tools(tmp_path, gateway=MemoryMcpGateway([_ref()]), enable_external=True)
    text = tools.arxiv_search("gpt-4")
    assert "GPT-4 Technical Report" in text
    assert tools.last_hits == []
    assert tools.last_external_refs[0].source == "arxiv"
    out = run_tools(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_arxiv", "name": "arxiv_search", "args": {"query": "gpt-4"}}
                    ],
                }
            ],
            "retrieval_keys": [],
        },
        tools,
    )
    assert out["evidence_bag"] == []
    assert out["external_refs"][0].identifier == "arxiv:2303.08774"
    assert "arxiv::gpt-4" in out["retrieval_keys"]
    assert "UNKNOWN_TOOL" not in out["messages"][0]["content"]


def test_arxiv_search_dedup(tmp_path) -> None:
    tools = _tools(tmp_path, gateway=MemoryMcpGateway([_ref()]), enable_external=True)
    out = run_tools(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_a", "name": "arxiv_search", "args": {"query": "gpt-4"}}
                    ],
                }
            ],
            "retrieval_keys": ["arxiv::gpt-4"],
        },
        tools,
    )
    assert "skipped duplicate" in out["messages"][0]["content"]
    assert out["external_refs"] == []
    assert out["tool_call_count"] == 0


def test_parse_orchestrator_reply_reads_arxiv_search() -> None:
    reply = parse_orchestrator_reply('{"tool":"arxiv_search","query":"diffusion policy"}')
    assert reply.tool_calls[0].name == "arxiv_search"
    assert reply.tool_calls[0].args["query"] == "diffusion policy"
    by_id = parse_orchestrator_reply('{"tool":"arxiv_search","arxiv_id":"2303.08774"}')
    assert by_id.tool_calls[0].args["arxiv_id"] == "2303.08774"
