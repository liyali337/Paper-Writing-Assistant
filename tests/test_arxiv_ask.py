from pathlib import Path

from fastapi.testclient import TestClient

from dl_agent.api.deps import get_knowledge, get_understand
from dl_agent.api.main import app
from dl_agent.config import Settings
from dl_agent.domain.models import ExternalRef, Paper, Section
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.mcp_gateway import MemoryMcpGateway
from dl_agent.understand.arxiv import NEED_ARXIV_TOPIC, NO_ARXIV_ANSWER, _search_args, ask_arxiv
from dl_agent.understand.service import UnderstandService


def _embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        low = text.lower()
        vectors.append(
            [
                1.0 if "contrastive" in low or "hrnet" in low else 0.0,
                1.0 if "diffusion" in low or "policy" in low else 0.0,
                1.0 if "transformer" in low else 0.0,
            ]
        )
    return vectors


def _settings(tmp_path: Path, **kwargs) -> Settings:
    data = {
        "data_dir": tmp_path,
        "formula_vision_enabled": False,
        "ask_mode": "simple",
        "ask_enable_external": False,
    }
    data.update(kwargs)
    return Settings(**data)


def _knowledge(tmp_path: Path) -> KnowledgeService:
    return KnowledgeService(
        FilePaperStore(tmp_path),
        _settings(tmp_path),
        embed_fn=_embed,
        vector_index=MemoryVectorIndex(),
    )


def _save_paper(
    knowledge: KnowledgeService,
    paper_id: str,
    *,
    title: str,
    abstract: str,
    method: str,
) -> None:
    knowledge.store.save_paper(
        Paper(
            paper_id=paper_id,
            sha256=paper_id,
            filename=f"{paper_id}.pdf",
            status="ready",
            title=title,
            authors=["Ada"],
            abstract=abstract,
            index_status="ready",
            embedding_version="test-embed",
        )
    )
    knowledge.store.save_sections(
        paper_id,
        [
            Section(
                section_id=f"{paper_id}-abs",
                paper_id=paper_id,
                title="Abstract",
                kind="abstract",
                level=1,
                page_start=1,
                page_end=1,
                text=abstract,
            ),
            Section(
                section_id=f"{paper_id}-m",
                paper_id=paper_id,
                title="Method",
                kind="method",
                level=1,
                page_start=3,
                page_end=5,
                text=method,
            ),
        ],
    )
    knowledge.index_paper(paper_id, force=True)


def _two_papers(tmp_path: Path) -> KnowledgeService:
    svc = _knowledge(tmp_path)
    _save_paper(
        svc,
        "paper-a",
        title="Contrastive HRNet",
        abstract="We study contrastive learning with HRNet on CIFAR-10.",
        method="The contrastive loss uses a momentum encoder on HRNet.",
    )
    _save_paper(
        svc,
        "paper-b",
        title="Diffusion Policy",
        abstract="We train a diffusion policy for robot manipulation.",
        method="The diffusion policy denoises actions over 100 steps.",
    )
    return svc


def _diffusion_ref() -> ExternalRef:
    return ExternalRef(
        source="arxiv",
        title="Diffusion Policy",
        url="https://arxiv.org/abs/2303.04137",
        identifier="arxiv:2303.04137",
        snippet="We train a diffusion policy for robot manipulation.",
        year=2023,
        authors=["Chi"],
    )


def test_search_args_prefers_latin_keywords_and_arxiv_id() -> None:
    query, ident = _search_args("在 arXiv 上检索 diffusion policy 相关论文")
    assert query.lower() == "diffusion policy"
    assert ident == ""
    query, ident = _search_args("看一下 2303.04137 这篇")
    assert ident == "2303.04137"
    query, ident = _search_args("在 arXiv 上检索相关论文")
    assert query == ""
    assert ident == ""
    query, ident = _search_args("search papers on arxiv")
    assert query == ""
    query, ident = _search_args("在 arXiv 上检索 超导 相关论文")
    assert "超导" in query
    query, ident = _search_args("cat:cs.LG attention")
    assert "cs.LG" in query or "cs.lg" in query.lower()


def test_ask_arxiv_returns_external_refs_not_citations(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    gateway = MemoryMcpGateway([_diffusion_ref()])

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        assert "[1]" not in joined or "ext-1" in joined
        assert "文献检索助手" in joined or "arXiv hits" in joined.lower() or "arxiv hits" in joined.lower()
        return '{"answer_zh":"arXiv 上有 Diffusion Policy，做机器人操作。"}'

    result = ask_arxiv(
        svc,
        "在 arXiv 上检索 diffusion policy",
        settings=_settings(tmp_path),
        chat_fn=chat,
        gateway=gateway,
        current_paper_id="paper-a",
    )
    assert result.mode == "arxiv"
    assert result.citations == []
    assert result.external_refs[0].identifier == "arxiv:2303.04137"
    assert result.library_hits[0].paper_id == "paper-b"
    assert "本地已有" in (result.library_hits[0].why or "")
    assert all(item.paper_id != "paper-a" for item in result.library_hits)
    assert gateway.calls and gateway.calls[0][0] == "arxiv_search"


def test_ask_arxiv_no_hits_skips_llm(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    gateway = MemoryMcpGateway([_diffusion_ref()])

    def chat(*args, **kwargs):
        raise AssertionError("LLM should not run")

    result = ask_arxiv(
        svc,
        "在 arXiv 上检索 xyzzy zebra",
        settings=_settings(tmp_path),
        chat_fn=chat,
        gateway=gateway,
        current_paper_id="paper-a",
    )
    assert result.no_evidence is True
    assert result.answer_zh == NO_ARXIV_ANSWER
    assert result.external_refs == []
    assert result.mode == "arxiv"


def test_ask_arxiv_generic_rewrites_query_and_drops_self(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    self_ref = ExternalRef(
        source="arxiv",
        title="Contrastive HRNet",
        url="https://arxiv.org/abs/2301.00001",
        identifier="arxiv:2301.00001",
        snippet="contrastive learning HRNet on CIFAR-10.",
        year=2023,
        authors=["Ada"],
    )
    related = ExternalRef(
        source="arxiv",
        title="Momentum Contrast for Visual Representation",
        url="https://arxiv.org/abs/1911.05722",
        identifier="arxiv:1911.05722",
        snippet="We study contrastive learning HRNet for representation.",
        year=2019,
        authors=["He"],
    )
    gateway = MemoryMcpGateway([self_ref, related])
    calls: list[str] = []

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        if "检索问句生成器" in joined:
            calls.append("rewrite")
            assert "Contrastive HRNet" in joined
            assert "contrastive learning" in joined.lower()
            return '{"queries":["contrastive learning HRNet"]}'
        calls.append("answer")
        assert "1911.05722" in joined
        assert "2301.00001" not in joined
        return '{"answer_zh":"相关工作有 Momentum Contrast。"}'

    result = ask_arxiv(
        svc,
        "在 arXiv 上检索相关论文",
        settings=_settings(tmp_path),
        chat_fn=chat,
        gateway=gateway,
        current_paper_id="paper-a",
    )
    assert calls == ["rewrite", "answer"]
    assert result.mode == "arxiv"
    assert result.no_evidence is False
    assert gateway.calls[0][1]["query"] == "contrastive learning HRNet"
    assert [item.identifier for item in result.external_refs] == ["arxiv:1911.05722"]


def test_ask_arxiv_generic_uses_abstract_when_metadata_missing(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    heading = (
        "MODAL: Multi-Modal Object Re-ID via Model-Driven Sparse Decoupling "
        "and Text-Image Differential Filtering"
    )
    svc.store.save_paper(
        Paper(
            paper_id="paper-modal",
            sha256="paper-modal",
            filename="MODEL.pdf",
            status="ready",
            title=None,
            abstract="",
            index_status="ready",
            embedding_version="test-embed",
        )
    )
    svc.store.save_sections(
        "paper-modal",
        [
            Section(
                section_id="s-title",
                paper_id="paper-modal",
                title=heading,
                kind="method",
                level=1,
                page_start=1,
                page_end=1,
                text="Chengbo Huang. Abstract - Multi-modal object re-identification uses sparse decoupling.",
            )
        ],
    )
    self_ref = ExternalRef(
        source="arxiv",
        title=heading,
        url="https://arxiv.org/abs/2401.00001",
        identifier="arxiv:2401.00001",
        snippet="multi-modal object re-identification sparse decoupling.",
        year=2024,
        authors=["Huang"],
    )
    related = ExternalRef(
        source="arxiv",
        title="RGB-NIR Object Re-Identification with Cross-Modal Fusion",
        url="https://arxiv.org/abs/2401.00002",
        identifier="arxiv:2401.00002",
        snippet="multi-modal object re-identification with incomplete modalities.",
        year=2024,
        authors=["Li"],
    )
    gateway = MemoryMcpGateway([self_ref, related])

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        if "检索问句生成器" in joined:
            assert "MODAL" in joined
            assert "Multi-modal object re-identification" in joined
            return '{"queries":["multi-modal object re-identification"]}'
        return '{"answer_zh":"相关工作有跨模态 Re-ID。"}'

    result = ask_arxiv(
        svc,
        "在 arXiv 上检索相关论文",
        settings=_settings(tmp_path),
        chat_fn=chat,
        gateway=gateway,
        current_paper_id="paper-modal",
    )
    assert result.mode == "arxiv"
    assert result.no_evidence is False
    assert gateway.calls[0][1]["query"] == "multi-modal object re-identification"
    assert gateway.calls[0][1]["query"] != heading
    assert [item.identifier for item in result.external_refs] == ["arxiv:2401.00002"]


def test_ask_arxiv_generic_without_paper_asks_for_topic(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    gateway = MemoryMcpGateway([_diffusion_ref()])

    def chat(*args, **kwargs):
        raise AssertionError("LLM should not run")

    result = ask_arxiv(
        svc,
        "在 arXiv 上检索相关论文",
        settings=_settings(tmp_path),
        chat_fn=chat,
        gateway=gateway,
    )
    assert result.no_evidence is True
    assert result.answer_zh == NEED_ARXIV_TOPIC
    assert result.external_refs == []
    assert gateway.calls == []


def test_arxiv_ask_api_and_paper_route(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    gateway = MemoryMcpGateway([_diffusion_ref()])

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        if "arXiv hits" in joined or "文献检索助手" in joined:
            return '{"answer_zh":"网上有 Diffusion Policy。"}'
        if "Library cards" in joined or "本地论文库" in joined:
            return '{"answer_zh":"库里有 Diffusion Policy。","used_papers":[1]}'
        return '{"answer_zh":"方法里有 contrastive loss[1]。","used_evidence":[1]}'

    understand = UnderstandService(
        svc,
        _settings(tmp_path),
        chat_fn=chat,
        arxiv_gateway=gateway,
    )
    app.dependency_overrides[get_knowledge] = lambda: svc
    app.dependency_overrides[get_understand] = lambda: understand
    client = TestClient(app)
    try:
        direct = client.post("/arxiv/ask", json={"question": "在 arXiv 上检索 diffusion policy"})
        assert direct.status_code == 200, direct.text
        body = direct.json()
        assert body["mode"] == "arxiv"
        assert body["citations"] == []
        assert body["external_refs"][0]["identifier"] == "arxiv:2303.04137"

        routed = client.post(
            "/papers/paper-a/ask",
            json={"question": "在 arXiv 上检索 diffusion policy"},
        )
        assert routed.status_code == 200, routed.text
        payload = routed.json()
        assert payload["mode"] == "arxiv"
        assert payload["citations"] == []
        assert payload["external_refs"][0]["source"] == "arxiv"
        ids = [item["paper_id"] for item in payload["library_hits"]]
        assert "paper-b" in ids
        assert "paper-a" not in ids

        close_read = client.post("/papers/paper-a/ask", json={"question": "核心方法是什么？"})
        assert close_read.json()["mode"] == "close_read"
    finally:
        app.dependency_overrides.clear()


def test_ask_agent_routes_arxiv_without_close_read_worker(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    gateway = MemoryMcpGateway([_diffusion_ref()])

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        assert "Pinned Evidence" not in joined
        return '{"answer_zh":"arXiv 上有 Diffusion Policy。"}'

    def worker(state):
        raise AssertionError("close-read worker should not run for arxiv mode")

    understand = UnderstandService(
        svc,
        _settings(tmp_path, ask_mode="agent", ask_enable_external=True),
        chat_fn=chat,
        worker_fn=worker,
        arxiv_gateway=gateway,
    )
    result = understand.ask_agent("paper-a", "在 arXiv 上检索 diffusion policy")
    assert result.mode == "arxiv"
    assert result.citations == []
    assert result.external_refs
    assert gateway.calls


def test_arxiv_ask_empty_query_is_400(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    understand = UnderstandService(svc, _settings(tmp_path), chat_fn=lambda *a, **k: "{}")
    app.dependency_overrides[get_knowledge] = lambda: svc
    app.dependency_overrides[get_understand] = lambda: understand
    client = TestClient(app)
    try:
        response = client.post("/arxiv/ask", json={"question": "  "})
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()
