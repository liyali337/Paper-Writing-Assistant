from pathlib import Path

from fastapi.testclient import TestClient

from dl_agent.api.deps import get_knowledge, get_understand
from dl_agent.api.main import app
from dl_agent.config import Settings
from dl_agent.domain.models import Paper, Section
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.understand.library import NO_LIBRARY_HITS, NO_PAPER_CARD, LibraryTools
from dl_agent.understand.route import classify_ask_mode, is_obvious_chat, suggest_task_kinds
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
    index_status: str = "ready",
    status: str = "ready",
) -> None:
    knowledge.store.save_paper(
        Paper(
            paper_id=paper_id,
            sha256=paper_id,
            filename=f"{paper_id}.pdf",
            status=status,  # type: ignore[arg-type]
            title=title,
            authors=["Ada"],
            abstract=abstract,
            index_status=index_status,  # type: ignore[arg-type]
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
    if status == "ready" and index_status == "ready":
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


def test_query_corpus_groups_by_paper_and_keeps_abstracts(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    hits = svc.query_corpus("contrastive HRNet")
    assert hits
    assert hits[0].paper_id == "paper-a"
    assert hits[0].title == "Contrastive HRNet"
    assert hits[0].abstract and "contrastive" in hits[0].abstract.lower()
    assert hits[0].openable is True
    assert all(item.paper_id != "paper-b" or "diffusion" in (item.abstract or "").lower() for item in hits)
    other = svc.query_corpus("diffusion policy")
    assert other
    assert other[0].paper_id == "paper-b"
    assert "diffusion" in (other[0].abstract or "").lower()
    assert "contrastive" not in (other[0].abstract or "").lower()


def test_query_corpus_skips_unindexed_paper(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    _save_paper(
        svc,
        "paper-c",
        title="Pending Paper",
        abstract="Pending contrastive notes.",
        method="Not indexed yet.",
        index_status="pending",
    )
    hits = svc.query_corpus("contrastive")
    ids = {item.paper_id for item in hits}
    assert "paper-c" not in ids
    assert "paper-a" in ids


def test_single_paper_query_still_does_not_cross(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    hits = svc.query("paper-a", "contrastive loss")
    assert hits
    assert all(item.section_id and item.section_id.startswith("paper-a") for item in hits)


def test_library_tools_search_and_card(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    tools = LibraryTools(svc)
    text = tools.search_library("diffusion policy")
    assert text != NO_LIBRARY_HITS
    assert "paper-b" in text
    assert tools.last_hits[0].paper_id == "paper-b"
    card = tools.get_paper_card("paper-b")
    assert "Diffusion Policy" in card
    assert tools.get_paper_card("missing") == NO_PAPER_CARD
    assert tools.search_library("   ") == NO_LIBRARY_HITS


def test_library_search_and_ask_api(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)

    def chat(messages, **kwargs):
        _ = (messages, kwargs)
        return '{"answer_zh":"库里有一篇 Diffusion Policy，讲机器人操作。细节需打开该篇精读。","used_papers":[1]}'

    understand = UnderstandService(svc, _settings(tmp_path), chat_fn=chat)
    app.dependency_overrides[get_knowledge] = lambda: svc
    app.dependency_overrides[get_understand] = lambda: understand
    client = TestClient(app)
    try:
        searched = client.get("/library/search", params={"q": "diffusion policy"})
        assert searched.status_code == 200
        body = searched.json()
        assert body[0]["paper_id"] == "paper-b"
        assert "diffusion" in (body[0]["abstract"] or "").lower()
        assert body[0]["openable"] is True

        asked = client.post("/library/ask", json={"question": "我研读过的 diffusion 论文"})
        assert asked.status_code == 200
        payload = asked.json()
        assert payload["mode"] == "library"
        assert payload["citations"] == []
        assert payload["library_hits"][0]["paper_id"] == "paper-b"
        assert "Diffusion Policy" in payload["answer_zh"]
        assert payload["no_evidence"] is False
    finally:
        app.dependency_overrides.clear()


def test_library_ask_empty_query_is_400(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    understand = UnderstandService(svc, _settings(tmp_path), chat_fn=lambda *a, **k: "{}")
    app.dependency_overrides[get_knowledge] = lambda: svc
    app.dependency_overrides[get_understand] = lambda: understand
    client = TestClient(app)
    try:
        response = client.post("/library/ask", json={"question": "  "})
        assert response.status_code == 400
    finally:
        app.dependency_overrides.clear()


def test_library_ask_no_hits(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    understand = UnderstandService(
        svc,
        _settings(tmp_path),
        chat_fn=lambda *a, **k: '{"answer_zh":"should not run"}',
    )
    result = understand.ask_library("xyzzy zebra not in corpus")
    assert result.no_evidence is True
    assert result.library_hits == []
    assert result.mode == "library"


def test_classify_ask_mode_library_vs_close_read() -> None:
    assert classify_ask_mode("核心方法是什么？") == "close_read"
    assert classify_ask_mode("这篇的实验结论？") == "close_read"
    assert classify_ask_mode("本文相关工作写了什么") == "close_read"
    assert classify_ask_mode("我之前研读过类似方向的论文吗？") == "library"
    assert classify_ask_mode("本地库里还有没有 diffusion") == "library"
    assert classify_ask_mode("Have I studied diffusion policy papers?") == "library"
    assert classify_ask_mode("在 arXiv 上检索相关论文") == "arxiv"
    assert classify_ask_mode("search papers on attention") == "arxiv"
    assert classify_ask_mode("检索本地库相关论文") == "library"


def test_is_obvious_chat_identity_not_paper_question() -> None:
    assert is_obvious_chat("你是干什么的")
    assert is_obvious_chat("Who are you?")
    assert is_obvious_chat("怎么用这个助手")
    assert not is_obvious_chat("这篇在干什么")
    assert not is_obvious_chat("核心方法是什么？")
    assert not is_obvious_chat("怎么用这个损失函数")


def test_suggest_task_kinds_allows_mixed_close_read_and_library() -> None:
    assert suggest_task_kinds("核心方法是什么？") == ["close_read"]
    assert suggest_task_kinds("我之前研读过类似方向的论文吗？") == ["library"]
    assert suggest_task_kinds("检索本地库相关论文") == ["library"]
    assert suggest_task_kinds("在 arXiv 上检索相关论文") == ["arxiv"]
    assert suggest_task_kinds("这篇方法和我库里那篇差在哪") == ["close_read", "library"]
    assert suggest_task_kinds("本文相关工作写了什么") == ["close_read"]


def test_query_corpus_excludes_current_paper(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    hits = svc.query_corpus("contrastive HRNet", exclude_paper_id="paper-a")
    assert all(item.paper_id != "paper-a" for item in hits)


def test_paper_ask_routes_library_question(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        if "Library cards" in joined or "本地论文库" in joined:
            return '{"answer_zh":"库里有 Diffusion Policy。","used_papers":[1]}'
        return '{"answer_zh":"方法里有 contrastive loss[1]。","used_evidence":[1]}'

    understand = UnderstandService(svc, _settings(tmp_path), chat_fn=chat)
    app.dependency_overrides[get_knowledge] = lambda: svc
    app.dependency_overrides[get_understand] = lambda: understand
    client = TestClient(app)
    try:
        close_read = client.post("/papers/paper-a/ask", json={"question": "核心方法是什么？"})
        assert close_read.status_code == 200, close_read.text
        assert close_read.json()["mode"] == "close_read"
        assert close_read.json().get("library_hits") in (None, [])

        library = client.post(
            "/papers/paper-a/ask",
            json={"question": "我之前研读过 diffusion 方向的论文吗？"},
        )
        assert library.status_code == 200, library.text
        payload = library.json()
        assert payload["mode"] == "library"
        assert payload["citations"] == []
        ids = [item["paper_id"] for item in payload["library_hits"]]
        assert "paper-a" not in ids
        assert "paper-b" in ids
    finally:
        app.dependency_overrides.clear()


def test_paper_ask_library_when_current_index_pending(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)
    paper = svc.get_paper("paper-a")
    paper.index_status = "pending"
    svc.store.save_paper(paper)

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        if "Library cards" in joined or "本地论文库" in joined:
            return '{"answer_zh":"库里有 Diffusion Policy。","used_papers":[1]}'
        raise AssertionError("close-read LLM should not run")

    understand = UnderstandService(svc, _settings(tmp_path), chat_fn=chat)
    app.dependency_overrides[get_knowledge] = lambda: svc
    app.dependency_overrides[get_understand] = lambda: understand
    client = TestClient(app)
    try:
        response = client.post(
            "/papers/paper-a/ask",
            json={"question": "我之前研读过 diffusion 方向的论文吗？"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["mode"] == "library"
        assert "paper-b" in [item["paper_id"] for item in payload["library_hits"]]
    finally:
        app.dependency_overrides.clear()


def test_paper_ask_mixed_fills_library_hits(tmp_path: Path) -> None:
    svc = _two_papers(tmp_path)

    def chat(messages, **kwargs):
        _ = kwargs
        joined = "\n".join(str(item.get("content") or "") for item in messages)
        if "Library cards" in joined or "本地论文库" in joined:
            return '{"answer_zh":"库里有 Diffusion Policy。","used_papers":[1]}'
        return '{"answer_zh":"方法里有 contrastive loss[1]。","used_evidence":[1]}'

    understand = UnderstandService(svc, _settings(tmp_path), chat_fn=chat)
    app.dependency_overrides[get_knowledge] = lambda: svc
    app.dependency_overrides[get_understand] = lambda: understand
    client = TestClient(app)
    try:
        response = client.post(
            "/papers/paper-a/ask",
            json={"question": "这篇方法和我库里那篇差在哪"},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["mode"] == "close_read"
        assert payload["citations"]
        assert "paper-b" in [item["paper_id"] for item in payload["library_hits"]]
        assert "库里有 Diffusion Policy" in payload["answer_zh"]
    finally:
        app.dependency_overrides.clear()
