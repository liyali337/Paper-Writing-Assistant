import json
from pathlib import Path

from fastapi.testclient import TestClient

from dl_agent.api.deps import get_knowledge, get_understand
from dl_agent.api.main import app
from dl_agent.config import Settings
from dl_agent.domain.models import Paper, Section
from dl_agent.harness.complete import LlmNotConfiguredError
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.understand.service import UnderstandService


def _embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        low = text.lower()
        vectors.append(
            [
                1.0 if "lambda" in low or "l_con" in low else 0.0,
                1.0 if "contrastive" in low or "cifar" in low else 0.0,
                1.0 if "xyzzy" in low or "zebra" in low else 0.0,
            ]
        )
    return vectors


def _settings(tmp_path: Path, **kwargs) -> Settings:
    data = {
        "data_dir": tmp_path,
        "formula_vision_enabled": False,
        "ask_mode": "agent",
    }
    data.update(kwargs)
    return Settings(**data)


def _knowledge(tmp_path: Path, settings: Settings | None = None) -> KnowledgeService:
    return KnowledgeService(
        FilePaperStore(tmp_path),
        settings or _settings(tmp_path),
        embed_fn=_embed,
        vector_index=MemoryVectorIndex(),
    )


def _save_ready_paper(
    knowledge: KnowledgeService,
    *,
    status: str = "ready",
    index_status: str = "ready",
    index_error: str | None = None,
    text: str = "The total loss is L = lambda * L_con + L_ce.",
) -> str:
    paper_id = "paper-a"
    knowledge.store.save_paper(
        Paper(
            paper_id=paper_id,
            sha256=paper_id,
            filename="paper-a.pdf",
            status=status,  # type: ignore[arg-type]
            index_status=index_status,  # type: ignore[arg-type]
            index_error=index_error,
            embedding_version="lexical-v1",
        )
    )
    knowledge.store.save_sections(
        paper_id,
        [
            Section(
                section_id="s-method",
                paper_id=paper_id,
                title="4. Approach",
                kind="method",
                level=1,
                page_start=6,
                page_end=8,
                text=text,
                figure_ids=["fig-1"],
            )
        ],
    )
    if status == "ready" and index_status == "ready":
        indexed = knowledge.index_paper(paper_id, force=True)
        assert indexed.index_status == "ready"
    return paper_id


def _fake_chat(messages, **_kwargs) -> str:
    joined = "\n".join(str(item.get("content") or "") for item in messages)
    if "对话调度器" in joined or "[Current question]" in joined or "改写成适合" in joined:
        query = joined.split("[Current question]", 1)[-1].strip().splitlines()[0].strip()
        if not query or query == joined.strip().splitlines()[0].strip():
            query = joined.split("User Query:", 1)[-1].strip().splitlines()[0].strip() if "User Query:" in joined else query
        return json.dumps(
            {
                "intent": "retrieve",
                "kinds": ["close_read"],
                "tasks": [{"kind": "close_read", "question": query or "lambda"}],
                "reply": "",
            },
            ensure_ascii=False,
        )
    if "合成一条" in joined:
        return json.dumps(
            {"answer_zh": "λ 设为与 L_con 相乘[1]。", "used_evidence": [1], "partial": False},
            ensure_ascii=False,
        )
    if "只根据下列编号证据" in joined or "Pinned Evidence" in joined:
        return json.dumps(
            {"answer_zh": "λ 与对比损失相乘[1]。", "used_evidence": [1], "partial": False},
            ensure_ascii=False,
        )
    return "根据检索，λ 与 L_con 相乘。"


def _client(knowledge: KnowledgeService, understand: UnderstandService) -> TestClient:
    app.dependency_overrides[get_knowledge] = lambda: knowledge
    app.dependency_overrides[get_understand] = lambda: understand
    return TestClient(app)


def _understand(knowledge: KnowledgeService, settings: Settings) -> UnderstandService:
    return UnderstandService(knowledge, settings, chat_fn=_fake_chat)


def test_ask_agent_returns_paper_answer(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ask_mode="agent")
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge)
    understand = _understand(knowledge, settings)
    client = _client(knowledge, understand)
    try:
        search = client.get(f"/papers/{paper_id}/search", params={"q": "lambda"})
        assert search.status_code == 200
        assert search.json()

        response = client.post(f"/papers/{paper_id}/ask", json={"question": "损失函数里的 lambda 怎么设？"})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["paper_id"] == paper_id
        assert payload["question"] == "损失函数里的 lambda 怎么设？"
        assert payload["no_evidence"] is False
        assert payload["answer_zh"]
        assert payload["prompt_version"] == "ask-agent-v1"
        assert payload["model"]
        assert payload["embedding_version"]
        assert "citations" in payload
        assert "figure_ids" in payload
        assert "partial" in payload
    finally:
        app.dependency_overrides.clear()


def test_ask_simple_mode_same_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ask_mode="simple")
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge)
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "损失函数里的 lambda 怎么设？"})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["prompt_version"] == "ask-v1"
        assert payload["no_evidence"] is False
        assert payload["answer_zh"]
        assert "citations" in payload
        assert "figure_ids" in payload
        assert "partial" in payload
        assert "model" in payload
        assert "embedding_version" in payload
    finally:
        app.dependency_overrides.clear()


def test_ask_falls_back_to_simple_without_langgraph(tmp_path: Path, monkeypatch) -> None:
    import dl_agent.api.routes.papers as papers_routes

    monkeypatch.setattr(papers_routes, "langgraph_available", lambda: False)
    settings = _settings(tmp_path, ask_mode="agent")
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge)
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "损失函数里的 lambda 怎么设？"})
        assert response.status_code == 200, response.text
        assert response.json()["prompt_version"] == "ask-v1"
    finally:
        app.dependency_overrides.clear()


def test_ask_empty_evidence_is_200(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ask_mode="agent")
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge)
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "xyzzy zebra unrelated"})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["no_evidence"] is True
        assert "没有找到直接证据" in payload["answer_zh"]
        assert payload["citations"] == []
    finally:
        app.dependency_overrides.clear()


def test_ask_404_unknown_paper(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    knowledge = _knowledge(tmp_path, settings)
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post("/papers/missing/ask", json={"question": "核心方法是什么？"})
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "not_found"
    finally:
        app.dependency_overrides.clear()


def test_ask_409_paper_not_ready(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge, status="parsing", index_status="pending")
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "核心方法是什么？"})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "not_ready"
    finally:
        app.dependency_overrides.clear()


def test_ask_409_index_pending(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge, status="ready", index_status="pending")
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "核心方法是什么？"})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "index_pending"
    finally:
        app.dependency_overrides.clear()


def test_ask_503_index_failed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(
        knowledge, status="ready", index_status="failed", index_error="embed timeout"
    )
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "核心方法是什么？"})
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "index_failed"
        assert "embed timeout" in detail["message"]
    finally:
        app.dependency_overrides.clear()


def test_ask_503_llm_not_configured(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ask_mode="simple")
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge)

    def boom(*_args, **_kwargs) -> str:
        raise LlmNotConfiguredError("未配置 OPENAI_API_KEY，无法调用翻译模型")

    understand = UnderstandService(knowledge, settings, chat_fn=boom)
    client = _client(knowledge, understand)
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "损失函数里的 lambda 怎么设？"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "llm_not_configured"
    finally:
        app.dependency_overrides.clear()


def test_ask_400_empty_question(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    knowledge = _knowledge(tmp_path, settings)
    paper_id = _save_ready_paper(knowledge)
    client = _client(knowledge, _understand(knowledge, settings))
    try:
        response = client.post(f"/papers/{paper_id}/ask", json={"question": "   "})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "invalid_request"
    finally:
        app.dependency_overrides.clear()
