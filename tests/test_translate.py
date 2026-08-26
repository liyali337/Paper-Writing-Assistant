import json

import pytest
from fastapi.testclient import TestClient

from dl_agent.api.deps import get_knowledge, get_translate
from dl_agent.api.main import app
from dl_agent.config import Settings
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.layout import LayoutItem
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.translate.service import TranslateService
from tests.test_papers_api import _minimal_text_pdf


def _parse_stub(_path: str) -> ParseResult:
    body = "Readable paper text for translation tests. " * 20
    return ParseResult(
        parser="pymupdf",
        page_count=2,
        items=[
            LayoutItem(kind="heading", page=1, text="Abstract", level=1),
            LayoutItem(kind="text", page=1, text=body),
            LayoutItem(kind="heading", page=1, text="1 Introduction", level=1),
            LayoutItem(kind="text", page=1, text=body),
            LayoutItem(kind="heading", page=2, text="References", level=1),
            LayoutItem(kind="text", page=2, text="[1] Alice et al. Example Paper. 2024."),
        ],
    )


def _chat_stub(messages: list[dict[str, str]]) -> str:
    user = messages[-1]["content"]
    if "References" in user or "Bibliography" in user:
        raise AssertionError("references must not call the LLM")
    if "Abstract" in user:
        return json.dumps(
            {
                "title_zh": "摘要",
                "text_zh": "这是摘要的中文翻译。",
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "title_zh": "1 引言",
            "text_zh": "这是引言的中文翻译。",
        },
        ensure_ascii=False,
    )


@pytest.fixture()
def translate_client(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
    )
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=_parse_stub)
    translate = TranslateService(store, knowledge, settings, chat_fn=_chat_stub)
    app.dependency_overrides[get_knowledge] = lambda: knowledge
    app.dependency_overrides[get_translate] = lambda: translate
    client = TestClient(app)
    yield client, knowledge, translate
    app.dependency_overrides.clear()


def test_translate_sections_background(translate_client) -> None:
    client, knowledge, translate = translate_client
    uploaded = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    )
    paper_id = uploaded.json()["paper_id"]
    knowledge.finish_ingest(paper_id)

    started = client.post(f"/papers/{paper_id}/translations")
    assert started.status_code == 202

    result = translate.finish_translation(paper_id)
    assert result.status == "ready"
    assert len(result.sections) == 3
    assert result.sections[0].title_zh == "摘要"
    refs = result.sections[-1]
    assert refs.title_zh == "参考文献"
    assert "Alice" in refs.text_zh

    fetched = client.get(f"/papers/{paper_id}/translations")
    assert fetched.status_code == 200
    payload = fetched.json()
    assert payload["status"] == "ready"
    assert payload["sections"][0]["title_zh"] == "摘要"


def test_translate_cached_without_refresh(translate_client) -> None:
    client, knowledge, translate = translate_client
    uploaded = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    )
    paper_id = uploaded.json()["paper_id"]
    knowledge.finish_ingest(paper_id)
    translate.finish_translation(paper_id)

    again = client.post(f"/papers/{paper_id}/translations")
    assert again.status_code == 200
    assert again.json()["status"] == "ready"


def test_translate_requires_api_key(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, openai_api_key="", translate_section_delay_s=0)
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=_parse_stub)
    translate = TranslateService(store, knowledge, settings)
    app.dependency_overrides[get_knowledge] = lambda: knowledge
    app.dependency_overrides[get_translate] = lambda: translate
    client = TestClient(app)
    try:
        uploaded = client.post(
            "/papers",
            files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
        )
        paper_id = uploaded.json()["paper_id"]
        knowledge.finish_ingest(paper_id)
        response = client.post(f"/papers/{paper_id}/translations")
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()
