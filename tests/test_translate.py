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
from dl_agent.domain.models import Section
from dl_agent.translate.service import TranslateService, _is_front_matter, _skip_section, _split_text_chunks
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


def _finish_translate(translate: TranslateService, paper_id: str, *, refresh: bool = False):
    _, _, run_id = translate.start_translation(paper_id, refresh=refresh)
    return translate.finish_translation(paper_id, run_id)


def _chat_stub(messages: list[dict[str, str]]) -> str:
    user = messages[-1]["content"]
    if "References" in user or "Bibliography" in user:
        raise AssertionError("references must not call the LLM")
    if "paper title only" in user:
        return json.dumps({"title_zh": "示例论文标题"}, ensure_ascii=False)
    if "text fragment" in user:
        return json.dumps({"text_zh": "这是分块的中文翻译。"}, ensure_ascii=False)
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
    paper = knowledge.get_paper(paper_id)
    paper.title = "Example Paper for Translation"
    knowledge.store.save_paper(paper)

    started = client.post(f"/papers/{paper_id}/translations")
    assert started.status_code == 200
    assert started.json()["status"] == "pending"

    result = _finish_translate(translate, paper_id)
    assert result.status == "ready"
    assert result.title_zh == "示例论文标题"
    assert len(result.sections) == 3
    assert result.sections[0].title_zh == "摘要"
    refs = result.sections[-1]
    assert refs.title_zh == "参考文献"
    assert "Alice" in refs.text_zh

    fetched = client.get(f"/papers/{paper_id}/translations")
    assert fetched.status_code == 200
    payload = fetched.json()
    assert payload["status"] == "ready"
    assert payload["title_zh"] == "示例论文标题"
    assert payload["sections"][0]["title_zh"] == "摘要"


def test_translate_refresh_reruns(translate_client) -> None:
    client, knowledge, translate = translate_client
    uploaded = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    )
    paper_id = uploaded.json()["paper_id"]
    knowledge.finish_ingest(paper_id)
    _finish_translate(translate, paper_id)

    refreshed = client.post(f"/papers/{paper_id}/translations?refresh=true")
    assert refreshed.status_code == 200
    assert refreshed.json()["status"] == "pending"


def test_translate_fatal_llm_marks_failed(translate_client) -> None:
    client, knowledge, translate = translate_client

    def boom(_messages: list[dict[str, str]]) -> str:
        from dl_agent.harness.complete import LlmRequestError

        raise LlmRequestError("LLM 返回 401: ACCESS_TOKEN_TYPE_UNSUPPORTED", status_code=401, fatal=True)

    translate.chat_fn = boom
    uploaded = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    )
    paper_id = uploaded.json()["paper_id"]
    knowledge.finish_ingest(paper_id)
    result = _finish_translate(translate, paper_id)
    assert result.status == "failed"
    fetched = client.get(f"/papers/{paper_id}/translations")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "failed"


def test_translate_cached_without_refresh(translate_client) -> None:
    client, knowledge, translate = translate_client
    uploaded = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    )
    paper_id = uploaded.json()["paper_id"]
    knowledge.finish_ingest(paper_id)
    _finish_translate(translate, paper_id)

    again = client.post(f"/papers/{paper_id}/translations")
    assert again.status_code == 200
    assert again.json()["status"] == "ready"


def test_translate_skips_front_matter_without_llm(tmp_path) -> None:
    author_block = "Yali Li 1, Qianru Han 1 Huazhong Agricultural University"
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
    )
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=_parse_stub)

    def boom_on_authors(messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"]
        if author_block in user:
            raise AssertionError("front matter authors must not call the LLM")
        return _chat_stub(messages)

    translate = TranslateService(store, knowledge, settings, chat_fn=boom_on_authors)
    app.dependency_overrides[get_knowledge] = lambda: knowledge
    app.dependency_overrides[get_translate] = lambda: translate
    client = TestClient(app)
    try:
        paper_id = client.post(
            "/papers",
            files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
        ).json()["paper_id"]
        knowledge.finish_ingest(paper_id)
        sections = knowledge.get_sections(paper_id)
        sections.insert(
            0,
            Section(
                section_id="sec-front",
                paper_id=paper_id,
                title="Front Matter",
                kind="other",
                level=1,
                page_start=1,
                page_end=1,
                text=author_block,
                parent_id=None,
                figure_ids=[],
            ),
        )
        store.save_sections(paper_id, sections)
        result = _finish_translate(translate, paper_id)
        front = next(item for item in result.sections if item.section_id == "sec-front")
        assert front.text_zh == author_block
        assert front.title_zh == "Front Matter"
        assert "李" not in front.text_zh
    finally:
        app.dependency_overrides.clear()


def test_skip_front_matter_section() -> None:
    section = Section(
        section_id="sec-front",
        paper_id="paper",
        title="Front Matter",
        kind="other",
        level=1,
        page_start=1,
        page_end=1,
        text="Alice One, Bob Two, Example University",
        parent_id=None,
        figure_ids=[],
    )
    assert _is_front_matter(section)
    assert _skip_section(section)


def test_split_text_chunks_by_paragraph() -> None:
    text = "A" * 900 + "\n\n" + "B" * 900 + "\n\n" + "C" * 900
    chunks = _split_text_chunks(text, 1000)
    assert len(chunks) == 3
    assert all(len(chunk.text) <= 1000 for chunk in chunks)
    assert all(chunk.paragraph_start for chunk in chunks)


def test_hard_split_marks_continuation() -> None:
    text = "word " * 300
    chunks = _split_text_chunks(text, 800)
    assert len(chunks) >= 2
    assert chunks[0].paragraph_start is True
    assert chunks[1].paragraph_start is False


def test_translate_long_section_in_chunks(tmp_path) -> None:
    long_body = ("Paragraph one. " * 120 + "\n\n") * 3
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
        translate_chunk_max_chars=800,
        translate_parallelism=1,
    )
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=_parse_stub)
    translate = TranslateService(store, knowledge, settings, chat_fn=_chat_stub)
    app.dependency_overrides[get_knowledge] = lambda: knowledge
    app.dependency_overrides[get_translate] = lambda: translate
    client = TestClient(app)
    try:
        paper_id = client.post(
            "/papers",
            files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
        ).json()["paper_id"]
        knowledge.finish_ingest(paper_id)
        sections = knowledge.get_sections(paper_id)
        intro = next(section for section in sections if "Introduction" in section.title)
        intro.text = long_body
        store.save_sections(paper_id, sections)

        result = _finish_translate(translate, paper_id)
        intro_zh = next(item for item in result.sections if item.section_id == intro.section_id)
        assert intro_zh.text_zh.count("这是") >= 2
        assert result.status == "ready"
    finally:
        app.dependency_overrides.clear()


def test_translate_resume_skips_cached_sections(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
        translate_parallelism=1,
    )
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=_parse_stub)
    calls: list[str] = []

    def counting_stub(messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"]
        if "paper title only" in user:
            return json.dumps({"title_zh": "示例论文标题"}, ensure_ascii=False)
        if "Abstract" in user:
            calls.append("abstract")
            return json.dumps({"title_zh": "摘要", "text_zh": "这是摘要的中文翻译。"}, ensure_ascii=False)
        calls.append("other")
        return json.dumps({"title_zh": "1 引言", "text_zh": "这是引言的中文翻译。"}, ensure_ascii=False)

    translate = TranslateService(store, knowledge, settings, chat_fn=counting_stub)
    uploaded = TestClient(app)
    app.dependency_overrides[get_knowledge] = lambda: knowledge
    app.dependency_overrides[get_translate] = lambda: translate
    client = TestClient(app)
    try:
        paper_id = client.post(
            "/papers",
            files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
        ).json()["paper_id"]
        knowledge.finish_ingest(paper_id)
        first = _finish_translate(translate, paper_id)
        assert first.status == "ready"
        abstract_id = first.sections[0].section_id
        store.save_translation(
            first.model_copy(
                update={
                    "sections": [item for item in first.sections if item.section_id != abstract_id],
                    "status": "pending",
                }
            )
        )
        calls.clear()
        _, _, run_id = translate.start_translation(paper_id)
        second = translate.finish_translation(paper_id, run_id)
        assert second.status == "ready"
        assert any(item.section_id == abstract_id for item in second.sections)
        assert "abstract" in calls
        assert calls.count("abstract") == 1
    finally:
        app.dependency_overrides.clear()


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
