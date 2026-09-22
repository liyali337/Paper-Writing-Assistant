import json
import re
import threading

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
from dl_agent.translate.service import (
    TranslateService,
    _is_front_matter,
    _is_markup_heavy,
    _is_paper_title_section,
    _looks_like_chinese,
    _repair_and_check,
    _skip_section,
    _split_text_chunks,
    _structure_issues,
)
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
    if "figure captions" in user:
        ids = re.findall(r'"figure_id":\s*"([^"]+)"', user)
        return json.dumps(
            {
                "captions": [
                    {"figure_id": figure_id, "caption_zh": f"图 {index + 1}. 中文图注。"}
                    for index, figure_id in enumerate(ids)
                ]
            },
            ensure_ascii=False,
        )
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
        formula_vision_enabled=False,
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


def test_harvest_figure_captions_from_section_text() -> None:
    from dl_agent.domain.models import Figure
    from dl_agent.translate.service import _harvest_figure_captions

    figures = [
        Figure(
            figure_id="fig-001",
            paper_id="p",
            page=1,
            label="Figure 1",
            caption="Figure 1. Overview of the method.",
        )
    ]
    harvested = _harvest_figure_captions(
        ["前文。\n图 1. 方法总览：三个视觉层级对齐。\n后文如图1所示。"],
        figures,
    )
    assert "fig-001" in harvested
    assert harvested["fig-001"].startswith("图 1.")
    assert "方法总览" in harvested["fig-001"]


def test_parse_figure_caption_payload_accepts_loose_ids() -> None:
    from dl_agent.domain.models import Figure
    from dl_agent.translate.service import _parse_figure_caption_payload

    figures = [
        Figure(
            figure_id="fig-001",
            paper_id="p",
            page=1,
            label="Figure 1",
            caption="Figure 1. Overview of the method.",
        ),
        Figure(
            figure_id="fig-002",
            paper_id="p",
            page=2,
            label="Fig. 2",
            caption="Fig. 2. Pipeline.",
        ),
    ]
    by_label = _parse_figure_caption_payload(
        {
            "captions": [
                {"figure_id": "Figure 1", "caption_zh": "图 1. 方法总览。"},
                {"id": "fig-002", "zh": "图 2. 流程。"},
            ]
        },
        figures,
    )
    assert by_label["fig-001"].caption_zh == "图 1. 方法总览。"
    assert by_label["fig-002"].caption_zh == "图 2. 流程。"

    by_order = _parse_figure_caption_payload(
        {"captions": ["图 1. 方法总览。", "图 2. 流程。"]},
        figures,
    )
    assert by_order["fig-001"].caption_zh.startswith("图 1.")
    assert by_order["fig-002"].caption_zh.startswith("图 2.")

    empty = _parse_figure_caption_payload(
        {
            "captions": [
                {"figure_id": "fig-001", "caption_zh": "Figure 1. Overview of the method."},
            ]
        },
        figures,
    )
    assert empty == {}


def test_translate_figure_captions(tmp_path) -> None:
    from tests.helpers import make_png

    png = make_png(140, 100)
    body = "Readable paper text for translation tests. " * 20

    def parse(_path: str) -> ParseResult:
        return ParseResult(
            parser="pymupdf",
            page_count=1,
            items=[
                LayoutItem(kind="heading", page=1, text="Abstract", level=1),
                LayoutItem(kind="text", page=1, text=body),
                LayoutItem(
                    kind="picture",
                    page=1,
                    image_bytes=png,
                    width_px=140,
                    height_px=100,
                    caption="Figure 1: Our pipeline.",
                ),
            ],
        )

    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
        formula_vision_enabled=False,
    )
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=parse)
    translate = TranslateService(store, knowledge, settings, chat_fn=_chat_stub)
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
        result = _finish_translate(translate, paper_id)
        assert result.status == "ready"
        assert result.figures
        assert result.figures[0].figure_id.startswith("fig-")
        assert "图" in result.figures[0].caption_zh
        assert "中文图注" in result.figures[0].caption_zh

        wiped = result.model_copy(update={"figures": []})
        store.save_translation(wiped)
        backfill = _finish_translate(translate, paper_id)
        assert backfill.figures
        assert "中文图注" in backfill.figures[0].caption_zh
        assert backfill.sections[0].title_zh == result.sections[0].title_zh
    finally:
        app.dependency_overrides.clear()


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
    assert refreshed.json()["sections"] == []


def test_translate_refresh_does_not_reuse_cached_sections(translate_client) -> None:
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
    first = _finish_translate(translate, paper_id)
    assert "这是摘要的中文翻译" in first.sections[0].text_zh
    calls = {"n": 0}

    def refreshed_stub(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        user = messages[-1]["content"]
        if "paper title only" in user:
            return json.dumps({"title_zh": "新的论文标题"}, ensure_ascii=False)
        if "text fragment" in user:
            return json.dumps({"text_zh": "这是重新翻译的分块。"}, ensure_ascii=False)
        if "Abstract" in user:
            return json.dumps(
                {"title_zh": "摘要", "text_zh": "这是重新翻译的摘要。"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"title_zh": "1 引言", "text_zh": "这是重新翻译的引言。"},
            ensure_ascii=False,
        )

    translate.chat_fn = refreshed_stub
    second = _finish_translate(translate, paper_id, refresh=True)
    assert second.status == "ready"
    assert calls["n"] >= 2
    assert "这是重新翻译的摘要" in second.sections[0].text_zh
    assert "这是摘要的中文翻译" not in second.sections[0].text_zh


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
        formula_vision_enabled=False,
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


def test_skip_paper_title_author_section() -> None:
    section = Section(
        section_id="sec-title",
        paper_id="paper",
        title="Chain-of-Thought Guided Multi-Modal Object Re-Identification",
        kind="other",
        level=1,
        page_start=1,
        page_end=1,
        text="Ya Gao 1, Shihao Li 1 Anhui University, China\n\n{gaoya615}@foxmail.com",
        parent_id=None,
        figure_ids=[],
    )
    assert _is_paper_title_section(section)
    assert _skip_section(section)


def test_skip_model_driven_title_with_ieee_authors() -> None:
    section = Section(
        section_id="sec-title",
        paper_id="paper",
        title=(
            "MODAL: Multi-Modal Object Re-ID via Model-Driven Sparse "
            "Decoupling and Text-Image Differential Filtering"
        ),
        kind="method",
        level=1,
        page_start=1,
        page_end=1,
        text="Chengbo Huang, Jun-Jie Huang, Senior Member, IEEE and Meng Wang, Fellow, IEEE",
        parent_id=None,
        figure_ids=[],
    )
    assert _is_paper_title_section(section)
    assert _skip_section(section)


def test_translate_skips_paper_title_authors_without_llm(tmp_path) -> None:
    author_block = "Ya Gao 1, Shihao Li 1 Anhui University, China {gaoya615}@foxmail.com"
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
        formula_vision_enabled=False,
    )
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=_parse_stub)

    def boom_on_authors(messages: list[dict[str, str]]) -> str:
        user = messages[-1]["content"]
        if author_block in user:
            raise AssertionError("paper title authors must not call the LLM")
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
                section_id="sec-title",
                paper_id=paper_id,
                title="Chain-of-Thought Guided Multi-Modal Object Re-Identification",
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
        title = next(item for item in result.sections if item.section_id == "sec-title")
        assert re.sub(r"\s+", " ", title.text_zh) == re.sub(r"\s+", " ", author_block)
        assert "安徽" not in title.text_zh
    finally:
        app.dependency_overrides.clear()


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


def test_hard_split_does_not_cut_math() -> None:
    prefix = "The proven lower bound is " + ("context " * 12)
    math = r"$I(F_i, F_j|T)$"
    suffix = " which provides a principled way to control fusion. " * 8
    text = prefix + math + suffix
    chunks = _split_text_chunks(text, len(prefix) + 8)
    assert any(math in chunk.text for chunk in chunks)
    assert all(chunk.text.count("$") % 2 == 0 for chunk in chunks)
    assert not any(chunk.text.lstrip().startswith("|T)") for chunk in chunks)
    naked = prefix + r"I(F_i, F_j|T)" + suffix
    naked_chunks = _split_text_chunks(naked, len(prefix) + 8)
    assert any("I(F_i, F_j|T)" in chunk.text for chunk in naked_chunks)
    assert not any(chunk.text.lstrip().startswith("|T)") for chunk in naked_chunks)
    display = ("Lead-in sentence. " * 8) + "$$\n\\mathcal{L}=1\n$$" + (" Tail sentence." * 8)
    shown = _split_text_chunks(display, 40)
    formula = next(chunk for chunk in shown if "\\mathcal{L}=1" in chunk.text)
    assert formula.text.count("$$") % 2 == 0


def test_translate_long_section_in_chunks(tmp_path) -> None:
    long_body = ("Paragraph one. " * 120 + "\n\n") * 3
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
        formula_vision_enabled=False,
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
        formula_vision_enabled=False,
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


def test_translate_ready_requeues_missing_section(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
        formula_vision_enabled=False,
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
        abstract = next(section for section in knowledge.get_sections(paper_id) if section.kind == "abstract")
        store.save_translation(
            first.model_copy(
                update={
                    "sections": [item for item in first.sections if item.section_id != abstract.section_id],
                    "status": "ready",
                }
            )
        )
        calls.clear()
        _, started, run_id = translate.start_translation(paper_id)
        assert started is True
        second = translate.finish_translation(paper_id, run_id)
        assert second.status == "ready"
        assert any(item.section_id == abstract.section_id for item in second.sections)
        assert "abstract" in calls
    finally:
        app.dependency_overrides.clear()


def test_translate_requires_api_key(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="",
        translate_section_delay_s=0,
        formula_vision_enabled=False,
    )
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


def test_cancel_stops_running_translation(translate_client) -> None:
    client, knowledge, translate = translate_client
    paper_id = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    ).json()["paper_id"]
    knowledge.finish_ingest(paper_id)

    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def blocking_chat(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        started.set()
        assert release.wait(timeout=5)
        return _chat_stub(messages)

    translate.chat_fn = blocking_chat
    _, _, run_id = translate.start_translation(paper_id)
    worker = threading.Thread(
        target=translate.finish_translation,
        args=(paper_id, run_id),
        daemon=True,
    )
    worker.start()
    assert started.wait(timeout=5)

    cancelled = client.post(f"/papers/{paper_id}/translations/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["error"] == "已终止翻译"

    release.set()
    worker.join(timeout=8)
    assert not worker.is_alive()

    fetched = client.get(f"/papers/{paper_id}/translations")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "cancelled"
    assert calls["n"] == 1

    translate.chat_fn = _chat_stub
    result = _finish_translate(translate, paper_id)
    assert result.status == "ready"


def test_cancel_when_idle_keeps_ready(translate_client) -> None:
    client, knowledge, translate = translate_client
    paper_id = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    ).json()["paper_id"]
    knowledge.finish_ingest(paper_id)
    _finish_translate(translate, paper_id)
    response = client.post(f"/papers/{paper_id}/translations/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_cancel_without_translation_404(translate_client) -> None:
    client, knowledge, _translate = translate_client
    paper_id = client.post(
        "/papers",
        files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
    ).json()["paper_id"]
    knowledge.finish_ingest(paper_id)
    response = client.post(f"/papers/{paper_id}/translations/cancel")
    assert response.status_code == 404


def test_markup_heavy_formula_chunk_keeps_short_chinese() -> None:
    source = "$$\n\\mathcal{L}_{i2j}=x\n$$\nwhere $x$ is the query."
    zh = "$$\n\\mathcal{L}_{i2j}=x\n$$\n其中 $x$."
    assert _is_markup_heavy(source)
    assert _looks_like_chinese(zh, source=source)
    copied = source
    repaired, issues = _repair_and_check(source, copied)
    assert "not_chinese" not in issues
    assert "verbatim" not in issues
    assert repaired.count("$$") == source.count("$$")


def test_prose_english_copy_is_still_rejected() -> None:
    source = (
        "Existing methods merely adopt descriptive representation learning for image-text, "
        "ignoring the relationships among the intrinsic logical hierarchies of semantic features."
    )
    assert not _is_markup_heavy(source)
    assert not _looks_like_chinese(source, source=source)
    _, issues = _repair_and_check(source, source)
    assert "verbatim" in issues or "not_chinese" in issues


def test_table_chunk_not_rejected_without_han() -> None:
    source = "\n".join(
        [
            "| Method | Venue | mAP | R-1 |",
            "|---|---|---|---|",
            "| BoT [18] | CVPRW'19 | 78.0 | 95.1 |",
            "| OSNet [48] | ICCV'19 | 75.0 | 95.6 |",
            "| AGW [37] | TPAMI'21 | 73.1 | 92.7 |",
        ]
    )
    assert _is_markup_heavy(source)
    assert _looks_like_chinese(source, source=source)
    _, issues = _repair_and_check(source, source)
    assert "not_chinese" not in issues
    assert "verbatim" not in issues


def test_structure_flags_dropped_math_and_paragraphs() -> None:
    source = "Lead sentence about the bound.\n\n$$\n\\mathcal{L}=1\n$$\n\nwhere $x$ is the query."
    dropped = "导语句在这里，公式和后文都被省略了。"
    issues = _structure_issues(source, dropped)
    assert any(item.startswith("display-math") for item in issues)
    assert any(item.startswith("paragraphs") for item in issues)


def test_repair_keeps_translated_latex() -> None:
    source = "Lead.\n$$\nL_g(F)=1\n$$\nTail."
    zh = "导语。\n$$\n\\mathcal{L}_{g}(F)=1\n$$\n结尾。"
    out, issues = _repair_and_check(source, zh)
    assert "display-math" not in ",".join(issues)
    assert "\\mathcal{L}_{g}(F)=1" in out
    assert "L_g(F)=1" not in out


def test_chunk_retries_when_paragraphs_or_math_drop(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="test-key",
        translate_section_delay_s=0,
        formula_vision_enabled=False,
    )
    store = FilePaperStore(tmp_path)
    knowledge = KnowledgeService(store, settings, parse_fn=_parse_stub)
    calls = {"n": 0}

    def stub(messages: list[dict[str, str]]) -> str:
        calls["n"] += 1
        user = messages[-1]["content"]
        if "Retry: match paragraph breaks" in user:
            return json.dumps(
                {
                    "text_zh": (
                        "导语句说明下界。\n\n"
                        "$$\n\\mathcal{L}=1\n$$\n\n"
                        "其中 $x$ 表示查询。"
                    )
                },
                ensure_ascii=False,
            )
        return json.dumps({"text_zh": "导语句说明下界，后面都漏了。"}, ensure_ascii=False)

    translate = TranslateService(store, knowledge, settings, chat_fn=stub)
    source = "Lead sentence about the bound.\n\n$$\n\\mathcal{L}=1\n$$\n\nwhere $x$ is the query."
    out = translate._translate_text_chunk_llm(source, paper_id="paper", run_id=1)
    assert calls["n"] == 2
    assert len(_structure_issues(source, out)) == 0
    assert "\\mathcal{L}=1" in out
    assert "其中" in out
