from pathlib import Path

from dl_agent.config import Settings
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.layout import LayoutItem
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from tests.helpers import make_png


def _service(tmp_path: Path, parse_fn) -> KnowledgeService:
    settings = Settings(data_dir=tmp_path, formula_vision_enabled=False)
    return KnowledgeService(FilePaperStore(tmp_path), settings, parse_fn=parse_fn)


def _rich_items() -> list[LayoutItem]:
    png = make_png(120, 90)
    body = "This sentence is long enough to count as paper text. " * 20
    return [
        LayoutItem(kind="heading", page=1, text="Abstract", level=1),
        LayoutItem(kind="text", page=1, text=body),
        LayoutItem(kind="heading", page=1, text="1. Introduction", level=1),
        LayoutItem(kind="text", page=1, text=body),
        LayoutItem(kind="heading", page=2, text="2. Approach", level=1),
        LayoutItem(kind="text", page=2, text=body),
        LayoutItem(
            kind="picture",
            page=2,
            image_bytes=png,
            width_px=120,
            height_px=90,
            caption="Figure 1: Architecture overview.",
        ),
        LayoutItem(kind="heading", page=3, text="References", level=1),
        LayoutItem(kind="text", page=3, text="[1] Prior work."),
    ]


def test_ingest_ready_binds_figure(tmp_path: Path) -> None:
    calls = {"n": 0}

    def parse(_path: str) -> ParseResult:
        calls["n"] += 1
        return ParseResult(parser="pymupdf", page_count=3, items=_rich_items())

    svc = _service(tmp_path, parse)
    pdf = b"%PDF-1.4 fake but magic only" + b"\n" * 100
    # validate_pdf_bytes opens with fitz; write a real tiny pdf instead
    pdf = _minimal_text_pdf()
    paper = svc.ingest(pdf, "demo.pdf")
    assert paper.status == "ready"
    assert paper.index_status == "ready"
    assert paper.parser == "pymupdf"
    assert paper.figure_count == 1
    assert paper.intro_status == "skipped"
    sections = svc.get_sections(paper.paper_id)
    figures = svc.get_figures(paper.paper_id)
    method = next(section for section in sections if "Approach" in section.title)
    assert figures[0].section_id == method.section_id
    assert svc.get_figure_path(paper.paper_id, figures[0].figure_id).exists()
    assert calls["n"] == 1

    again = svc.ingest(pdf, "demo.pdf")
    assert again.paper_id == paper.paper_id
    assert calls["n"] == 1

    first, run_first = svc.start_ingest(pdf, "demo.pdf")
    second, run_second = svc.start_ingest(pdf, "demo.pdf")
    assert first.paper_id == second.paper_id == paper.paper_id
    assert run_first is False and run_second is False

    svc.delete_index(paper.paper_id)
    assert svc.get_paper(paper.paper_id).index_status == "pending"
    rebuilt = svc.ingest(pdf, "demo.pdf")
    assert rebuilt.paper_id == paper.paper_id
    assert rebuilt.index_status == "ready"
    assert calls["n"] == 1


def test_reparse_ignores_sha_dedup(tmp_path: Path) -> None:
    calls = {"n": 0}

    def parse(_path: str) -> ParseResult:
        calls["n"] += 1
        return ParseResult(parser="pymupdf", page_count=3, items=_rich_items())

    svc = _service(tmp_path, parse)
    pdf = _minimal_text_pdf()
    paper = svc.ingest(pdf, "demo.pdf")
    assert calls["n"] == 1

    queued, should = svc.start_reparse(paper.paper_id)
    assert should is True
    assert queued.status == "queued"
    _, again = svc.start_reparse(paper.paper_id)
    assert again is False

    ready = svc.finish_ingest(paper.paper_id)
    assert ready.status == "ready"
    assert calls["n"] == 2


def test_scan_marked_needs_ocr(tmp_path: Path) -> None:
    def parse(_path: str) -> ParseResult:
        items = [
            LayoutItem(kind="text", page=i + 1, text="abcdefghij") for i in range(5)
        ]
        return ParseResult(parser="pymupdf", page_count=5, items=items)

    svc = _service(tmp_path, parse)
    paper = svc.ingest(_minimal_text_pdf(), "scan.pdf")
    assert paper.status == "needs_ocr"
    assert paper.index_status == "skipped"
    assert paper.intro_status == "skipped"
    assert paper.method_status == "skipped"


def test_ingest_repairs_broken_formula_with_vision(tmp_path: Path) -> None:
    body = "This sentence is long enough to count as paper text. " * 20

    def parse(_path: str) -> ParseResult:
        return ParseResult(
            parser="pymupdf",
            page_count=1,
            items=[
                LayoutItem(kind="heading", page=1, text="1. Method", level=1),
                LayoutItem(kind="text", page=1, text=body),
                LayoutItem(
                    kind="formula",
                    page=1,
                    text=r"L_{C} F_{i} _{j}. i,j\in",
                    image_bytes=make_png(240, 48),
                    width_px=240,
                    height_px=48,
                    bbox=(80, 200, 320, 230),
                ),
            ],
        )

    def fake_chat(messages, **_kwargs):
        return r'{"latex": "\\sum_{i,j} L_{C}(F_i, F_j)"}'

    settings = Settings(
        data_dir=tmp_path,
        openai_api_key="sk-test",
        formula_vision_enabled=True,
        formula_vision_model="qwen-vl-plus",
    )
    svc = KnowledgeService(
        FilePaperStore(tmp_path),
        settings,
        parse_fn=parse,
        chat_fn=fake_chat,
    )
    paper = svc.ingest(_minimal_text_pdf(), "demo.pdf")
    assert paper.status == "ready"
    text = svc.get_sections(paper.paper_id)[0].text
    assert r"\sum_{i,j}" in text
    assert "<!--fig:" not in text
    assert paper.figure_count == 0


def _minimal_text_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Placeholder page for ingest tests.")
    data = doc.tobytes()
    doc.close()
    return data
