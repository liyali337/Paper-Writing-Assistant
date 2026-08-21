from pathlib import Path

from dl_agent.config import Settings
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.layout import LayoutItem
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from tests.helpers import make_png


def _service(tmp_path: Path, parse_fn) -> KnowledgeService:
    settings = Settings(data_dir=tmp_path)
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
    assert paper.intro_status == "skipped"
    assert paper.method_status == "skipped"


def _minimal_text_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Placeholder page for ingest tests.")
    data = doc.tobytes()
    doc.close()
    return data
