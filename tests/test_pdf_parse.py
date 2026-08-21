from unittest.mock import patch

from dl_agent.knowledge.adapters.pdf import parse_pdf
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.layout import LayoutItem


def _docling_result() -> ParseResult:
    body = "This sentence is long enough to count as paper text. " * 20
    return ParseResult(
        parser="docling",
        page_count=1,
        items=[LayoutItem(kind="text", page=1, text=body)],
    )


def test_parse_pdf_skips_formula_vlm_by_default(tmp_path) -> None:
    calls: list[bool] = []

    def fake_docling(pdf_path, *, timeout_s=120, images_scale=2.0, formula_enrichment=True):
        calls.append(formula_enrichment)
        return _docling_result()

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    with (
        patch("dl_agent.knowledge.adapters.pdf.docling_adapter.parse_pdf", fake_docling),
        patch(
            "dl_agent.knowledge.adapters.pdf.pymupdf_adapter.clip_missing_captions",
            lambda _path, items: items,
        ),
        patch(
            "dl_agent.knowledge.adapters.pdf.pymupdf_adapter.clip_formula_items",
            lambda _path, items: items,
        ),
        patch(
            "dl_agent.knowledge.formula_latex.inject_inline_math",
            lambda _path, items: items,
        ),
    ):
        parsed = parse_pdf(str(pdf_path))
    assert calls == [False]
    assert parsed.parser == "docling"


def test_parse_pdf_retries_without_vlm_when_enrichment_fails(tmp_path) -> None:
    calls: list[bool] = []

    def fake_docling(pdf_path, *, timeout_s=120, images_scale=2.0, formula_enrichment=True):
        calls.append(formula_enrichment)
        if formula_enrichment:
            raise RuntimeError("ConnectTimeout")
        return _docling_result()

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    with (
        patch("dl_agent.knowledge.adapters.pdf.docling_adapter.parse_pdf", fake_docling),
        patch(
            "dl_agent.knowledge.adapters.pdf.pymupdf_adapter.clip_missing_captions",
            lambda _path, items: items,
        ),
        patch(
            "dl_agent.knowledge.adapters.pdf.pymupdf_adapter.clip_formula_items",
            lambda _path, items: items,
        ),
        patch(
            "dl_agent.knowledge.formula_latex.inject_inline_math",
            lambda _path, items: items,
        ),
    ):
        parsed = parse_pdf(str(pdf_path), formula_enrichment=True)
    assert calls == [True, False]
    assert parsed.parser == "docling"
