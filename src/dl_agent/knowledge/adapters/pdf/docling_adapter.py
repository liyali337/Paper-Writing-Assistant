"""Docling 主路径：关 OCR、开抽图、带超时。一次 convert 同时出章节与图。"""

from __future__ import annotations

import io
import logging

from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.classify import is_page_chrome, parse_caption
from dl_agent.knowledge.layout import LayoutItem
from dl_agent.knowledge.pdf_io import ParseError

logger = logging.getLogger(__name__)


def parse_pdf(
    pdf_path: str,
    *,
    timeout_s: float = 120,
    images_scale: float = 2.0,
    formula_enrichment: bool = False,
) -> ParseResult:
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if formula_enrichment:
        logger.info("docling convert with formula enrichment path=%s", pdf_path)
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        do_formula_enrichment=formula_enrichment,
        images_scale=images_scale,
        document_timeout=timeout_s,
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    result = converter.convert(pdf_path)
    if result.status == ConversionStatus.FAILURE:
        raise ParseError("docling conversion failed")
    if hasattr(result, "has_timeout_errors") and result.has_timeout_errors():
        raise ParseError("docling timed out")

    doc = result.document
    items: list[LayoutItem] = []
    title = None

    from docling_core.types.doc import PictureItem, TableItem, TextItem

    for element, tree_level in doc.iterate_items():
        page = _page_no(element)
        bbox = _bbox(element)

        if isinstance(element, PictureItem):
            image = element.get_image(doc)
            if image is None:
                continue
            png = _pil_to_png(image)
            caption = ""
            try:
                caption = (element.caption_text(doc=doc) or "").strip()
            except TypeError:
                caption = (element.caption_text(doc) or "").strip()
            label, caption_text, _kind = parse_caption(caption)
            items.append(
                LayoutItem(
                    kind="picture",
                    page=page,
                    text=caption,
                    image_bytes=png,
                    width_px=int(image.width),
                    height_px=int(image.height),
                    bbox=bbox,
                    caption=caption_text or caption or None,
                    label=label,
                    source="docling_picture",
                )
            )
            continue

        if isinstance(element, TableItem):
            text = _table_markdown(element, doc)
            if text:
                items.append(LayoutItem(kind="table", page=page, text=text, bbox=bbox))
            continue

        if _is_formula(element):
            latex = (element.text or "").strip()
            orig = (getattr(element, "orig", None) or "").strip()
            from dl_agent.knowledge.formula_latex import (
                is_garbled_math,
                looks_like_latex,
                normalize_latex,
            )

            blob = normalize_latex(latex) if looks_like_latex(latex) else (latex or orig)
            if is_garbled_math(blob):
                blob = orig if looks_like_latex(orig) else ""
            if _looks_like_prose(blob) and not looks_like_latex(blob):
                items.append(LayoutItem(kind="text", page=page, text=blob, bbox=bbox))
            else:
                items.append(
                    LayoutItem(
                        kind="formula",
                        page=page,
                        text=blob,
                        bbox=bbox,
                        source="page_clip",
                    )
                )
            continue

        if not isinstance(element, TextItem):
            continue
        text = (element.text or "").strip()
        if not text:
            continue
        label = str(getattr(element, "label", "")).lower()
        if "page_header" in label or "page_footer" in label or "footnote" in label:
            continue
        if is_page_chrome(text):
            continue
        if "title" in label and title is None and page <= 1:
            title = text
            items.append(LayoutItem(kind="heading", page=page, text=text, level=1, bbox=bbox))
            continue
        if "section_header" in label or label.endswith("title"):
            items.append(
                LayoutItem(
                    kind="heading",
                    page=page,
                    text=text,
                    level=int(tree_level or 1),
                    bbox=bbox,
                )
            )
            continue
        if "caption" in label:
            items.append(LayoutItem(kind="caption", page=page, text=text, bbox=bbox, caption=text))
            continue
        items.append(LayoutItem(kind="text", page=page, text=text, bbox=bbox))

    page_count = 0
    try:
        page_count = len(doc.pages) if getattr(doc, "pages", None) else 0
    except Exception:
        page_count = 0
    if page_count == 0:
        page_count = max((item.page for item in items), default=0)

    return ParseResult(
        parser="docling",
        page_count=page_count,
        items=items,
        title=title,
        language=_guess_language(items),
    )


def _page_no(element) -> int:
    prov = getattr(element, "prov", None) or []
    if not prov:
        return 1
    return int(getattr(prov[0], "page_no", 1) or 1)


def _bbox(element) -> tuple[float, float, float, float] | None:
    prov = getattr(element, "prov", None) or []
    if not prov:
        return None
    box = getattr(prov[0], "bbox", None)
    if box is None:
        return None
    try:
        return (float(box.l), float(box.t), float(box.r), float(box.b))
    except Exception:
        return None


def _pil_to_png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _table_markdown(table, doc) -> str:
    for args in ((), (doc,), (),):
        try:
            if args:
                text = table.export_to_markdown(*args)
            else:
                text = table.export_to_markdown()
            if isinstance(text, str) and text.strip():
                return text.strip()
        except TypeError:
            continue
        except Exception:
            break
    return getattr(table, "text", "") or ""


def _is_formula(element) -> bool:
    from docling_core.types.doc import FormulaItem, TextItem

    if isinstance(element, FormulaItem):
        return True
    if not isinstance(element, TextItem):
        return False
    return str(getattr(element, "label", "")).lower() == "formula"


def _looks_like_prose(text: str) -> bool:
    blob = text.strip()
    if len(blob) < 80:
        return False
    if "\\" in blob or "=" in blob:
        return False
    return blob.count(" ") >= 12


def _guess_language(items: list[LayoutItem]) -> str:
    sample = " ".join(item.text for item in items[:40])
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    return "zho" if cjk > 20 else "eng"
