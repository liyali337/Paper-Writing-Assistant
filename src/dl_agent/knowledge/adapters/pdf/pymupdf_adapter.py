"""PyMuPDF 降级：按页 sort=True 抽字 + 标题启发式，xref / 题注邻域 clip 抽图。"""

from __future__ import annotations

import logging
import re
import statistics

from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.classify import CAPTION_RE, is_page_chrome, parse_caption
from dl_agent.knowledge.layout import LayoutItem
from dl_agent.knowledge.pdf_io import ParseError

logger = logging.getLogger(__name__)

_SECTION_NAME = (
    r"(abstract|introduction|background|related work|preliminaries|"
    r"method(?:ology)?|approach|model|architecture|proposed method|"
    r"experiments?|evaluation|results?|ablation|"
    r"discussion|limitations?|conclusion(?:s)?|"
    r"references|bibliography|参考文献|致谢|acknowledgements?)"
)
_NUMBERED_HEADING = re.compile(
    rf"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[.\s:-]+{_SECTION_NAME}\s*$",
    re.IGNORECASE,
)
_BARE_HEADING = re.compile(rf"^{_SECTION_NAME}\s*$", re.IGNORECASE)


def parse_pdf(pdf_path: str) -> ParseResult:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise ParseError("pymupdf is not installed") from exc

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise ParseError("pymupdf cannot open pdf") from exc

    try:
        items: list[LayoutItem] = []
        title = None
        for page_index, page in enumerate(document):
            page_no = page_index + 1
            items.extend(_page_items(page, document, page_no))
            if title is None:
                title = _first_large_line(page)
        items = _clip_missing_captions(document, items)
        return ParseResult(
            parser="pymupdf",
            page_count=document.page_count,
            items=items,
            title=title,
            language=_guess_language(items),
        )
    finally:
        document.close()


def clip_missing_captions(pdf_path: str, items: list[LayoutItem]) -> list[LayoutItem]:
    """给 Docling 漏掉的矢量图做题注邻域 clip。"""
    import pymupdf as fitz

    document = fitz.open(pdf_path)
    try:
        return _clip_missing_captions(document, items)
    finally:
        document.close()


def clip_formula_items(pdf_path: str, items: list[LayoutItem]) -> list[LayoutItem]:
    """按 Docling 公式框从 PDF 裁 PNG；无 LaTeX 时也能看见公式。"""
    import pymupdf as fitz

    document = fitz.open(pdf_path)
    try:
        return _clip_formula_items(document, items)
    finally:
        document.close()


def _page_items(page, document, page_no: int) -> list[LayoutItem]:
    payload = page.get_text("dict", sort=True)
    sizes = [
        span["size"]
        for block in payload.get("blocks", [])
        if block.get("type") == 0
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    ]
    median = statistics.median(sizes) if sizes else 11.0
    heading_size = max(median * 1.18, median + 1.2)
    page_h = float(page.rect.y1)
    items: list[LayoutItem] = []

    for block in payload.get("blocks", []):
        bbox = _block_bbox(block)
        if block.get("type") == 1:
            item = _image_block(block, document, page, page_no, bbox)
            if item:
                items.append(item)
            continue
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            line_bbox = tuple(float(v) for v in (line.get("bbox") or (0, 0, 0, 0)))
            y0 = line_bbox[1] if len(line_bbox) == 4 else 0
            y1 = line_bbox[3] if len(line_bbox) == 4 else 0
            if (y0 < 36 or y1 > page_h - 36) and len(text) < 80:
                continue
            if is_page_chrome(text):
                continue
            size = max(span.get("size", 0) for span in spans)
            if _is_heading(text, size, heading_size):
                items.append(
                    LayoutItem(kind="heading", page=page_no, text=text, level=1, bbox=line_bbox)
                )
                continue
            if CAPTION_RE.match(text):
                label, caption, _kind = parse_caption(text)
                items.append(
                    LayoutItem(
                        kind="caption",
                        page=page_no,
                        text=text,
                        bbox=line_bbox,
                        caption=caption,
                        label=label,
                    )
                )
                continue
            items.append(LayoutItem(kind="text", page=page_no, text=text, bbox=line_bbox))
    return items


def _is_heading(text: str, size: float, heading_size: float) -> bool:
    if len(text) > 120:
        return False
    stripped = text.strip()
    if _NUMBERED_HEADING.match(stripped):
        return True
    if _BARE_HEADING.match(stripped) and stripped[:1].isupper():
        return True
    if size >= heading_size and len(text) <= 90:
        return True
    return False


def _image_block(block, document, page, page_no: int, bbox) -> LayoutItem | None:
    import pymupdf as fitz

    xref = block.get("xref") or 0
    png = None
    width = int(block.get("width") or 0)
    height = int(block.get("height") or 0)
    try:
        if xref:
            pix = fitz.Pixmap(document, xref)
            if pix.n - pix.alpha > 3:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            width, height = pix.width, pix.height
            png = pix.tobytes("png")
        elif block.get("image"):
            png = block["image"] if isinstance(block["image"], bytes) else None
    except Exception:
        logger.debug("skip pymupdf xref image page=%s", page_no, exc_info=True)
        png = None
    if png is None:
        return None
    caption = _caption_near(page, bbox)
    label, caption_text, _kind = parse_caption(caption)
    return LayoutItem(
        kind="picture",
        page=page_no,
        text=caption,
        image_bytes=png,
        width_px=width,
        height_px=height,
        bbox=bbox,
        caption=caption_text or caption or None,
        label=label,
        source="pymupdf_xref",
    )


def _caption_near(page, bbox: tuple[float, float, float, float] | None) -> str:
    if not bbox:
        return ""
    import pymupdf as fitz

    below = fitz.Rect(bbox[0] - 8, bbox[3], bbox[2] + 8, min(bbox[3] + 80, page.rect.y1))
    text = page.get_text("text", clip=below).strip()
    first = text.splitlines()[0] if text else ""
    if CAPTION_RE.match(first):
        return " ".join(text.splitlines()[:4])
    above = fitz.Rect(bbox[0] - 8, max(0, bbox[1] - 80), bbox[2] + 8, bbox[1])
    text = page.get_text("text", clip=above).strip()
    last_lines = text.splitlines()[-4:] if text else []
    blob = " ".join(last_lines)
    return blob if CAPTION_RE.search(blob) else ""


def _clip_missing_captions(document, items: list[LayoutItem]) -> list[LayoutItem]:
    import pymupdf as fitz

    for index, item in enumerate(items):
        if item.kind != "caption":
            continue
        label, caption, kind = parse_caption(item.text)
        if not label or kind == "table_snapshot":
            continue
        _bind_caption_to_nearby_picture(items, index, label, caption)

    existing_labels = {
        item.label for item in items if item.kind == "picture" and item.label
    }
    extras: list[LayoutItem] = []
    insert_after: list[tuple[int, LayoutItem]] = []

    for index, item in enumerate(items):
        if item.kind != "caption":
            continue
        label, caption, kind = parse_caption(item.text)
        if not label or label in existing_labels:
            continue
        if kind == "table_snapshot":
            continue
        page = document[item.page - 1]
        clip = _caption_above_clip(page, item.bbox)
        if clip is None:
            extras.append(
                LayoutItem(
                    kind="picture",
                    page=item.page,
                    caption=caption,
                    label=label,
                    source="page_clip",
                    width_px=0,
                    height_px=0,
                )
            )
            insert_after.append((index, extras[-1]))
            existing_labels.add(label)
            continue
        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2), alpha=False)
        if pix.width < 80 or pix.height < 80:
            continue
        extras.append(
            LayoutItem(
                kind="picture",
                page=item.page,
                text=item.text,
                image_bytes=pix.tobytes("png"),
                width_px=pix.width,
                height_px=pix.height,
                bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
                caption=caption,
                label=label,
                source="page_clip",
            )
        )
        insert_after.append((index, extras[-1]))
        existing_labels.add(label)

    if not insert_after:
        return items
    merged = list(items)
    offset = 0
    for index, extra in insert_after:
        merged.insert(index + offset, extra)
        offset += 1
    return merged


def _bind_caption_to_nearby_picture(
    items: list[LayoutItem], caption_index: int, label: str, caption: str | None
) -> bool:
    """题注前面同页的未标注插图视为已抽出，避免再按固定高度误切。"""
    page = items[caption_index].page
    for index in range(caption_index - 1, max(-1, caption_index - 10), -1):
        prev = items[index]
        if prev.page != page:
            return False
        if prev.kind == "heading":
            return False
        if prev.kind == "picture":
            if prev.label and prev.label != label:
                return False
            prev.label = label
            if caption:
                prev.caption = caption
            return True
    return False


def _clip_formula_items(document, items: list[LayoutItem]) -> list[LayoutItem]:
    import io

    import pymupdf as fitz
    from PIL import Image

    for item in items:
        if item.kind != "formula" or item.image_bytes:
            continue
        page = document[item.page - 1]
        clip = _formula_clip_rect(page, item.bbox)
        if clip is None:
            continue
        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(2, 2), alpha=False)
        png = _trim_png(pix.tobytes("png"))
        if png is None:
            continue
        image = Image.open(io.BytesIO(png))
        if image.width < 24 or image.height < 16:
            continue
        item.image_bytes = png
        item.width_px = image.width
        item.height_px = image.height
        item.bbox = (clip.x0, clip.y0, clip.x1, clip.y1)
        item.source = "page_clip"
    return items


def _formula_clip_rect(page, bbox: tuple[float, float, float, float] | None):
    import pymupdf as fitz

    if not bbox:
        return None
    left, y_a, right, y_b = bbox
    if y_a > y_b:
        top = page.rect.y1 - y_a
        bottom = page.rect.y1 - y_b
    else:
        top, bottom = y_a, y_b
    pad = 3.0
    rect = fitz.Rect(left - pad, top - pad, right + pad, bottom + pad) & page.rect
    if rect.width < 12 or rect.height < 10:
        return None
    if rect.height > page.rect.height * 0.35 or rect.width > page.rect.width * 0.95:
        return None
    return rect


def _trim_png(png: bytes, pad: int = 4) -> bytes | None:
    import io

    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(png)).convert("RGB")
    gray = ImageOps.grayscale(image)
    mask = gray.point(lambda pixel: 0 if pixel > 248 else 255)
    box = mask.getbbox()
    if box is None:
        return None
    left, top, right, bottom = box
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image.width, right + pad)
    bottom = min(image.height, bottom + pad)
    cropped = image.crop((left, top, right, bottom))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def _caption_above_clip(page, bbox: tuple[float, float, float, float] | None):
    import pymupdf as fitz

    if not bbox:
        return None
    top = max(36.0, bbox[1] - 280)
    rect = fitz.Rect(max(36.0, bbox[0] - 12), top, min(page.rect.x1 - 36, bbox[2] + 12), bbox[1] - 2)
    if rect.height < 40 or rect.width < 40:
        return None
    return rect


def _block_bbox(block) -> tuple[float, float, float, float] | None:
    box = block.get("bbox")
    if not box or len(box) != 4:
        return None
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _first_large_line(page) -> str | None:
    payload = page.get_text("dict", sort=True)
    best = ""
    best_size = 0.0
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text or len(text) > 180:
                continue
            size = max(span.get("size", 0) for span in spans)
            y0 = (line.get("bbox") or [0, 999])[1]
            if y0 > page.rect.y1 * 0.45:
                continue
            if size > best_size:
                best_size = size
                best = text
    return best or None


def _guess_language(items: list[LayoutItem]) -> str:
    sample = " ".join(item.text for item in items[:40])
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    return "zho" if cjk > 20 else "eng"
