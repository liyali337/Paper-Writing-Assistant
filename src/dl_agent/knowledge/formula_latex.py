"""把公式变成 LaTeX：Docling 识别整行，数学字体还原行内。"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass

from dl_agent.knowledge.layout import LayoutItem

logger = logging.getLogger(__name__)

_MATH_FONT = re.compile(
    r"(cmmi|cmsy|cmex|msam|msbm|eufm|eusm|txmi|txsy|xits|stix|asana|"
    r"lmmath|latinmodernmath|texgyremath|mtmi|mtsy|symbol|cambria\s*math)",
    re.IGNORECASE,
)
_ITALIC_FONT = re.compile(r"italic|oblique", re.IGNORECASE)
_SMALL_WORDS = {"of", "in", "to", "is", "or", "an", "as", "be", "we", "it", "at", "by", "if", "on"}
_MATH_OPS = set("×·⋅−±=∈∉⊂⊃∪∩→↔≤≥≠≈∼∘⊙∗+")
_EQ_NUMBER = re.compile(r"^\(\s*\d+\s*\)$")
_WRAP = re.compile(r"^\s*(\${1,2}|\\\(|\\\[)\s*(.*?)\s*(\${1,2}|\\\)|\\\])\s*$", re.DOTALL)
_ITALIC_WORD = re.compile(r"^[A-Za-z]{2,}$")
_WORD_TOKEN = re.compile(r"[A-Za-z]{2,}")


def is_garbled_math(text: str | None) -> bool:
    blob = text or ""
    if "\ufffd" in blob:
        return True
    if "\\backslash" in blob:
        return True
    if any(0xE000 <= ord(char) <= 0xF8FF for char in blob):
        return True
    return False


def normalize_latex(text: str | None) -> str:
    blob = (text or "").strip()
    if not blob:
        return ""
    match = _WRAP.match(blob)
    if match:
        blob = match.group(2).strip()
    blob = blob.replace("\n", " ").replace("$", "").strip()
    blob = re.sub(r"\\([A-Za-z]+)\s+\{", r"\\\1{", blob)
    return blob


def looks_like_latex(text: str | None) -> bool:
    blob = (text or "").strip()
    if len(blob) < 1 or is_garbled_math(blob):
        return False
    if _EQ_NUMBER.match(blob):
        return False
    if "\\" in blob or "_{" in blob or "^{" in blob:
        return True
    if re.search(r"[_^]", blob) and re.search(r"[A-Za-z]", blob):
        return True
    return False


_DOUBLE_SUP = re.compile(r"\^(?:\{[^{}]*\}|[A-Za-z0-9])\s*\^")
_DOUBLE_SUB = re.compile(r"_(?:\{[^{}]*\}|[A-Za-z0-9])\s*_")
_CMD_TOKEN = re.compile(r"\\([A-Za-z]+)")
_TRAILING_OP = re.compile(r"\\(?:in|sum|prod|int|oplus|otimes)\s*$")
_RAW_MATH_CHARS = set("⊕⊗⊖⊙∘⋅×∈∉⊂⊃∪∩≤≥≠≈→↔∞∑∏∫")
_GLUED_HEADS = frozenset(
    {
        "times",
        "cdot",
        "oplus",
        "otimes",
        "ominus",
        "cap",
        "cup",
        "in",
        "notin",
        "subset",
        "subseteq",
        "leq",
        "geq",
        "neq",
        "approx",
        "pm",
        "to",
        "rightarrow",
        "leftarrow",
        "infty",
        "sum",
        "prod",
        "int",
        "mathbb",
        "mathcal",
        "mathrm",
        "mathbf",
        "mathit",
        "operatorname",
        "text",
        "textbf",
        "textrm",
        "textit",
    }
)


def is_renderable_latex(text: str | None) -> bool:
    """粗判能否过 KaTeX；过不了的交给视觉重识别。"""
    blob = normalize_latex(text)
    if not blob or is_garbled_math(blob) or is_eq_number(blob):
        return False
    if not looks_like_latex(blob):
        return False
    if blob.count("{") != blob.count("}"):
        return False
    if not _braces_balanced(blob):
        return False
    if _DOUBLE_SUP.search(blob) or _DOUBLE_SUB.search(blob):
        return False
    if _has_glued_cmd(blob) or _TRAILING_OP.search(blob):
        return False
    if blob[:1] in "^_" or any(char in _RAW_MATH_CHARS for char in blob):
        return False
    if any(0x2200 <= ord(char) <= 0x22FF for char in blob):
        return False
    return True


def latex_needs_vision(text: str | None) -> bool:
    if not is_renderable_latex(text):
        return True
    return latex_looks_incomplete(text)


_IDENT_TOKEN = re.compile(
    r"[A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?(?:\^\{[^{}]+\}|\^[A-Za-z0-9']+)?"
)
_BINOP = re.compile(r"(?<![A-Za-z])\+|\\times|\\cdot|\\oplus|\\otimes|\\pm")


def latex_looks_incomplete(text: str | None) -> bool:
    """KaTeX 能渲染、但运算符/括号已丢的残式，例如 L_g L_{CE} L_{Tri}。"""
    blob = normalize_latex(text)
    if not blob:
        return True
    idents = _IDENT_TOKEN.findall(blob)
    has_eq = "=" in blob
    has_op = bool(_BINOP.search(blob))
    has_paren = "(" in blob or r"\left" in blob
    has_set = r"\{" in blob
    has_in = r"\in" in blob
    if len(idents) >= 3 and not has_eq and not has_op and not has_in and not has_paren:
        return True
    if has_eq and len(idents) >= 3 and not has_op and not has_paren and not has_set:
        return True
    if len(re.findall(r"(?<![A-Za-z])L[_^]", blob)) >= 2 and not has_op and not has_eq:
        return True
    return False


def latex_score(text: str | None) -> int:
    blob = normalize_latex(text)
    return (
        blob.count("=") * 3
        + blob.count("+")
        + blob.count("(")
        + blob.count(")")
        + blob.count(r"\mathcal") * 2
        + blob.count(r"\mathbb") * 2
        + blob.count(r"\in")
        + blob.count(r"\times")
        + blob.count(r"\oplus")
        + blob.count(r"\tag")
    )


def _has_glued_cmd(blob: str) -> bool:
    for match in _CMD_TOKEN.finditer(blob):
        name = match.group(1)
        if name in _GLUED_HEADS:
            continue
        for size in range(len(name) - 1, 0, -1):
            if name[:size] in _GLUED_HEADS:
                return True
    return False


def _braces_balanced(blob: str) -> bool:
    depth = 0
    index = 0
    while index < len(blob):
        char = blob[index]
        if char == "\\" and index + 1 < len(blob) and blob[index + 1] in "{}":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
        index += 1
    return depth == 0


def is_eq_number(text: str | None) -> bool:
    return bool(_EQ_NUMBER.match((text or "").strip()))


def is_display_formula(item: LayoutItem) -> bool:
    if not item.bbox:
        return True
    width = abs(item.bbox[2] - item.bbox[0])
    height = abs(item.bbox[3] - item.bbox[1])
    return height >= 18 or width >= 120


def inject_inline_math(pdf_path: str, items: list[LayoutItem]) -> list[LayoutItem]:
    """把未进 Docling FORMULA 的行内数学字体还原成 $LaTeX$。"""
    try:
        runs = _extract_math_runs(pdf_path)
    except Exception:
        logger.warning("inline math extract failed", exc_info=True)
        return items
    if not runs:
        return items
    _fill_formula_from_runs(items, runs)
    formula_boxes = [
        (item.page, item.bbox)
        for item in items
        if item.kind == "formula" and item.bbox
    ]
    for run in runs:
        if is_garbled_math(run.latex) or is_eq_number(run.latex) or not run.latex:
            continue
        if _ITALIC_WORD.match("".join(run.raw_parts).replace(" ", "")) and "_{" not in run.latex and "^{" not in run.latex:
            continue
        if any(
            page == run.page and bbox and _overlap(run.bbox, bbox) > 0.2
            for page, bbox in formula_boxes
        ):
            continue
        target = _nearest_text(items, run)
        if target is None:
            continue
        replaced = _splice_latex(target.text, run.raw_parts, run.latex)
        if replaced is not None:
            target.text = replaced
    return items


@dataclass
class _MathRun:
    page: int
    bbox: tuple[float, float, float, float]
    latex: str
    raw_parts: list[str]


def _extract_math_runs(pdf_path: str) -> list[_MathRun]:
    import pymupdf as fitz

    document = fitz.open(pdf_path)
    runs: list[_MathRun] = []
    try:
        for page_index, page in enumerate(document):
            payload = page.get_text("dict", sort=True)
            for block in payload.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    current: list[dict] = []
                    last_x1 = None
                    for span in line.get("spans") or []:
                        text = (span.get("text") or "").strip()
                        font = str(span.get("font") or "")
                        bbox = span.get("bbox") or None
                        if not text or not bbox or len(bbox) != 4:
                            if current:
                                runs.extend(_flush_run(page_index + 1, current))
                                current = []
                                last_x1 = None
                            continue
                        if not _is_math_span(text, font):
                            if current:
                                runs.extend(_flush_run(page_index + 1, current))
                                current = []
                                last_x1 = None
                            continue
                        x0 = float(bbox[0])
                        if last_x1 is not None and x0 - last_x1 > 10:
                            runs.extend(_flush_run(page_index + 1, current))
                            current = []
                        current.append(
                            {
                                "text": text,
                                "bbox": tuple(float(v) for v in bbox),
                                "size": float(span.get("size") or 10),
                            }
                        )
                        last_x1 = float(bbox[2])
                    if current:
                        runs.extend(_flush_run(page_index + 1, current))
    finally:
        document.close()
    return runs


def _flush_run(page: int, spans: list[dict]) -> list[_MathRun]:
    if not spans:
        return []
    parts = [str(span["text"]) for span in spans]
    glued = "".join(part.strip() for part in parts)
    if not glued or all(part.strip() in ".,;:()[]{}_^\\ " for part in parts):
        return []
    if len(spans) == 1 and glued.isalpha() and len(glued) <= 2:
        return []
    if _ITALIC_WORD.match(glued):
        sizes = {round(float(span["size"]), 1) for span in spans}
        if len(sizes) == 1:
            return []
    xs0 = [span["bbox"][0] for span in spans]
    ys0 = [span["bbox"][1] for span in spans]
    xs1 = [span["bbox"][2] for span in spans]
    ys1 = [span["bbox"][3] for span in spans]
    latex = spans_to_latex(spans)
    if not latex or is_garbled_math(latex):
        return []
    return [
        _MathRun(
            page=page,
            bbox=(min(xs0), min(ys0), max(xs1), max(ys1)),
            latex=latex,
            raw_parts=parts,
        )
    ]


def spans_to_latex(spans: list[dict]) -> str:
    """用字号和竖向位置还原上/下标。"""
    if not spans:
        return ""
    sizes = [float(span["size"]) for span in spans]
    mids = [(span["bbox"][1] + span["bbox"][3]) / 2 for span in spans]
    base_size = max(sizes)
    base_mids = [
        mid
        for span, mid in zip(spans, mids)
        if float(span["size"]) >= base_size * 0.92
    ]
    base_mid = statistics.median(base_mids) if base_mids else statistics.median(mids)
    chunks: list[tuple[str, str]] = []
    for span in spans:
        token = _escape_token(str(span["text"]))
        if not token:
            continue
        size = float(span["size"])
        mid = (span["bbox"][1] + span["bbox"][3]) / 2
        role = "base"
        if size <= base_size * 0.88:
            if mid < base_mid - 0.8:
                role = "sup"
            elif mid > base_mid + 0.8:
                role = "sub"
        if chunks and chunks[-1][0] == role:
            chunks[-1] = (role, chunks[-1][1] + token)
        else:
            chunks.append((role, token))
    return _join_script_chunks(chunks)


def _join_script_chunks(chunks: list[tuple[str, str]]) -> str:
    out: list[str] = []
    index = 0
    while index < len(chunks):
        role, token = chunks[index]
        if role == "base":
            out.append(token)
            index += 1
            continue
        grouped = {"sub": "", "sup": ""}
        while index < len(chunks) and chunks[index][0] != "base":
            script, piece = chunks[index]
            grouped[script] += piece
            index += 1
        if grouped["sup"]:
            out.append("^{" + grouped["sup"] + "}")
        if grouped["sub"]:
            out.append("_{" + grouped["sub"] + "}")
    return "".join(out).strip()


def _is_math_span(text: str, font: str) -> bool:
    """数学字体、运算符，或斜体短符号（Times-Italic 的 L、p）都算行内公式片段。"""
    blob = text.strip()
    if not blob:
        return False
    if _MATH_FONT.search(font):
        return True
    if all(char in _MATH_OPS or char.isspace() for char in blob):
        return True
    if (
        _ITALIC_FONT.search(font)
        and len(blob) <= 2
        and blob.lower() not in _SMALL_WORDS
        and (blob.isalnum() or len(blob) == 1)
    ):
        return True
    return False


def _escape_token(text: str) -> str:
    specials = {
        "\\": r"\backslash ",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "×": r"\times ",
        "·": r"\cdot ",
        "⋅": r"\cdot ",
        "−": "-",
        "±": r"\pm ",
        "∈": r"\in ",
        "≤": r"\leq ",
        "≥": r"\geq ",
        "≠": r"\neq ",
        "≈": r"\approx ",
        "→": r"\rightarrow ",
        "⊙": r"\odot ",
        "∘": r"\circ ",
    }
    return "".join(specials.get(char, char) for char in text)


def _nearest_text(items: list[LayoutItem], run: _MathRun) -> LayoutItem | None:
    best: LayoutItem | None = None
    best_dist = 1e9
    run_mid = (run.bbox[1] + run.bbox[3]) / 2
    for item in items:
        if item.kind != "text" or item.page != run.page or not item.bbox:
            continue
        mid = (item.bbox[1] + item.bbox[3]) / 2
        dist = abs(mid - run_mid)
        if dist < best_dist:
            best = item
            best_dist = dist
    if best is None or best_dist > 18:
        return None
    return best


def _fill_formula_from_runs(items: list[LayoutItem], runs: list[_MathRun]) -> None:
    for item in items:
        if item.kind != "formula" or not item.bbox:
            continue
        if looks_like_latex(item.text):
            continue
        pieces = [
            run.latex
            for run in runs
            if run.page == item.page and looks_like_latex(run.latex) and _overlap(run.bbox, item.bbox) > 0.15
        ]
        if pieces:
            item.text = " ".join(pieces)


def _splice_latex(text: str, parts: list[str], latex: str) -> str | None:
    wrapped = f"${latex}$"
    if wrapped in text:
        return text
    glued = "".join(part.strip() for part in parts if part.strip())
    spaced = " ".join(part.strip() for part in parts if part.strip())
    for token in (spaced, glued):
        if not token or token not in text:
            continue
        if len(token) == 1 and token.isalpha() and _WORD_TOKEN.search(text):
            pattern = re.compile(rf"(?<![A-Za-z$\\]){re.escape(token)}(?![A-Za-z])")
            match = pattern.search(text)
            if not match or _inside_math(text, match.start()):
                continue
            return pattern.sub(wrapped, text, count=1)
        index = text.find(token)
        if index >= 0 and _inside_math(text, index):
            continue
        return text.replace(token, wrapped, 1)
    return None


def _inside_math(text: str, index: int) -> bool:
    before = text[:index]
    if before.count("$$") % 2 == 1:
        return True
    dollars = before.replace("$$", "")
    return dollars.count("$") % 2 == 1


def _overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx0, by0, bx1, by1 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    return inter / area
