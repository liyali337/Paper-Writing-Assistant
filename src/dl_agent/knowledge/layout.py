from __future__ import annotations

import re
from dataclasses import dataclass, field

from dl_agent.domain.models import Figure, FigureSource, Section
from dl_agent.knowledge.classify import (
    classify_kind,
    heading_level,
    is_page_chrome,
    is_page_number,
    is_plausible_heading,
    is_references_heading,
    is_skip_heading,
    looks_like_paper_title_heading,
    parse_caption,
    strip_page_chrome,
)


@dataclass
class LayoutItem:
    kind: str
    page: int
    text: str = ""
    level: int = 1
    image_bytes: bytes | None = None
    width_px: int = 0
    height_px: int = 0
    bbox: tuple[float, float, float, float] | None = None
    caption: str | None = None
    label: str | None = None
    source: FigureSource = "docling_picture"


@dataclass
class AssembleResult:
    sections: list[Section]
    figures: list[Figure]
    figure_pngs: dict[str, bytes] = field(default_factory=dict)
    title: str | None = None
    abstract: str | None = None


def assemble(
    paper_id: str,
    items: list[LayoutItem],
    *,
    min_figure_px: int = 80,
    paper_title: str | None = None,
) -> AssembleResult:
    """按阅读序把 heading 开成 Section，Picture 挂到当前未关闭的节。"""
    sections: list[Section] = []
    figures: list[Figure] = []
    pngs: dict[str, bytes] = {}
    stack: list[Section] = []
    after_refs = False
    skip_until_heading = False
    derived_title = paper_title
    seen_numbered = False

    def current() -> Section | None:
        return stack[-1] if stack else None

    def close_to(level: int) -> None:
        while stack and stack[-1].level >= level:
            closed = stack.pop()
            closed.page_end = max(closed.page_end, closed.page_start)

    def open_section(title: str, page: int, level: int) -> Section:
        close_to(level)
        parent = stack[-1] if stack else None
        section = Section(
            section_id=f"sec-{len(sections) + 1:03d}",
            paper_id=paper_id,
            title=title.strip(),
            kind=classify_kind(title),
            level=level,
            page_start=page,
            page_end=page,
            text="",
            parent_id=parent.section_id if parent else None,
        )
        sections.append(section)
        stack.append(section)
        return section

    def append_text(text: str, page: int) -> None:
        chunk = text.strip()
        if not chunk:
            return
        section = current()
        if section is None:
            section = open_section("Front Matter", page, 1)
            section.kind = "other"
        if is_page_chrome(chunk) and not (
            section.kind == "references" and _is_ref_number(chunk)
        ):
            return
        if section.text.endswith(chunk) or (len(chunk) > 48 and chunk in section.text):
            section.page_end = max(section.page_end, page)
            return
        if section.kind == "references" and _glue_ref_number(section, chunk, page):
            return
        if section.text:
            prev_line = section.text.rsplit("\n\n", 1)[-1].split("\n")[-1]
            if (
                section.kind != "references"
                and _is_list_item(chunk)
                and _is_list_item(prev_line)
            ):
                section.text += "\n" + chunk
            else:
                section.text += "\n\n" + chunk
        else:
            section.text = chunk
        section.page_end = max(section.page_end, page)

    def append_inline(text: str, page: int) -> None:
        chunk = text.strip()
        if not chunk or is_page_chrome(chunk):
            return
        section = current()
        if section is None:
            section = open_section("Front Matter", page, 1)
            section.kind = "other"
        if section.text:
            section.text = section.text.rstrip() + " " + chunk
        else:
            section.text = chunk
        section.page_end = max(section.page_end, page)

    def absorb_false_heading(title: str, page: int) -> None:
        section = current()
        if section is None:
            append_text(title, page)
            return
        body = title.strip()
        if body[:1].islower() and section.text and not section.text.rstrip().endswith((".", "!", "?")):
            section.text = section.text.rstrip() + " " + body
        else:
            append_text(body, page)
        section.page_end = max(section.page_end, page)

    preamble_texts: list[str] = []

    for index, item in enumerate(items):
        if after_refs and item.kind in {"picture", "formula"}:
            continue
        if skip_until_heading and item.kind != "heading":
            continue

        if item.kind == "heading":
            skip_until_heading = False
            title = item.text.strip()
            if not title:
                continue
            if is_skip_heading(title):
                skip_until_heading = True
                continue
            numbered = bool(title[:1].isdigit()) or title[:1] in "IVXivx"
            if (
                not seen_numbered
                and not numbered
                and item.page == 1
                and not sections
                and looks_like_paper_title_heading(title)
            ):
                derived_title = derived_title or title
                continue
            if numbered:
                seen_numbered = True
            if not is_plausible_heading(title):
                absorb_false_heading(title, item.page)
                continue
            if is_references_heading(title):
                open_section(title, item.page, heading_level(title, item.level))
                after_refs = True
                continue
            if after_refs:
                continue
            open_section(title, item.page, heading_level(title, item.level))
            continue

        if after_refs and current() is not None and current().kind == "references":
            if item.kind in {"text", "table", "caption"}:
                append_text(item.text, item.page)
            continue
        if after_refs:
            continue

        if item.kind in {"text", "table"}:
            if not sections and item.page == 1:
                preamble_texts.append(item.text)
            append_text(item.text, item.page)
            continue

        if item.kind == "caption":
            label, _, kind = parse_caption(item.text)
            if label and kind != "table_snapshot":
                continue
            append_text(item.text, item.page)
            continue

        if item.kind == "formula":
            _append_formula(
                item,
                paper_id,
                current,
                open_section,
                append_text,
                append_inline,
                figures,
                pngs,
            )
            continue

        if item.kind != "picture":
            continue

        if item.width_px < min_figure_px or item.height_px < min_figure_px:
            if not (item.source == "page_clip" and not item.image_bytes and (item.label or item.caption)):
                continue

        label, caption, fig_kind = parse_caption(item.caption or item.text or "")
        if not caption:
            lookahead = _next_caption(items, index)
            if lookahead:
                label, caption, fig_kind = lookahead

        if fig_kind == "table_snapshot":
            if caption:
                append_text(caption, item.page)
            continue

        if current() is None:
            open_section("Front Matter", item.page, 1)

        figure_id = f"fig-{len(figures) + 1:03d}"
        storage_key = None
        if item.image_bytes:
            storage_key = f"papers/{paper_id}/figures/{figure_id}.png"
            pngs[figure_id] = item.image_bytes

        figure = Figure(
            figure_id=figure_id,
            paper_id=paper_id,
            section_id=current().section_id if current() else None,
            page=item.page,
            kind=fig_kind,
            label=label or item.label,
            caption=caption or item.caption,
            storage_key=storage_key,
            source=item.source,
            width_px=item.width_px,
            height_px=item.height_px,
            bbox=item.bbox,
        )
        figures.append(figure)
        if current() is not None:
            current().figure_ids.append(figure_id)
            current().page_end = max(current().page_end, item.page)

    sections, figures, pngs = apply_figure_dedupe(sections, figures, pngs)
    sections, figures, pngs = polish_formulas(sections, figures, pngs)
    sections = repair_front_matter(collapse_false_headings(sections))

    abstract = None
    for section in sections:
        if section.kind == "abstract" and section.text:
            abstract = section.text
            break
    if abstract is None and preamble_texts:
        joined = "\n\n".join(t.strip() for t in preamble_texts if t.strip())
        if 80 < len(joined) < 4000:
            abstract = joined

    return AssembleResult(
        sections=sections,
        figures=figures,
        figure_pngs=pngs,
        title=derived_title,
        abstract=abstract,
    )


_LIST_ITEM = re.compile(r"^(?:\(\d+\)|\d+\.(?!\d)|[-–•])\s+\S")
_REF_NUMBER = re.compile(r"^[\[\(]?\d{1,3}[\]\).]?\s*$")


def _is_list_item(text: str) -> bool:
    return bool(_LIST_ITEM.match(text.strip()))


def _is_ref_number(text: str) -> bool:
    return bool(_REF_NUMBER.match(text.strip()))


def _ref_mark(text: str) -> str:
    digits = re.sub(r"\D", "", text.strip())
    return f"[{digits}]" if digits else text.strip()


def _glue_ref_number(section: Section, chunk: str, page: int) -> bool:
    """PDF 常把 [1] 和条目正文拆成两块，粘回 [1] Author..."""
    if not section.text or _is_ref_number(chunk):
        return False
    parts = section.text.rsplit("\n\n", 1)
    prev = parts[-1].strip()
    if not _is_ref_number(prev):
        return False
    head = f"{parts[0]}\n\n" if len(parts) == 2 else ""
    section.text = f"{head}{_ref_mark(prev)} {chunk}".strip()
    section.page_end = max(section.page_end, page)
    return True


def formula_marker(figure_id: str) -> str:
    return f"<!--fig:{figure_id}-->"


def _append_formula(
    item: LayoutItem,
    paper_id,
    current,
    open_section,
    append_text,
    append_inline,
    figures,
    pngs,
) -> None:
    from dl_agent.knowledge.formula_latex import (
        is_display_formula,
        is_eq_number,
        looks_like_latex,
        normalize_latex,
    )

    latex = normalize_latex(item.text)
    if latex and looks_like_latex(latex) and not is_eq_number(latex):
        if current() is None:
            open_section("Front Matter", item.page, 1)
        if is_display_formula(item):
            append_text(f"$$\n{latex}\n$$", item.page)
        else:
            append_inline(f"${latex}$", item.page)
        return

    has_image = bool(item.image_bytes) and item.width_px >= 90 and item.height_px >= 16
    if not has_image:
        return
    if current() is None:
        open_section("Front Matter", item.page, 1)
    section = current()
    figure_id = f"fig-{len(figures) + 1:03d}"
    storage_key = f"papers/{paper_id}/figures/{figure_id}.png"
    pngs[figure_id] = item.image_bytes
    figures.append(
        Figure(
            figure_id=figure_id,
            paper_id=paper_id,
            section_id=section.section_id if section else None,
            page=item.page,
            kind="formula",
            storage_key=storage_key,
            source=item.source,
            width_px=item.width_px,
            height_px=item.height_px,
            bbox=item.bbox,
        )
    )
    if section is not None:
        section.figure_ids.append(figure_id)
        section.page_end = max(section.page_end, item.page)
    append_text(formula_marker(figure_id), item.page)


def apply_figure_dedupe(
    sections: list[Section],
    figures: list[Figure],
    pngs: dict[str, bytes] | None = None,
) -> tuple[list[Section], list[Figure], dict[str, bytes] | None]:
    """同一题注只留一张图：优先 Docling 裁剪，丢掉题注邻域误切。"""
    winner: dict[str, Figure] = {}
    for figure in figures:
        if not figure.label:
            continue
        prev = winner.get(figure.label)
        if prev is None or _better_figure(figure, prev):
            winner[figure.label] = figure

    kept: list[Figure] = []
    seen: set[str] = set()
    redirect: dict[str, str] = {}
    for figure in figures:
        if not figure.label:
            kept.append(figure)
            continue
        champ = winner[figure.label]
        if champ.figure_id not in seen:
            kept.append(champ)
            seen.add(champ.figure_id)
        if figure.figure_id != champ.figure_id:
            redirect[figure.figure_id] = champ.figure_id

    keep_ids = {figure.figure_id for figure in kept}
    for section in sections:
        mapped: list[str] = []
        for figure_id in section.figure_ids:
            figure_id = redirect.get(figure_id, figure_id)
            if figure_id in keep_ids and figure_id not in mapped:
                mapped.append(figure_id)
        section.figure_ids = mapped
    if pngs is not None:
        pngs = {key: blob for key, blob in pngs.items() if key in keep_ids}
    return sections, kept, pngs


def polish_formulas(
    sections: list[Section],
    figures: list[Figure],
    pngs: dict[str, bytes] | None = None,
) -> tuple[list[Section], list[Figure], dict[str, bytes] | None]:
    """丢掉公式碎片图，并把正文里被拆开的行内符号收紧。"""
    drop = {
        figure.figure_id
        for figure in figures
        if figure.kind == "formula" and figure.width_px < 90
    }
    kept = [figure for figure in figures if figure.figure_id not in drop]
    keep_ids = {figure.figure_id for figure in kept}
    for section in sections:
        section.figure_ids = [fid for fid in section.figure_ids if fid in keep_ids]
        text = section.text
        for figure_id in drop:
            text = text.replace(formula_marker(figure_id), "")
        section.text = strip_page_chrome(tidy_math_prose(text))
    if pngs is not None:
        pngs = {key: blob for key, blob in pngs.items() if key in keep_ids}
    return sections, kept, pngs


_TEX_JUNK = re.compile(r"\\(?:protect|big|Big|left|right|displaystyle)\s*")
_HAT = re.compile(r"ˆ\s*([A-Za-z])(?:\s+([A-Za-z]))?")
_SPACED_ACCENT = re.compile(r"([¨ˆ˜¯´˙`˚])\s+([A-Za-z])\s+([A-Za-z])\b")
_ACCENT_BEFORE_MATH = re.compile(r"([¨ˆ˜¯´˙`˚])\s*\$([A-Za-z])([^$]*)\$")
_COMBINING_ACCENT = re.compile(
    r"(?<![A-Za-z])([A-Za-z])([\u0300\u0301\u0302\u0303\u0304\u0307\u0308\u030a])"
    r"(?:_\{?([A-Za-z0-9]+)\}?)?(?![A-Za-z])"
)
_TEXT_UMLAUT = re.compile(r"(?<=[A-Za-z])\s*¨\s*([aouAOUe])(?=[A-Za-z])")
_UMLAUT_LETTER = {
    "a": "ä",
    "o": "ö",
    "u": "ü",
    "e": "ë",
    "A": "Ä",
    "O": "Ö",
    "U": "Ü",
    "E": "Ë",
}
_ACCENT_CMD = {
    "¨": "ddot",
    "ˆ": "hat",
    "˜": "tilde",
    "¯": "bar",
    "´": "acute",
    "˙": "dot",
    "`": "grave",
    "˚": "mathring",
    "\u0300": "grave",
    "\u0301": "acute",
    "\u0302": "hat",
    "\u0303": "tilde",
    "\u0304": "bar",
    "\u0307": "dot",
    "\u0308": "ddot",
    "\u030a": "mathring",
}
_TRIPLE = re.compile(r"\b([A-Za-z])\s+([A-Za-z0-9]+)\s+([A-Za-z])\b")
_NAMED_TRIPLE = re.compile(r"\b([A-Z][A-Za-z]{2,})\s+([A-Z])\s+([a-z])\b")
_WS_SCRIPT = re.compile(r"(?<![A-Za-z\\])([WFTX])([SPRNIT])_\{?([A-Za-z0-9]+)\}?")
_REAL_SPACE = re.compile(r"(?<![A-Za-z\\])R([A-Z])\s*(?:\\times|×)\s*([A-Z])")
_REAL_DIM = re.compile(
    r"(?<!mathbb\{)(?<![A-Za-z\\])R(\d+)\s*(?:\\times|×)\s*"
    r"([A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?)"
)
_LOG_STACKED_FRAC = re.compile(
    r"(?:\\log|\blog)\s*\(\s*([A-Za-z])_\1\^1\s*0\s*\)"
)
_DELTA_LOG_FRAC = re.compile(
    r"(?<![$])(?:([Δδ]|\\Delta)\s*=\s*)?(?:\\log|\blog)\s*\(\s*([A-Za-z])_\2\^1\s*0\s*\)"
)
_BARE_ASSIGN = re.compile(
    r"(?<![$\\])("
    r"[A-Za-z]"
    r"(?:_\{[^{}]{2,}\}|\^\{\{?[^{}]+\}\}?){1,6}"
    r"\s*=\s*"
    r"[^\n$]{8,240}"
    r")",
    re.M,
)
_GREEK_LATEX = {
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "Δ": r"\Delta ",
    "θ": r"\theta ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "π": r"\pi ",
    "σ": r"\sigma ",
    "φ": r"\phi ",
    "ω": r"\omega ",
    "⊕": r"\oplus ",
}
_PROSE_REAL_DIM = re.compile(
    r"(?<![$\\])(?:([A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?)\s*)?"
    r"(∈|\\in)\s*R(\d+)\s*(?:×|\\times)\s*"
    r"([A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?)"
)
_REAL_BASE = r"(?:\\mathbb\{R\}|ℝ|(?<!mathbb\{)(?<![A-Za-z\\])R)"
_STRAY_REAL_SUB_BRACED = re.compile(
    _REAL_BASE + r"_\{([A-Za-z])\}\^\s*\{([^{}]*?(?:\\times|×)\s*)([A-Za-z])\}"
)
_STRAY_REAL_SUB_BARE = re.compile(
    _REAL_BASE + r"_([A-Za-z])\^\s*\{([^{}]*?(?:\\times|×)\s*)([A-Za-z])\}"
)
_STRAY_REAL_TRAIL_BRACED = re.compile(
    _REAL_BASE + r"\^\s*\{([^{}]*?(?:\\times|×)\s*)([A-Za-z])\}_\{([A-Za-z])\}"
)
_STRAY_REAL_TRAIL_BARE = re.compile(
    _REAL_BASE + r"\^\s*\{([^{}]*?(?:\\times|×)\s*)([A-Za-z])\}_([A-Za-z])(?![A-Za-z0-9{])"
)
_GLUED_TENSOR = re.compile(
    r"(?<![A-Za-z\\])([QFHWXZPK])([A-Z])([A-Z])\s*(?:\\in|∈)\s*"
    r"(?:\\mathbb\{R\}|ℝ)"
    r"(?=(?:_\{?[A-Za-z]\}?)?\s*\^\{[^{}]{0,80}(?:\\times|×))"
)
_GLUED_TENSOR_SKIP = frozenset(
    {
        "RNN",
        "CNN",
        "GAN",
        "MLP",
        "SVD",
        "PCA",
        "RGB",
        "NLP",
        "NMS",
        "ROI",
        "FPS",
        "GPU",
        "CPU",
        "BERT",
        "LSTM",
        "VAE",
        "GNN",
        "ViT",
    }
)
_ATTENTION_TRIO = frozenset("QKV")
_SPACE_PUNCT = re.compile(r"\s+([,;:.)\]}])")
_OPEN_SPACE = re.compile(r"([(\[{])\s+")


def _should_unglue_tensor(token: str) -> bool:
    blob = token.upper()
    if len(blob) != 3 or not blob.isalpha():
        return False
    if blob in _GLUED_TENSOR_SKIP or blob[0] not in "QFHWXZPK":
        return False
    return not set(blob) <= _ATTENTION_TRIO


def _real_shape(prefix: str, last: str, sub: str) -> str:
    prefix = prefix.replace("×", r"\times ")
    script = f"_{sub}" if len(sub) == 1 else f"_{{{sub}}}"
    return rf"\mathbb{{R}}^{{{prefix}{last}{script}}}"


def _repair_tensor_shapes(text: str) -> str:
    """把 QVT∈ℝ_v^{B×C} 收成 Q_T^V ∈ ℝ^{B×C_v}。"""
    if not text:
        return text
    cleaned = text.replace("ℝ", r"\mathbb{R}")
    cleaned = _STRAY_REAL_SUB_BRACED.sub(
        lambda m: _real_shape(m.group(2), m.group(3), m.group(1)), cleaned
    )
    cleaned = _STRAY_REAL_SUB_BARE.sub(
        lambda m: _real_shape(m.group(2), m.group(3), m.group(1)), cleaned
    )
    cleaned = _STRAY_REAL_TRAIL_BRACED.sub(
        lambda m: _real_shape(m.group(1), m.group(2), m.group(3)), cleaned
    )
    cleaned = _STRAY_REAL_TRAIL_BARE.sub(
        lambda m: _real_shape(m.group(1), m.group(2), m.group(3)), cleaned
    )

    def unglue(match: re.Match[str]) -> str:
        token = match.group(1) + match.group(2) + match.group(3)
        if not _should_unglue_tensor(token):
            return match.group(0)
        return rf"{match.group(1)}_{{{match.group(3)}}}^{{{match.group(2)}}} \in \mathbb{{R}}"

    return _GLUED_TENSOR.sub(unglue, cleaned)


def tidy_math_prose(text: str) -> str:
    """把 PDF 抽字收成 $LaTeX$。数学片段按 KaTeX/Docling 方式整段切分，内部不再套 $。"""
    if not text:
        return text
    text = text.replace("\ufffd", "")
    text = _WORD_EMBEDDED_MATH.sub(_unwrap_word_math, text)
    text = _EMBEDDED_MATH.sub(_unwrap_embedded_math, text)
    text = _GLUED_IDENTIFIER.sub(_merge_glued_identifier, text)
    text = _TEXT_UMLAUT.sub(lambda m: _UMLAUT_LETTER.get(m.group(1), m.group(0)), text)
    text = _ACCENT_BEFORE_MATH.sub(_merge_accent_before_math, text)
    text = _FLOOR_DOLLARS.sub(lambda m: r"$\lfloor " + m.group(1).strip() + r" \rfloor$", text)
    text = _K_FLOOR.sub(lambda m: f"${m.group(1)} = {m.group(2)}$", text)
    rebuilt: list[str] = []
    for kind, body in _split_math_segments(text):
        if kind == "display":
            rebuilt.append(_emit_display(body))
        elif kind == "inline":
            rebuilt.append(_emit_inline(body))
        else:
            rebuilt.append(_tidy_text_segment(body))
    out = _join_math_pieces(rebuilt)
    out = _wrap_bare_inline_math(out)
    out = _collapse_shattered_formulas(out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_COND_AFTER_MATH = re.compile(
    r"\$([^$\n]{1,80})\$\s*([|∣]\s*[A-Za-z][A-Za-z0-9]*)([)）]?)"
)


def _strip_stray_dollar_after_word(text: str) -> str:
    """去掉 p$ neg 这种把后半句切进数学体的多余 $。"""
    out: list[str] = []
    dollars = 0
    index = 0
    length = len(text)
    while index < length:
        if text.startswith("$$", index):
            close = text.find("$$", index + 2)
            if close < 0:
                out.append(text[index:])
                break
            out.append(text[index : close + 2])
            dollars = 0
            index = close + 2
            continue
        char = text[index]
        if char == "$":
            if dollars % 2 == 0 and index > 0 and text[index - 1].isalpha():
                rest = text[index + 1 : index + 8]
                if re.match(r"\s+[\u4e00-\u9fffA-Za-z]", rest):
                    index += 1
                    continue
            dollars += 1
        out.append(char)
        index += 1
    return "".join(out)


def _absorb_trailing_condition(text: str) -> str:
    """把 $F_j$|T) 收成 $F_j | T$，避免 |T) 掉到下一段。"""

    def absorb(match: re.Match[str]) -> str:
        body = match.group(1).rstrip()
        if "|" in body or "\\mid" in body:
            return match.group(0)
        cond = re.sub(r"\s+", "", match.group(2).replace("∣", "|"))
        return f"${body} {cond}${match.group(3)}"

    return _COND_AFTER_MATH.sub(absorb, text)


def tidy_translation_math(text: str) -> str:
    """整理译文里已有的 $ / $$，不要按 PDF 抽字去包新的独立公式。"""
    if not text:
        return text
    text = _strip_stray_dollar_after_word(text)
    text = _absorb_trailing_condition(text)
    rebuilt: list[str] = []
    for kind, body in _split_math_segments(text):
        if kind == "display":
            math, prose = _peel_math_prose(body)
            cleaned = _sanitize_math_body(math or body)
            rebuilt.append(f"$$\n{cleaned}\n$$" if cleaned else "")
            if prose:
                rebuilt.append(("" if rebuilt[-1].endswith("\n") else " ") + _wrap_bare_zh_math(prose))
        elif kind == "inline":
            if re.search(r"[\u4e00-\u9fff]", body):
                rebuilt.append(_wrap_bare_zh_math(body))
                continue
            cleaned = _sanitize_math_body(body)
            rebuilt.append(f"${cleaned}$" if cleaned else "")
        else:
            rebuilt.append(_wrap_bare_zh_math(body))
    return "".join(rebuilt)


_BARE_ZH_MATH = re.compile(
    r"(?<![$\\])("
    r"(?:[A-Za-z]\s+)?\\in\s*\\\{(?:[^{}]|\{[^{}]{0,40}\})*\\\}"
    r"|\\frac\{[^{}]{1,40}\}\{[^{}]{1,40}\}"
    r"|\([A-Za-z](?:[_^](?:\{(?:[^{}]|\{[^{}]{0,40}\})+\}|[A-Za-z0-9]+))"
    r"(?:,\s*[A-Za-z](?:[_^](?:\{(?:[^{}]|\{[^{}]{0,40}\})+\}|[A-Za-z0-9]+)))+\)"
    r"|[A-Za-z](?:[_^](?:\{(?:[^{}]|\{[^{}]{0,40}\})+\}|[A-Za-z0-9'\\]+))+"
    r"|\\\{(?:[^{}]|\{[^{}]{0,40}\})*\\\}"
    r"|\\[A-Za-z]+(?:\s*\{[^{}]{0,80}\})+(?:[_^](?:\{[^{}]{0,40}\}|[A-Za-z0-9]+))*"
    r"|\\(?:in|cdot|times|mid|oplus|otimes|leq|geq|neq|pm|infty|ldots|dots)(?![A-Za-z])"
    r")"
)


def _wrap_bare_zh_math(text: str) -> str:
    """译文正文里的裸 LaTeX 包进 $，已在 $...$ 里的不要再包。"""
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        if _in_math_span(match.string, match.start()):
            return match.group(0)
        return f"${match.group(1)}$"

    return _BARE_ZH_MATH.sub(repl, text)


_WORD_EMBEDDED_MATH = re.compile(
    r"(?<=[A-Za-z])\$([A-Za-z]+(?:[\^_]\{[A-Za-z]+\})*)\$"
    r"|\$([A-Za-z]+(?:[\^_]\{[A-Za-z]+\})*)\$(?=[A-Za-z])"
)
_EQ_NUMBER_TAIL = re.compile(r"[.,]?\s*\((\d+[a-z]?)\)\s*$")
_CMD_SPACE = re.compile(r"\\([A-Za-z]+)\s+\{")
_BARE_TEX_CMD = re.compile(
    r"(?<![$\\])("
    r"\\[A-Za-z]+(?:\s*\{[^{}]{0,80}\})+"
    r"(?:[_^](?:\{[^{}]{0,40}\}|[A-Za-z0-9]+))*"
    r")"
)
_BEGIN_ENV = re.compile(
    r"\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?|aligned)\}"
)


def _join_math_pieces(pieces: list[str]) -> str:
    out: list[str] = []
    for piece in pieces:
        if (
            out
            and out[-1].endswith("$")
            and not out[-1].endswith("$$")
            and piece[:1].isalpha()
        ):
            out.append(" ")
        if out and piece.startswith("$") and not piece.startswith("$$") and out[-1][-1:].isalpha():
            inner = piece.strip("$")[:2]
            if not inner.startswith(("-", "−")):
                out.append(" ")
        out.append(piece)
    return "".join(out)


def _split_math_segments(text: str) -> list[tuple[str, str]]:
    """按最长定界符扫描（$$ 优先于 $），空的 $$ 包装直接丢掉。"""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    index = 0
    length = len(text)

    def flush_text() -> None:
        if buf:
            out.append(("text", "".join(buf)))
            buf.clear()

    def skip_extra_dollars(pos: int) -> int:
        while pos < length:
            cursor = pos
            while cursor < length and text[cursor] in " \t\n":
                cursor += 1
            if text.startswith("$$", cursor):
                pos = cursor + 2
                continue
            break
        return pos

    def skip_empty_trailers(pos: int) -> int:
        while pos < length:
            cursor = pos
            while cursor < length and text[cursor] in " \t\n":
                cursor += 1
            if not text.startswith("$$", cursor):
                break
            nxt = cursor + 2
            while nxt < length and text[nxt] in " \t\n":
                nxt += 1
            if nxt >= length or text.startswith("$$", nxt):
                pos = cursor + 2
                continue
            break
        return pos

    while index < length:
        if text.startswith("\\$", index):
            buf.append("\\$")
            index += 2
            continue
        env_match = _BEGIN_ENV.match(text, index)
        if env_match:
            env = env_match.group(1)
            close = f"\\end{{{env}}}"
            end = text.find(close, env_match.end())
            if end >= 0:
                flush_text()
                out.append(("display", text[index : end + len(close)]))
                index = end + len(close)
                continue
        if text.startswith("\\[", index):
            end = text.find("\\]", index + 2)
            if end >= 0:
                flush_text()
                out.append(("display", text[index + 2 : end]))
                index = end + 2
                continue
        if text.startswith("\\(", index):
            end = text.find("\\)", index + 2)
            if end >= 0:
                flush_text()
                out.append(("inline", text[index + 2 : end]))
                index = end + 2
                continue
        if text.startswith("$$", index):
            start = skip_extra_dollars(index + 2)
            close = text.find("$$", start)
            if close < 0:
                buf.append(text[index:])
                break
            flush_text()
            out.append(("display", text[start:close]))
            index = skip_empty_trailers(close + 2)
            continue
        if text[index] == "$":
            close = text.find("$", index + 1)
            if close >= 0 and not text.startswith("$$", close):
                flush_text()
                out.append(("inline", text[index + 1 : close]))
                index = close + 1
                continue
        buf.append(text[index])
        index += 1
    flush_text()
    return out or [("text", text)]


_LEAD_DICT_TERM = re.compile(
    r"(?<![A-Za-z0-9])(?:\{\})?_\{([A-Za-z0-9]+)\}\s*([A-Za-z])_\{([A-Za-z0-9]+)\}"
)
_DOUBLE_SUP = re.compile(
    r"([A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?\s*\^\{[^{}]+\})\s+\^\{([^{}]+)\}"
)
_ORPHAN_SUP = re.compile(r"(^|,\s*)(?:\{\})?\^\{([^{}]+)\}")


def _repair_pdf_script_artifacts(body: str) -> str:
    """PDF 丢掉矩阵字母后会留下 _{r}h_{r}、z^{v} ^{Qv}，KaTeX 无法解析。"""
    cleaned = (body or "").replace("\\ ", " ")
    cleaned = _DOUBLE_SUP.sub(r"\1 W^{\2}", cleaned)
    cleaned = _ORPHAN_SUP.sub(r"\1 W^{\2}", cleaned)
    cleaned = _repair_leading_dict_terms(cleaned)
    cleaned = re.sub(r"=([A-Za-z])", r"= \1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _repair_leading_dict_terms(body: str) -> str:
    if not _LEAD_DICT_TERM.search(body):
        return body

    def repair_row(row: str) -> str:
        dict_letter: str | None = None

        def repl(match: re.Match[str]) -> str:
            nonlocal dict_letter
            sub, base, hsub = match.group(1), match.group(2), match.group(3)
            if dict_letter is None:
                dict_letter = sub[0].upper()
            return rf"{dict_letter}_{{{sub}}} {base}_{{{hsub}}}"

        out = _LEAD_DICT_TERM.sub(repl, row)
        return re.sub(r"([a-z]_\{[^{}]+\})\s+([A-Z]_\{)", r"\1 + \2", out)

    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in body:
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(char)
    parts.append("".join(buf))
    hits = sum(1 for part in parts if _LEAD_DICT_TERM.search(part))
    if hits >= 2:
        return ", ".join(repair_row(part.strip()) for part in parts if part.strip())
    return repair_row(body)


def _sanitize_math_body(body: str) -> str:
    cleaned = _CMD_SPACE.sub(r"\\\1{", body.replace("$", "").replace("\ufffd", ""))
    for char, latex in _GREEK_LATEX.items():
        cleaned = cleaned.replace(char, latex)
    cleaned = re.sub(r"([_^])\{([^{}\\]*)\\\}", r"\1{\2}", cleaned)
    cleaned = _LEAKED_SUB.sub(r"\1_\2 \3", cleaned)
    cleaned = re.sub(r"(?<!mathbb\{)(?<![A-Za-z\\])R\^\{", r"\\mathbb{R}^{", cleaned)
    cleaned = _WS_SCRIPT.sub(r"\1^{\2}_{\3}", cleaned)
    cleaned = _REAL_DIM.sub(r"\\mathbb{R}^{\1 \\times \2}", cleaned)
    cleaned = _REAL_SPACE.sub(r"\\mathbb{R}^{\1 \\times \2}", cleaned)
    cleaned = _LOG_STACKED_FRAC.sub(r"\\log(\\frac{\1_1}{\1_0})", cleaned)
    cleaned = re.sub(r"(\}_{[A-Za-z0-9]+}),(?=[A-Z])", r"\1, ", cleaned)
    cleaned = re.sub(r"\\times(?=[A-Za-z])", r"\\times ", cleaned)
    cleaned = re.sub(r"(\\[A-Za-z]+)\s+([_^])", r"\1\2", cleaned)
    cleaned = re.sub(r"\^\{\{([^{}]+)\}\}", r"^{\1}", cleaned)
    cleaned = _repair_tensor_shapes(cleaned)
    cleaned = _repair_pdf_script_artifacts(cleaned)
    if cleaned.endswith(".") and not cleaned.endswith(r"\ldots"):
        cleaned = cleaned[:-1]
    if "\\begin{" not in cleaned:
        cleaned = re.sub(r"[ \t]*\n[ \t]*", " ", cleaned)
    return cleaned.strip()


def _with_eq_tag(body: str) -> str:
    match = _EQ_NUMBER_TAIL.search(body)
    if not match:
        return body
    return body[: match.start()].rstrip(" .,") + rf" \tag{{{match.group(1)}}}"


def _keep_as_display(body: str) -> bool:
    from dl_agent.knowledge.formula_latex import is_garbled_math, looks_like_latex

    blob = body.strip()
    if not blob or is_garbled_math(blob) or not looks_like_latex(blob):
        return False
    if re.match(r"^[)）\]|,，、;；]", blob):
        return False
    if re.match(r"^\\?\|[A-Za-z]{1,8}\\?\)$", blob.replace(" ", "")):
        return False
    if re.match(r"^[A-Za-z]{1,4}:\s*", blob):
        return False
    if blob.count("{") != blob.count("}"):
        return False
    plain = re.sub(r"\\(?:mathrm|text|operatorname|mathbf|mathit)\{[^{}]*\}", "", blob)
    plain = re.sub(r"\\[A-Za-z]+", "", plain)
    if re.search(r"\b[A-Za-z]{5,}\b", plain):
        return False
    if "=" in blob:
        return True
    if "+" in blob and re.search(r"[_^]", blob):
        return True
    return bool("\\{" in blob or re.search(r"\\[A-Za-z]{2,}", blob))


_PROSE_SPLIT = re.compile(
    r"(?i)(?<![\\$])\s+(where|denotes?|represents?|indicates?|indexes|is the|are the)\b"
)
_TAG_THEN_PROSE = re.compile(r"(\\tag\{[^}]+\})\s+(?=[A-Za-z]{3,})")
_TAG_THEN_ZH = re.compile(r"(\\tag\{[^}]+\})\s*(?=其中(?!的)|式中)")
_ZH_WHERE_SPLIT = re.compile(r"(?<![\\$])\s+(其中(?!的)|式中)")


def _peel_math_prose(body: str) -> tuple[str, str]:
    """公式后粘了英文/「其中」时拆开，避免 where / 其中 把 $$ 整段毁掉。"""
    blob = (body or "").strip()
    if not blob:
        return "", ""
    tagged_zh = _TAG_THEN_ZH.search(blob)
    if tagged_zh:
        return blob[: tagged_zh.end(1)].strip(), blob[tagged_zh.end() :].strip()
    tagged = _TAG_THEN_PROSE.search(blob)
    if tagged:
        return blob[: tagged.end(1)].strip(), blob[tagged.end() :].strip()
    zh_split = _ZH_WHERE_SPLIT.search(blob)
    if zh_split and zh_split.start() > 6:
        from dl_agent.knowledge.formula_latex import looks_like_latex

        left = blob[: zh_split.start()].strip()
        right = blob[zh_split.start() :].strip()
        if looks_like_latex(left) or "=" in left:
            return left, right
    split = _PROSE_SPLIT.search(blob)
    if split and split.start() > 6:
        from dl_agent.knowledge.formula_latex import looks_like_latex

        left = blob[: split.start()].strip()
        right = blob[split.start() :].strip()
        if looks_like_latex(left):
            return left, right
    return blob, ""


def _emit_display(body: str) -> str:
    from dl_agent.knowledge.formula_latex import looks_like_latex

    cleaned = _with_eq_tag(_sanitize_math_body(body))
    if not cleaned:
        return "\n"
    math, prose = _peel_math_prose(cleaned)
    math = _with_eq_tag(_sanitize_math_body(math)) if math else ""
    can_display = bool(math) and (
        _keep_as_display(math) or (bool(prose) and looks_like_latex(math))
    )
    if can_display:
        out = f"\n$$\n{math}\n$$\n"
        if prose:
            out += prose.strip() + "\n"
        return out
    if math and looks_like_latex(math):
        plain = re.sub(r"\\[A-Za-z]+", "", math)
        if not re.search(r"\b[A-Za-z]{5,}\b", plain):
            return f"\n$$\n{math}\n$$\n"
    if _looks_like_assignment(cleaned) or _display_looks_continuation(cleaned):
        return f"\n$$\n{cleaned}\n$$\n"
    if len(re.sub(r"\s+", "", cleaned)) > 24:
        return "\n" + cleaned + "\n"
    return "\n"


def _emit_inline(body: str) -> str:
    raw = body.strip()
    trailing_dot = raw.endswith(".") and not raw.endswith(r"\ldots")
    cleaned = _sanitize_math_body(body)
    if _is_junk_inline(cleaned):
        return ""
    return f"${cleaned}$." if trailing_dot else f"${cleaned}$"


_OP_INLINE = re.compile(
    r"^(?:\\(?:times|cdot|leq|geq|in|oplus|otimes|pm)|[×·⋅≤≥∈⊕]|"
    r"\\tag\{[^}]+\})$"
)
_SINGLE_SYM = re.compile(r"^[A-Za-z](?:[_^]\{[^{}]+\})?$")


def _is_operator_inline(blob: str) -> bool:
    cleaned = _sanitize_math_body(blob)
    return bool(_OP_INLINE.match(cleaned.replace(" ", "")) or _OP_INLINE.match(cleaned))


def _display_looks_truncated(body: str) -> bool:
    blob = _sanitize_math_body(body)
    return bool(re.search(r"(?:\\in|∈|=)\s*$", blob))


def _display_looks_continuation(body: str) -> bool:
    blob = _sanitize_math_body(body)
    return bool(re.match(r"(?:\\in|∈)", blob))


def _is_shattered_piece(kind: str, body: str, *, in_run: bool) -> bool:
    blob = body.strip()
    if not blob:
        return in_run
    if kind == "display":
        cleaned = _sanitize_math_body(blob)
        if _keep_as_display(cleaned):
            return in_run and (
                _display_looks_truncated(cleaned) or _display_looks_continuation(cleaned)
            )
        from dl_agent.knowledge.formula_latex import looks_like_latex

        return looks_like_latex(cleaned) or r"\tag" in cleaned or _display_looks_continuation(cleaned)
    if kind == "inline":
        cleaned = _sanitize_math_body(blob)
        if r"\tag" in cleaned:
            return True
        if _is_operator_inline(cleaned):
            return True
        if in_run and r"\mathbb" in cleaned:
            return True
        if in_run and re.fullmatch(r"\([^)]{1,48}\)", cleaned.replace(" ", "")):
            return True
        if in_run and (_SINGLE_SYM.match(cleaned) or len(cleaned) <= 24 and re.search(r"[_^\\]", cleaned)):
            return True
        return False
    if re.search(r"\\(?:tag|mathbb)|[_^]\{", blob):
        return True
    if in_run and re.fullmatch(r"[,;:.\s]+", blob):
        return True
    if in_run and len(blob) < 16 and re.search(r"[_^\\()]", blob):
        return True
    return False


def _is_shattered_start(kind: str, body: str) -> bool:
    blob = body.strip()
    if kind == "display":
        cleaned = _sanitize_math_body(blob)
        if _display_looks_truncated(blob) or _display_looks_continuation(cleaned):
            return True
        if "=" in cleaned and not (r"\in" in cleaned or "∈" in cleaned):
            return True
        return not _keep_as_display(cleaned) and (
            "\\tag" in blob or "\\mathbb" in blob or "=" in blob
        )
    if kind == "inline":
        cleaned = _sanitize_math_body(blob)
        return r"\tag" in cleaned or _is_operator_inline(cleaned)
    return bool(re.search(r"\\(?:tag|mathbb)", blob) or _looks_like_assignment(blob))


def _looks_like_assignment(blob: str) -> bool:
    """赋值式，含 F_{out}^{{(l)}}= 这种双层括号。"""
    cleaned = _sanitize_math_body(blob)
    match = re.search(r"[A-Za-z](?:_\{[^{}]+\}|\^\{[^{}]+\})+\s*=", cleaned)
    if not match:
        return False
    head = cleaned[: match.start()].strip()
    return len(head) < 8


def _collapse_shattered_formulas(text: str) -> str:
    """把被拆成 × / ≤ / \\tag 碎片的展示公式重新收成一块。"""
    if not text:
        return text
    segs = _split_math_segments(text)
    rebuilt: list[str] = []
    index = 0
    while index < len(segs):
        kind, body = segs[index]
        if not _is_shattered_start(kind, body):
            rebuilt.append(_emit_segment(kind, body))
            index += 1
            continue
        chunks = [(kind, body)]
        cursor = index + 1
        while cursor < len(segs) and _is_shattered_piece(*segs[cursor], in_run=True):
            nxt_kind, nxt_body = segs[cursor]
            if (
                nxt_kind == "display"
                and _keep_as_display(_sanitize_math_body(nxt_body))
                and not _display_looks_truncated(nxt_body)
                and not _display_looks_continuation(nxt_body)
            ):
                break
            chunks.append((nxt_kind, nxt_body))
            cursor += 1
        if len(chunks) == 1 and kind == "display" and _keep_as_display(_sanitize_math_body(body)):
            rebuilt.append(_emit_display(body))
            index += 1
            continue
        combined = " ".join(_sanitize_math_body(part) for _, part in chunks)
        math, prose = _peel_math_prose(combined)
        from dl_agent.knowledge.formula_latex import looks_like_latex

        if math and looks_like_latex(math) and (
            "=" in math or r"\in" in math or r"\tag" in math or "∈" in math
        ):
            rebuilt.append(_emit_display(math))
            if prose:
                rebuilt.append(prose.strip() + "\n")
        else:
            for piece_kind, piece_body in chunks:
                rebuilt.append(_emit_segment(piece_kind, piece_body))
        index = cursor
    return _join_math_pieces(rebuilt)


def _emit_segment(kind: str, body: str) -> str:
    if kind == "display":
        return _emit_display(body)
    if kind == "inline":
        return _emit_inline(body)
    return body


def _unwrap_word_math(match: re.Match[str]) -> str:
    inner = match.group(1) or match.group(2) or ""
    return re.sub(r"[\^_{}$]", "", inner)


def _is_junk_inline(body: str) -> bool:
    from dl_agent.knowledge.formula_latex import is_garbled_math

    stripped = body.strip()
    if not stripped or is_garbled_math(stripped):
        return True
    if re.fullmatch(r"[_^{},\s.\\]+", stripped):
        return True
    if stripped in {"=", "-", "+", "*", "/", ",", ".", ";", ":"}:
        return True
    if stripped.startswith("_") or stripped.startswith("^"):
        return True
    if stripped.count("{") != stripped.count("}"):
        return True
    if r"\_" in stripped or r"\backslash" in stripped:
        return True
    if re.match(r"[a-z]{3,}[\s\\({]", stripped):
        return True
    return False


_BARE_INLINE_MATH = re.compile(
    r"(?<![$\\])(?<![A-Za-z0-9])"
    r"("
    r"[A-Za-z]_(?:[A-Za-z]{1,3}|\d{1,4})(?:\^(?:[A-Za-z]{1,3}|\d{1,4}))?"
    r"(?:\s*=\s*\{[^{}\n]{1,200}\})?"
    r")(?![A-Za-z0-9])"
)
_ABS_MATH = re.compile(r"\|([^|]{3,80})\|")
_FLOOR_DOLLARS = re.compile(r"⌊\s*\$([^$]+)\$\s*⌋")
_BARE_TEX_SYMBOL = re.compile(
    r"(?<![$\\])(\\"
    r"(?:alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|"
    r"iota|kappa|lambda|mu|nu|xi|pi|varpi|rho|varrho|sigma|varsigma|tau|"
    r"upsilon|phi|varphi|chi|psi|omega|ell|infty|times|cdot|in|leq|geq|"
    r"neq|pm|cap|cup|circ|odot)"
    r"(?:[_^](?:\{[^{}]+\}|[A-Za-z0-9]+))*"
    r")(?![A-Za-z])"
)
_K_FLOOR = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z])\s*=\s*\$(\\lfloor[^$]+\\rfloor)\$"
)
_EMBEDDED_MATH = re.compile(
    r"([A-Za-z]{2,})\s*\$([-−]?)([^$\n]{1,40})\$\s*([A-Za-z]{2,})"
)
_GLUED_IDENTIFIER = re.compile(
    r"\b([A-Z][A-Za-z]?)\s+([A-Za-z]{1,8})\s*\$(\\(?:in|subset|subseteq|notin|leq|geq)[^$]*)\$"
)
_SPACED_SUB = re.compile(r"\b([A-Z])\s+([a-z])\b")
_LEAKED_SUB = re.compile(
    r"([A-Za-z])\}([a-z]{1,4})\s*\^\{((?:\\times|\\cdot|\\otimes)\s*)"
)
_LOSS_CAL = re.compile(r"(?<![$\\])\bL(?:CE|Tri|\s+Tri|\s+C(?![A-Za-z]))\b")
_ENGLISH_CAPS = {"A", "I"}


def _replace_loss_cal(match: re.Match[str]) -> str:
    name = re.sub(r"\s+", "", match.group(0))[1:]
    return rf"$\mathcal{{L}}_{{{name}}}$"


def _unwrap_embedded_math(match: re.Match[str]) -> str:
    left, hyphen, body, right = match.groups()
    blob = body.strip().lstrip("-−")
    if not re.fullmatch(r"[A-Za-z]+(?:\^\{[A-Za-z]+\}|\^[A-Za-z])?", blob):
        return match.group(0)
    letters = re.sub(r"[^A-Za-z]", "", blob)
    glue = "-" if hyphen else ""
    return f"{left}{glue}{letters}{right}"


def _merge_glued_identifier(match: re.Match[str]) -> str:
    base, sub, rest = match.group(1), match.group(2), match.group(3)
    return f"${base}_{{\\mathrm{{{sub}}}}} {rest}$"


def _accent_cmd(mark: str) -> str | None:
    return _ACCENT_CMD.get(mark)


def _merge_accent_before_math(match: re.Match[str]) -> str:
    cmd = _accent_cmd(match.group(1))
    if not cmd:
        return match.group(0)
    return f"$\\{cmd}{{{match.group(2)}}}{match.group(3)}$"


def _replace_spaced_accent(match: re.Match[str]) -> str:
    cmd = _accent_cmd(match.group(1))
    if not cmd:
        return match.group(0)
    base, sub = match.group(2), match.group(3)
    latex = f"\\{cmd}{{{base}}}"
    if sub:
        latex += f"_{sub}"
    return f"${latex}$"


def _replace_spaced_sub(match: re.Match[str]) -> str:
    if match.group(1) in _ENGLISH_CAPS:
        return match.group(0)
    return f"${match.group(1)}_{match.group(2)}$"


_LEAKED_SET_MEMBER = re.compile(
    r"(?<![$\\])"
    r"([A-Za-z]{1,8}(?:_\{[^{}]+\})?(?:\^\{[^{}]+\})?)\s*"
    r"(\\in|∈)\s*"
    r"(\\mathbb\{[^}]+\})"
    r"((?:\s*[\^_](?:\{[^{}]*\}|[A-Za-z0-9]+)){0,4})"
)
_LEAKED_LEQ_GROUP = re.compile(
    r"(?<![$])(\((?:[^$()\n]{0,16})(?:\\leq|\\geq|\\le|\\ge|≤|≥)(?:[^$()\n]{0,32})\))"
)


def _in_math_span(text: str, index: int) -> bool:
    before = text[:index]
    if before.count("$$") % 2 == 1:
        return True
    return before.replace("$$", "").count("$") % 2 == 1


def _wrap_set_member(match: re.Match[str]) -> str:
    if _in_math_span(match.string, match.start()):
        return match.group(0)
    ident, inn, bb, scripts = match.group(1), match.group(2), match.group(3), match.group(4)
    if re.fullmatch(r"[A-Z]{2}", ident):
        ident = f"{ident[0]}_{{{ident[1]}}}"
    elif re.fullmatch(r"[A-Z]{3}", ident) and _should_unglue_tensor(ident):
        ident = f"{ident[0]}_{{{ident[2]}}}^{{{ident[1]}}}"
    inn = r"\in" if inn == "∈" else inn
    return f"${ident} {inn} {bb}{scripts}$"


def _wrap_leq_group(match: re.Match[str]) -> str:
    if _in_math_span(match.string, match.start()):
        return match.group(0)
    return f"${match.group(1)}$"


def _wrap_tex_cmd(match: re.Match[str]) -> str:
    if _in_math_span(match.string, match.start()):
        return match.group(0)
    return f"${_sanitize_math_body(match.group(1))}$"


def _wrap_tex_symbol(match: re.Match[str]) -> str:
    if _in_math_span(match.string, match.start()):
        return match.group(0)
    return f"${match.group(1)}$"


def _wrap_prose_real_dim(match: re.Match[str]) -> str:
    if _in_math_span(match.string, match.start()):
        return match.group(0)
    ident, inn, dim, rest = match.group(1), match.group(2), match.group(3), match.group(4)
    inn = r"\in" if inn == "∈" else inn
    body = rf"{inn} \mathbb{{R}}^{{{dim} \times {rest}}}"
    if ident:
        body = f"{ident} {body}"
    return f"${body}$"


def _wrap_delta_log(match: re.Match[str]) -> str:
    if _in_math_span(match.string, match.start()):
        return match.group(0)
    ident = match.group(1)
    letter = match.group(2)
    body = rf"\log(\frac{{{letter}_1}}{{{letter}_0}})"
    if ident:
        ident = r"\Delta" if ident in {"Δ", "δ"} else ident
        body = f"{ident} = {body}"
    return f"${body}$"


def _wrap_bare_assign(match: re.Match[str]) -> str:
    if _in_math_span(match.string, match.start()):
        return match.group(0)
    from dl_agent.knowledge.formula_latex import looks_like_latex

    blob = match.group(1).rstrip()
    split = re.search(
        r"(?i)\s+(where|and|is|are|the|with|for|denotes?|represents?)\b",
        blob,
    )
    rest = ""
    if split and split.start() > 8:
        rest = blob[split.start() :]
        blob = blob[: split.start()].rstrip()
    if not looks_like_latex(blob) or "=" not in blob:
        return match.group(0)
    return f"\n$$\n{_sanitize_math_body(blob)}\n$$\n{rest}"


def _wrap_prose_math(text: str) -> str:
    """把正文里的 x_m^i、|a-b|、\\mu、⌊$n\\times r$⌋ 收成行内公式。"""
    text = _FLOOR_DOLLARS.sub(lambda m: r"$\lfloor " + m.group(1).strip() + r" \rfloor$", text)
    text = _K_FLOOR.sub(lambda m: f"${m.group(1)} = {m.group(2)}$", text)
    text = _ABS_MATH.sub(_replace_abs_math, text)
    rebuilt: list[str] = []
    for kind, body in _split_math_segments(text):
        if kind == "display":
            rebuilt.append(_emit_display(body))
        elif kind == "inline":
            rebuilt.append(_emit_inline(body))
        else:
            part = _repair_tensor_shapes(body)
            part = _BARE_ASSIGN.sub(_wrap_bare_assign, part)
            part = _DELTA_LOG_FRAC.sub(_wrap_delta_log, part)
            part = _PROSE_REAL_DIM.sub(_wrap_prose_real_dim, part)
            part = _LEAKED_SET_MEMBER.sub(_wrap_set_member, part)
            part = _LEAKED_LEQ_GROUP.sub(_wrap_leq_group, part)
            part = _LOSS_CAL.sub(_replace_loss_cal, part)
            part = _BARE_TEX_CMD.sub(_wrap_tex_cmd, part)
            part = _BARE_TEX_SYMBOL.sub(_wrap_tex_symbol, part)
            part = _SPACED_SUB.sub(_replace_spaced_sub, part)
            part = _BARE_INLINE_MATH.sub(_replace_bare_inline, part)
            rebuilt.append(part)
    return _join_math_pieces(rebuilt)


def _replace_abs_math(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not re.search(r"[_^\\]", inner):
        return match.group(0)
    return r"$|" + re.sub(r"\s+", " ", inner) + "|$"


def _wrap_bare_inline_math(text: str) -> str:
    """把混在英文里的 x_m^p、x_m^p = {...} 收成 $LaTeX$，让前端 KaTeX 能渲染。"""
    if not text:
        return text
    rebuilt: list[str] = []
    for kind, body in _split_math_segments(text):
        if kind == "display":
            rebuilt.append(_emit_display(body))
        elif kind == "inline":
            rebuilt.append(_emit_inline(body))
        elif body.strip().startswith("|"):
            rebuilt.append(body)
        else:
            rebuilt.append(_wrap_prose_math(body))
    return _join_math_pieces(rebuilt)


def _replace_bare_inline(match: re.Match[str]) -> str:
    expr = match.group(1)
    if match.string[: match.start()].count("$") % 2 == 1:
        return expr
    before = match.string[max(0, match.start() - 12) : match.start()]
    after = match.string[match.end() : match.end() + 8]
    if re.search(r"\\[A-Za-z]*\{?$", before) or after.startswith("\\}") or after.startswith("}"):
        return expr
    body = re.sub(r"\s+", " ", expr.strip())
    body = body.replace("...", r"\ldots")
    body = body.replace("{", r"\{").replace("}", r"\}")
    return f"${body}$"


def _tidy_text_segment(text: str) -> str:
    if not text:
        return text
    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|"):
            lines.append(line)
            continue
        if _keep_as_display(_sanitize_math_body(stripped)):
            lines.append(_emit_display(stripped).strip("\n"))
            continue
        lines.append(_tidy_math_paragraph(line))
    cleaned = "\n".join(lines)
    cleaned = _COMBINING_ACCENT.sub(_replace_combining_accent, cleaned)
    return _map_plain(cleaned, _wrap_prose_math)


def _map_plain(text: str, transform) -> str:
    rebuilt: list[str] = []
    for kind, body in _split_math_segments(text):
        if kind == "display":
            rebuilt.append(_emit_display(body))
        elif kind == "inline":
            rebuilt.append(_emit_inline(body))
        else:
            rebuilt.append(transform(body))
    return _join_math_pieces(rebuilt)


def _replace_combining_accent(match: re.Match[str]) -> str:
    base, mark, sub = match.group(1), match.group(2), match.group(3)
    cmd = _accent_cmd(mark) or "hat"
    latex = f"\\{cmd}{{{base}}}"
    if sub:
        latex += f"_{{{sub}}}" if len(sub) > 1 else f"_{sub}"
    return f"${latex}$"


_PROSE_NAMES = {
    "the", "and", "for", "with", "from", "this", "that", "where", "when",
    "then", "than", "each", "both", "such", "also", "into", "over",
}


def _replace_named_triple(match: re.Match[str]) -> str:
    name, sup, sub = match.group(1), match.group(2), match.group(3)
    if name.lower() in _PROSE_NAMES:
        return match.group(0)
    return rf"\mathrm{{{name}}}_{sub}^{{{sup}}}"


def _tidy_math_paragraph(text: str) -> str:
    def replace_hat(match: re.Match[str]) -> str:
        base = match.group(1)
        sub = match.group(2)
        return f"{base}\u0302_{sub}" if sub else f"{base}\u0302"

    def replace_triple(match: re.Match[str]) -> str:
        base, mid, sub = match.group(1), match.group(2), match.group(3)
        if len(mid) > 1 and not mid.isdigit():
            return match.group(0)
        return f"{base}_{sub}^{mid}"

    cleaned = _TEX_JUNK.sub("", text)
    cleaned = _repair_tensor_shapes(cleaned)
    cleaned = _PROSE_REAL_DIM.sub(_wrap_prose_real_dim, cleaned)
    cleaned = _DELTA_LOG_FRAC.sub(_wrap_delta_log, cleaned)
    cleaned = _LEAKED_SET_MEMBER.sub(_wrap_set_member, cleaned)
    cleaned = _HAT.sub(replace_hat, cleaned)
    cleaned = _SPACED_ACCENT.sub(_replace_spaced_accent, cleaned)
    cleaned = _fold_spaced_math(cleaned)
    cleaned = _TRIPLE.sub(replace_triple, cleaned)
    cleaned = _NAMED_TRIPLE.sub(_replace_named_triple, cleaned)
    cleaned = re.sub(r"\s*\|\s*", "|", cleaned)
    cleaned = _SPACE_PUNCT.sub(r"\1", cleaned)
    cleaned = _OPEN_SPACE.sub(r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned)


_MATH_OP_CHARS = "×·⋅−±=∈∉⊂⊃∪∩→↔≤≥≠≈∼∘⊙∗+"
_GREEK_CHARS = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩℓ"
_MATH_TOKEN = re.compile(
    rf"([A-Za-z]{{2,}})|([{_GREEK_CHARS}])|([A-Za-z])|(\d+)|([{re.escape(_MATH_OP_CHARS)}])|(\s+)|(.)"
)
_OP_TO_LATEX = {
    "×": r"\times ",
    "·": r"\cdot ",
    "⋅": r"\cdot ",
    "−": "-",
    "±": r"\pm ",
    "∈": r"\in ",
    "∉": r"\notin ",
    "≤": r"\leq ",
    "≥": r"\geq ",
    "≠": r"\neq ",
    "≈": r"\approx ",
    "→": r"\rightarrow ",
    "⊂": r"\subset ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "∘": r"\circ ",
    "⊙": r"\odot ",
    "∗": r"\ast ",
    "+": "+",
    "=": "=",
}


def _fold_spaced_math(text: str) -> str:
    """把 L p × D、α ∈ [0,1] 这类「单字母+运算符」收成一段 $LaTeX$，不逐条写正则。"""
    rebuilt: list[str] = []
    for kind, body in _split_math_segments(text):
        if kind == "display":
            rebuilt.append(f"$$\n{body}\n$$" if body.strip() else "")
        elif kind == "inline":
            rebuilt.append(f"${body}$" if body.strip() else "")
        else:
            rebuilt.append(_fold_spaced_math_run(body))
    return "".join(rebuilt)


def _fold_spaced_math_run(text: str) -> str:
    pieces: list[str] = []
    run: list[tuple[str, str]] = []

    def flush() -> None:
        if not run:
            return
        if _run_looks_like_math(run):
            pieces.append("$" + _run_to_latex(run).strip() + "$")
        else:
            pieces.append("".join(value for _, value in run))
        run.clear()

    for match in _MATH_TOKEN.finditer(text):
        word, greek, letter, number, op, space, other = match.groups()
        if word:
            flush()
            pieces.append(word)
        elif space:
            if run:
                run.append(("space", space))
            else:
                pieces.append(space)
        elif greek:
            run.append(("greek", greek))
        elif letter:
            run.append(("letter", letter))
        elif number:
            run.append(("number", number))
        elif op:
            run.append(("op", op))
        elif other in "()[]{},.":
            if run:
                run.append(("punct", other))
            else:
                pieces.append(other)
        else:
            flush()
            pieces.append(other or "")
    flush()
    return "".join(pieces)


_SYMBOL_OPS = set("×·⋅−±∈∉⊂⊃∪∩→↔≤≥≠≈∼∘⊙∗")


def _run_looks_like_math(run: list[tuple[str, str]]) -> bool:
    atoms = [(kind, value) for kind, value in run if kind != "space"]
    if len(atoms) < 3:
        return False
    return any(kind == "op" and value in _SYMBOL_OPS for kind, value in atoms)


def _run_to_latex(run: list[tuple[str, str]]) -> str:
    seq = [(kind, value) for kind, value in run if kind != "space"]
    out: list[str] = []
    index = 0
    while index < len(seq):
        kind, value = seq[index]
        nxt = seq[index + 1] if index + 1 < len(seq) else None
        if (
            kind == "letter"
            and nxt
            and nxt[0] == "letter"
            and len(value) == 1
            and len(nxt[1]) == 1
            and nxt[1].islower()
        ):
            out.append(f"{value}_{{{nxt[1]}}}")
            index += 2
            continue
        if kind == "op":
            out.append(_OP_TO_LATEX.get(value, value))
            index += 1
            continue
        if kind == "punct" and value in "{}":
            out.append("\\" + value)
            index += 1
            continue
        out.append(value)
        index += 1
    return "".join(out)


def _better_figure(new: Figure, old: Figure) -> bool:
    if old.source == "page_clip" and new.source != "page_clip":
        return True
    if new.source == "page_clip" and old.source != "page_clip":
        return False
    if new.storage_key and not old.storage_key:
        return True
    return False


_ABSTRACT_RUNIN = re.compile(
    r"(?:^|\n\n)\s*(abstract|摘要)\s*[-—–:.\u2013\u2014]*\s*",
    re.I,
)
_INDEX_TERMS_RUNIN = re.compile(
    r"(?:^|\n\n)\s*(index\s+terms|keywords?|关键[词字])\s*[-—–:.\u2013\u2014]*\s*",
    re.I,
)
_AFFILIATION_PARA = re.compile(
    r"(?ix)"
    r"\b(?:are|is|was)\s+with\b|"
    r"e-?mail\s*:|"
    r"corresponding author|"
    r"senior member,\s*ieee|fellow,\s*ieee"
)
_DROP_CAP = re.compile(r"^([A-Z])\s+([A-Z][a-z]{2,})\b")
_DROP_CAP_STOP = frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "there",
        "novel",
        "new",
        "recent",
        "traditional",
        "however",
        "although",
        "existing",
        "current",
        "many",
        "most",
        "several",
        "deep",
        "multi",
    }
)
_BODY_SECTION_KINDS = frozenset({"abstract", "intro", "related", "conclusion", "references"})


def repair_front_matter(sections: list[Section]) -> list[Section]:
    """拆出 Abstract— 跑题摘要，并把 IEEE 单位脚注从引言挪回作者区。GET 旧数据也能修好。"""
    if not sections:
        return sections
    repaired = [section.model_copy(deep=True) for section in sections]
    has_abstract = any(section.kind == "abstract" for section in repaired)
    if not has_abstract:
        repaired = _split_run_in_abstract(repaired)

    affiliations: list[str] = []
    for section in repaired:
        if section.kind in {"abstract", "references"}:
            continue
        if _is_title_or_front(section):
            continue
        kept, peeled = _peel_affiliation_paragraphs(section.text)
        if peeled:
            section.text = kept
            affiliations.extend(peeled)
        section.text = _fix_drop_caps(section.text)

    if affiliations:
        target = _title_or_front_section(repaired)
        extra = "\n\n".join(affiliations)
        if extra and extra not in (target.text or ""):
            target.text = f"{target.text}\n\n{extra}".strip() if target.text else extra
    return repaired


def _is_title_or_front(section: Section) -> bool:
    title = section.title.strip()
    if title.lower() == "front matter":
        return True
    if section.level != 1 or section.page_start > 1:
        return False
    if section.kind in _BODY_SECTION_KINDS:
        return False
    return looks_like_paper_title_heading(title)


def _title_or_front_section(sections: list[Section]) -> Section:
    for section in sections:
        if _is_title_or_front(section):
            return section
    return sections[0]


def _split_run_in_abstract(sections: list[Section]) -> list[Section]:
    for index, section in enumerate(sections):
        if section.kind == "abstract":
            return sections
        authors, abstract, index_terms = _split_title_blob(section.text)
        if not abstract and not index_terms:
            continue
        if _is_title_or_front(section) or index == 0:
            section.text = authors
            if looks_like_paper_title_heading(section.title) and section.kind not in _BODY_SECTION_KINDS:
                section.kind = "other"
            abstract_text = abstract
            if index_terms:
                abstract_text = f"{abstract}\n\n{index_terms}" if abstract else index_terms
            inserted = Section(
                section_id=_next_section_id(sections),
                paper_id=section.paper_id,
                title="Abstract",
                kind="abstract",
                level=1,
                page_start=section.page_start,
                page_end=section.page_end,
                text=abstract_text,
                parent_id=None,
                figure_ids=[],
            )
            return sections[: index + 1] + [inserted] + sections[index + 1 :]
    return sections


def _split_title_blob(text: str) -> tuple[str, str, str]:
    blob = text or ""
    match = _ABSTRACT_RUNIN.search(blob)
    if not match:
        return blob.strip(), "", ""
    authors = blob[: match.start()].strip()
    rest = blob[match.end() :].strip()
    index_match = _INDEX_TERMS_RUNIN.search(rest)
    if not index_match:
        return authors, rest, ""
    abstract = rest[: index_match.start()].strip()
    terms = rest[index_match.start() :].strip()
    return authors, abstract, terms


def _peel_affiliation_paragraphs(text: str) -> tuple[str, list[str]]:
    if not text.strip():
        return text, []
    kept: list[str] = []
    peeled: list[str] = []
    for part in re.split(r"\n\s*\n", text):
        paragraph = part.strip()
        if not paragraph:
            continue
        if _is_affiliation_paragraph(paragraph):
            peeled.append(paragraph)
        else:
            kept.append(paragraph)
    return "\n\n".join(kept).strip(), peeled


def _is_affiliation_paragraph(text: str) -> bool:
    blob = re.sub(r"\s+", " ", text).strip()
    if len(blob) > 700:
        return False
    return bool(_AFFILIATION_PARA.search(blob))


def _fix_drop_caps(text: str) -> str:
    if not text:
        return text
    parts: list[str] = []
    for part in re.split(r"(\n\s*\n)", text):
        if not part.strip() or part.startswith("\n"):
            parts.append(part)
            continue
        match = _DROP_CAP.match(part)
        if match and match.group(2).lower() not in _DROP_CAP_STOP:
            letter, rest = match.group(1), match.group(2)
            parts.append(letter + rest[0].lower() + rest[1:] + part[match.end() :])
        else:
            parts.append(part)
    return "".join(parts)


def _next_section_id(sections: list[Section]) -> str:
    used = {section.section_id for section in sections}
    max_n = 0
    for section_id in used:
        match = re.match(r"sec-(\d+)$", section_id)
        if match:
            max_n = max(max_n, int(match.group(1)))
    n = max_n + 1
    while f"sec-{n:03d}" in used:
        n += 1
    return f"sec-{n:03d}"


def collapse_false_headings(sections: list[Section]) -> list[Section]:
    """把已落盘的断行假标题并回上一节，刷新 GET 时也能修好旧数据。"""
    kept: list[Section] = []
    for section in sections:
        copy = section.model_copy(deep=True)
        if kept and not is_plausible_heading(copy.title):
            prev = kept[-1]
            extra = copy.title.strip()
            body = copy.text.strip()
            if extra[:1].islower() and prev.text and not prev.text.rstrip().endswith((".", "!", "?")):
                prev.text = prev.text.rstrip() + " " + extra
                extra = ""
            addition = "\n\n".join(part for part in (extra, body) if part)
            if addition:
                prev.text = f"{prev.text}\n\n{addition}" if prev.text else addition
            for figure_id in copy.figure_ids:
                if figure_id not in prev.figure_ids:
                    prev.figure_ids.append(figure_id)
            prev.page_end = max(prev.page_end, copy.page_end)
            continue
        kept.append(copy)
    return kept


def _next_caption(
    items: list[LayoutItem], index: int
) -> tuple[str | None, str | None, str] | None:
    page = items[index].page
    for later in items[index + 1 : index + 6]:
        if later.page != page:
            break
        if later.kind == "heading":
            break
        blob = later.caption or later.text
        parsed = parse_caption(blob)
        if parsed[0]:
            return parsed
    return None
