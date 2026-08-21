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
        if not chunk or is_page_chrome(chunk):
            return
        section = current()
        if section is None:
            section = open_section("Front Matter", page, 1)
            section.kind = "other"
        if section.text.endswith(chunk) or (len(chunk) > 48 and chunk in section.text):
            section.page_end = max(section.page_end, page)
            return
        if section.text:
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
                and classify_kind(title) == "other"
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
        sections=collapse_false_headings(sections),
        figures=figures,
        figure_pngs=pngs,
        title=derived_title,
        abstract=abstract,
    )


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
_TRIPLE = re.compile(r"\b([A-Za-z])\s+([A-Za-z0-9]+)\s+([A-Za-z])\b")
_SPACE_PUNCT = re.compile(r"\s+([,;:.)\]}])")
_OPEN_SPACE = re.compile(r"([(\[{])\s+")


def tidy_math_prose(text: str) -> str:
    """把 PDF 抽字造成的 'x c m'、'ˆ x m'、TeX 残渣收成可读符号。已是 $LaTeX$ 的片段不动。"""
    if not text:
        return text
    text = _scrub_broken_math(text)
    pieces = re.split(r"(\$\$[\s\S]*?\$\$|\$[^$]*\$)", text)
    cleaned: list[str] = []
    for piece in pieces:
        if piece.startswith("$"):
            cleaned.append(piece)
            continue
        paragraphs = [
            _tidy_math_paragraph(part) if not part.strip().startswith("|") else part
            for part in piece.split("\n")
        ]
        cleaned.append("\n".join(paragraphs))
    out = "".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_BROKEN_INLINE = re.compile(r"\$([^$\n]+)\$")
_WORD_EMBEDDED_MATH = re.compile(
    r"(?<=[A-Za-z])\$([A-Za-z]+(?:[\^_]\{[A-Za-z]+\})*)\$"
    r"|\$([A-Za-z]+(?:[\^_]\{[A-Za-z]+\})*)\$(?=[A-Za-z])"
)


def _scrub_broken_math(text: str) -> str:
    text = re.sub(r"\$\$([\s\S]*?)\$\$", _drop_broken_display, text)
    text = _BROKEN_INLINE.sub(_drop_broken_inline, text)
    text = text.replace("\ufffd", "")
    text = _WORD_EMBEDDED_MATH.sub(_unwrap_word_math, text)
    text = _BROKEN_INLINE.sub(_drop_broken_inline, text)
    text = _keep_valid_display(text)
    text = _BROKEN_INLINE.sub(_drop_broken_inline, text)
    return _wrap_bare_equations(text)


def _keep_valid_display(text: str) -> str:
    from dl_agent.knowledge.formula_latex import is_garbled_math, looks_like_latex

    pieces: list[str] = []
    last = 0
    for match in re.finditer(r"\$\$([\s\S]*?)\$\$", text):
        body = match.group(1).strip()
        pieces.append(text[last:match.start()])
        if (
            body
            and looks_like_latex(body)
            and not is_garbled_math(body)
            and ("=" in body or re.search(r"\\[A-Za-z]+", body))
        ):
            pieces.append(f"$$\n{body}\n$$")
        else:
            pieces.append(body)
        last = match.end()
    pieces.append(text[last:].replace("$$", ""))
    return "".join(pieces)


def _drop_broken_display(match: re.Match[str]) -> str:
    from dl_agent.knowledge.formula_latex import is_garbled_math

    body = match.group(1)
    if not is_garbled_math(body):
        if not body.strip():
            return "\n"
        return match.group(0)
    cleaned = _BROKEN_INLINE.sub(_drop_broken_inline, body)
    cleaned = cleaned.replace("\ufffd", "")
    cleaned = re.sub(r"\$+", "", cleaned)
    prose = cleaned.strip()
    if len(re.sub(r"\s+", "", prose)) > 24:
        return "\n" + prose + "\n"
    return "\n"


def _drop_broken_inline(match: re.Match[str]) -> str:
    body = match.group(1)
    if _is_junk_inline(body):
        return ""
    return match.group(0)


def _unwrap_word_math(match: re.Match[str]) -> str:
    inner = match.group(1) or match.group(2) or ""
    return re.sub(r"[\^_{}$]", "", inner)
    leftover = text.split("$$")
    if len(leftover) == 2:
        text = leftover[0] + leftover[1]
    elif len(leftover) > 2:
        rebuilt: list[str] = [leftover[0]]
        for index, chunk in enumerate(leftover[1:], start=1):
            if index % 2 == 1 and looks_like_latex(chunk) and not is_garbled_math(chunk):
                rebuilt.append(f"$$\n{chunk.strip()}\n$$")
            else:
                rebuilt.append(chunk)
        text = "".join(rebuilt)
    text = re.sub(r"\$\$\s*\$\$", "\n", text)
    return _wrap_bare_equations(text)


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


def _wrap_bare_equations(text: str) -> str:
    from dl_agent.knowledge.formula_latex import looks_like_latex

    lines = text.split("\n")
    wrapped: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("$$")
            and not stripped.startswith("|")
            and "=" in stripped
            and "\\" in stripped
            and looks_like_latex(stripped)
            and len(stripped) < 500
            and not re.search(r"\b[A-Za-z]{5,}\b", re.sub(r"\\[A-Za-z]+", "", stripped))
        ):
            wrapped.append(f"$$\n{stripped}\n$$")
        else:
            wrapped.append(line)
    return "\n".join(wrapped)


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
    cleaned = _HAT.sub(replace_hat, cleaned)
    cleaned = _TRIPLE.sub(replace_triple, cleaned)
    cleaned = re.sub(r"\s*\|\s*", "|", cleaned)
    cleaned = _SPACE_PUNCT.sub(r"\1", cleaned)
    cleaned = _OPEN_SPACE.sub(r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def _better_figure(new: Figure, old: Figure) -> bool:
    if old.source == "page_clip" and new.source != "page_clip":
        return True
    if new.source == "page_clip" and old.source != "page_clip":
        return False
    if new.storage_key and not old.storage_key:
        return True
    return False


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
