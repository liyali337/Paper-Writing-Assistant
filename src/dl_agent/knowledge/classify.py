from __future__ import annotations

import re

from dl_agent.domain.models import FigureKind, SectionKind

CAPTION_RE = re.compile(
    r"^(fig(?:ure)?|table|algorithm)\s*\.?\s*([A-Z]?\d+)",
    re.IGNORECASE,
)

_NUMBER_PREFIX = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+)|(?:[A-Z]))[.\s:-]+",
    re.IGNORECASE,
)

_SKIP_HEADING = re.compile(
    r"^(acknowledgements?|acknowledgments?|致谢|running\s+title)\s*$",
    re.IGNORECASE,
)

_PAGE_NUM = re.compile(r"^\d{1,4}$")
_CORRESPONDING = re.compile(
    r"(?i)(?:\n+\s*)?\*{0,2}\s*corresponding author\.?\s*"
)
_CONFERENCE_OA = re.compile(
    r"(?i)^this\s+(cvpr|iccv|eccv|neurips|nips|icml|aaai|iclr|acl)\b.{0,160}open access",
)
_FOOTNOTE_MARK = re.compile(r"^\*+$")

_KIND_RULES: list[tuple[SectionKind, re.Pattern[str]]] = [
    ("abstract", re.compile(r"\babstract\b|摘要", re.I)),
    ("intro", re.compile(r"\bintroduct|\bbackground\b|\bmotivation\b", re.I)),
    ("related", re.compile(r"related\s+work|prior\s+work|literature\s+review", re.I)),
    (
        "method",
        re.compile(
            r"\bmethod|\bapproach\b|\bmodel\b|\barchitecture\b|\bproposed\b|\bframework\b",
            re.I,
        ),
    ),
    ("experiment", re.compile(r"\bexperiment|\bevaluat|\bresult|\bablation", re.I)),
    ("conclusion", re.compile(r"\bconclusion|\bdiscussion\b|\blimitation", re.I)),
    ("references", re.compile(r"\breferences\b|\bbibliography\b|参考文献", re.I)),
]


def strip_heading_number(title: str) -> str:
    return _NUMBER_PREFIX.sub("", title.strip()).strip()


def heading_level(title: str, tree_level: int = 1) -> int:
    match = re.match(r"^(\d+(?:\.\d+)*)\b", title.strip())
    if match:
        return match.group(1).count(".") + 1
    return max(1, min(int(tree_level or 1), 6))


def classify_kind(title: str) -> SectionKind:
    body = strip_heading_number(title)
    for kind, pattern in _KIND_RULES:
        if pattern.search(body):
            return kind
    return "other"


def is_skip_heading(title: str) -> bool:
    return bool(_SKIP_HEADING.match(strip_heading_number(title)))


def is_references_heading(title: str) -> bool:
    return classify_kind(title) == "references"


def is_page_number(text: str) -> bool:
    return bool(_PAGE_NUM.match(text.strip()))


def is_page_chrome(text: str) -> bool:
    """页脚、脚注、会议页眉，不应进入章节正文。"""
    blob = re.sub(r"\s+", " ", text).strip()
    if not blob:
        return True
    if is_page_number(blob) or _FOOTNOTE_MARK.match(blob):
        return True
    if re.fullmatch(r"(?i)\*{0,2}\s*corresponding author\.?", blob):
        return True
    if _CONFERENCE_OA.match(blob):
        return True
    return False


def strip_page_chrome(text: str) -> str:
    """从已拼好的章节里拿掉对应作者脚注，并补上被它切断的句子。"""
    if not text:
        return text
    cleaned = _CORRESPONDING.sub(" ", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"([A-Za-z]{2,})-\n+([a-z]{2,})", _join_hyphen_break, cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"([^\s.!?])\n\n+([a-z])", r"\1 \2", cleaned)
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", cleaned)
        if part.strip() and not is_page_chrome(part)
    ]
    return "\n\n".join(paragraphs).strip()


_HYPHEN_KEEP = {"of", "the", "and", "or", "to", "in", "on", "for", "at", "by"}


def _join_hyphen_break(match: re.Match[str]) -> str:
    left, right = match.group(1), match.group(2)
    if right in _HYPHEN_KEEP:
        return f"{left}-{right}"
    return f"{left}{right}"


_NUMBERED_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*|[IVXLC]{1,6}|[A-Z])[.\s:-]+\S",
    re.IGNORECASE,
)


def is_plausible_heading(title: str) -> bool:
    """挡住双栏断行被当成标题的碎片，例如 Introduction 末行的 'challenge.'。"""
    text = title.strip()
    if not text or len(text) > 220:
        return False
    if parse_caption(text)[0]:
        return False
    if classify_kind(text) != "other":
        return True
    if _NUMBERED_HEADING.match(text):
        return True
    if text[:1].islower():
        return False
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]+", text)
    if text.endswith((".", ",", ";", ":")) and len(words) <= 8:
        return False
    return len(text) >= 4


def parse_caption(text: str) -> tuple[str | None, str | None, FigureKind]:
    stripped = re.sub(r"\s+", " ", text).strip()
    match = CAPTION_RE.match(stripped)
    if not match:
        return None, None, "figure"
    kind_word = match.group(1).lower()
    number = match.group(2)
    if kind_word.startswith("table"):
        kind: FigureKind = "table_snapshot"
        label = f"Table {number}"
    elif kind_word.startswith("algorithm"):
        kind = "algorithm"
        label = f"Algorithm {number}"
    else:
        kind = "figure"
        label = f"Figure {number}"
    return label, stripped, kind
