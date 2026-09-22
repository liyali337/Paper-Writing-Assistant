"""问答引用装配：去掉扉页误引，并清理表格角标。"""

from __future__ import annotations

import re

from dl_agent.domain.models import Evidence

_FRONT_TITLE = re.compile(r"^front\s*matter$", re.I)
_CIRC_MATH = re.compile(
    r"\$\s*(?:\^\{?(?:\\circ|circ|\\ast|\*|\\dagger|\\ddagger|\\bullet)\}?|"
    r"\\circ|\\bullet|\\dagger|\\ddagger|\\ast)\s*\$",
    re.I,
)
_TRAIL_MARK = re.compile(r"(?<=[A-Za-z0-9])[ \t]*[◦○●†‡※]")
_GAP_MARK = "尚未覆盖的部分"
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_NUMBER = re.compile(r"\d+\.\d+|\d{2,}")
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "are",
    "was",
    "were",
    "have",
    "has",
    "been",
    "not",
    "but",
    "its",
    "our",
    "their",
    "using",
    "used",
    "also",
    "than",
    "into",
    "over",
    "section",
}
_HEADING_BODY = re.compile(r"(\*\*[一二三四五六七八九十0-9]+[、.．][^*]*\*\*)[ \t]+(?=\S)")
_SECTION_BREAK = re.compile(r"(?<=[。！？])\s*(?=\*\*[一二三四五六七八九十0-9])")


def sanitize_answer_zh(text: str) -> str:
    blob = (text or "").strip()
    if not blob:
        return blob
    blob = _CIRC_MATH.sub("", blob)
    blob = _TRAIL_MARK.sub("", blob)
    blob = re.sub(r"[ \t]{2,}", " ", blob)
    blob = _SECTION_BREAK.sub("\n\n", blob)
    blob = _HEADING_BODY.sub(r"\1\n\n", blob)
    blob = _trim_gap_list(blob)
    return blob.strip()


def resolve_citations(
    answer: str,
    evidence: list[Evidence],
    used_indices: list[int],
    *,
    limit: int = 6,
) -> list[Evidence]:
    if not evidence:
        return []
    ranked = sorted(evidence, key=lambda item: _cite_score(item, answer), reverse=True)
    strong = [
        item
        for item in ranked
        if _cite_score(item, answer) > 0 and not is_weak_citation(item, answer=answer)
    ]
    picked: list[Evidence] = []
    seen: set[str] = set()

    def add(item: Evidence) -> None:
        key = item.section_id or item.chunk_id or f"{item.page}:{item.quote[:48]}"
        if key in seen:
            return
        seen.add(key)
        picked.append(item)

    for index in used_indices:
        if index < 1 or index > len(evidence):
            continue
        item = evidence[index - 1]
        if strong and (_cite_score(item, answer) == 0 or is_weak_citation(item, answer=answer)):
            continue
        add(item)

    if not picked:
        fallback = strong or [item for item in ranked if not is_weak_citation(item, answer=answer)] or ranked
        for item in fallback:
            add(item)
            if len(picked) >= min(4, limit):
                break
    return picked[:limit]


def rank_evidence_for_prompt(evidence: list[Evidence], context: str) -> list[Evidence]:
    return sorted(
        evidence,
        key=lambda item: (
            is_weak_citation(item, answer=context),
            -_cite_score(item, context),
            item.page,
        ),
    )


def is_weak_citation(item: Evidence, *, answer: str) -> bool:
    title = (item.section_title or "").strip()
    if _FRONT_TITLE.match(title):
        return True
    if _cite_score(item, answer) > 0:
        return False
    if item.page <= 1:
        return True
    return False


def _cite_score(item: Evidence, answer: str) -> int:
    hay = f"{item.quote} {item.section_title or ''}"
    nums = _numeric_tokens(answer) & _numeric_tokens(hay)
    latin = _latin_tokens(answer) & _latin_tokens(hay)
    return 3 * len(nums) + len(latin)


def _numeric_tokens(text: str) -> set[str]:
    return set(_NUMBER.findall(text or ""))


def _latin_tokens(text: str) -> set[str]:
    return {token.lower() for token in _LATIN.findall(text or "")} - _STOP


def _trim_gap_list(text: str) -> str:
    mark = text.find(_GAP_MARK)
    if mark < 0:
        return text
    start = mark
    prefix = text[:mark]
    heading = re.search(r"(?:\n{0,2}\s*)?(?:\*\*)?(?:三、|3[、.．)]?\s*)$", prefix)
    if heading:
        start = heading.start()
    head = text[:start].rstrip(" \t\n*")
    rest = text[mark + len(_GAP_MARK) :].lstrip(" \t*：:").strip()
    rest = re.sub(r"^以下内容在当前证据中原文未给出[：:]\s*", "", rest)
    compact = re.sub(r"\s+", " ", rest)
    if not compact:
        return head
    if len(compact) > 80:
        compact = compact[:77].rstrip("；;，,、 ") + "…"
    note = f"部分细节原文未给出：{compact}"
    if not head:
        return note
    return f"{head}\n\n{note}"
