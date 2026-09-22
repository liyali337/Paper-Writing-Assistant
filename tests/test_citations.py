from dl_agent.domain.models import Evidence
from dl_agent.understand.citations import resolve_citations, sanitize_answer_zh


def _ev(quote: str, *, title: str, page: int, section_id: str) -> Evidence:
    return Evidence(
        page=page,
        section_title=title,
        quote=quote,
        sourced=True,
        section_id=section_id,
        chunk_id=f"{section_id}:{page}",
    )


def test_sanitize_strips_circ_and_splits_sections() -> None:
    raw = (
        "主要结论可分为三部分。 **一、与 SOTA 对比** 相较 DeMo$^{\\circ}$ 与 IDEA ◦，"
        "mAP 提升 3.0%。 **三、尚未覆盖的部分** 以下内容在当前证据中原文未给出：Table 1 完整数值。"
    )
    out = sanitize_answer_zh(raw)
    assert "◦" not in out
    assert "$^{\\circ}$" not in out
    assert "尚未覆盖的部分" not in out
    assert "部分细节原文未给出" in out
    assert "\n\n" in out
    assert "**一、与 SOTA 对比**" in out


def test_resolve_citations_drops_front_matter() -> None:
    front = _ev("Chain-of-Thought Guided Multi-Modal Object Re-Identification", title="Front Matter", page=1, section_id="s-front")
    title = _ev("Chain-of-Thought Guided Multi-Modal Object Re-Identification", title="Chain-of-Thought Guided Multi-Modal Object Re-Identification", page=1, section_id="s-title")
    exp = _ev(
        "On RGBNT201, CoT-ReID reaches mAP 83.3 and Rank-1 86.1, outperforming IDEA.",
        title="4.3 Comparison with State-of-the-art Methods",
        page=8,
        section_id="s-exp",
    )
    answer = "在 RGBNT201 上 CoT 达到 mAP 83.3、R-1 86.1，优于 IDEA[1][2]。"
    picked = resolve_citations(answer, [front, title, exp], [1, 2])
    assert [item.section_id for item in picked] == ["s-exp"]


def test_resolve_citations_keeps_explicit_used_match() -> None:
    method = _ev("lambda = 0.5", title="Approach", page=6, section_id="s-method")
    exp = _ev("trained on CIFAR-10", title="Experiments", page=9, section_id="s-exp")
    picked = resolve_citations("数据是 CIFAR-10[1]。", [method, exp], [2])
    assert [item.section_id for item in picked] == ["s-exp"]
