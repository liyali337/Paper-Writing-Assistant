from dl_agent.knowledge.formula_latex import (
    is_display_formula,
    looks_like_latex,
    normalize_latex,
    spans_to_latex,
)
from dl_agent.knowledge.layout import LayoutItem, assemble, tidy_math_prose
from tests.helpers import make_png


def test_spans_to_latex_builds_sub_and_super() -> None:
    spans = [
        {"text": "x", "size": 11, "bbox": (10, 20, 16, 32)},
        {"text": "c", "size": 7.5, "bbox": (16, 16, 21, 24)},
        {"text": "m", "size": 7.5, "bbox": (16, 28, 22, 36)},
    ]
    assert spans_to_latex(spans) == "x^{c}_{m}"


def test_normalize_strips_dollar_wrappers() -> None:
    assert normalize_latex("$$ F_m = x $$") == "F_m = x"
    assert looks_like_latex(r"F_{m}=\{x_m^c\}")


def test_display_latex_becomes_block_math() -> None:
    items = [
        LayoutItem(kind="heading", page=3, text="3.1. Feature Extraction", level=2),
        LayoutItem(kind="text", page=3, text="The encoder extracts features:"),
        LayoutItem(
            kind="formula",
            page=3,
            text=r"F_m=\{x_m^c,x_m^p\}",
            bbox=(60, 200, 280, 218),
        ),
        LayoutItem(kind="text", page=3, text="Here the CLS token is"),
        LayoutItem(
            kind="formula",
            page=3,
            text=r"x_m^c",
            bbox=(90, 240, 120, 252),
        ),
        LayoutItem(kind="text", page=3, text="in the sentence."),
    ]
    result = assemble("p1", items)
    text = result.sections[0].text
    assert "$$\nF_m=\\{x_m^c,x_m^p\\}\n$$" in text
    assert "$x_m^c$" in text
    assert result.figures == []


def test_tidy_does_not_break_latex_dollars() -> None:
    raw = "Here, $x^{c}_{m}$ denotes the CLS token, while x p m stays."
    out = tidy_math_prose(raw)
    assert "$x^{c}_{m}$" in out
    assert "x_m^p" in out


def test_looks_like_latex_rejects_replacement_chars() -> None:
    assert not looks_like_latex("\ufffd_m^c")
    assert looks_like_latex(r"F_m=\{x_m^c\}")


def test_tidy_strips_garbled_math_and_unwraps_italic_words() -> None:
    raw = (
        "respectively: $$\ufffd$_{m}$ $x_{$x_{m}$p}$\n\n"
        "$$\nF_m = \\{x_m^c, x_m^p\\}, \\quad m \\in \\{R, N, T\\}. (1)\n$$\n\n"
        "Here, x c m denotes the CLS token. adaptively $f$use the global "
        "structure. Mul$t$imodal features. Proj$ec^{t}$ion."
    )
    out = tidy_math_prose(raw)
    assert "\ufffd" not in out
    assert "$f$use" not in out
    assert "fuse" in out
    assert "Multimodal" in out
    assert "Projection" in out
    assert "CLS token" in out
    assert "$$" in out
    assert "F_m" in out
    assert "x_m^c" in out


def test_tidy_keeps_prose_trapped_between_broken_dollars() -> None:
    trapped = "Here the CLS token is important. $\ufffd_{m}$ more prose."
    out = tidy_math_prose(f"lead-in.\n$$\n{trapped}\n$$")
    assert "CLS token" in out
    assert "\ufffd" not in out


def test_splice_does_not_break_english_words() -> None:
    from dl_agent.knowledge.formula_latex import _splice_latex

    assert _splice_latex("adaptively fuse the global", ["f"], "f") is None
    assert "$x_m^c$" in (_splice_latex("token x denotes", ["x"], "x_m^c") or "")


def test_formula_image_still_used_without_latex() -> None:
    items = [
        LayoutItem(kind="heading", page=1, text="1. Method", level=1),
        LayoutItem(
            kind="formula",
            page=1,
            image_bytes=make_png(240, 48),
            width_px=240,
            height_px=48,
            source="page_clip",
        ),
    ]
    result = assemble("p1", items)
    assert result.figures[0].kind == "formula"
    assert "<!--fig:fig-001-->" in result.sections[0].text


def test_display_vs_inline_by_bbox() -> None:
    wide = LayoutItem(kind="formula", page=1, bbox=(80, 100, 260, 118))
    tiny = LayoutItem(kind="formula", page=1, bbox=(90, 100, 118, 112))
    assert is_display_formula(wide)
    assert not is_display_formula(tiny)
