from dl_agent.knowledge.layout import LayoutItem, assemble, collapse_false_headings, tidy_math_prose
from tests.helpers import make_png


def test_pictures_bind_to_current_heading() -> None:
    png = make_png(120, 90)
    items = [
        LayoutItem(kind="heading", page=1, text="1. Introduction", level=1),
        LayoutItem(kind="text", page=1, text="Intro body " * 20),
        LayoutItem(
            kind="picture",
            page=1,
            image_bytes=png,
            width_px=120,
            height_px=90,
            caption="Figure 1: Overview.",
        ),
        LayoutItem(kind="heading", page=2, text="2. Approach", level=1),
        LayoutItem(kind="text", page=2, text="Method body " * 20),
        LayoutItem(
            kind="picture",
            page=2,
            image_bytes=png,
            width_px=160,
            height_px=100,
            caption="Figure 2: Pipeline.",
        ),
    ]
    result = assemble("p1", items)
    assert [section.title for section in result.sections] == ["1. Introduction", "2. Approach"]
    intro, method = result.sections
    assert intro.figure_ids == ["fig-001"]
    assert method.figure_ids == ["fig-002"]
    assert result.figures[0].section_id == intro.section_id
    assert result.figures[1].section_id == method.section_id
    assert result.figures[0].label == "Figure 1"
    assert result.figures[1].page == 2


def test_tiny_pictures_are_dropped() -> None:
    items = [
        LayoutItem(kind="heading", page=1, text="1. Method", level=1),
        LayoutItem(
            kind="picture",
            page=1,
            image_bytes=make_png(40, 40),
            width_px=40,
            height_px=40,
            caption="Figure 1: Logo.",
        ),
        LayoutItem(
            kind="picture",
            page=1,
            image_bytes=make_png(200, 120),
            width_px=200,
            height_px=120,
            caption="Figure 2: Architecture.",
        ),
    ]
    result = assemble("p1", items, min_figure_px=80)
    assert len(result.figures) == 1
    assert result.figures[0].label == "Figure 2"
    assert result.sections[0].figure_ids == ["fig-001"]


def test_references_cut_off_later_figures() -> None:
    png = make_png(100, 100)
    items = [
        LayoutItem(kind="heading", page=1, text="3. Method", level=1),
        LayoutItem(
            kind="picture",
            page=1,
            image_bytes=png,
            width_px=100,
            height_px=100,
            caption="Figure 1: Model.",
        ),
        LayoutItem(kind="heading", page=8, text="References", level=1),
        LayoutItem(kind="text", page=8, text="[1] Someone. Paper."),
        LayoutItem(kind="heading", page=9, text="A. Extra Appendix", level=1),
        LayoutItem(
            kind="picture",
            page=9,
            image_bytes=png,
            width_px=100,
            height_px=100,
            caption="Figure 8: Appendix plot.",
        ),
    ]
    result = assemble("p1", items)
    titles = [section.title for section in result.sections]
    assert "A. Extra Appendix" not in titles
    assert all(figure.page != 9 for figure in result.figures)
    assert result.figures[0].label == "Figure 1"


def test_acknowledgements_are_skipped() -> None:
    items = [
        LayoutItem(kind="heading", page=1, text="5. Conclusion", level=1),
        LayoutItem(kind="text", page=1, text="We conclude."),
        LayoutItem(kind="heading", page=1, text="Acknowledgements", level=1),
        LayoutItem(kind="text", page=1, text="We thank everyone."),
        LayoutItem(kind="heading", page=2, text="References", level=1),
        LayoutItem(kind="text", page=2, text="[1] Cite."),
    ]
    result = assemble("p1", items)
    assert [section.title for section in result.sections] == ["5. Conclusion", "References"]
    assert "thank" not in result.sections[0].text


def test_wrapped_line_is_not_a_section() -> None:
    items = [
        LayoutItem(kind="heading", page=1, text="1. Introduction", level=1),
        LayoutItem(kind="text", page=1, text="Fusion remains a core"),
        LayoutItem(kind="heading", page=1, text="challenge.", level=1),
        LayoutItem(kind="text", page=1, text="Recent advances have witnessed a shift."),
        LayoutItem(kind="heading", page=2, text="2. Method", level=1),
        LayoutItem(kind="text", page=2, text="We propose a module."),
    ]
    result = assemble("p1", items)
    assert [section.title for section in result.sections] == ["1. Introduction", "2. Method"]
    intro = result.sections[0]
    assert "core challenge." in intro.text
    assert "Recent advances" in intro.text


def test_duplicate_caption_is_not_appended_twice() -> None:
    caption = "Figure 1. Motivation and overall framework of the proposed method."
    items = [
        LayoutItem(kind="heading", page=1, text="1. Introduction", level=1),
        LayoutItem(kind="caption", page=1, text=caption),
        LayoutItem(
            kind="picture",
            page=1,
            image_bytes=make_png(120, 90),
            width_px=120,
            height_px=90,
            caption=caption,
        ),
    ]
    result = assemble("p1", items)
    assert "Figure 1." not in result.sections[0].text
    assert result.figures[0].caption is not None
    assert result.figures[0].caption.startswith("Figure 1")


def test_table_pictures_are_dropped_markdown_kept() -> None:
    items = [
        LayoutItem(kind="heading", page=1, text="4. Experiments", level=1),
        LayoutItem(
            kind="table",
            page=1,
            text="| Methods | mAP |\n|---|---|\n| Ours | 81.1 |",
        ),
        LayoutItem(kind="caption", page=1, text="Table 1. Results."),
        LayoutItem(
            kind="picture",
            page=1,
            image_bytes=make_png(200, 180),
            width_px=200,
            height_px=180,
            caption="Table 1. Results.",
        ),
        LayoutItem(
            kind="picture",
            page=1,
            image_bytes=make_png(120, 90),
            width_px=120,
            height_px=90,
            caption="Figure 1. Pipeline.",
        ),
    ]
    result = assemble("p1", items)
    assert [figure.label for figure in result.figures] == ["Figure 1"]
    assert result.sections[0].figure_ids == ["fig-001"]
    assert "| Methods | mAP |" in result.sections[0].text
    assert result.sections[0].text.count("Table 1.") == 1


def test_collapse_false_headings_repairs_saved_tree() -> None:
    from dl_agent.domain.models import Section

    sections = [
        Section(
            section_id="sec-001",
            paper_id="p",
            title="1. Introduction",
            kind="intro",
            level=1,
            page_start=1,
            page_end=1,
            text="Fusion remains a core",
        ),
        Section(
            section_id="sec-002",
            paper_id="p",
            title="challenge.",
            kind="other",
            level=1,
            page_start=1,
            page_end=2,
            text="Recent advances have witnessed a shift.",
            figure_ids=["fig-001"],
        ),
    ]
    out = collapse_false_headings(sections)
    assert [item.title for item in out] == ["1. Introduction"]
    assert "core challenge." in out[0].text
    assert "Recent advances" in out[0].text
    assert out[0].figure_ids == ["fig-001"]
    assert out[0].page_end == 2


def test_duplicate_figure_label_keeps_docling_crop() -> None:
    png = make_png(120, 90)
    items = [
        LayoutItem(kind="heading", page=8, text="4.5. Visualization", level=2),
        LayoutItem(
            kind="picture",
            page=8,
            image_bytes=png,
            width_px=457,
            height_px=413,
            caption="Figure 4. Qualitative comparison.",
            label="Figure 4",
            source="docling_picture",
        ),
        LayoutItem(
            kind="picture",
            page=8,
            image_bytes=make_png(200, 200),
            width_px=522,
            height_px=557,
            caption="Figure 4. Qualitative comparison.",
            label="Figure 4",
            source="page_clip",
        ),
        LayoutItem(
            kind="picture",
            page=8,
            image_bytes=png,
            width_px=464,
            height_px=317,
            caption="Figure 5. t-SNE visualization.",
            label="Figure 5",
            source="docling_picture",
        ),
    ]
    result = assemble("p1", items)
    assert [figure.label for figure in result.figures] == ["Figure 4", "Figure 5"]
    assert result.figures[0].source == "docling_picture"
    assert result.figures[0].width_px == 457
    assert result.sections[0].figure_ids == ["fig-001", "fig-003"]


def test_formula_is_inlined_in_section_text() -> None:
    png = make_png(240, 48)
    items = [
        LayoutItem(kind="heading", page=3, text="3.2. Shared Visual Encoder", level=2),
        LayoutItem(kind="text", page=3, text="The encoder extracts features from three modalities:"),
        LayoutItem(
            kind="formula",
            page=3,
            image_bytes=png,
            width_px=240,
            height_px=48,
            source="page_clip",
        ),
        LayoutItem(kind="text", page=3, text="Here, x_m denotes the CLS token."),
        LayoutItem(kind="formula", page=3, text=r"\mathcal{L} = \ell_{ce} + \ell_{tri}"),
    ]
    result = assemble("p1", items)
    section = result.sections[0]
    assert [figure.kind for figure in result.figures] == ["formula"]
    assert "<!--fig:fig-001-->" in section.text
    assert "Here, x_m denotes the CLS token." in section.text
    assert r"\mathcal{L} = \ell_{ce} + \ell_{tri}" in section.text
    assert section.figure_ids == ["fig-001"]


def test_empty_formula_without_crop_is_dropped() -> None:
    items = [
        LayoutItem(kind="heading", page=1, text="1. Method", level=1),
        LayoutItem(kind="text", page=1, text="We define the loss as follows."),
        LayoutItem(kind="formula", page=1),
    ]
    result = assemble("p1", items)
    assert result.figures == []
    assert "<!--fig:" not in result.sections[0].text


def test_tiny_formula_fragment_is_dropped() -> None:
    items = [
        LayoutItem(kind="heading", page=5, text="3.3. Fusion", level=2),
        LayoutItem(kind="text", page=5, text="Then we inject:"),
        LayoutItem(
            kind="formula",
            page=5,
            image_bytes=make_png(58, 42),
            width_px=58,
            height_px=42,
            source="page_clip",
        ),
        LayoutItem(kind="text", page=5, text="into the token space."),
    ]
    result = assemble("p1", items)
    assert result.figures == []
    assert "<!--fig:" not in result.sections[0].text


def test_tidy_math_prose_repairs_broken_inline_symbols() -> None:
    raw = (
        "Here, x c m denotes the CLS token, while x p m = { x 1 m , x 2 m } . "
        "The hat ˆ x m and \\protect \\big |x_m^i - x_f^i\\big | , "
        "with α ∈ [0 , 1] and m ∈ { R,N,T } ."
    )
    out = tidy_math_prose(raw)
    assert "x_m^c" in out
    assert "x_m^p" in out
    assert "x_m^1" in out
    assert "x̂_m" in out or "x\u0302_m" in out
    assert "\\protect" not in out
    assert "|x_m^i - x_f^i|" in out
    assert "[0, 1]" in out
    assert "{R,N,T}" in out


def test_corresponding_author_footnote_is_dropped_and_sentence_stitched() -> None:
    from dl_agent.knowledge.classify import is_page_chrome, strip_page_chrome

    assert is_page_chrome("* Corresponding author")
    assert is_page_chrome("This CVPR Findings paper is the Open Access version, provided by the Computer Vision")
    items = [
        LayoutItem(kind="heading", page=1, text="1. Introduction", level=1),
        LayoutItem(kind="text", page=1, text="Fusion of these modalities remains a core"),
        LayoutItem(kind="text", page=1, text="* Corresponding author"),
        LayoutItem(kind="text", page=1, text="challenge."),
        LayoutItem(
            kind="text",
            page=1,
            text="Recent advances have witnessed a paradigm shift from CNNs.",
        ),
    ]
    result = assemble("p1", items)
    text = result.sections[0].text
    assert "Corresponding author" not in text
    assert "core challenge." in text
    assert "Recent advances" in text
    stitched = strip_page_chrome(
        "remains a core\n\n* Corresponding author challenge.\n\nRecent advances began."
    )
    assert "Corresponding author" not in stitched
    assert "core challenge." in stitched
