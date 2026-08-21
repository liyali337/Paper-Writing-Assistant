from dl_agent.knowledge.adapters.pdf.pymupdf_adapter import parse_pdf
from tests.helpers import make_png


def test_pymupdf_splits_headings_and_extracts_image(tmp_path) -> None:
    import pymupdf

    pdf_path = tmp_path / "toy.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Introduction", fontsize=16)
    page.insert_text((72, 100), ("This paper studies a toy method. " * 12), fontsize=11)
    page.insert_text((72, 220), "2. Approach", fontsize=16)
    page.insert_text((72, 248), ("We propose a pipeline with two stages. " * 10), fontsize=11)
    page.insert_image(pymupdf.Rect(72, 360, 320, 500), stream=make_png(200, 120))
    page.insert_text((72, 510), "Figure 1: Architecture of the proposed model.", fontsize=10)
    page.insert_text((72, 560), "3. Experiments", fontsize=16)
    page.insert_text((72, 588), ("Results are omitted in this fixture. " * 8), fontsize=11)
    doc.save(pdf_path)
    doc.close()

    parsed = parse_pdf(str(pdf_path))
    headings = [item.text for item in parsed.items if item.kind == "heading"]
    pictures = [item for item in parsed.items if item.kind == "picture"]
    assert any("Introduction" in text for text in headings)
    assert any("Approach" in text for text in headings)
    assert parsed.page_count == 1
    assert parsed.char_count > 200
    assert pictures
    assert pictures[0].width_px >= 80
    assert pictures[0].image_bytes
    labeled = [item.label for item in pictures if item.label]
    assert len(labeled) == len(set(labeled))


def test_pymupdf_does_not_treat_linebroken_experiments_as_heading(tmp_path) -> None:
    import pymupdf

    pdf_path = tmp_path / "break.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Introduction", fontsize=16)
    page.insert_text((72, 110), "Extensive", fontsize=11)
    page.insert_text((72, 126), "experiments", fontsize=11)
    page.insert_text((72, 142), "on three public datasets.", fontsize=11)
    doc.save(pdf_path)
    doc.close()

    parsed = parse_pdf(str(pdf_path))
    headings = [item.text for item in parsed.items if item.kind == "heading"]
    assert any("Introduction" in text for text in headings)
    assert "experiments" not in headings
    assert "Experiments" not in headings


def test_caption_binds_to_unlabeled_docling_picture_instead_of_clipping() -> None:
    from dl_agent.knowledge.adapters.pdf.pymupdf_adapter import _bind_caption_to_nearby_picture
    from dl_agent.knowledge.layout import LayoutItem

    picture = LayoutItem(
        kind="picture",
        page=8,
        image_bytes=b"png",
        width_px=457,
        height_px=413,
        source="docling_picture",
    )
    caption = LayoutItem(
        kind="caption",
        page=8,
        text="Figure 4. Qualitative comparison between baseline and FusionBridge.",
    )
    items = [picture, caption]
    bound = _bind_caption_to_nearby_picture(items, 1, "Figure 4", "Qualitative comparison.")
    assert bound
    assert items[0].label == "Figure 4"


def test_clip_formula_items_renders_png(tmp_path) -> None:
    import pymupdf

    from dl_agent.knowledge.adapters.pdf.pymupdf_adapter import clip_formula_items
    from dl_agent.knowledge.layout import LayoutItem

    pdf_path = tmp_path / "eq.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 200), "E = mc^2", fontsize=18)
    doc.save(pdf_path)
    doc.close()

    items = [LayoutItem(kind="formula", page=1, bbox=(60, 180, 180, 220))]
    out = clip_formula_items(str(pdf_path), items)
    assert out[0].image_bytes
    assert out[0].width_px >= 24
    assert out[0].height_px >= 16
    assert out[0].source == "page_clip"
