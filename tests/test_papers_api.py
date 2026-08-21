from fastapi.testclient import TestClient

from dl_agent.api.deps import get_knowledge
from dl_agent.api.main import app
from dl_agent.config import Settings
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.layout import LayoutItem
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from tests.helpers import make_png


def _minimal_text_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Placeholder page for API ingest tests.")
    data = doc.tobytes()
    doc.close()
    return data


def test_upload_and_fetch_sections_figures(tmp_path) -> None:
    png = make_png(140, 100)
    body = "Readable paper text for the parser gate. " * 30

    def parse(_path: str) -> ParseResult:
        return ParseResult(
            parser="pymupdf",
            page_count=2,
            items=[
                LayoutItem(kind="heading", page=1, text="Abstract", level=1),
                LayoutItem(kind="text", page=1, text=body),
                LayoutItem(kind="heading", page=1, text="1. Introduction", level=1),
                LayoutItem(kind="text", page=1, text=body),
                LayoutItem(kind="heading", page=2, text="2. Method", level=1),
                LayoutItem(kind="text", page=2, text=body),
                LayoutItem(
                    kind="picture",
                    page=2,
                    image_bytes=png,
                    width_px=140,
                    height_px=100,
                    caption="Figure 1: Our pipeline.",
                ),
            ],
        )

    svc = KnowledgeService(FilePaperStore(tmp_path), Settings(data_dir=tmp_path), parse_fn=parse)
    app.dependency_overrides[get_knowledge] = lambda: svc
    client = TestClient(app)
    try:
        uploaded = client.post(
            "/papers",
            files={"file": ("toy.pdf", _minimal_text_pdf(), "application/pdf")},
        )
        assert uploaded.status_code == 200
        paper_id = uploaded.json()["paper_id"]

        paper = client.get(f"/papers/{paper_id}")
        assert paper.status_code == 200
        body_json = paper.json()
        if body_json["status"] in {"queued", "parsing"}:
            svc.finish_ingest(paper_id)
            body_json = client.get(f"/papers/{paper_id}").json()
        assert body_json["status"] == "ready"
        assert body_json["figure_count"] == 1
        assert body_json["parser"] == "pymupdf"

        sections = client.get(f"/papers/{paper_id}/sections").json()
        titles = [item["title"] for item in sections]
        assert "Abstract" in titles
        assert any("Method" in title for title in titles)
        method = next(item for item in sections if "Method" in item["title"])
        assert method["figure_ids"]

        figures = client.get(f"/papers/{paper_id}/figures").json()
        assert len(figures) == 1
        filtered = client.get(
            f"/papers/{paper_id}/figures",
            params={"section_id": method["section_id"]},
        ).json()
        assert len(filtered) == 1

        png_response = client.get(f"/papers/{paper_id}/figures/{figures[0]['figure_id']}")
        assert png_response.status_code == 200
        assert png_response.headers["content-type"].startswith("image/png")
        assert png_response.content[:8] == b"\x89PNG\r\n\x1a\n"

        source = client.get(f"/papers/{paper_id}/source")
        assert source.status_code == 200
        assert source.headers["content-type"].startswith("application/pdf")
        assert source.content.startswith(b"%PDF")

        reparsing = client.post(f"/papers/{paper_id}/reparse")
        assert reparsing.status_code == 200
        payload = reparsing.json()
        assert payload["paper_id"] == paper_id
        if payload["status"] in {"queued", "parsing"}:
            svc.finish_ingest(paper_id)
        assert client.get(f"/papers/{paper_id}").json()["status"] == "ready"
    finally:
        app.dependency_overrides.clear()
