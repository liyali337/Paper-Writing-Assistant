from fastapi.testclient import TestClient

from dl_agent.api.deps import get_knowledge
from dl_agent.api.main import app
from dl_agent.config import Settings
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.layout import LayoutItem
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from tests.helpers import make_png

PAPER_ID = "00000000-0000-0000-0000-000000000000"


def _assert_not_implemented(response, stage: str) -> None:
    assert response.status_code == 501
    detail = response.json()["detail"]
    assert detail["code"] == "not_implemented"
    assert detail["stage"] == stage


def test_unknown_paper_reparse_is_404() -> None:
    client = TestClient(app)
    response = client.post(f"/papers/{PAPER_ID}/reparse")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "not_found"


def test_intro_returns_501() -> None:
    _assert_not_implemented(TestClient(app).get(f"/papers/{PAPER_ID}/intro"), "intro")


def test_method_returns_501() -> None:
    _assert_not_implemented(TestClient(app).get(f"/papers/{PAPER_ID}/method"), "method")


def test_upload_invalid_bytes_400() -> None:
    client = TestClient(app)
    response = client.post(
        "/papers",
        files={"file": ("sample.pdf", b"not-a-pdf", "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_pdf"
