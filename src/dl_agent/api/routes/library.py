from fastapi import APIRouter, Depends, Query

from dl_agent.api.deps import get_knowledge, get_understand
from dl_agent.api.errors import api_error
from dl_agent.domain.models import AskRequest, PaperAnswer
from dl_agent.harness.complete import LlmNotConfiguredError, LlmRequestError
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.understand.service import UnderstandService

router = APIRouter(tags=["library"])


@router.get("/library/search")
def search_library(
    q: str = Query(..., min_length=1),
    k: int | None = Query(default=None, ge=1, le=16),
    svc: KnowledgeService = Depends(get_knowledge),
):
    return svc.query_corpus(q, k=k)


@router.post("/library/ask", response_model=PaperAnswer)
def ask_library(
    body: AskRequest,
    understand: UnderstandService = Depends(get_understand),
):
    question = (body.question or "").strip()
    if not question:
        raise api_error(400, "invalid_request", "library_ask", "问题不能为空")
    try:
        return understand.ask_library(question, body.history)
    except LlmNotConfiguredError as exc:
        raise api_error(503, "llm_not_configured", "library_ask", str(exc)) from exc
    except LlmRequestError as exc:
        raise api_error(503, "llm_request_failed", "library_ask", str(exc)) from exc
    except ValueError as exc:
        raise api_error(400, "invalid_request", "library_ask", str(exc)) from exc
