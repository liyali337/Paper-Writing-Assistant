from fastapi import APIRouter, Depends

from dl_agent.api.deps import get_understand
from dl_agent.api.errors import api_error
from dl_agent.domain.models import AskRequest, PaperAnswer
from dl_agent.harness.complete import LlmNotConfiguredError, LlmRequestError
from dl_agent.understand.service import UnderstandService

router = APIRouter(tags=["arxiv"])


@router.post("/arxiv/ask", response_model=PaperAnswer)
def ask_arxiv(
    body: AskRequest,
    understand: UnderstandService = Depends(get_understand),
):
    question = (body.question or "").strip()
    if not question:
        raise api_error(400, "invalid_request", "arxiv_ask", "问题不能为空")
    try:
        return understand.ask_arxiv(question, body.history)
    except LlmNotConfiguredError as exc:
        raise api_error(503, "llm_not_configured", "arxiv_ask", str(exc)) from exc
    except LlmRequestError as exc:
        raise api_error(503, "llm_request_failed", "arxiv_ask", str(exc)) from exc
    except ValueError as exc:
        raise api_error(400, "invalid_request", "arxiv_ask", str(exc)) from exc
