import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse

from dl_agent.api.deps import get_knowledge, get_translate, get_understand
from dl_agent.api.errors import api_error, not_implemented
from dl_agent.domain.models import AskRequest, PaperAnswer
from dl_agent.harness.complete import LlmNotConfiguredError, LlmRequestError
from dl_agent.knowledge.pdf_io import InvalidPdfError
from dl_agent.knowledge.service import (
    IndexNotReadyError,
    KnowledgeService,
    PaperNotFoundError,
    PaperNotReadyError,
)
from dl_agent.translate.service import TranslateService
from dl_agent.understand.service import PaperNotReadyError as UnderstandNotReady
from dl_agent.understand.service import UnderstandService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["papers"])

_M2 = "中文介绍与方法详解在 M2 实现"


@router.post("/papers")
async def upload_paper(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    svc: KnowledgeService = Depends(get_knowledge),
):
    filename = file.filename or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise api_error(400, "invalid_pdf", "ingest", "只接受 PDF 文件")
    data = await file.read()
    try:
        paper, should_parse = svc.start_ingest(data, filename)
    except InvalidPdfError as exc:
        raise api_error(400, "invalid_pdf", "ingest", str(exc)) from exc
    if should_parse:
        background.add_task(svc.finish_ingest, paper.paper_id)
    elif svc.needs_index(paper):
        background.add_task(svc.index_paper, paper.paper_id, force=True)
    return {"paper_id": paper.paper_id, "status": paper.status}


@router.get("/papers")
def list_papers(svc: KnowledgeService = Depends(get_knowledge)):
    return svc.list_papers()


@router.post("/papers/{paper_id}/reparse")
def reparse_paper(
    paper_id: str,
    background: BackgroundTasks,
    svc: KnowledgeService = Depends(get_knowledge),
):
    try:
        paper, should_parse = svc.start_reparse(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "reparse", "论文不存在") from None
    if should_parse:
        background.add_task(svc.finish_ingest, paper.paper_id)
    return {"paper_id": paper.paper_id, "status": paper.status}


@router.get("/papers/{paper_id}")
def get_paper(paper_id: str, svc: KnowledgeService = Depends(get_knowledge)):
    try:
        return svc.get_paper(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "get_paper", "论文不存在") from None


@router.get("/papers/{paper_id}/sections")
def list_sections(paper_id: str, svc: KnowledgeService = Depends(get_knowledge)):
    try:
        return svc.get_sections(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "sections", "论文不存在") from None


@router.get("/papers/{paper_id}/figures")
def list_figures(
    paper_id: str,
    section_id: str | None = Query(default=None),
    svc: KnowledgeService = Depends(get_knowledge),
):
    try:
        return svc.get_figures(paper_id, section_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "figures", "论文不存在") from None


@router.get("/papers/{paper_id}/source")
def get_source_pdf(paper_id: str, svc: KnowledgeService = Depends(get_knowledge)):
    try:
        path = svc.get_source_path(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "source", "原文不存在") from None
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline")


@router.get("/papers/{paper_id}/figures/{figure_id}")
def get_figure_bytes(
    paper_id: str,
    figure_id: str,
    svc: KnowledgeService = Depends(get_knowledge),
):
    try:
        path = svc.get_figure_path(paper_id, figure_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "figure_bytes", "图片不存在") from None
    return FileResponse(path, media_type="image/png")


@router.get("/papers/{paper_id}/intro")
def get_intro(paper_id: str):
    _ = paper_id
    raise not_implemented("intro", _M2)


@router.post("/papers/{paper_id}/intro")
def refresh_intro(paper_id: str, refresh: bool = Query(default=False)):
    _ = (paper_id, refresh)
    raise not_implemented("intro", _M2)


@router.get("/papers/{paper_id}/method")
def get_method(paper_id: str):
    _ = paper_id
    raise not_implemented("method", _M2)


@router.post("/papers/{paper_id}/method")
def refresh_method(paper_id: str, refresh: bool = Query(default=False)):
    _ = (paper_id, refresh)
    raise not_implemented("method", _M2)


@router.get("/papers/{paper_id}/translations")
def get_translations(
    paper_id: str,
    svc: KnowledgeService = Depends(get_knowledge),
    translate: TranslateService = Depends(get_translate),
):
    try:
        svc.get_paper(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "translations", "论文不存在") from None
    cached = translate.get_translation(paper_id)
    if cached is None:
        raise api_error(404, "not_found", "translations", "尚未开始翻译")
    # pending / failed / ready / partial 都直接返回正文，避免前端空转 202
    return cached


@router.post("/papers/{paper_id}/translations")
def start_translations(
    paper_id: str,
    background: BackgroundTasks,
    refresh: bool = Query(default=False),
    svc: KnowledgeService = Depends(get_knowledge),
    translate: TranslateService = Depends(get_translate),
):
    try:
        paper = svc.get_paper(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "translations", "论文不存在") from None
    if paper.status != "ready":
        raise api_error(409, "not_ready", "translations", "论文尚未解析完成")
    try:
        cached, should_run, run_id = translate.start_translation(paper_id, refresh=refresh)
    except LlmNotConfiguredError as exc:
        raise api_error(503, "llm_not_configured", "translations", str(exc)) from exc
    if should_run:
        background.add_task(translate.finish_translation, paper_id, run_id)
    return cached


@router.post("/papers/{paper_id}/translations/cancel")
def cancel_translations(
    paper_id: str,
    svc: KnowledgeService = Depends(get_knowledge),
    translate: TranslateService = Depends(get_translate),
):
    try:
        svc.get_paper(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "translations", "论文不存在") from None
    try:
        return translate.cancel_translation(paper_id)
    except ValueError:
        raise api_error(404, "not_found", "translations", "尚未开始翻译") from None


@router.post("/papers/{paper_id}/index")
def rebuild_index(
    paper_id: str,
    background: BackgroundTasks,
    svc: KnowledgeService = Depends(get_knowledge),
):
    try:
        paper = svc.get_paper(paper_id)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "index", "论文不存在") from None
    if paper.status != "ready":
        raise api_error(409, "not_ready", "index", "论文尚未解析完成")
    paper.index_status = "pending"
    paper.index_error = None
    svc.store.save_paper(paper)
    background.add_task(svc.index_paper, paper_id, force=True)
    return paper


@router.get("/papers/{paper_id}/search")
def search_paper(
    paper_id: str,
    q: str = Query(..., min_length=1),
    k: int | None = Query(default=None, ge=1, le=32),
    svc: KnowledgeService = Depends(get_knowledge),
):
    try:
        return svc.query(paper_id, q, k=k)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "search", "论文不存在") from None
    except PaperNotReadyError:
        raise api_error(409, "not_ready", "search", "论文尚未解析完成") from None
    except IndexNotReadyError as exc:
        if exc.status == "pending":
            raise api_error(202, "index_pending", "search", "索引尚未完成") from exc
        raise api_error(503, "index_failed", "search", exc.error or "索引失败") from exc


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


def should_use_agent(settings) -> bool:
    if settings.ask_mode != "agent":
        return False
    if langgraph_available():
        return True
    logger.warning("ASK_MODE=agent 但未安装 langgraph，回退 simple ask")
    return False


@router.post("/papers/{paper_id}/ask", response_model=PaperAnswer)
def ask_paper(
    paper_id: str,
    body: AskRequest,
    understand: UnderstandService = Depends(get_understand),
):
    question = (body.question or "").strip()
    if not question:
        raise api_error(400, "invalid_request", "ask", "问题不能为空")
    try:
        if should_use_agent(understand.settings):
            return understand.ask_agent(paper_id, question, body.history)
        return understand.ask(paper_id, question, body.history)
    except PaperNotFoundError:
        raise api_error(404, "not_found", "ask", "论文不存在") from None
    except (PaperNotReadyError, UnderstandNotReady):
        raise api_error(409, "not_ready", "ask", "论文尚未解析完成") from None
    except IndexNotReadyError as exc:
        if exc.status == "pending":
            raise api_error(409, "index_pending", "ask", "索引尚未完成") from exc
        raise api_error(503, "index_failed", "ask", exc.error or "索引失败") from exc
    except LlmNotConfiguredError as exc:
        raise api_error(503, "llm_not_configured", "ask", str(exc)) from exc
    except LlmRequestError as exc:
        raise api_error(503, "llm_request_failed", "ask", str(exc)) from exc
    except ValueError as exc:
        raise api_error(400, "invalid_request", "ask", str(exc)) from exc
