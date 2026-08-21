from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse

from dl_agent.api.deps import get_knowledge
from dl_agent.api.errors import api_error, not_implemented
from dl_agent.knowledge.pdf_io import InvalidPdfError
from dl_agent.knowledge.service import KnowledgeService, PaperNotFoundError

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
    return {"paper_id": paper.paper_id, "status": paper.status}


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
