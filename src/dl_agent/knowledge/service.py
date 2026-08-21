"""知识层：上传校验、按标题切节、裁图、按 sha256 去重。"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from dl_agent.config import Settings, get_settings
from dl_agent.domain.models import Figure, Paper, Section
from dl_agent.knowledge.adapters.pdf import parse_pdf as default_parse_pdf
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.layout import assemble, collapse_false_headings, apply_figure_dedupe, polish_formulas
from dl_agent.knowledge.pdf_io import InvalidPdfError, ParseError, sha256_bytes, validate_pdf_bytes
from dl_agent.knowledge.store import FilePaperStore

logger = logging.getLogger(__name__)

ParseFn = Callable[[str], ParseResult]


class PaperNotFoundError(KeyError):
    pass


class KnowledgeService:
    def __init__(
        self,
        store: FilePaperStore,
        settings: Settings | None = None,
        parse_fn: ParseFn | None = None,
    ):
        self.store = store
        self.settings = settings or get_settings()
        self.parse_fn = parse_fn or self._parse
        self._lock = threading.Lock()

    def _parse(self, pdf_path: str) -> ParseResult:
        return default_parse_pdf(
            pdf_path,
            timeout_s=self.settings.docling_timeout_s,
            images_scale=self.settings.images_scale,
            min_chars=self.settings.min_chars_total,
            formula_enrichment=self.settings.docling_formula_enrichment,
        )

    def start_ingest(self, pdf_bytes: bytes, filename: str) -> tuple[Paper, bool]:
        validate_pdf_bytes(pdf_bytes, max_bytes=self.settings.max_pdf_bytes)
        digest = sha256_bytes(pdf_bytes)
        with self._lock:
            existing_id = self.store.get_paper_id_by_sha(digest)
            if existing_id:
                existing = self.store.get_paper(existing_id)
                if existing and existing.status in {"ready", "needs_ocr", "ingest_failed"}:
                    logger.info(
                        "ingest dedup paper_id=%s status=%s sha256=%s",
                        existing.paper_id,
                        existing.status,
                        digest[:12],
                    )
                    return existing, False
                if existing:
                    # 进程重启后 parsing/queued 没有后台任务，需要重新跑
                    if existing.status == "parsing":
                        existing.status = "queued"
                        self.store.save_paper(existing)
                    logger.info(
                        "ingest resume paper_id=%s status=%s sha256=%s",
                        existing.paper_id,
                        existing.status,
                        digest[:12],
                    )
                    return existing, True

            paper_id = existing_id or str(uuid.uuid4())
            paper = Paper(
                paper_id=paper_id,
                sha256=digest,
                filename=filename,
                status="queued",
                intro_status="pending",
                method_status="pending",
            )
            self.store.save_paper(paper)
            self.store.save_source_pdf(paper_id, pdf_bytes)
            self.store.index_sha(digest, paper_id)
            logger.info(
                "ingest queued paper_id=%s filename=%s bytes=%s",
                paper_id,
                filename,
                len(pdf_bytes),
            )
            return paper, True

    def finish_ingest(self, paper_id: str) -> Paper:
        with self._lock:
            paper = self.store.get_paper(paper_id)
            if paper is None:
                raise PaperNotFoundError(paper_id)
            if paper.status != "queued":
                logger.info("ingest skip paper_id=%s status=%s", paper_id, paper.status)
                return paper
            paper.status = "parsing"
            self.store.save_paper(paper)
        started = time.perf_counter()
        pdf_path = self.store.source_pdf_path(paper_id)
        try:
            parsed = self.parse_fn(str(pdf_path))
        except ParseError as exc:
            logger.warning("ingest_failed paper_id=%s error=%s", paper_id, exc)
            paper.status = "ingest_failed"
            paper.intro_status = "skipped"
            paper.method_status = "skipped"
            self.store.save_paper(paper)
            self._audit(paper_id, "parse", started, extra={"status": "ingest_failed", "error": str(exc)})
            return paper
        except Exception as exc:
            logger.exception("ingest_failed paper_id=%s", paper_id)
            paper.status = "ingest_failed"
            paper.intro_status = "skipped"
            paper.method_status = "skipped"
            self.store.save_paper(paper)
            self._audit(paper_id, "parse", started, extra={"status": "ingest_failed", "error": str(exc)})
            return paper

        assembled = assemble(
            paper_id,
            parsed.items,
            min_figure_px=self.settings.min_figure_px,
            paper_title=parsed.title,
        )
        self.store.delete_figures(paper_id)
        self.store.save_sections(paper_id, assembled.sections)
        self.store.save_figures(paper_id, assembled.figures, assembled.figure_pngs)

        paper.parser = parsed.parser
        paper.page_count = parsed.page_count
        paper.figure_count = len(assembled.figures)
        paper.title = assembled.title or parsed.title
        paper.authors = parsed.authors
        paper.abstract = assembled.abstract
        paper.language = parsed.language
        paper.intro_status = "skipped"
        paper.method_status = "skipped"

        chars = parsed.char_count
        per_page = chars / parsed.page_count if parsed.page_count else 0
        if (
            parsed.page_count <= 0
            or chars < self.settings.min_chars_total
            or per_page < self.settings.min_chars_per_page
        ):
            paper.status = "needs_ocr"
            logger.info(
                "needs_ocr paper_id=%s parser=%s pages=%s chars=%s",
                paper_id,
                parsed.parser,
                parsed.page_count,
                chars,
            )
        else:
            paper.status = "ready"
            logger.info(
                "parse done paper_id=%s parser=%s pages=%s sections=%s figures=%s chars=%s",
                paper_id,
                parsed.parser,
                parsed.page_count,
                len(assembled.sections),
                len(assembled.figures),
                chars,
            )

        self.store.save_paper(paper)
        self._audit(
            paper_id,
            "parse",
            started,
            extra={
                "status": paper.status,
                "parser": paper.parser,
                "section_count": len(assembled.sections),
                "figure_count": paper.figure_count,
                "char_count": chars,
            },
        )
        return paper

    def start_reparse(self, paper_id: str) -> tuple[Paper, bool]:
        """用已存 PDF 强制再跑解析，忽略 sha256 去重。"""
        with self._lock:
            paper = self.store.get_paper(paper_id)
            if paper is None:
                raise PaperNotFoundError(paper_id)
            if not self.store.source_pdf_path(paper_id).exists():
                raise PaperNotFoundError(paper_id)
            if paper.status in {"queued", "parsing"}:
                return paper, False
            paper.status = "queued"
            self.store.save_paper(paper)
            logger.info("reparse queued paper_id=%s", paper_id)
            return paper, True

    def ingest(self, pdf_bytes: bytes, filename: str) -> Paper:
        paper, should_parse = self.start_ingest(pdf_bytes, filename)
        if should_parse:
            return self.finish_ingest(paper.paper_id)
        return paper

    def get_paper(self, paper_id: str) -> Paper:
        paper = self.store.get_paper(paper_id)
        if paper is None:
            raise PaperNotFoundError(paper_id)
        return paper

    def get_sections(self, paper_id: str) -> list[Section]:
        self.get_paper(paper_id)
        sections = collapse_false_headings(self.store.get_sections(paper_id))
        figures = self.store.get_figures(paper_id)
        sections, figures, _ = apply_figure_dedupe(sections, figures)
        sections, _, _ = polish_formulas(sections, figures)
        return sections

    def get_figures(self, paper_id: str, section_id: str | None = None) -> list[Figure]:
        self.get_paper(paper_id)
        sections = collapse_false_headings(self.store.get_sections(paper_id))
        figures = self.store.get_figures(paper_id)
        sections, figures, _ = apply_figure_dedupe(sections, figures)
        _, figures, _ = polish_formulas(sections, figures)
        if section_id:
            figures = [item for item in figures if item.section_id == section_id]
        return figures

    def get_source_path(self, paper_id: str) -> Path:
        self.get_paper(paper_id)
        path = self.store.source_pdf_path(paper_id)
        if not path.exists():
            raise PaperNotFoundError(paper_id)
        return path

    def get_figure_path(self, paper_id: str, figure_id: str) -> Path:
        self.get_paper(paper_id)
        figures = self.store.get_figures(paper_id)
        match = next((item for item in figures if item.figure_id == figure_id), None)
        if match is None or not match.storage_key:
            raise PaperNotFoundError(f"{paper_id}/{figure_id}")
        path = self.store.figure_png_path(paper_id, figure_id)
        if path is None:
            raise PaperNotFoundError(f"{paper_id}/{figure_id}")
        return path

    def _audit(self, paper_id: str, stage: str, started: float, extra: dict) -> None:
        record = {
            "paper_id": paper_id,
            "stage": stage,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            **extra,
        }
        self.store.append_audit(record)


_service: KnowledgeService | None = None


def get_service() -> KnowledgeService:
    global _service
    if _service is None:
        settings = get_settings()
        _service = KnowledgeService(FilePaperStore(settings.data_dir), settings)
    return _service


def reset_service(service: KnowledgeService | None = None) -> None:
    global _service
    _service = service


def ingest(pdf_bytes: bytes, filename: str) -> Paper:
    return get_service().ingest(pdf_bytes, filename)


def get_paper(paper_id: str) -> Paper:
    return get_service().get_paper(paper_id)


def get_sections(paper_id: str) -> list[Section]:
    return get_service().get_sections(paper_id)


def get_figures(paper_id: str, section_id: str | None = None) -> list[Figure]:
    return get_service().get_figures(paper_id, section_id)
