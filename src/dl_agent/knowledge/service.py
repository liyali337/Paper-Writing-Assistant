"""知识层：上传校验、按标题切节、裁图、按 sha256 去重。"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from dl_agent.config import Settings, get_settings
from dl_agent.domain.models import ChildChunk, Evidence, Figure, LibraryHit, Paper, Section
from dl_agent.knowledge.adapters.pdf import parse_pdf as default_parse_pdf
from dl_agent.knowledge.adapters.pdf.types import ParseResult
from dl_agent.knowledge.adapters.vector import try_qdrant_index
from dl_agent.knowledge.adapters.vector.embed import EmbedFn
from dl_agent.knowledge.adapters.vector.hybrid_embed import HybridEncoder, SparseFn, try_hybrid_encoder
from dl_agent.knowledge.chunk import build_chunks, cap_chunks, chunks_source_hash
from dl_agent.knowledge.formula_vision import repair_formula_items
from dl_agent.knowledge.layout import assemble, collapse_false_headings, apply_figure_dedupe, polish_formulas, repair_front_matter
from dl_agent.knowledge.pdf_io import InvalidPdfError, ParseError, sha256_bytes, validate_pdf_bytes
from dl_agent.knowledge.retrieve import retrieve_corpus_from_chunks, retrieve_from_chunks, to_evidence, tokenize
from dl_agent.knowledge.store import FilePaperStore

logger = logging.getLogger(__name__)

ParseFn = Callable[[str], ParseResult]
ChatFn = Callable[..., str]


class PaperNotFoundError(KeyError):
    pass


class PaperNotReadyError(RuntimeError):
    def __init__(self, status: str):
        super().__init__(status)
        self.status = status


class IndexNotReadyError(RuntimeError):
    def __init__(self, status: str, error: str | None = None):
        super().__init__(status)
        self.status = status
        self.error = error


class KnowledgeService:
    def __init__(
        self,
        store: FilePaperStore,
        settings: Settings | None = None,
        parse_fn: ParseFn | None = None,
        chat_fn: ChatFn | None = None,
        embed_fn: EmbedFn | None = None,
        sparse_fn: SparseFn | None = None,
        vector_index=None,
        hybrid_encoder: HybridEncoder | None = None,
    ):
        self.store = store
        self.settings = settings or get_settings()
        self.parse_fn = parse_fn or self._parse
        self.chat_fn = chat_fn
        self.embed_fn = embed_fn
        self.sparse_fn = sparse_fn
        self._vector_index = vector_index
        self._vector_attempted = vector_index is not None
        self._hybrid_encoder = hybrid_encoder
        self._encoder_attempted = hybrid_encoder is not None
        self._encoder_lock = threading.Lock()
        self._encoder_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._indexing: set[str] = set()

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
            try:
                parsed.items = repair_formula_items(
                    parsed.items,
                    settings=self.settings,
                    chat_fn=self.chat_fn,
                )
            except Exception:
                logger.warning("formula vision repair failed paper_id=%s", paper_id, exc_info=True)
        except ParseError as exc:
            logger.warning("ingest_failed paper_id=%s error=%s", paper_id, exc)
            paper.status = "ingest_failed"
            paper.intro_status = "skipped"
            paper.method_status = "skipped"
            paper.index_status = "skipped"
            paper.index_error = "ingest_failed"
            self.store.save_paper(paper)
            self._audit(paper_id, "parse", started, extra={"status": "ingest_failed", "error": str(exc)})
            return paper
        except Exception as exc:
            logger.exception("ingest_failed paper_id=%s", paper_id)
            paper.status = "ingest_failed"
            paper.intro_status = "skipped"
            paper.method_status = "skipped"
            paper.index_status = "skipped"
            paper.index_error = "ingest_failed"
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
        paper.translate_status = "pending"
        paper.index_status = "pending"
        paper.index_error = None
        self.store.delete_translation(paper_id)
        self.store.delete_chunks(paper_id)

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
        if paper.status == "ready":
            return self.index_paper(paper.paper_id, force=True)
        paper.index_status = "skipped"
        paper.index_error = "needs_ocr"
        self.store.save_paper(paper)
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
            paper.index_status = "pending"
            paper.index_error = None
            paper.embedding_version = None
            self.store.save_paper(paper)
            logger.info("reparse queued paper_id=%s", paper_id)
        self.delete_index(paper_id)
        paper = self.store.get_paper(paper_id)
        if paper is None:
            raise PaperNotFoundError(paper_id)
        return paper, True

    def ingest(self, pdf_bytes: bytes, filename: str) -> Paper:
        paper, should_parse = self.start_ingest(pdf_bytes, filename)
        if should_parse:
            return self.finish_ingest(paper.paper_id)
        if paper.status == "ready" and paper.index_status == "pending":
            return self.index_paper(paper.paper_id, force=True)
        return paper

    def get_paper(self, paper_id: str) -> Paper:
        paper = self.store.get_paper(paper_id)
        if paper is None:
            raise PaperNotFoundError(paper_id)
        return paper

    def list_papers(self) -> list[Paper]:
        return self.store.list_papers()

    def get_sections(self, paper_id: str) -> list[Section]:
        self.get_paper(paper_id)
        sections = repair_front_matter(collapse_false_headings(self.store.get_sections(paper_id)))
        figures = self.store.get_figures(paper_id)
        sections, figures, _ = apply_figure_dedupe(sections, figures)
        sections, _, _ = polish_formulas(sections, figures)
        return sections

    def get_figures(self, paper_id: str, section_id: str | None = None) -> list[Figure]:
        self.get_paper(paper_id)
        sections = repair_front_matter(collapse_false_headings(self.store.get_sections(paper_id)))
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

    def needs_index(self, paper: Paper) -> bool:
        return paper.status == "ready" and paper.index_status == "pending"

    def index_paper(self, paper_id: str, *, force: bool = False) -> Paper:
        with self._lock:
            if paper_id in self._indexing:
                return self.get_paper(paper_id)
            self._indexing.add(paper_id)
        try:
            return self._index_paper(paper_id, force=force)
        finally:
            with self._lock:
                self._indexing.discard(paper_id)

    def _index_paper(self, paper_id: str, *, force: bool = False) -> Paper:
        paper = self.get_paper(paper_id)
        if paper.status != "ready":
            if paper.status == "needs_ocr":
                return self._set_index(paper, "skipped", error="needs_ocr")
            return paper
        version = self._index_version()
        sections = self.get_sections(paper_id)
        chunks = cap_chunks(
            build_chunks(
                sections,
                chunk_size=self.settings.chunk_size,
                overlap=self.settings.chunk_overlap,
            ),
            self.settings.max_chunks,
        )
        digest = chunks_source_hash(chunks, version)
        manifest = self.store.get_chunk_manifest(paper_id)
        if (
            not force
            and paper.index_status == "ready"
            and manifest
            and manifest.get("source_hash") == digest
            and manifest.get("embedding_version") == version
        ):
            return paper
        paper.index_status = "pending"
        paper.index_error = None
        self.store.save_paper(paper)
        started = time.perf_counter()
        skipped = sum(1 for section in sections if not (section.text or "").strip())
        try:
            if not chunks:
                paper = self._set_index(paper, "skipped", error="no_indexable_sections", version=version)
                self._audit(
                    paper_id,
                    "index",
                    started,
                    extra={"status": "skipped", "section_count": len(sections), "chunk_count": 0},
                )
                return paper
            backend = self._vector_backend()
            encoder = self._encoder(wait=True)
            dense_fn = self.embed_fn or (encoder.embed_dense_documents if encoder else None)
            sparse_fn = self.sparse_fn or (encoder.embed_sparse_documents if encoder else None)
            if backend is not None and dense_fn is not None:
                texts = [item.text for item in chunks]
                dense = dense_fn(texts)
                sparse = sparse_fn(texts) if sparse_fn is not None else None
                backend.delete_paper(paper_id)
                backend.upsert(paper_id, chunks, dense, sparse)
            elif encoder is not None and backend is None:
                logger.warning("hybrid encoder ready but no Qdrant, lexical index paper_id=%s", paper_id)
                version = "lexical-v1"
                digest = chunks_source_hash(chunks, version)
            self.store.save_chunks(
                paper_id,
                chunks,
                embedding_version=version,
                source_hash=digest,
            )
            paper = self._set_index(paper, "ready", version=version)
            self._audit(
                paper_id,
                "index",
                started,
                extra={
                    "status": "ready",
                    "section_count": len(sections),
                    "chunk_count": len(chunks),
                    "skipped_empty": skipped,
                    "embedding_version": version,
                },
            )
            logger.info(
                "index ready paper_id=%s chunks=%s version=%s",
                paper_id,
                len(chunks),
                version,
            )
            return paper
        except Exception as exc:
            logger.exception("index_failed paper_id=%s", paper_id)
            paper = self._set_index(paper, "failed", error=str(exc)[:500], version=version)
            self._audit(
                paper_id,
                "index",
                started,
                extra={"status": "failed", "error": str(exc)[:200]},
            )
            return paper

    def delete_index(self, paper_id: str) -> None:
        self.store.delete_chunks(paper_id)
        backend = self._vector_backend()
        if backend is not None:
            backend.delete_paper(paper_id)
        paper = self.store.get_paper(paper_id)
        if paper is None:
            return
        paper.index_status = "pending"
        paper.index_error = None
        paper.embedding_version = None
        self.store.save_paper(paper)

    def query(
        self,
        paper_id: str,
        question: str,
        k: int | None = None,
        kinds: list[str] | None = None,
    ) -> list[Evidence]:
        paper = self.get_paper(paper_id)
        if paper.status != "ready":
            raise PaperNotReadyError(paper.status)
        if paper.index_status == "pending":
            raise IndexNotReadyError("pending")
        if paper.index_status == "failed":
            raise IndexNotReadyError("failed", paper.index_error)
        if paper.index_status == "skipped":
            return []
        chunks = self.store.get_chunks(paper_id)
        if not chunks:
            chunks = cap_chunks(
                build_chunks(
                    self.get_sections(paper_id),
                    chunk_size=self.settings.chunk_size,
                    overlap=self.settings.chunk_overlap,
                ),
                self.settings.max_chunks,
            )
        if kinds:
            allowed = set(kinds)
            chunks = [item for item in chunks if item.section_kind in allowed]
        limit = k if k is not None else self.settings.retrieve_k
        backend = self._vector_backend()
        encoder = self._encoder(wait=True)
        dense_query = None
        sparse_query = None
        if encoder is not None:
            dense_query = encoder.embed_dense_query(question)
            sparse_query = encoder.embed_sparse_query(question)
        elif self.embed_fn is not None:
            dense_query = self.embed_fn([question])[0]
            if self.sparse_fn is not None:
                sparse_query = self.sparse_fn([question])[0]
        if backend is not None and dense_query is not None:
            try:
                hits = backend.search_hybrid(
                    paper_id,
                    dense_query,
                    sparse_query,
                    limit,
                    kinds=kinds,
                )
            except Exception:
                logger.warning("hybrid search failed, lexical only paper_id=%s", paper_id, exc_info=True)
                hits = []
            if hits:
                by_id = {item.chunk_id: item for item in chunks}
                out: list[Evidence] = []
                for hit in hits:
                    if hit.paper_id != paper_id:
                        continue
                    chunk = by_id.get(hit.chunk_id)
                    if chunk is None:
                        continue
                    out.append(to_evidence(chunk, hit.score))
                    if len(out) >= limit:
                        break
                if out:
                    return out
        return retrieve_from_chunks(chunks, question, k=limit)

    def query_corpus(
        self,
        question: str,
        k: int | None = None,
        chunks_per_paper: int | None = None,
        kinds: list[str] | None = None,
        exclude_paper_id: str | None = None,
    ) -> list[LibraryHit]:
        """跨篇检索：按块命中，再聚成论文卡片（标题 + 摘要 + 命中原因）。"""
        needle = (question or "").strip()
        if not needle:
            return []
        paper_limit = k if k is not None else self.settings.library_paper_k
        paper_limit = max(1, min(int(paper_limit), 16))
        per_paper = chunks_per_paper if chunks_per_paper is not None else self.settings.library_chunks_per_paper
        per_paper = max(1, min(int(per_paper), 4))
        papers = self._indexed_papers()
        if exclude_paper_id:
            papers = [item for item in papers if item.paper_id != exclude_paper_id]
        if not papers:
            return []
        chunks = self._corpus_chunks(papers, kinds=kinds)
        if not chunks:
            return []
        grouped = self._hybrid_corpus_hits(needle, chunks, paper_limit, per_paper, kinds=kinds)
        if not grouped:
            grouped = retrieve_corpus_from_chunks(
                chunks,
                needle,
                paper_limit=paper_limit,
                chunks_per_paper=per_paper,
            )
        catalog = {item.paper_id: item for item in papers}
        out: list[LibraryHit] = []
        for paper_id, pairs in grouped:
            paper = catalog.get(paper_id) or self.store.get_paper(paper_id)
            if paper is None or paper.status != "ready":
                continue
            out.append(self._to_library_hit(paper, pairs))
            if len(out) >= paper_limit:
                break
        return out

    def paper_card(self, paper_id: str) -> LibraryHit | None:
        paper = self.store.get_paper(paper_id)
        if paper is None:
            return None
        return self._to_library_hit(paper, [])

    def _indexed_papers(self) -> list[Paper]:
        return [
            paper
            for paper in self.store.list_papers()
            if paper.status == "ready" and paper.index_status == "ready"
        ]

    def _corpus_chunks(
        self,
        papers: list[Paper],
        *,
        kinds: list[str] | None = None,
    ) -> list[ChildChunk]:
        allowed = set(kinds) if kinds else None
        chunks: list[ChildChunk] = []
        for paper in papers:
            items = self.store.get_chunks(paper.paper_id)
            if not items:
                items = cap_chunks(
                    build_chunks(
                        self.get_sections(paper.paper_id),
                        chunk_size=self.settings.chunk_size,
                        overlap=self.settings.chunk_overlap,
                    ),
                    self.settings.max_chunks,
                )
            if allowed:
                items = [item for item in items if item.section_kind in allowed]
            chunks.extend(items)
        return chunks

    def _hybrid_corpus_hits(
        self,
        question: str,
        chunks: list,
        paper_limit: int,
        per_paper: int,
        *,
        kinds: list[str] | None = None,
    ) -> list[tuple[str, list[tuple]]]:
        backend = self._vector_backend()
        encoder = self._encoder(wait=True)
        dense_query = None
        sparse_query = None
        if encoder is not None:
            dense_query = encoder.embed_dense_query(question)
            sparse_query = encoder.embed_sparse_query(question)
        elif self.embed_fn is not None:
            dense_query = self.embed_fn([question])[0]
            if self.sparse_fn is not None:
                sparse_query = self.sparse_fn([question])[0]
        if backend is None or dense_query is None:
            return []
        fetch = max(paper_limit * per_paper * 4, paper_limit * 2)
        try:
            hits = backend.search_hybrid(None, dense_query, sparse_query, fetch, kinds=kinds)
        except Exception:
            logger.warning("corpus hybrid search failed, lexical only", exc_info=True)
            return []
        if not hits:
            return []
        by_key = {(item.paper_id, item.chunk_id): item for item in chunks}
        grouped: dict[str, list[tuple]] = {}
        for hit in hits:
            chunk = by_key.get((hit.paper_id, hit.chunk_id))
            if chunk is None:
                continue
            if not _chunk_matches_query(question, chunk):
                continue
            bucket = grouped.setdefault(hit.paper_id, [])
            if len(bucket) >= per_paper:
                continue
            bucket.append((chunk, hit.score))
        ranked = sorted(
            grouped.items(),
            key=lambda item: item[1][0][1] if item[1] else 0.0,
            reverse=True,
        )
        return ranked[:paper_limit]

    def _to_library_hit(self, paper: Paper, pairs: list[tuple]) -> LibraryHit:
        abstract = self._abstract_text(paper)
        why = ""
        section_title = None
        score = None
        if pairs:
            chunk, score = pairs[0]
            section_title = chunk.section_title
            quote = (chunk.text or "").strip()
            quote = " ".join(quote.split())
            if len(quote) > 220:
                quote = quote[:220].rstrip() + "…"
            why = f"{chunk.section_title}: {quote}" if quote else (chunk.section_title or "")
        return LibraryHit(
            paper_id=paper.paper_id,
            title=paper.title,
            filename=paper.filename,
            authors=list(paper.authors or []),
            abstract=abstract or None,
            score=None if score is None else round(float(score), 4),
            why=why,
            section_title=section_title,
            openable=paper.status == "ready",
            index_status=paper.index_status,
        )

    def _abstract_text(self, paper: Paper) -> str:
        text = (paper.abstract or "").strip()
        if text:
            return text
        try:
            sections = self.store.get_sections(paper.paper_id)
        except Exception:
            return ""
        match = next(
            (
                item
                for item in sections
                if item.kind == "abstract" or (item.title or "").strip().lower() == "abstract"
            ),
            None,
        )
        return (match.text or "").strip() if match is not None else ""

    def _index_version(self) -> str:
        encoder = self._encoder()
        if encoder is not None:
            return encoder.version
        if self.embed_fn is not None and self._vector_index is not None:
            return self.settings.embedding_version()
        return "lexical-v1"

    def _encoder(self, *, wait: bool = False) -> HybridEncoder | None:
        if self.embed_fn is not None:
            return None
        if self._hybrid_encoder is not None or self._encoder_attempted:
            return self._hybrid_encoder
        self._start_encoder_load()
        if wait and self._encoder_thread is not None:
            self._encoder_thread.join(timeout=30)
        return self._hybrid_encoder

    def _start_encoder_load(self) -> None:
        with self._encoder_lock:
            if self._hybrid_encoder is not None or self._encoder_attempted:
                return
            if self._encoder_thread is not None and self._encoder_thread.is_alive():
                return

            def _load() -> None:
                try:
                    encoder = try_hybrid_encoder(self.settings)
                    self._hybrid_encoder = encoder
                    if encoder is not None:
                        logger.info("hybrid encoder version=%s", encoder.version)
                except Exception:
                    logger.warning("hybrid encoder failed to load", exc_info=True)
                finally:
                    self._encoder_attempted = True

            self._encoder_thread = threading.Thread(target=_load, daemon=True, name="hybrid-encoder")
            self._encoder_thread.start()

    def _vector_backend(self):
        if self._vector_index is not None:
            return self._vector_index
        if self._vector_attempted:
            return None
        self._vector_attempted = True
        if self._encoder(wait=True) is None and self.embed_fn is None:
            return None
        name = f"{self.settings.qdrant_collection}_{self._index_version()}"
        self._vector_index = try_qdrant_index(self.settings.resolved_qdrant_path(), name)
        return self._vector_index

    def _set_index(
        self,
        paper: Paper,
        status: str,
        *,
        error: str | None = None,
        version: str | None = None,
    ) -> Paper:
        paper.index_status = status  # type: ignore[assignment]
        paper.index_error = error
        if version is not None:
            paper.embedding_version = version
        self.store.save_paper(paper)
        return paper

    def _audit(self, paper_id: str, stage: str, started: float, extra: dict) -> None:
        record = {
            "paper_id": paper_id,
            "stage": stage,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            **extra,
        }
        self.store.append_audit(record)


def _chunk_matches_query(question: str, chunk: ChildChunk) -> bool:
    latin = {token for token in tokenize(question) if any(ch.isascii() and ch.isalnum() for ch in token)}
    if not latin:
        return True
    hay = set(tokenize(f"{chunk.section_title} {chunk.text}"))
    return bool(latin & hay)


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


def list_papers() -> list[Paper]:
    return get_service().list_papers()


def get_sections(paper_id: str) -> list[Section]:
    return get_service().get_sections(paper_id)


def get_figures(paper_id: str, section_id: str | None = None) -> list[Figure]:
    return get_service().get_figures(paper_id, section_id)
