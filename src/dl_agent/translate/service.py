"""章节全文翻译。"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable

from dl_agent.config import Settings, get_settings
from dl_agent.domain.models import (
    Figure,
    FigureTranslation,
    PaperTranslation,
    Section,
    SectionTranslation,
    TranslateStatus,
)
from dl_agent.harness.complete import LlmNotConfiguredError, LlmRequestError, chat
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore

logger = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class TextChunk:
    text: str
    paragraph_start: bool = True


SYSTEM_PROMPT = """You are an academic paper translator. Translate English academic text into Simplified Chinese.

Rules:
- Output MUST be Simplified Chinese (简体中文). Do NOT copy the English source as the translation.
- Translate ALL prose. Do not leave English sentences untranslated. Do not summarize or omit paragraphs.
- Math must be valid KaTeX-compatible LaTeX that renders. Do NOT copy broken PDF/OCR math unchanged.
  Fix garbled symbols, for example:
  - `b proj ∈ RC_v` or `b proj $\\in RC_{v}$` → `$b_{\\mathrm{proj}} \\in \\mathbb{R}^{C_v}$`
  - `W proj $\\in \\mathbb{R}^{512 \\times C_{v}}$` → `$W_{\\mathrm{proj}} \\in \\mathbb{R}^{512 \\times C_{v}}$`
  - `QVT` used as a tensor → `$Q_{T}^{V}$`; keep CNN/MLP/ReID as names
  - `set to $4),l(1\\leq l\\leq L)$` → `set to $4$, $1 \\leq l \\leq L$`
  - `p$ neg` / `p neg（$F_i$|$T$）` → `$p_{\\mathrm{neg}}(F_i \\mid T)$`
  - `N_N^1 0` or `N 1` as counts → `$N_1$` / `$N_0$`
- Numbered or standalone equations MUST use `$$...$$`. Inline symbols use `$...$`. Same $$ count as the source.
- Do not wrap Chinese prose (其中 / 表示 / 是) inside `$...$` or `$$...$$`.
- Keep paragraph breaks (`\\n\\n`) identical to the source.
- Preserve markdown tables, list markers, <!--fig:fig-N-->, citations like [1], and model/dataset names.
- Return ONLY valid JSON, no markdown fences"""

FIGURE_CAPTION_PROMPT = """You are an academic caption translator. Translate figure/algorithm captions into Simplified Chinese.

Rules:
- Output MUST be Simplified Chinese. Do NOT copy the English caption unchanged.
- Keep 图 N / 算法 N numbering, (a)(b) markers, dataset/model names, and math.
- figure_id in the output MUST copy the input figure_id exactly (for example fig-001).
- Return ONLY JSON: {"captions":[{"figure_id":"fig-001","caption_zh":"图 1. ..."}]}"""

_DISPLAY_MATH_RE = re.compile(r"\$\$[\s\S]*?\$\$")
_FIG_MARK_RE = re.compile(r"<!--fig:fig-\d+-->")
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)


class TranslateService:
    def __init__(
        self,
        store: FilePaperStore,
        knowledge: KnowledgeService,
        settings: Settings | None = None,
        chat_fn: ChatFn | None = None,
    ):
        self.store = store
        self.knowledge = knowledge
        self.settings = settings or get_settings()
        self.chat_fn = chat_fn
        self._lock = threading.RLock()
        self._running: set[str] = set()
        self._run_seq: dict[str, int] = {}
        self._active_runs: set[tuple[str, int]] = set()
        self._cancel_events: dict[tuple[str, int], threading.Event] = {}
        self._fresh_runs: set[tuple[str, int]] = set()

    def get_translation(self, paper_id: str) -> PaperTranslation | None:
        self.knowledge.get_paper(paper_id)
        cached = self.store.get_translation(paper_id)
        if cached is None:
            return None
        # pending / cancelled 即使版本旧也要返回，否则前端轮询会 404
        if cached.status in {"pending", "cancelled"}:
            return _tidy_translation(cached)
        if cached.prompt_version == self.settings.translate_prompt_version:
            return _tidy_translation(cached)
        return None

    def start_translation(
        self, paper_id: str, *, refresh: bool = False
    ) -> tuple[PaperTranslation, bool, int]:
        if not self.settings.openai_api_key.strip():
            raise LlmNotConfiguredError("未配置 OPENAI_API_KEY，无法调用翻译模型")
        paper = self.knowledge.get_paper(paper_id)
        if paper.status != "ready":
            raise ValueError("paper_not_ready")

        cached = self.get_translation(paper_id)
        if cached and not refresh:
            same_version = cached.prompt_version == self.settings.translate_prompt_version
            run_id = self._run_seq.get(paper_id, 0)
            if cached.status in {"ready", "partial"} and same_version:
                if not self._missing_figure_captions(
                    paper_id, cached
                ) and not self._missing_section_translations(paper_id, cached):
                    return cached, False, run_id
            if cached.status == "pending":
                # 进程仍在跑 → 只轮询；reload 后 BackgroundTask 会丢 → POST 可续跑
                if self._is_active(paper_id):
                    return cached, False, run_id
                logger.warning(
                    "translate stale pending paper_id=%s sections=%s version=%s; re-queue",
                    paper_id,
                    len(cached.sections),
                    cached.prompt_version,
                )
                return cached, True, run_id
            # cancelled：再次 POST（非 refresh）续跑，保留已译章节
            # failed / 旧版 ready：默认允许再次 POST 自动重试

        with self._lock:
            paper = self.knowledge.get_paper(paper_id)
            running = self.store.get_translation(paper_id)
            if (
                running
                and running.status == "pending"
                and running.prompt_version == self.settings.translate_prompt_version
                and not refresh
                and self._is_active(paper_id)
            ):
                return running, False, self._run_seq.get(paper_id, 0)
            keep_sections: list[SectionTranslation] = []
            keep_title: str | None = None
            keep_figures: list[FigureTranslation] = []
            if (
                not refresh
                and running
                and running.prompt_version == self.settings.translate_prompt_version
                and running.status in {"cancelled", "ready", "partial"}
            ):
                keep_sections = list(running.sections)
                keep_title = running.title_zh
                keep_figures = list(running.figures)
            previous = self._run_seq.get(paper_id, 0)
            if previous:
                old = self._cancel_events.get((paper_id, previous))
                if old:
                    old.set()
            run_id = previous + 1
            self._run_seq[paper_id] = run_id
            self._cancel_event(paper_id, run_id)
            if refresh:
                self._fresh_runs.add((paper_id, run_id))
                keep_sections = []
                keep_title = None
                keep_figures = []
                logger.info("translate refresh paper_id=%s run_id=%s", paper_id, run_id)
            paper.translate_status = "pending"
            self.store.save_paper(paper)
            pending = PaperTranslation(
                paper_id=paper_id,
                status="pending",
                model=self.settings.model_name,
                prompt_version=self.settings.translate_prompt_version,
                title_zh=keep_title,
                sections=keep_sections,
                figures=keep_figures,
                error=None,
            )
            self.store.save_translation(pending)
            return pending, True, run_id

    def finish_translation(self, paper_id: str, run_id: int) -> PaperTranslation:
        if self._is_stale_run(paper_id, run_id):
            logger.info(
                "translate skip stale run paper_id=%s run_id=%s current=%s",
                paper_id,
                run_id,
                self._run_seq.get(paper_id, 0),
            )
            existing = self.store.get_translation(paper_id)
            if existing:
                return existing
        with self._lock:
            key = (paper_id, run_id)
            if key in self._active_runs:
                existing = self.store.get_translation(paper_id)
                if existing:
                    return existing
            self._active_runs.add(key)
        self._running.add(paper_id)
        try:
            reuse_cache = (paper_id, run_id) not in self._fresh_runs
            return self._finish_translation_locked(paper_id, run_id, reuse_cache=reuse_cache)
        finally:
            with self._lock:
                self._active_runs.discard((paper_id, run_id))
                self._fresh_runs.discard((paper_id, run_id))
                if not any(pid == paper_id for pid, _rid in self._active_runs):
                    self._running.discard(paper_id)

    def cancel_translation(self, paper_id: str) -> PaperTranslation:
        self.knowledge.get_paper(paper_id)
        existing = self.store.get_translation(paper_id)
        if existing is None:
            raise ValueError("not_started")
        if existing.status != "pending":
            return existing
        run_id = self._run_seq.get(paper_id, 0)
        self._cancel_event(paper_id, run_id).set()
        logger.info("translate cancel paper_id=%s run_id=%s", paper_id, run_id)
        result = existing.model_copy(update={"status": "cancelled", "error": "已终止翻译"})
        self.store.save_translation(result)
        paper = self.knowledge.get_paper(paper_id)
        paper.translate_status = "cancelled"
        self.store.save_paper(paper)
        return result

    def _cancel_event(self, paper_id: str, run_id: int) -> threading.Event:
        key = (paper_id, run_id)
        with self._lock:
            event = self._cancel_events.get(key)
            if event is None:
                event = threading.Event()
                self._cancel_events[key] = event
            return event

    def _is_cancelled(self, paper_id: str, run_id: int) -> bool:
        event = self._cancel_events.get((paper_id, run_id))
        return bool(event and event.is_set())

    def _is_active(self, paper_id: str) -> bool:
        with self._lock:
            return any(pid == paper_id for pid, _rid in self._active_runs)

    def _should_stop(self, paper_id: str, run_id: int) -> bool:
        return self._is_stale_run(paper_id, run_id) or self._is_cancelled(paper_id, run_id)

    def _is_stale_run(self, paper_id: str, run_id: int) -> bool:
        return self._run_seq.get(paper_id, 0) != run_id

    def _invoke_chat(self, paper_id: str, run_id: int, messages: list[dict[str, str]]) -> str:
        if self._is_cancelled(paper_id, run_id):
            raise LlmRequestError("翻译已中止", aborted=True)
        if self.chat_fn is not None:
            return self.chat_fn(messages)
        return chat(
            messages,
            settings=self.settings,
            cancel_event=self._cancel_event(paper_id, run_id),
        )

    def _wait_or_cancel(self, paper_id: str, run_id: int, seconds: float) -> None:
        if seconds <= 0:
            return
        event = self._cancel_event(paper_id, run_id)
        if event.wait(timeout=seconds):
            raise LlmRequestError("翻译已中止", aborted=True)

    def _finish_translation_locked(
        self, paper_id: str, run_id: int, *, reuse_cache: bool = True
    ) -> PaperTranslation:
        paper = self.knowledge.get_paper(paper_id)
        sections = self.knowledge.get_sections(paper_id)
        section_by_id = {section.section_id: section for section in sections}
        existing = self.store.get_translation(paper_id)
        translated_by_id: dict[str, SectionTranslation] = {}
        failed = False
        fatal = False
        fail_message: str | None = None
        llm_ok = 0
        same_version = bool(
            existing
            and reuse_cache
            and existing.prompt_version == self.settings.translate_prompt_version
        )
        title_zh: str | None = existing.title_zh if existing and same_version else None
        started = time.perf_counter()
        parallelism = max(1, self.settings.translate_parallelism)
        chunk_max = max(400, self.settings.translate_chunk_max_chars)

        if existing and same_version:
            for item in existing.sections:
                section = section_by_id.get(item.section_id)
                if section and (_is_front_matter(section) or _is_paper_title_section(section)):
                    continue
                if _is_cached_section_translation(item, section):
                    translated_by_id[item.section_id] = item

        logger.info(
            "translate start paper_id=%s run_id=%s sections=%s cached=%s model=%s timeout_s=%s "
            "parallelism=%s chunk_max=%s",
            paper_id,
            run_id,
            len(sections),
            len(translated_by_id),
            self.settings.model_name,
            self.settings.llm_timeout_s,
            parallelism,
            chunk_max,
        )

        paper_title = (paper.title or "").strip()
        if not paper_title:
            meta = next((section for section in sections if _is_paper_title_section(section)), None)
            if meta:
                paper_title = meta.title.strip()

        if paper_title and not title_zh:
            if self._should_stop(paper_id, run_id):
                return self._stopped_result(
                    paper_id,
                    run_id,
                    translated_by_id,
                    sections,
                    title_zh=title_zh,
                    llm_ok=llm_ok,
                    started=started,
                )
            try:
                title_zh = self._translate_paper_title_llm(paper_title, paper_id=paper_id, run_id=run_id)
                llm_ok += 1
                self._save_progress_if_current(
                    paper_id, run_id, list(translated_by_id.values()), status="pending", title_zh=title_zh
                )
            except LlmNotConfiguredError as exc:
                failed = True
                fatal = True
                fail_message = str(exc)
                logger.error("translate title aborted paper_id=%s error=%s", paper_id, exc)
            except LlmRequestError as exc:
                if getattr(exc, "aborted", False) or self._is_cancelled(paper_id, run_id):
                    return self._stopped_result(
                        paper_id,
                        run_id,
                        translated_by_id,
                        sections,
                        title_zh=title_zh,
                        llm_ok=llm_ok,
                        started=started,
                    )
                failed = True
                fail_message = str(exc)
                logger.warning(
                    "translate title failed paper_id=%s fatal=%s error=%s",
                    paper_id,
                    getattr(exc, "fatal", False),
                    exc,
                )
                if getattr(exc, "fatal", False):
                    fatal = True
            except Exception as exc:
                failed = True
                fail_message = str(exc)
                logger.exception("translate title failed paper_id=%s", paper_id)

        if self._is_cancelled(paper_id, run_id):
            return self._stopped_result(
                paper_id,
                run_id,
                translated_by_id,
                sections,
                title_zh=title_zh,
                llm_ok=llm_ok,
                started=started,
            )

        if fatal:
            return self._finalize_translation(
                paper_id,
                run_id,
                translated_by_id,
                sections,
                title_zh=title_zh,
                status="failed",
                fail_message=fail_message,
                llm_ok=llm_ok,
                started=started,
            )

        pending_sections: list[Section] = []
        for section in sections:
            if _skip_section(section):
                if section.section_id not in translated_by_id:
                    prev = None
                    if existing:
                        prev = next(
                            (item for item in existing.sections if item.section_id == section.section_id),
                            None,
                        )
                    translated_by_id[section.section_id] = _skipped_translation(section, prev)
            elif section.section_id not in translated_by_id:
                pending_sections.append(section)

        if pending_sections:
            section_llm_ok, section_failed, section_fatal, section_error = self._translate_sections_parallel(
                paper_id,
                run_id,
                pending_sections,
                translated_by_id,
                title_zh=title_zh,
                parallelism=parallelism,
                chunk_max=chunk_max,
            )
            llm_ok += section_llm_ok
            failed = failed or section_failed
            if section_fatal:
                fatal = True
            if section_error and not fail_message:
                fail_message = section_error

        missing = [
            section
            for section in sections
            if not _skip_section(section) and section.section_id not in translated_by_id
        ]
        if missing:
            failed = True
            if not fail_message:
                fail_message = f"仍有 {len(missing)} 个章节未译完"

        if self._is_cancelled(paper_id, run_id):
            return self._stopped_result(
                paper_id,
                run_id,
                translated_by_id,
                sections,
                title_zh=title_zh,
                llm_ok=llm_ok,
                started=started,
            )

        if failed and fatal:
            status: TranslateStatus = "failed"
        elif failed and llm_ok == 0 and not translated_by_id:
            status = "failed"
        elif failed:
            status = "partial"
        else:
            status = "ready"

        figure_items: list[FigureTranslation] = list(existing.figures) if existing and same_version else []
        if status in {"ready", "partial"} and not self._should_stop(paper_id, run_id):
            try:
                figure_items = self._translate_figures(
                    paper_id,
                    run_id,
                    section_translations=list(translated_by_id.values()),
                    existing=figure_items,
                )
            except LlmRequestError as exc:
                if getattr(exc, "aborted", False) or self._is_cancelled(paper_id, run_id):
                    return self._stopped_result(
                        paper_id,
                        run_id,
                        translated_by_id,
                        sections,
                        title_zh=title_zh,
                        llm_ok=llm_ok,
                        started=started,
                        figures=figure_items,
                    )
                logger.warning("translate figures failed paper_id=%s error=%s", paper_id, exc)
            except Exception:
                logger.exception("translate figures failed paper_id=%s", paper_id)

        return self._finalize_translation(
            paper_id,
            run_id,
            translated_by_id,
            sections,
            title_zh=title_zh,
            status=status,
            fail_message=fail_message,
            llm_ok=llm_ok,
            started=started,
            figures=figure_items,
        )

    def _translate_sections_parallel(
        self,
        paper_id: str,
        run_id: int,
        sections: list[Section],
        translated_by_id: dict[str, SectionTranslation],
        *,
        title_zh: str | None,
        parallelism: int,
        chunk_max: int,
    ) -> tuple[int, bool, bool, str | None]:
        llm_ok = 0
        failed = False
        fatal = False
        fail_message: str | None = None
        stop = threading.Event()
        failed_sections: list[Section] = []

        def run(section: Section) -> SectionTranslation:
            if stop.is_set() or self._should_stop(paper_id, run_id):
                raise LlmRequestError("翻译已中止", aborted=True)
            existing = translated_by_id.get(section.section_id)
            logger.info(
                "translate section paper_id=%s section_id=%s title=%s chars=%s chunks=%s resume=%s",
                paper_id,
                section.section_id,
                section.title[:80],
                len(section.text),
                len(_split_text_chunks(section.text, chunk_max)),
                bool(existing and existing.partial),
            )

            def on_progress(item: SectionTranslation) -> None:
                if self._should_stop(paper_id, run_id):
                    stop.set()
                    return
                translated_by_id[section.section_id] = item
                self._save_progress_if_current(
                    paper_id,
                    run_id,
                    list(translated_by_id.values()),
                    status="pending",
                    title_zh=title_zh,
                )

            return self._translate_section(
                section,
                paper_id=paper_id,
                run_id=run_id,
                chunk_max=chunk_max,
                existing=existing if existing and existing.partial else None,
                on_progress=on_progress,
                is_stale=lambda: self._should_stop(paper_id, run_id),
            )

        with ThreadPoolExecutor(max_workers=parallelism) as pool:
            futures: dict[Future[SectionTranslation], Section] = {
                pool.submit(run, section): section for section in sections
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    section = futures[future]
                    try:
                        item = future.result()
                    except LlmNotConfiguredError as exc:
                        failed = True
                        fatal = True
                        fail_message = str(exc)
                        stop.set()
                        for other in pending:
                            other.cancel()
                        logger.error("translate aborted paper_id=%s error=%s", paper_id, exc)
                        return llm_ok, failed, fatal, fail_message
                    except LlmRequestError as exc:
                        if getattr(exc, "aborted", False) or self._is_cancelled(paper_id, run_id):
                            stop.set()
                            for other in pending:
                                other.cancel()
                            logger.info("translate aborted paper_id=%s", paper_id)
                            return llm_ok, True, False, "已终止翻译"
                        failed = True
                        failed_sections.append(section)
                        if not fail_message:
                            fail_message = str(exc)
                        logger.warning(
                            "translate section failed paper_id=%s section_id=%s fatal=%s error=%s",
                            paper_id,
                            section.section_id,
                            getattr(exc, "fatal", False),
                            exc,
                        )
                        if getattr(exc, "fatal", False):
                            fatal = True
                            stop.set()
                            for other in pending:
                                other.cancel()
                            return llm_ok, failed, fatal, fail_message
                        continue
                    except Exception as exc:
                        failed = True
                        failed_sections.append(section)
                        if not fail_message:
                            fail_message = str(exc)
                        logger.exception(
                            "translate section failed paper_id=%s section_id=%s",
                            paper_id,
                            section.section_id,
                        )
                        if llm_ok == 0 and not translated_by_id:
                            fatal = True
                            stop.set()
                            for other in pending:
                                other.cancel()
                            return llm_ok, failed, fatal, fail_message
                        continue

                    translated_by_id[section.section_id] = item
                    llm_ok += 1
                    self._save_progress_if_current(
                        paper_id,
                        run_id,
                        list(translated_by_id.values()),
                        status="pending",
                        title_zh=title_zh,
                    )

        if failed_sections and not fatal and not self._is_cancelled(paper_id, run_id):
            logger.info(
                "translate retry failed sections paper_id=%s count=%s",
                paper_id,
                len(failed_sections),
            )
            for section in failed_sections:
                if section.section_id in translated_by_id:
                    cached = translated_by_id[section.section_id]
                    if not cached.partial:
                        continue

                def on_retry_progress(item: SectionTranslation, sid: str = section.section_id) -> None:
                    if self._should_stop(paper_id, run_id):
                        return
                    translated_by_id[sid] = item
                    self._save_progress_if_current(
                        paper_id,
                        run_id,
                        list(translated_by_id.values()),
                        status="pending",
                        title_zh=title_zh,
                    )

                try:
                    item = self._translate_section(
                        section,
                        paper_id=paper_id,
                        run_id=run_id,
                        chunk_max=chunk_max,
                        existing=translated_by_id.get(section.section_id),
                        on_progress=on_retry_progress,
                        is_stale=lambda: self._should_stop(paper_id, run_id),
                    )
                except LlmNotConfiguredError as exc:
                    failed = True
                    fatal = True
                    fail_message = str(exc)
                    break
                except LlmRequestError as exc:
                    if getattr(exc, "aborted", False) or self._is_cancelled(paper_id, run_id):
                        fail_message = "已终止翻译"
                        break
                    failed = True
                    if not fail_message:
                        fail_message = str(exc)
                    logger.warning(
                        "translate retry failed paper_id=%s section_id=%s error=%s",
                        paper_id,
                        section.section_id,
                        exc,
                    )
                    continue
                except Exception as exc:
                    failed = True
                    if not fail_message:
                        fail_message = str(exc)
                    logger.exception(
                        "translate retry failed paper_id=%s section_id=%s",
                        paper_id,
                        section.section_id,
                    )
                    continue
                translated_by_id[section.section_id] = item
                llm_ok += 1
                self._save_progress_if_current(
                    paper_id,
                    run_id,
                    list(translated_by_id.values()),
                    status="pending",
                    title_zh=title_zh,
                )

        return llm_ok, failed, fatal, fail_message

    def _translate_section(
        self,
        section: Section,
        *,
        paper_id: str,
        run_id: int,
        chunk_max: int,
        existing: SectionTranslation | None = None,
        on_progress: Callable[[SectionTranslation], None] | None = None,
        is_stale: Callable[[], bool] | None = None,
    ) -> SectionTranslation:
        stale = is_stale or (lambda: False)
        chunks = _split_text_chunks(section.text, chunk_max)
        total = len(chunks)
        if not chunks:
            item = SectionTranslation(
                section_id=section.section_id,
                title_zh=section.title,
                text_zh=section.text,
            )
            if on_progress:
                on_progress(item)
            return item
        if total == 1:
            item = self._translate_section_llm(section, paper_id=paper_id, run_id=run_id)
            item = item.model_copy(update={"partial": False, "chunks_done": 1, "chunks_total": 1})
            if on_progress:
                on_progress(item)
            return item

        start_index = 0
        title_zh = section.title
        accumulated = ""
        if existing and existing.partial and existing.chunks_done < existing.chunks_total:
            start_index = existing.chunks_done
            accumulated = existing.text_zh
            title_zh = existing.title_zh

        for index in range(start_index, total):
            if stale():
                raise LlmRequestError("翻译已中止", aborted=True)
            chunk = chunks[index].text
            if index == 0:
                part = self._translate_section_llm(
                    Section(
                        section_id=section.section_id,
                        paper_id=section.paper_id,
                        title=section.title,
                        kind=section.kind,
                        level=section.level,
                        page_start=section.page_start,
                        page_end=section.page_end,
                        text=chunk,
                        parent_id=section.parent_id,
                        figure_ids=section.figure_ids,
                    ),
                    paper_id=paper_id,
                    run_id=run_id,
                )
                title_zh = part.title_zh
                piece = part.text_zh
            else:
                piece = self._translate_text_chunk_llm(chunk, paper_id=paper_id, run_id=run_id)
            if not accumulated:
                accumulated = piece
            elif chunks[index].paragraph_start:
                accumulated = f"{accumulated}\n\n{piece}"
            else:
                accumulated = accumulated + piece
            done = index + 1
            item = SectionTranslation(
                section_id=section.section_id,
                title_zh=title_zh,
                text_zh=accumulated,
                partial=done < total,
                chunks_done=done,
                chunks_total=total,
            )
            if on_progress:
                on_progress(item)
            if done < total and self.settings.translate_section_delay_s > 0:
                self._wait_or_cancel(paper_id, run_id, self.settings.translate_section_delay_s)
        return item

    def _stopped_result(
        self,
        paper_id: str,
        run_id: int,
        translated_by_id: dict[str, SectionTranslation],
        sections: list[Section],
        *,
        title_zh: str | None,
        llm_ok: int,
        started: float,
        figures: list[FigureTranslation] | None = None,
    ) -> PaperTranslation:
        if self._is_stale_run(paper_id, run_id):
            existing = self.store.get_translation(paper_id)
            if existing:
                return existing
        return self._finalize_translation(
            paper_id,
            run_id,
            translated_by_id,
            sections,
            title_zh=title_zh,
            status="cancelled",
            fail_message="已终止翻译",
            llm_ok=llm_ok,
            started=started,
            figures=figures,
        )

    def _finalize_translation(
        self,
        paper_id: str,
        run_id: int,
        translated_by_id: dict[str, SectionTranslation],
        sections: list[Section],
        *,
        title_zh: str | None,
        status: TranslateStatus,
        fail_message: str | None,
        llm_ok: int,
        started: float,
        figures: list[FigureTranslation] | None = None,
    ) -> PaperTranslation:
        if self._is_stale_run(paper_id, run_id):
            existing = self.store.get_translation(paper_id)
            if existing:
                return existing
        ordered = [
            translated_by_id[section.section_id]
            for section in sections
            if section.section_id in translated_by_id
        ]
        if figures is None:
            prev = self.store.get_translation(paper_id)
            figures = list(prev.figures) if prev else []
        result = PaperTranslation(
            paper_id=paper_id,
            status=status,
            model=self.settings.model_name,
            prompt_version=self.settings.translate_prompt_version,
            title_zh=title_zh,
            sections=ordered,
            figures=figures,
            error=fail_message,
        )
        self.store.save_translation(result)
        paper = self.knowledge.get_paper(paper_id)
        paper.translate_status = status
        self.store.save_paper(paper)
        self.knowledge._audit(
            paper_id,
            "translate",
            started,
            extra={
                "status": status,
                "section_count": len(ordered),
                "llm_ok": llm_ok,
                "model": self.settings.model_name,
                "error": fail_message,
                "parallelism": self.settings.translate_parallelism,
                "chunk_max": self.settings.translate_chunk_max_chars,
            },
        )
        logger.info(
            "translate done paper_id=%s status=%s llm_ok=%s sections=%s error=%s",
            paper_id,
            status,
            llm_ok,
            len(ordered),
            fail_message,
        )
        return result

    def _save_progress_if_current(
        self,
        paper_id: str,
        run_id: int,
        sections: list[SectionTranslation],
        *,
        status: TranslateStatus,
        title_zh: str | None = None,
    ) -> None:
        if self._should_stop(paper_id, run_id):
            return
        self._save_progress(paper_id, sections, status=status, title_zh=title_zh)

    def _save_progress(
        self,
        paper_id: str,
        sections: list[SectionTranslation],
        *,
        status: TranslateStatus,
        title_zh: str | None = None,
    ) -> None:
        with self._lock:
            current = self.store.get_translation(paper_id)
            self.store.save_translation(
                PaperTranslation(
                    paper_id=paper_id,
                    status=status,
                    model=self.settings.model_name,
                    prompt_version=self.settings.translate_prompt_version,
                    title_zh=title_zh,
                    sections=list(sections),
                    figures=list(current.figures) if current else [],
                    error=None,
                )
            )

    def _missing_figure_captions(self, paper_id: str, translation: PaperTranslation) -> bool:
        needed = _captioned_figures(self.knowledge.get_figures(paper_id))
        if not needed:
            return False
        have = {
            item.figure_id for item in translation.figures if _looks_like_chinese(item.caption_zh)
        }
        return any(figure.figure_id not in have for figure in needed)

    def _missing_section_translations(self, paper_id: str, translation: PaperTranslation) -> bool:
        sections = self.knowledge.get_sections(paper_id)
        by_id = {item.section_id: item for item in translation.sections}
        for section in sections:
            if _skip_section(section):
                continue
            item = by_id.get(section.section_id)
            if item is None or not _is_cached_section_translation(item, section):
                return True
        return False

    def _translate_figures(
        self,
        paper_id: str,
        run_id: int,
        *,
        section_translations: list[SectionTranslation],
        existing: list[FigureTranslation],
    ) -> list[FigureTranslation]:
        figures = _captioned_figures(self.knowledge.get_figures(paper_id))
        if not figures:
            return []
        by_id = {
            item.figure_id: item
            for item in existing
            if item.figure_id and _looks_like_chinese(item.caption_zh)
        }
        harvested = _harvest_figure_captions(
            [item.text_zh for item in section_translations],
            figures,
        )
        for figure in figures:
            if figure.figure_id in by_id:
                continue
            caption_zh = harvested.get(figure.figure_id)
            if caption_zh:
                by_id[figure.figure_id] = FigureTranslation(
                    figure_id=figure.figure_id,
                    caption_zh=caption_zh,
                )
        missing = [figure for figure in figures if figure.figure_id not in by_id]
        if missing:
            translated = self._translate_figure_captions_llm(
                missing, paper_id=paper_id, run_id=run_id
            )
            by_id.update(translated)
        return [by_id[figure.figure_id] for figure in figures if figure.figure_id in by_id]

    def _translate_figure_captions_llm(
        self,
        figures: list[Figure],
        *,
        paper_id: str,
        run_id: int,
    ) -> dict[str, FigureTranslation]:
        payload = json.dumps(
            {
                "figures": [
                    {
                        "figure_id": figure.figure_id,
                        "caption": (figure.caption or figure.label or "").strip(),
                    }
                    for figure in figures
                ]
            },
            ensure_ascii=False,
        )
        user_content = (
            "Translate these paper figure captions into Simplified Chinese. "
            "Keep Figure/Fig/Algorithm numbers as 图 N / 算法 N. "
            "Keep (a)(b) markers, dataset/model names, and math. "
            "Copy each figure_id exactly. "
            'Respond with JSON {"captions":[{"figure_id":"fig-001","caption_zh":"图 1. ..."}]} only.\n\n'
            f"{payload}"
        )
        data = self._figure_caption_json(paper_id, run_id, user_content)
        out = _parse_figure_caption_payload(data, figures)
        if out:
            return out
        logger.warning(
            "translate figures retry paper_id=%s keys=%s sample=%s",
            paper_id,
            list(data)[:8] if isinstance(data, dict) else type(data).__name__,
            str(data)[:400],
        )
        retry = (
            "Your previous JSON could not be matched. "
            "Use the exact figure_id values from the input (fig-001, fig-002, …). "
            "caption_zh MUST be Simplified Chinese. "
            'JSON only: {"captions":[{"figure_id":"fig-001","caption_zh":"图 1. ..."}]}\n\n'
            f"{payload}"
        )
        data = self._figure_caption_json(paper_id, run_id, retry)
        out = _parse_figure_caption_payload(data, figures)
        if not out:
            logger.warning(
                "translate figures empty paper_id=%s keys=%s sample=%s",
                paper_id,
                list(data)[:8] if isinstance(data, dict) else type(data).__name__,
                str(data)[:400],
            )
            raise LlmRequestError("图注翻译结果仍是英文或为空")
        return out

    def _figure_caption_json(self, paper_id: str, run_id: int, user_content: str) -> dict:
        raw = self._invoke_chat(
            paper_id,
            run_id,
            [
                {"role": "system", "content": FIGURE_CAPTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        try:
            data = _parse_json_object(raw)
        except LlmRequestError:
            text = raw.strip()
            fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
            if fence:
                text = fence.group(1).strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LlmRequestError("图注翻译结果不是 JSON 对象") from exc
            if isinstance(parsed, list):
                return {"captions": parsed}
            if isinstance(parsed, dict):
                return parsed
            raise LlmRequestError("图注翻译结果不是 JSON 对象")
        if isinstance(data.get("captions"), str):
            try:
                nested = json.loads(data["captions"])
                data = {**data, "captions": nested}
            except json.JSONDecodeError:
                pass
        return data

    def _translate_paper_title_llm(self, title: str, *, paper_id: str, run_id: int) -> str:
        if _looks_like_chinese(title):
            return title.strip()
        payload = json.dumps({"title": title}, ensure_ascii=False)
        raw = self._invoke_chat(
            paper_id,
            run_id,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Translate this paper title only. Do NOT translate author names. "
                        'Respond with JSON {"title_zh":"..."} only.\n\n'
                        f"{payload}"
                    ),
                },
            ],
        )
        data = _parse_json_object(raw)
        title_zh = str(data.get("title_zh") or "").strip()
        if not title_zh:
            raise LlmRequestError("翻译结果缺少 title_zh")
        if not _looks_like_chinese(title_zh):
            raise LlmRequestError("论文标题翻译结果仍是英文")
        if title_zh == title.strip():
            raise LlmRequestError("论文标题翻译结果与英文原文相同")
        return title_zh

    def _chat_json(self, paper_id: str, run_id: int, user_content: str) -> dict:
        raw = self._invoke_chat(
            paper_id,
            run_id,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        return _parse_json_object(raw)

    def _translate_with_structure_retry(
        self,
        paper_id: str,
        run_id: int,
        user_content: str,
        *,
        source: str,
    ) -> dict:
        data = self._chat_json(paper_id, run_id, user_content)
        text_zh = str(data.get("text_zh") or "").strip()
        if not text_zh:
            raise LlmRequestError("翻译结果缺少 text_zh")
        repaired, issues = _repair_and_check(source, text_zh)
        data = {**data, "text_zh": repaired}
        if not issues:
            return data
        logger.info(
            "translate retry structure paper_id=%s issues=%s",
            paper_id,
            ",".join(issues),
        )
        retry_content = _structure_retry_prompt(source) + "\n\n" + user_content
        try:
            retry_data = self._chat_json(paper_id, run_id, retry_content)
        except LlmRequestError:
            if _can_keep_translation(source, repaired, issues):
                logger.warning("translate retry aborted, keeping first draft paper_id=%s", paper_id)
                return data
            raise
        retry_zh = str(retry_data.get("text_zh") or "").strip()
        if not retry_zh:
            if _can_keep_translation(source, repaired, issues):
                return data
            raise LlmRequestError("翻译结果缺少 text_zh")
        repaired2, issues2 = _repair_and_check(source, retry_zh)
        retry_data = {**retry_data, "text_zh": repaired2}
        if _fatal_translation_issues(source, issues2):
            if _can_keep_translation(source, repaired, issues):
                logger.warning(
                    "translate retry still invalid, keeping first draft paper_id=%s issues=%s",
                    paper_id,
                    ",".join(issues2),
                )
                return data
            raise LlmRequestError(_fatal_message(issues2))
        if issues2:
            logger.warning(
                "translate structure still off paper_id=%s issues=%s",
                paper_id,
                ",".join(issues2),
            )
        return retry_data

    def _translate_section_llm(
        self, section: Section, *, paper_id: str, run_id: int
    ) -> SectionTranslation:
        payload = json.dumps(
            {"title": section.title, "text": section.text},
            ensure_ascii=False,
        )
        user_content = (
            "Translate this section. Rewrite every formula as valid KaTeX LaTeX "
            "(fix broken PDF symbols). Respond with JSON "
            '{"title_zh":"...","text_zh":"..."} only.\n\n'
            f"{payload}"
        )
        data = self._translate_with_structure_retry(
            paper_id, run_id, user_content, source=section.text
        )
        title_zh = str(data.get("title_zh") or section.title).strip()
        text_zh = str(data.get("text_zh") or "").strip()
        return SectionTranslation(
            section_id=section.section_id,
            title_zh=title_zh if _looks_like_chinese(title_zh) else section.title,
            text_zh=text_zh,
        )

    def _translate_text_chunk_llm(self, text: str, *, paper_id: str, run_id: int) -> str:
        payload = json.dumps({"text": text}, ensure_ascii=False)
        user_content = (
            "Translate this text fragment from a paper section. "
            "Rewrite every formula as valid KaTeX LaTeX (fix broken PDF symbols). "
            'Respond with JSON {"text_zh":"..."} only.\n\n'
            f"{payload}"
        )
        data = self._translate_with_structure_retry(
            paper_id, run_id, user_content, source=text
        )
        return str(data.get("text_zh") or "").strip()


def _split_text_chunks(text: str, max_chars: int) -> list[TextChunk]:
    body = text.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [TextChunk(body)]

    chunks: list[TextChunk] = []
    for paragraph in re.split(r"\n\n+", body):
        piece = paragraph.strip()
        if not piece:
            continue
        if len(piece) <= max_chars:
            chunks.append(TextChunk(piece))
        else:
            chunks.extend(_hard_split_text(piece, max_chars))
    return chunks


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """$...$、$$...$$、\\begin...\\end 以及带下标的括号式，切块时不能从中间断开。"""
    spans: list[tuple[int, int]] = []
    index = 0
    length = len(text)
    while index < length:
        begin = re.match(r"\\begin\{([A-Za-z*]+)\}", text[index:])
        if begin:
            env = begin.group(1)
            start = index
            index += begin.end()
            end_token = f"\\end{{{env}}}"
            found = text.find(end_token, index)
            if found < 0:
                spans.append((start, length))
                break
            index = found + len(end_token)
            spans.append((start, index))
            continue
        if text.startswith("$$", index):
            found = text.find("$$", index + 2)
            if found < 0:
                spans.append((index, length))
                break
            spans.append((index, found + 2))
            index = found + 2
            continue
        if text[index] == "$":
            found = text.find("$", index + 1)
            if found < 0:
                spans.append((index, length))
                break
            spans.append((index, found + 1))
            index = found + 1
            continue
        if text[index] == "(":
            depth = 1
            cursor = index + 1
            while cursor < length and depth:
                if text[cursor] == "(":
                    depth += 1
                elif text[cursor] == ")":
                    depth -= 1
                cursor += 1
            blob = text[index:cursor]
            if depth == 0 and re.search(r"[_^\\|]", blob) and cursor - index <= 180:
                lead = index
                while lead > 0 and (text[lead - 1].isalnum() or text[lead - 1] in "_"):
                    lead -= 1
                spans.append((lead, cursor))
                index = cursor
                continue
        index += 1
    return spans


def _inside_protected(spans: list[tuple[int, int]], pos: int) -> tuple[int, int] | None:
    for start, end in spans:
        if start < pos < end:
            return start, end
    return None


def _safe_split_end(text: str, start: int, max_chars: int) -> int:
    limit = min(len(text), start + max_chars)
    if limit >= len(text):
        return len(text)
    spans = _protected_spans(text)
    covered = _inside_protected(spans, limit)
    if covered:
        span_start, span_end = covered
        if span_start > start:
            limit = span_start
        else:
            limit = min(len(text), max(span_end, start + 1))
            if limit - start > max_chars * 2:
                limit = min(len(text), start + max_chars)
    window = text[start:limit]
    for sep in ("\n\n", "\n", ". ", "? ", "! ", "。", " "):
        pos = window.rfind(sep)
        if pos <= 0:
            continue
        abs_pos = start + pos
        if sep != " " and sep != "。":
            abs_pos += len(sep) if sep.startswith("\n") else 0
            if sep in {". ", "? ", "! "}:
                abs_pos = start + pos + 1
        if sep == "。":
            abs_pos = start + pos + 1
        if _inside_protected(spans, abs_pos):
            continue
        if abs_pos > start:
            return abs_pos
    return limit if limit > start else min(len(text), start + max_chars)


def _hard_split_text(text: str, max_chars: int) -> list[TextChunk]:
    if len(text) <= max_chars:
        return [TextChunk(text)]
    parts: list[str] = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start] in " \t\n":
            start += 1
        if start >= len(text):
            break
        end = _safe_split_end(text, start, max_chars)
        if end <= start:
            end = min(len(text), start + max_chars)
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    return [TextChunk(part, paragraph_start=index == 0) for index, part in enumerate(parts)]


def _tidy_translation(payload: PaperTranslation) -> PaperTranslation:
    from dl_agent.knowledge.layout import tidy_translation_math

    return payload.model_copy(
        update={
            "title_zh": tidy_translation_math(payload.title_zh) if payload.title_zh else payload.title_zh,
            "sections": [
                item.model_copy(
                    update={
                        "title_zh": tidy_translation_math(item.title_zh) if item.title_zh else item.title_zh,
                        "text_zh": tidy_translation_math(item.text_zh) if item.text_zh else item.text_zh,
                    }
                )
                for item in payload.sections
            ],
            "figures": [
                item.model_copy(
                    update={
                        "caption_zh": tidy_translation_math(item.caption_zh)
                        if item.caption_zh
                        else item.caption_zh,
                    }
                )
                for item in payload.figures
            ],
        }
    )


def _is_cached_section_translation(
    item: SectionTranslation,
    section: Section | None,
) -> bool:
    if item.partial:
        return False
    if section is not None and _skip_section(section):
        return True
    if not item.text_zh.strip():
        return False
    source = section.text if section is not None else None
    return _looks_like_chinese(item.text_zh, source=source)


def _skipped_translation(
    section: Section, existing: SectionTranslation | None = None
) -> SectionTranslation:
    title_zh = section.title
    if existing and _looks_like_chinese(existing.title_zh) and existing.title_zh.strip() != section.title.strip():
        title_zh = existing.title_zh.strip()
    elif section.kind == "references" or _looks_like_references(section.title):
        title_zh = "参考文献"
    return SectionTranslation(
        section_id=section.section_id,
        title_zh=title_zh,
        text_zh=section.text,
    )


def _looks_like_references(title: str) -> bool:
    return bool(re.search(r"\breferences\b|\bbibliography\b|参考文献", title, re.I))


def _looks_like_acknowledgements(title: str) -> bool:
    return bool(re.search(r"acknowledgements?|acknowledgments?|致谢", title, re.I))


def _is_front_matter(section: Section) -> bool:
    return section.title.strip().lower() == "front matter"


_AUTHOR_HINT = re.compile(
    r"@|university|universit[aä]t|institute|laboratory|\blab\b|"
    r"corresponding author|department of|学院|大学|"
    r"e-?mail\s*:|\bare with\b|\bis with\b|ieee",
    re.I,
)


def _is_paper_title_section(section: Section) -> bool:
    if _is_front_matter(section):
        return False
    if section.level != 1 or section.page_start > 1:
        return False
    if section.kind in {"abstract", "intro", "related", "conclusion", "references"}:
        return False
    title = section.title.strip()
    if not title or re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]{1,6})[.\s:-]", title, re.I):
        return False
    if section.kind in {"method", "experiment"} and len(title) < 60:
        return False
    return bool(_AUTHOR_HINT.search(section.text or ""))


def _skip_section(section: Section) -> bool:
    if not section.text.strip():
        return True
    if _is_front_matter(section) or _is_paper_title_section(section):
        return True
    if section.kind == "references":
        return True
    if _looks_like_references(section.title) or _looks_like_acknowledgements(section.title):
        return True
    return False


def _han_count(text: str) -> int:
    return sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")


def _latin_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isascii() and ch.isalpha())


def _strip_markup(text: str) -> str:
    cleaned = _DISPLAY_MATH_RE.sub(" ", text)
    cleaned = re.sub(r"\$[^$]*\$", " ", cleaned)
    cleaned = re.sub(r"\\begin\{[A-Za-z*]+\}[\s\S]*?\\end\{[A-Za-z*]+\}", " ", cleaned)
    cleaned = _FIG_MARK_RE.sub(" ", cleaned)
    cleaned = _TABLE_LINE_RE.sub(" ", cleaned)
    return cleaned


def _paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\n+", text.strip()) if part.strip()])


def _display_math_blocks(text: str) -> list[str]:
    return _DISPLAY_MATH_RE.findall(text or "")


def _fig_markers(text: str) -> list[str]:
    return _FIG_MARK_RE.findall(text or "")


def _is_markup_heavy(text: str) -> bool:
    """公式、表格或插图标记占主体时，不能因为汉字少就整块作废。"""
    stripped = (text or "").strip()
    if not stripped:
        return False
    math_len = sum(len(block) for block in _display_math_blocks(stripped))
    table_lines = _TABLE_LINE_RE.findall(stripped)
    table_len = sum(len(line) for line in table_lines)
    fig_len = sum(len(mark) for mark in _fig_markers(stripped))
    markup = math_len + table_len + fig_len
    latin = _latin_letters(_strip_markup(stripped))
    if len(table_lines) >= 3 and latin < 120:
        return True
    if markup >= max(len(stripped), 1) * 0.45 and latin < 120:
        return True
    if latin < 40 and (math_len or table_lines or fig_len):
        return True
    return False


_FIG_LABEL_RE = re.compile(
    r"^(?:fig(?:ure)?|algorithm|图|算法)\s*\.?\s*([A-Z]?\d+)\b",
    re.I,
)
_FIG_CAPTION_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:图|算法)\s*\.?\s*([A-Z]?\d+)\s*[.:：]\s*(\S[^\n]*)",
)


def _figure_id_aliases(figures: list[Figure]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for figure in figures:
        aliases[figure.figure_id.lower()] = figure.figure_id
        digits = re.search(r"(\d+)$", figure.figure_id)
        if digits:
            num = digits.group(1)
            aliases[num] = figure.figure_id
            aliases[num.lstrip("0") or "0"] = figure.figure_id
            aliases[f"fig-{num}"] = figure.figure_id
            aliases[f"fig-{int(num)}"] = figure.figure_id
        key = _figure_ref_key(figure)
        if key:
            aliases[key] = figure.figure_id
            num = key.split()[-1]
            aliases[num] = figure.figure_id
            aliases[f"figure {num}"] = figure.figure_id
            aliases[f"fig. {num}"] = figure.figure_id
            aliases[f"fig {num}"] = figure.figure_id
            aliases[f"图 {num}"] = figure.figure_id
            aliases[f"图{num}"] = figure.figure_id
    return aliases


def _resolve_figure_id(raw: str, aliases: dict[str, str]) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    lowered = re.sub(r"[_:]+", " ", text.lower()).strip()
    lowered = re.sub(r"\s+", " ", lowered)
    compact = lowered.replace(" ", "")
    if text in aliases:
        return aliases[text]
    for candidate in (lowered, compact, text.lower()):
        if candidate in aliases:
            return aliases[candidate]
    match = _FIG_LABEL_RE.match(text.strip())
    if match:
        num = match.group(1).lower()
        return aliases.get(num) or aliases.get(f"figure {num}")
    digits = re.search(r"(\d+)$", compact)
    if digits:
        return aliases.get(digits.group(1)) or aliases.get(digits.group(1).lstrip("0") or "0")
    return None


def _caption_zh_from_item(item: dict) -> str:
    for key in ("caption_zh", "caption", "text_zh", "zh", "translation"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_figure_caption_payload(
    data: dict, figures: list[Figure]
) -> dict[str, FigureTranslation]:
    if not figures:
        return {}
    aliases = _figure_id_aliases(figures)
    raw_items = data.get("captions") if isinstance(data, dict) else None
    if isinstance(raw_items, dict):
        items: list[object] = [
            {"figure_id": key, "caption_zh": value} for key, value in raw_items.items()
        ]
    elif isinstance(raw_items, list):
        items = raw_items
    elif isinstance(data, dict) and any(
        key in data for key in ("figure_id", "caption_zh", "caption")
    ):
        items = [data]
    else:
        items = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key in {"title_zh", "text_zh", "status"}:
                    continue
                if isinstance(value, str) and _han_count(value) >= 1:
                    items.append({"figure_id": key, "caption_zh": value})
    out: dict[str, FigureTranslation] = {}
    for index, item in enumerate(items):
        figure_id: str | None = None
        caption_zh = ""
        if isinstance(item, str):
            figure_id = figures[index].figure_id if index < len(figures) else None
            caption_zh = item.strip()
        elif isinstance(item, dict):
            raw_id = item.get("figure_id") or item.get("id") or item.get("figureId") or ""
            figure_id = _resolve_figure_id(str(raw_id), aliases)
            if figure_id is None and index < len(figures):
                figure_id = figures[index].figure_id
            caption_zh = _caption_zh_from_item(item)
        if not figure_id or not caption_zh:
            continue
        if _han_count(caption_zh) < 1:
            continue
        out[figure_id] = FigureTranslation(figure_id=figure_id, caption_zh=caption_zh)
    return out


def _captioned_figures(figures: list[Figure]) -> list[Figure]:
    return [
        figure
        for figure in figures
        if figure.kind != "formula" and (figure.caption or "").strip()
    ]


def _figure_ref_key(figure: Figure) -> str | None:
    for text in (figure.label, figure.caption):
        if not text:
            continue
        match = _FIG_LABEL_RE.match(text.strip())
        if not match:
            continue
        kind = "algorithm" if re.match(r"^(algorithm|算法)", text.strip(), re.I) else "figure"
        return f"{kind} {match.group(1).lower()}"
    return None


def _harvest_figure_captions(texts: list[str], figures: list[Figure]) -> dict[str, str]:
    by_key: dict[str, str] = {}
    for text in texts:
        if not text:
            continue
        for match in _FIG_CAPTION_LINE_RE.finditer(text):
            num = match.group(1).lower()
            body = match.group(2).strip()
            if not _looks_like_chinese(body):
                continue
            blob = match.group(0).lstrip()
            prefix = "算法" if blob.startswith("算法") else "图"
            key_kind = "algorithm" if prefix == "算法" else "figure"
            by_key.setdefault(f"{key_kind} {num}", f"{prefix} {match.group(1)}. {body}")
    out: dict[str, str] = {}
    for figure in figures:
        key = _figure_ref_key(figure)
        if key and key in by_key:
            out[figure.figure_id] = by_key[key]
    return out


def _looks_like_chinese(text: str, *, source: str | None = None) -> bool:
    """至少含汉字，避免把英文原文当成译文。公式/表格块按原文结构放宽。"""
    if not text.strip():
        return False
    prose = _strip_markup(text)
    han = _han_count(prose)
    if han >= 4:
        return True
    if han >= 2 and len(text.strip()) <= 40:
        return True
    if source is not None and _is_markup_heavy(source):
        return True
    return False


def _is_forbidden_verbatim(source: str, translated: str) -> bool:
    if translated.strip() != source.strip():
        return False
    if _is_markup_heavy(source):
        return False
    return _latin_letters(_strip_markup(source)) >= 40


def _structure_issues(source: str, translated: str) -> list[str]:
    issues: list[str] = []
    src_math = _display_math_blocks(source)
    zh_math = _display_math_blocks(translated)
    if len(src_math) != len(zh_math):
        issues.append(f"display-math {len(zh_math)}!={len(src_math)}")
    src_paras = _paragraph_count(source)
    zh_paras = _paragraph_count(translated)
    if src_paras > 1 and zh_paras < src_paras:
        issues.append(f"paragraphs {zh_paras}<{src_paras}")
    if _fig_markers(source) != _fig_markers(translated):
        issues.append("figure-markers")
    return issues


def _repair_and_check(source: str, text_zh: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    if not text_zh.strip():
        return text_zh, ["empty"]
    if _is_forbidden_verbatim(source, text_zh):
        issues.append("verbatim")
    if not _looks_like_chinese(text_zh, source=source):
        issues.append("not_chinese")
    issues.extend(_structure_issues(source, text_zh))
    return text_zh, issues


def _fatal_translation_issues(source: str, issues: list[str]) -> bool:
    if "empty" in issues:
        return True
    markup = _is_markup_heavy(source)
    if "not_chinese" in issues and not markup:
        return True
    if "verbatim" in issues and not markup:
        return True
    return False


def _can_keep_translation(source: str, text_zh: str, issues: list[str]) -> bool:
    if not text_zh.strip() or "empty" in issues:
        return False
    if "verbatim" in issues and not _is_markup_heavy(source):
        return False
    return _looks_like_chinese(text_zh, source=source)


def _fatal_message(issues: list[str]) -> str:
    if "empty" in issues:
        return "翻译结果缺少 text_zh"
    if "not_chinese" in issues:
        return "翻译结果仍是英文，未得到简体中文"
    if "verbatim" in issues:
        return "翻译结果与英文原文相同"
    return "翻译结构与原文不一致"


def _structure_retry_prompt(source: str) -> str:
    return (
        "Retry: match paragraph breaks and $$ count exactly. "
        "The previous translation omitted sentences or broke structure. "
        f"Required paragraph breaks (\\n\\n groups): {_paragraph_count(source)}. "
        f"Required $$ display-math blocks: {len(_display_math_blocks(source))}. "
        "Rewrite every formula as valid KaTeX LaTeX; fix OCR/PDF symbol errors. "
        "Do not copy broken math from the source. Do not wrap Chinese prose in $...$. "
        "Translate ALL prose; do not summarize."
    )


def _parse_json_object(raw: str) -> dict:
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise LlmRequestError("翻译结果不是 JSON 对象")
    return data


_service: TranslateService | None = None


def get_translate_service() -> TranslateService:
    global _service
    if _service is None:
        from dl_agent.knowledge.service import get_service

        settings = get_settings()
        knowledge = get_service()
        store = knowledge.store
        _service = TranslateService(store, knowledge, settings)
    return _service


def reset_translate_service(service: TranslateService | None = None) -> None:
    global _service
    _service = service
