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
from dl_agent.domain.models import PaperTranslation, Section, SectionTranslation, TranslateStatus
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
- Preserve LaTeX math ($...$, $$...$$) exactly unchanged
- Preserve markdown table syntax, list markers, and <!--fig:fig-N--> markers exactly
- Preserve author names, affiliations, and email addresses exactly unchanged
- Preserve model names, dataset names, method names, and citation markers like [1] where appropriate
- Keep paragraph breaks and structure identical to the source
- Return ONLY valid JSON, no markdown fences"""


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
        self.chat_fn = chat_fn or (lambda messages: chat(messages, settings=self.settings))
        self._lock = threading.Lock()
        self._running: set[str] = set()
        self._run_seq: dict[str, int] = {}
        self._active_runs: set[tuple[str, int]] = set()

    def get_translation(self, paper_id: str) -> PaperTranslation | None:
        self.knowledge.get_paper(paper_id)
        cached = self.store.get_translation(paper_id)
        if cached is None:
            return None
        # pending 即使版本旧也要返回，否则前端轮询会 404
        if cached.status == "pending":
            return cached
        if cached.prompt_version == self.settings.translate_prompt_version:
            return cached
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
                return cached, False, run_id
            if cached.status == "pending":
                # 进程仍在跑 → 只轮询；reload 后 BackgroundTask 会丢 → POST 可续跑
                if paper_id in self._running:
                    return cached, False, run_id
                logger.warning(
                    "translate stale pending paper_id=%s sections=%s version=%s; re-queue",
                    paper_id,
                    len(cached.sections),
                    cached.prompt_version,
                )
                return cached, True, run_id
            # failed / 旧版 ready：默认允许再次 POST 自动重试

        with self._lock:
            paper = self.knowledge.get_paper(paper_id)
            running = self.store.get_translation(paper_id)
            if (
                running
                and running.status == "pending"
                and running.prompt_version == self.settings.translate_prompt_version
                and not refresh
                and paper_id in self._running
            ):
                return running, False, self._run_seq.get(paper_id, 0)
            run_id = self._run_seq.get(paper_id, 0) + 1
            self._run_seq[paper_id] = run_id
            if refresh:
                logger.info("translate refresh paper_id=%s run_id=%s", paper_id, run_id)
            paper.translate_status = "pending"
            self.store.save_paper(paper)
            pending = PaperTranslation(
                paper_id=paper_id,
                status="pending",
                model=self.settings.model_name,
                prompt_version=self.settings.translate_prompt_version,
                title_zh=None,
                sections=[],
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
            return self._finish_translation_locked(paper_id, run_id)
        finally:
            self._running.discard(paper_id)
            with self._lock:
                self._active_runs.discard((paper_id, run_id))

    def _is_stale_run(self, paper_id: str, run_id: int) -> bool:
        return self._run_seq.get(paper_id, 0) != run_id

    def _finish_translation_locked(self, paper_id: str, run_id: int) -> PaperTranslation:
        paper = self.knowledge.get_paper(paper_id)
        sections = self.knowledge.get_sections(paper_id)
        section_by_id = {section.section_id: section for section in sections}
        existing = self.store.get_translation(paper_id)
        translated_by_id: dict[str, SectionTranslation] = {}
        failed = False
        fatal = False
        fail_message: str | None = None
        llm_ok = 0
        title_zh: str | None = existing.title_zh if existing else None
        started = time.perf_counter()
        parallelism = max(1, self.settings.translate_parallelism)
        chunk_max = max(400, self.settings.translate_chunk_max_chars)

        if existing:
            for item in existing.sections:
                section = section_by_id.get(item.section_id)
                if section and _is_front_matter(section):
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

        if paper.title and paper.title.strip() and not title_zh:
            if self._is_stale_run(paper_id, run_id):
                existing = self.store.get_translation(paper_id)
                return existing or self._finalize_translation(
                    paper_id, {}, sections, title_zh=None, status="pending", fail_message=None, llm_ok=0, started=started
                )
            try:
                title_zh = self._translate_paper_title_llm(paper.title)
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

        if fatal:
            return self._finalize_translation(
                paper_id,
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
                    translated_by_id[section.section_id] = _skipped_translation(section)
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

        if failed and (fatal or llm_ok == 0):
            status: TranslateStatus = "failed"
        elif failed:
            status = "partial"
        else:
            status = "ready"

        return self._finalize_translation(
            paper_id,
            translated_by_id,
            sections,
            title_zh=title_zh,
            status=status,
            fail_message=fail_message,
            llm_ok=llm_ok,
            started=started,
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
            if stop.is_set() or self._is_stale_run(paper_id, run_id):
                raise LlmRequestError("翻译已中止", fatal=False)
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
                if self._is_stale_run(paper_id, run_id):
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
                chunk_max=chunk_max,
                existing=existing if existing and existing.partial else None,
                on_progress=on_progress,
                is_stale=lambda: self._is_stale_run(paper_id, run_id),
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

        if failed_sections and not fatal:
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
                    if self._is_stale_run(paper_id, run_id):
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
                        chunk_max=chunk_max,
                        existing=translated_by_id.get(section.section_id),
                        on_progress=on_retry_progress,
                        is_stale=lambda: self._is_stale_run(paper_id, run_id),
                    )
                except LlmNotConfiguredError as exc:
                    failed = True
                    fatal = True
                    fail_message = str(exc)
                    break
                except LlmRequestError as exc:
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
            item = self._translate_section_llm(section)
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
                raise LlmRequestError("翻译已中止", fatal=False)
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
                    )
                )
                title_zh = part.title_zh
                piece = part.text_zh
            else:
                piece = self._translate_text_chunk_llm(chunk)
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
                time.sleep(self.settings.translate_section_delay_s)
        return item

    def _finalize_translation(
        self,
        paper_id: str,
        translated_by_id: dict[str, SectionTranslation],
        sections: list[Section],
        *,
        title_zh: str | None,
        status: TranslateStatus,
        fail_message: str | None,
        llm_ok: int,
        started: float,
    ) -> PaperTranslation:
        ordered = [
            translated_by_id[section.section_id]
            for section in sections
            if section.section_id in translated_by_id
        ]
        result = PaperTranslation(
            paper_id=paper_id,
            status=status,
            model=self.settings.model_name,
            prompt_version=self.settings.translate_prompt_version,
            title_zh=title_zh,
            sections=ordered,
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
        if self._is_stale_run(paper_id, run_id):
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
            self.store.save_translation(
                PaperTranslation(
                    paper_id=paper_id,
                    status=status,
                    model=self.settings.model_name,
                    prompt_version=self.settings.translate_prompt_version,
                    title_zh=title_zh,
                    sections=list(sections),
                    error=None,
                )
            )

    def _translate_paper_title_llm(self, title: str) -> str:
        if _looks_like_chinese(title):
            return title.strip()
        payload = json.dumps({"title": title}, ensure_ascii=False)
        raw = self.chat_fn(
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
            ]
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

    def _translate_section_llm(self, section: Section) -> SectionTranslation:
        payload = json.dumps(
            {"title": section.title, "text": section.text},
            ensure_ascii=False,
        )
        raw = self.chat_fn(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Translate this section. Respond with JSON "
                        '{"title_zh":"...","text_zh":"..."} only.\n\n'
                        f"{payload}"
                    ),
                },
            ]
        )
        data = _parse_json_object(raw)
        title_zh = str(data.get("title_zh") or section.title).strip()
        text_zh = str(data.get("text_zh") or "").strip()
        if not text_zh:
            raise LlmRequestError("翻译结果缺少 text_zh")
        if not _looks_like_chinese(text_zh):
            raise LlmRequestError("翻译结果仍是英文，未得到简体中文")
        if text_zh == section.text.strip():
            raise LlmRequestError("翻译结果与英文原文相同")
        return SectionTranslation(
            section_id=section.section_id,
            title_zh=title_zh if _looks_like_chinese(title_zh) else section.title,
            text_zh=text_zh,
        )

    def _translate_text_chunk_llm(self, text: str) -> str:
        payload = json.dumps({"text": text}, ensure_ascii=False)
        raw = self.chat_fn(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Translate this text fragment from a paper section. "
                        'Respond with JSON {"text_zh":"..."} only.\n\n'
                        f"{payload}"
                    ),
                },
            ]
        )
        data = _parse_json_object(raw)
        text_zh = str(data.get("text_zh") or "").strip()
        if not text_zh:
            raise LlmRequestError("翻译结果缺少 text_zh")
        if not _looks_like_chinese(text_zh):
            raise LlmRequestError("翻译结果仍是英文，未得到简体中文")
        if text_zh == text.strip():
            raise LlmRequestError("翻译结果与英文原文相同")
        return text_zh


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


def _hard_split_text(text: str, max_chars: int) -> list[TextChunk]:
    if len(text) <= max_chars:
        return [TextChunk(text)]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        piece = text[start:end].strip()
        if piece:
            parts.append(piece)
        start = end
    return [TextChunk(part, paragraph_start=index == 0) for index, part in enumerate(parts)]


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
    return _looks_like_chinese(item.text_zh)


def _skipped_translation(section: Section) -> SectionTranslation:
    title_zh = section.title
    if section.kind == "references" or _looks_like_references(section.title):
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


def _skip_section(section: Section) -> bool:
    if not section.text.strip():
        return True
    if _is_front_matter(section):
        return True
    if section.kind == "references":
        return True
    if _looks_like_references(section.title) or _looks_like_acknowledgements(section.title):
        return True
    return False


def _looks_like_chinese(text: str) -> bool:
    """至少含汉字，避免把英文原文当成译文。"""
    if not text.strip():
        return False
    # 去掉公式与常见符号后再判断
    cleaned = re.sub(r"\$\$[\s\S]*?\$\$|\$[^$]+\$|<!--fig:fig-\d+-->", " ", text)
    han = sum(1 for ch in cleaned if "\u4e00" <= ch <= "\u9fff")
    if han >= 4:
        return True
    # 短标题如「摘要」
    return han >= 2 and len(text.strip()) <= 40


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
