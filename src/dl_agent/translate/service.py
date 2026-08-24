"""章节全文翻译。"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Callable

from dl_agent.config import Settings, get_settings
from dl_agent.domain.models import PaperTranslation, Section, SectionTranslation, TranslateStatus
from dl_agent.harness.complete import LlmNotConfiguredError, LlmRequestError, chat
from dl_agent.knowledge.service import KnowledgeService, PaperNotFoundError
from dl_agent.knowledge.store import FilePaperStore

logger = logging.getLogger(__name__)

ChatFn = Callable[[list[dict[str, str]]], str]

SYSTEM_PROMPT = """You are an academic paper translator. Translate English academic text into Simplified Chinese.

Rules:
- Preserve LaTeX math ($...$, $$...$$) exactly unchanged
- Preserve markdown table syntax, list markers, and <!--fig:fig-N--> markers exactly
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

    def get_translation(self, paper_id: str) -> PaperTranslation | None:
        self.knowledge.get_paper(paper_id)
        cached = self.store.get_translation(paper_id)
        if cached and cached.prompt_version == self.settings.translate_prompt_version:
            return cached
        return None

    def start_translation(self, paper_id: str, *, refresh: bool = False) -> tuple[PaperTranslation, bool]:
        if not self.settings.openai_api_key.strip():
            raise LlmNotConfiguredError("未配置 OPENAI_API_KEY，无法调用翻译模型")
        paper = self.knowledge.get_paper(paper_id)
        if paper.status != "ready":
            raise ValueError("paper_not_ready")

        cached = self.get_translation(paper_id)
        if cached and not refresh:
            if cached.status in {"ready", "partial", "failed"}:
                return cached, False
            if cached.status == "pending":
                return cached, False

        with self._lock:
            paper = self.knowledge.get_paper(paper_id)
            running = self.store.get_translation(paper_id)
            if running and running.status == "pending" and not refresh:
                return running, False
            paper.translate_status = "pending"
            self.store.save_paper(paper)
            pending = PaperTranslation(
                paper_id=paper_id,
                status="pending",
                model=self.settings.model_name,
                prompt_version=self.settings.translate_prompt_version,
                sections=[],
            )
            self.store.save_translation(pending)
            return pending, True

    def finish_translation(self, paper_id: str) -> PaperTranslation:
        sections = self.knowledge.get_sections(paper_id)
        translated: list[SectionTranslation] = []
        failed = False
        started = time.perf_counter()

        for index, section in enumerate(sections):
            try:
                item = self._translate_section(section)
                translated.append(item)
            except (LlmNotConfiguredError, LlmRequestError) as exc:
                logger.warning(
                    "translate section failed paper_id=%s section_id=%s error=%s",
                    paper_id,
                    section.section_id,
                    exc,
                )
                failed = True
                break
            except Exception:
                logger.exception(
                    "translate section failed paper_id=%s section_id=%s",
                    paper_id,
                    section.section_id,
                )
                failed = True
                break
            if index + 1 < len(sections):
                time.sleep(self.settings.translate_section_delay_s)

        status: TranslateStatus
        if failed and not translated:
            status = "failed"
        elif failed:
            status = "partial"
        else:
            status = "ready"

        result = PaperTranslation(
            paper_id=paper_id,
            status=status,
            model=self.settings.model_name,
            prompt_version=self.settings.translate_prompt_version,
            sections=translated,
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
                "section_count": len(translated),
                "model": self.settings.model_name,
            },
        )
        return result

    def _translate_section(self, section: Section) -> SectionTranslation:
        if section.kind == "references" or not section.text.strip():
            return SectionTranslation(
                section_id=section.section_id,
                title_zh=section.title,
                text_zh=section.text,
            )
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
        return SectionTranslation(
            section_id=section.section_id,
            title_zh=title_zh,
            text_zh=text_zh,
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
