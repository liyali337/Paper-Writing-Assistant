"""本地知识库问答：跨篇检索卡片，不进本篇 citations。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from dl_agent.config import Settings
from dl_agent.domain.models import AskTurn, LibraryHit, PaperAnswer
from dl_agent.harness.complete import LlmRequestError
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.understand.citations import sanitize_answer_zh

logger = logging.getLogger(__name__)

ChatFn = Callable[..., str]

NO_LIBRARY_HITS = "NO_LIBRARY_HITS"
NO_PAPER_CARD = "NO_PAPER_CARD"
NO_LIBRARY_ANSWER = "本地知识库里没有找到相关论文。可以换个关键词，或先上传并完成索引。"
ABSTRACT_CARD_MAX = 800
ABSTRACT_FULL_MAX = 4000
LIBRARY_PAPER_ID = "library"

LIBRARY_SYSTEM = """你是本地论文库助手。用户正在精读某一篇，但这个问题是在问自己研读库里还有没有相关论文。
根据编号论文卡片（标题、摘要、命中原因）用中文回答。

硬约束：
- 只用卡片里出现的信息；不要编造实验数字、模块名或数据集。
- 细节不足就写「需打开该篇精读」。
- 不要用 [n] 当作当前 PDF 的页码引用。
- 不要把当前正在读的那一篇再推荐一遍。
- 中文讲解；术语、模型名、数据集名保留英文。
- 只输出一个 JSON 对象：{"answer_zh":"...","used_papers":[1,2]}，不要 Markdown 围栏。
used_papers 是卡片编号，可省略。"""


class LibraryTools:
    """模型只能传 query / paper_id；范围是已索引的本地库，不是当前打开的一篇。"""

    def __init__(
        self,
        knowledge: KnowledgeService,
        *,
        default_limit: int | None = None,
    ):
        self.knowledge = knowledge
        self.default_limit = default_limit
        self.last_hits: list[LibraryHit] = []
        self.last_card: LibraryHit | None = None

    def search_library(
        self,
        query: str,
        limit: int | None = None,
        exclude_paper_id: str | None = None,
    ) -> str:
        question = (query or "").strip()
        if not question:
            self.last_hits = []
            return NO_LIBRARY_HITS
        k = limit if limit is not None else self.default_limit
        try:
            hits = self.knowledge.query_corpus(question, k=k, exclude_paper_id=exclude_paper_id)
        except Exception as exc:
            self.last_hits = []
            return f"LIBRARY_SEARCH_ERROR: {exc}"
        if not hits:
            self.last_hits = []
            return NO_LIBRARY_HITS
        self.last_hits = hits
        return "\n\n".join(_format_card(index, item, abstract_max=ABSTRACT_CARD_MAX) for index, item in enumerate(hits, start=1))

    def get_paper_card(self, paper_id: str) -> str:
        pid = (paper_id or "").strip()
        if not pid:
            self.last_card = None
            return NO_PAPER_CARD
        try:
            hit = self.knowledge.paper_card(pid)
        except Exception as exc:
            self.last_card = None
            return f"LIBRARY_CARD_ERROR: {exc}"
        if hit is None:
            self.last_card = None
            return NO_PAPER_CARD
        self.last_card = hit
        return _format_card(1, hit, abstract_max=ABSTRACT_FULL_MAX)


def ask_library(
    knowledge: KnowledgeService,
    question: str,
    history: list[AskTurn] | None = None,
    *,
    settings: Settings,
    chat_fn: ChatFn,
    exclude_paper_id: str | None = None,
    current_paper_id: str | None = None,
) -> PaperAnswer:
    question = (question or "").strip()
    if not question:
        raise ValueError("问题不能为空")
    version = settings.library_prompt_version
    model = settings.model_name
    tools = LibraryTools(knowledge, default_limit=settings.library_paper_k)
    observation = tools.search_library(question, exclude_paper_id=exclude_paper_id)
    hits = list(tools.last_hits)
    for hit in hits:
        if not (hit.abstract or "").strip():
            extra = tools.get_paper_card(hit.paper_id)
            if tools.last_card is not None and extra != NO_PAPER_CARD:
                hit.abstract = tools.last_card.abstract
    answer_paper_id = current_paper_id or LIBRARY_PAPER_ID
    if not hits:
        return PaperAnswer(
            paper_id=answer_paper_id,
            question=question,
            answer_zh=NO_LIBRARY_ANSWER,
            library_hits=[],
            mode="library",
            no_evidence=True,
            model=model,
            prompt_version=version,
        )
    raw = chat_fn(
        [
            {"role": "system", "content": LIBRARY_SYSTEM},
            {
                "role": "user",
                "content": _user_prompt(
                    question,
                    observation,
                    history,
                    settings.ask_history_turns,
                    exclude_paper_id=exclude_paper_id,
                ),
            },
        ],
        settings=settings,
    )
    payload = _parse_json_object(raw)
    answer = sanitize_answer_zh(
        str(payload.get("answer_zh") or "").strip() or "模型没有给出可用回答，请换个问法重试。"
    )
    ordered = _order_hits(hits, payload.get("used_papers"))
    return PaperAnswer(
        paper_id=answer_paper_id,
        question=question,
        answer_zh=answer,
        library_hits=ordered,
        mode="library",
        model=model,
        prompt_version=version,
    )


def _format_card(index: int, hit: LibraryHit, *, abstract_max: int) -> str:
    title = (hit.title or "").strip() or hit.filename or hit.paper_id
    abstract = (hit.abstract or "").strip() or "(no abstract)"
    if len(abstract) > abstract_max:
        abstract = abstract[:abstract_max].rstrip() + "…"
    lines = [
        f"[{index}] paper_id={hit.paper_id} openable={str(hit.openable).lower()} title={title}",
    ]
    if hit.authors:
        lines.append("authors: " + ", ".join(hit.authors[:8]))
    lines.append(f"abstract: {abstract}")
    if (hit.why or "").strip():
        lines.append(f"why: {hit.why.strip()}")
    return "\n".join(lines)


def _user_prompt(
    question: str,
    observation: str,
    history: list[AskTurn] | None,
    history_turns: int,
    *,
    exclude_paper_id: str | None = None,
) -> str:
    lines = ["[Library cards]", observation, ""]
    if exclude_paper_id:
        lines.append(f"[Current paper] paper_id={exclude_paper_id}（正在精读，不要再推荐这一篇）")
        lines.append("")
    recent = _recent_history(history, history_turns)
    if recent:
        lines.append("[Recent history]")
        for turn in recent:
            prefix = "Q" if turn.role == "user" else "A"
            content = re.sub(r"\s+", " ", turn.content).strip()[:400]
            lines.append(f"{prefix}: {content}")
        lines.append("")
    lines.append("[User question]")
    lines.append(question)
    return "\n".join(lines)


def _recent_history(history: list[AskTurn] | None, turns: int) -> list[AskTurn]:
    if not history or turns <= 0:
        return []
    cleaned = [item for item in history if item.content.strip()]
    return cleaned[-max(turns, 1) * 2 :]


def _order_hits(hits: list[LibraryHit], used: Any) -> list[LibraryHit]:
    if not isinstance(used, list) or not hits:
        return hits
    seen: set[int] = set()
    ordered: list[LibraryHit] = []
    for item in used:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > len(hits) or index in seen:
            continue
        seen.add(index)
        ordered.append(hits[index - 1])
    for index, hit in enumerate(hits, start=1):
        if index not in seen:
            ordered.append(hit)
    return ordered


def _parse_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LlmRequestError("问答结果不是合法 JSON") from exc
    if not isinstance(data, dict):
        raise LlmRequestError("问答结果不是 JSON 对象")
    return data
