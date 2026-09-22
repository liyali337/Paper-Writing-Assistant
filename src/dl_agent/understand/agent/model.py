"""用现有 harness.chat 驱动主图 / Worker（测试可注入 chat_fn）。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from dl_agent.config import Settings
from dl_agent.domain.models import Evidence
from dl_agent.harness.complete import ChatTurn, LlmRequestError, uses_gemini_native
from dl_agent.understand.agent.nodes import (
    ToolCall,
    WorkerReply,
    build_aggregate_messages,
    build_compress_messages,
    build_dialogue_compress_messages,
    build_dialogue_messages,
    build_fallback_messages,
    build_orchestrator_messages,
    parse_dialogue_act,
    parse_orchestrator_reply,
)
from dl_agent.understand.agent.schemas import DialogueAct, QueryAnalysis
from dl_agent.understand.agent.tools import worker_tool_schemas

logger = logging.getLogger(__name__)

ChatTurnFn = Callable[..., ChatTurn]


class ChatAskModel:
    def __init__(self, chat_fn, settings: Settings, chat_turn_fn: ChatTurnFn | None = None):
        self.chat_fn = chat_fn
        self.chat_turn_fn = chat_turn_fn
        self.settings = settings

    def dialogue(self, query: str, recent_turns: str) -> DialogueAct:
        raw = self.chat_fn(build_dialogue_messages(query, recent_turns), settings=self.settings)
        return parse_dialogue_act(raw)

    def compress_dialogue(self, older_turns: str, query: str) -> str:
        return self.chat_fn(
            build_dialogue_compress_messages(older_turns, query),
            settings=self.settings,
        ).strip()

    def rewrite(self, query: str, summary: str) -> QueryAnalysis:
        act = self.dialogue(query, summary)
        questions = [item.question for item in act.tasks if item.kind == "close_read"]
        return QueryAnalysis(
            is_clear=act.intent == "retrieve",
            questions=questions,
            clarification_needed=act.reply if act.intent == "clarify" else "",
        )

    def orchestrate(self, messages, question, context_summary) -> WorkerReply:
        if self._use_native_tools():
            packed = build_orchestrator_messages(
                question,
                context_summary,
                messages,
                native=True,
                enable_external=self.settings.ask_enable_external,
            )
            try:
                turn = self.chat_turn_fn(
                    packed,
                    settings=self.settings,
                    tools=worker_tool_schemas(enable_external=self.settings.ask_enable_external),
                )
            except LlmRequestError as exc:
                if getattr(exc, "status_code", None) == 400:
                    logger.warning("native function calling 返回 400，回落 json 协议")
                    return self._orchestrate_json(messages, question, context_summary)
                raise
            return _reply_from_turn(turn)
        return self._orchestrate_json(messages, question, context_summary)

    def compress(self, conversation_text: str) -> str:
        return self.chat_fn(build_compress_messages(conversation_text), settings=self.settings).strip()

    def fallback(self, question: str, context_text: str) -> str:
        return self.chat_fn(build_fallback_messages(question, context_text), settings=self.settings).strip()

    def aggregate(self, original: str, answers: list[dict[str, Any]], evidence: list[Evidence]) -> str:
        return self.chat_fn(
            build_aggregate_messages(original, answers, evidence),
            settings=self.settings,
        )

    def _use_native_tools(self) -> bool:
        if self.settings.ask_tool_protocol != "native":
            return False
        if self.chat_turn_fn is None:
            return False
        return not uses_gemini_native(self.settings.openai_base_url, self.settings.openai_api_key)

    def _orchestrate_json(self, messages, question, context_summary) -> WorkerReply:
        packed = build_orchestrator_messages(
            question,
            context_summary,
            messages,
            native=False,
            enable_external=self.settings.ask_enable_external,
        )
        text = self.chat_fn(packed, settings=self.settings)
        return parse_orchestrator_reply(text)


def _reply_from_turn(turn: ChatTurn) -> WorkerReply:
    calls = [
        ToolCall(name=item.name, args=dict(item.arguments or {}), id=item.id)
        for item in turn.tool_calls
        if item.name
    ]
    if calls:
        return WorkerReply(content=turn.content or "", tool_calls=calls)
    return parse_orchestrator_reply(turn.content or "")


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
