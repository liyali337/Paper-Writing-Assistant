from typing import Literal

from pydantic import BaseModel, Field

AskTaskKind = Literal["close_read", "library", "arxiv"]
DialogueIntent = Literal["chat", "clarify", "retrieve"]


class AskTask(BaseModel):
    kind: AskTaskKind = "close_read"
    question: str = ""


class QueryAnalysis(BaseModel):
    is_clear: bool = True
    questions: list[str] = Field(default_factory=list)
    clarification_needed: str = ""


class DialogueAct(BaseModel):
    intent: DialogueIntent = "retrieve"
    kinds: list[AskTaskKind] = Field(default_factory=list)
    tasks: list[AskTask] = Field(default_factory=list)
    reply: str = ""
