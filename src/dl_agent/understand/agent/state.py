"""Worker 子图与主图状态。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from dl_agent.domain.models import Evidence, ExternalRef, LibraryHit


def _append_messages(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    left = list(left or [])
    right = list(right or [])
    if right and right[0].get("__reset__"):
        return right[1:]
    return left + right


def _unique_keys(left: list[str] | None, right: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(left or []) + list(right or []):
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _merge_evidence(left: list[Evidence] | None, right: list[Evidence] | None) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[tuple[str | None, str | None, str]] = set()
    for item in list(left or []) + list(right or []):
        key = (item.chunk_id, item.section_id, item.quote)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _merge_external_refs(
    left: list[ExternalRef] | None,
    right: list[ExternalRef] | None,
) -> list[ExternalRef]:
    out: list[ExternalRef] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in list(left or []) + list(right or []):
        key = (item.source, item.identifier or "", item.url or "", item.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


class WorkerState(TypedDict, total=False):
    paper_id: str
    question: str
    question_index: int
    messages: Annotated[list[dict[str, Any]], _append_messages]
    context_summary: str
    retrieval_keys: Annotated[list[str], _unique_keys]
    evidence_bag: Annotated[list[Evidence], _merge_evidence]
    external_refs: Annotated[list[ExternalRef], _merge_external_refs]
    final_answer: str
    tool_call_count: Annotated[int, operator.add]
    iteration_count: Annotated[int, operator.add]


def _append_answers(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    left = list(left or [])
    right = list(right or [])
    if right and any(item.get("__reset__") for item in right):
        return [item for item in right if not item.get("__reset__")]
    return left + right


def _merge_library_hits(left: list[LibraryHit] | None, right: list[LibraryHit] | None) -> list[LibraryHit]:
    out: list[LibraryHit] = []
    seen: set[str] = set()
    for item in list(left or []) + list(right or []):
        if item.paper_id in seen:
            continue
        seen.add(item.paper_id)
        out.append(item)
    return out


class AskState(TypedDict, total=False):
    paper_id: str
    originalQuery: str
    history: list[Any]
    conversation_summary: str
    dialogue_intent: str
    skip_workers: bool
    rewrittenQuestions: list[str]
    plan_tasks: list[dict[str, Any]]
    index_status: str
    agent_answers: Annotated[list[dict[str, Any]], _append_answers]
    answer_zh: str
    citations: list[Evidence]
    external_refs: list[ExternalRef]
    library_hits: list[LibraryHit]
    mode: str
    no_evidence: bool
    partial: bool
