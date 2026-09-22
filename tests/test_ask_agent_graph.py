import json
from pathlib import Path

import pytest

from dl_agent.config import Settings
from dl_agent.domain.models import AskTurn, Evidence, ExternalRef, Paper
from dl_agent.knowledge.service import IndexNotReadyError, KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.understand.agent.graph import run_ask
from dl_agent.understand.agent.nodes import (
    EARLIER_COMPRESSED_MARKER,
    pack_dialogue_memory,
)
from dl_agent.understand.agent.prompts import CHAT_IDENTITY_REPLY, NO_EVIDENCE_ANSWER
from dl_agent.understand.agent.schemas import AskTask, DialogueAct, QueryAnalysis
from dl_agent.understand.service import PaperNotReadyError, UnderstandService


class ScriptedAskModel:
    def __init__(self, analysis: QueryAnalysis | DialogueAct, *, aggregate_text: str | None = None):
        self.analysis = analysis
        self.aggregate_text = aggregate_text or json.dumps(
            {"answer_zh": "合成回答", "used_evidence": [1], "partial": False},
            ensure_ascii=False,
        )
        self.summarize_calls = 0
        self.rewrite_calls = 0
        self.dialogue_calls = 0
        self.compress_dialogue_calls = 0
        self.aggregate_calls = 0
        self.last_recent_turns = ""
        self.last_older_turns = ""

    def summarize(self, conversation: str) -> str:
        self.summarize_calls += 1
        return ""

    def compress_dialogue(self, older_turns: str, query: str) -> str:
        self.compress_dialogue_calls += 1
        self.last_older_turns = older_turns
        return "焦点：此前讨论的方法；术语：λ、公式 (3)；未决：取值方式。"

    def dialogue(self, query: str, recent_turns: str) -> DialogueAct:
        self.dialogue_calls += 1
        self.rewrite_calls += 1
        self.last_recent_turns = recent_turns
        if isinstance(self.analysis, DialogueAct):
            return self.analysis
        if not self.analysis.is_clear:
            return DialogueAct(
                intent="clarify",
                reply=self.analysis.clarification_needed or "请把问题说具体一点。",
            )
        return DialogueAct(
            intent="retrieve",
            tasks=[AskTask(kind="close_read", question=item) for item in self.analysis.questions],
        )

    def rewrite(self, query: str, summary: str) -> QueryAnalysis:
        return self.analysis if isinstance(self.analysis, QueryAnalysis) else QueryAnalysis()

    def orchestrate(self, messages, question, context_summary):
        raise AssertionError("worker 应被 worker_fn 替换，不应调用 orchestrate")

    def compress(self, conversation_text: str) -> str:
        raise AssertionError("不应调用 compress")

    def fallback(self, question: str, context_text: str) -> str:
        raise AssertionError("不应调用 fallback")

    def aggregate(self, original: str, answers: list[dict], evidence: list[Evidence]) -> str:
        self.aggregate_calls += 1
        return self.aggregate_text


class RecordingWorker:
    def __init__(
        self,
        payloads: list[tuple[str, list[Evidence]] | tuple[str, list[Evidence], list[ExternalRef]]] | None = None,
    ):
        self.calls: list[tuple[int, str, str]] = []
        self.payloads = payloads or []

    def __call__(self, state: dict) -> dict:
        index = int(state.get("question_index") or 0)
        question = state.get("question") or ""
        kind = str(state.get("kind") or "close_read")
        self.calls.append((index, question, kind))
        assert "history" not in state
        refs: list[ExternalRef] = []
        if index < len(self.payloads):
            payload = self.payloads[index]
            answer, evidence = payload[0], payload[1]
            if len(payload) == 3:
                refs = list(payload[2])
        else:
            answer, evidence = "占位回答", [_evidence(f"quote-{index}")]
        return {
            "agent_answers": [
                {
                    "index": index,
                    "question": question,
                    "answer": answer,
                    "evidence": evidence,
                    "external_refs": refs,
                }
            ]
        }


def _settings(tmp_path: Path, **kwargs) -> Settings:
    data = {
        "data_dir": tmp_path,
        "formula_vision_enabled": False,
        "ask_max_subquestions": 3,
    }
    data.update(kwargs)
    return Settings(**data)


def _evidence(quote: str, *, page: int = 6, section_id: str | None = None) -> Evidence:
    return Evidence(
        page=page,
        section_title="Approach",
        quote=quote,
        sourced=True,
        section_id=section_id,
        chunk_id=f"{section_id or 'anon'}:{page}",
    )


def _ready_knowledge(tmp_path: Path, *, index_status: str = "ready", status: str = "ready") -> KnowledgeService:
    store = FilePaperStore(tmp_path)
    store.save_paper(
        Paper(
            paper_id="paper-a",
            sha256="paper-a",
            filename="paper-a.pdf",
            status=status,  # type: ignore[arg-type]
            index_status=index_status,  # type: ignore[arg-type]
            embedding_version="lexical-v1",
        )
    )
    return KnowledgeService(store, _settings(tmp_path))


def test_rewrite_splits_two_unrelated_questions(tmp_path: Path) -> None:
    model = ScriptedAskModel(
        QueryAnalysis(
            is_clear=True,
            questions=["这篇的对比损失怎么定义？", "实验用了哪个数据集？"],
        )
    )
    worker = RecordingWorker()
    question = "对比损失怎么定义，以及实验用了哪个数据集？"
    state = run_ask(
        model,
        _settings(tmp_path),
        "paper-a",
        question,
        worker_fn=worker,
    )
    assert model.rewrite_calls == 1
    assert state["rewrittenQuestions"] == ["这篇的对比损失怎么定义？", "实验用了哪个数据集？"]
    assert len(state["rewrittenQuestions"]) == 2
    assert sorted(item[1] for item in worker.calls) == sorted(state["rewrittenQuestions"])
    assert len(worker.calls) == 2


def test_unclear_query_asks_for_clarification(tmp_path: Path) -> None:
    original = "这个呢？"
    model = ScriptedAskModel(
        QueryAnalysis(is_clear=False, questions=[], clarification_needed="请说明问的是哪一部分"),
    )
    worker = RecordingWorker()
    state = run_ask(model, _settings(tmp_path), "paper-a", original, worker_fn=worker)
    assert worker.calls == []
    assert state["skip_workers"] is True
    assert state["dialogue_intent"] == "clarify"
    assert "请说明问的是哪一部分" in state["answer_zh"]


def test_aggregate_keeps_only_used_evidence(tmp_path: Path) -> None:
    first = _evidence("lambda = 0.5", page=6, section_id="s-method")
    second = _evidence("trained on CIFAR-10", page=9, section_id="s-exp")
    model = ScriptedAskModel(
        QueryAnalysis(is_clear=True, questions=["损失怎么写？", "用了什么数据？"]),
        aggregate_text=json.dumps(
            {"answer_zh": "数据是 CIFAR-10[1]。", "used_evidence": [2], "partial": False},
            ensure_ascii=False,
        ),
    )
    worker = RecordingWorker(
        [
            ("子答一：lambda = 0.5", [first]),
            ("子答二：CIFAR-10", [second]),
        ]
    )
    svc = UnderstandService(
        _ready_knowledge(tmp_path),
        _settings(tmp_path),
        ask_model=model,
        worker_fn=worker,
    )
    answer = svc.ask_agent("paper-a", "损失怎么写，数据是什么？")
    assert model.aggregate_calls == 1
    assert answer.no_evidence is False
    assert answer.answer_zh == "数据是 CIFAR-10[1]。"
    assert [item.quote for item in answer.citations] == [second.quote]
    assert answer.prompt_version == "ask-agent-v1"
    assert answer.embedding_version == "lexical-v1"


def test_empty_evidence_skips_aggregate(tmp_path: Path) -> None:
    model = ScriptedAskModel(
        QueryAnalysis(is_clear=True, questions=["损失怎么写？", "用了什么数据？"]),
        aggregate_text="SHOULD_NOT_RUN",
    )
    worker = RecordingWorker([("没有找到", []), ("也没有", [])])
    svc = UnderstandService(
        _ready_knowledge(tmp_path),
        _settings(tmp_path),
        ask_model=model,
        worker_fn=worker,
    )
    answer = svc.ask_agent("paper-a", "损失怎么写，数据是什么？")
    assert model.aggregate_calls == 0
    assert answer.no_evidence is True
    assert answer.answer_zh == NO_EVIDENCE_ANSWER
    assert answer.citations == []


def test_external_refs_do_not_become_citations(tmp_path: Path) -> None:
    local = _evidence("lambda = 0.5", page=6, section_id="s-method")
    external = ExternalRef(
        source="arxiv",
        title="GPT-4 Technical Report",
        url="https://arxiv.org/abs/2303.08774",
        identifier="arxiv:2303.08774",
        snippet="We report the development of GPT-4.",
        year=2023,
    )
    model = ScriptedAskModel(
        QueryAnalysis(is_clear=True, questions=["损失怎么写？"]),
        aggregate_text=json.dumps(
            {"answer_zh": "λ 为 0.5[1]。相关工作见 arXiv 条目，不是原文。", "used_evidence": [1], "partial": False},
            ensure_ascii=False,
        ),
    )
    worker = RecordingWorker([("子答：lambda = 0.5", [local], [external])])
    svc = UnderstandService(
        _ready_knowledge(tmp_path),
        _settings(tmp_path),
        ask_model=model,
        worker_fn=worker,
    )
    answer = svc.ask_agent("paper-a", "损失怎么写，有没有相关论文？")
    assert [item.quote for item in answer.citations] == [local.quote]
    assert [item.identifier for item in answer.external_refs] == ["arxiv:2303.08774"]
    assert all(item.section_id == "s-method" for item in answer.citations)


def test_empty_evidence_keeps_external_refs(tmp_path: Path) -> None:
    external = ExternalRef(
        source="arxiv",
        title="GPT-4 Technical Report",
        identifier="arxiv:2303.08774",
        url="https://arxiv.org/abs/2303.08774",
    )
    model = ScriptedAskModel(
        QueryAnalysis(is_clear=True, questions=["后续工作？"]),
        aggregate_text="SHOULD_NOT_RUN",
    )
    worker = RecordingWorker([("没有本篇证据", [], [external])])
    svc = UnderstandService(
        _ready_knowledge(tmp_path),
        _settings(tmp_path),
        ask_model=model,
        worker_fn=worker,
    )
    answer = svc.ask_agent("paper-a", "有没有后续工作？")
    assert model.aggregate_calls == 0
    assert answer.no_evidence is True
    assert answer.answer_zh == NO_EVIDENCE_ANSWER
    assert answer.citations == []
    assert [item.identifier for item in answer.external_refs] == ["arxiv:2303.08774"]


def test_rewrite_caps_at_max_subquestions(tmp_path: Path) -> None:
    model = ScriptedAskModel(
        QueryAnalysis(is_clear=True, questions=["问A", "问B", "问C", "问D"]),
    )
    worker = RecordingWorker()
    state = run_ask(model, _settings(tmp_path), "paper-a", "一次问四件事", worker_fn=worker)
    assert state["rewrittenQuestions"] == ["问A", "问B", "问C"]
    assert len(worker.calls) == 3


def test_pending_index_raises(tmp_path: Path) -> None:
    model = ScriptedAskModel(QueryAnalysis(is_clear=True, questions=["随便"]))
    svc = UnderstandService(
        _ready_knowledge(tmp_path, index_status="pending"),
        _settings(tmp_path),
        ask_model=model,
        worker_fn=RecordingWorker(),
    )
    with pytest.raises(IndexNotReadyError) as exc:
        svc.ask_agent("paper-a", "核心方法是什么？")
    assert exc.value.status == "pending"


def test_paper_not_ready_raises(tmp_path: Path) -> None:
    svc = UnderstandService(
        _ready_knowledge(tmp_path, status="parsing"),
        _settings(tmp_path),
        ask_model=ScriptedAskModel(QueryAnalysis(is_clear=True, questions=["x"])),
        worker_fn=RecordingWorker(),
    )
    with pytest.raises(PaperNotReadyError):
        svc.ask_agent("paper-a", "核心方法是什么？")


def test_skipped_index_returns_no_evidence(tmp_path: Path) -> None:
    model = ScriptedAskModel(QueryAnalysis(is_clear=True, questions=["x"]))
    worker = RecordingWorker()
    svc = UnderstandService(
        _ready_knowledge(tmp_path, index_status="skipped"),
        _settings(tmp_path),
        ask_model=model,
        worker_fn=worker,
    )
    answer = svc.ask_agent("paper-a", "核心方法是什么？")
    assert worker.calls == []
    assert model.rewrite_calls == 0
    assert answer.no_evidence is True
    assert answer.answer_zh == NO_EVIDENCE_ANSWER


def test_arxiv_only_plan_skips_close_read_rewrite(tmp_path: Path) -> None:
    model = ScriptedAskModel(QueryAnalysis(is_clear=True, questions=["不应改写"]))
    worker = RecordingWorker()
    question = "在 arXiv 上检索相关论文"
    state = run_ask(model, _settings(tmp_path), "paper-a", question, worker_fn=worker)
    assert model.rewrite_calls == 1
    assert worker.calls == []
    assert [item["kind"] for item in state.get("plan_tasks") or []] == ["arxiv"]


def test_identity_chat_skips_workers(tmp_path: Path) -> None:
    model = ScriptedAskModel(QueryAnalysis(is_clear=True, questions=["不应检索"]))
    worker = RecordingWorker()
    state = run_ask(model, _settings(tmp_path), "paper-a", "你是干什么的", worker_fn=worker)
    assert model.dialogue_calls == 0
    assert worker.calls == []
    assert state["skip_workers"] is True
    assert state["dialogue_intent"] == "chat"
    assert "精读助手" in state["answer_zh"]


def test_identity_chat_works_when_index_pending(tmp_path: Path) -> None:
    model = ScriptedAskModel(QueryAnalysis(is_clear=True, questions=["不应检索"]))
    worker = RecordingWorker()
    svc = UnderstandService(
        _ready_knowledge(tmp_path, index_status="pending"),
        _settings(tmp_path),
        ask_model=model,
        worker_fn=worker,
    )
    answer = svc.ask_agent("paper-a", "你是干什么的")
    assert worker.calls == []
    assert answer.no_evidence is False
    assert "精读助手" in answer.answer_zh
    assert answer.answer_zh == CHAT_IDENTITY_REPLY


def test_dialogue_sees_short_history_without_summary(tmp_path: Path) -> None:
    model = ScriptedAskModel(
        QueryAnalysis(is_clear=True, questions=["解释公式 (3) 中 λ 如何取值"]),
    )
    worker = RecordingWorker()
    history = [
        AskTurn(role="user", content="这篇的核心方法是什么？"),
        AskTurn(role="assistant", content="本文用 Transformer encoder，公式 (3) 中的 λ 控制权重。"),
    ]
    state = run_ask(
        model,
        _settings(tmp_path),
        "paper-a",
        "那这个怎么取的？",
        history,
        worker_fn=worker,
    )
    assert model.summarize_calls == 0
    assert model.compress_dialogue_calls == 0
    assert "公式 (3)" in model.last_recent_turns
    assert "λ" in model.last_recent_turns
    assert worker.calls == [(0, "解释公式 (3) 中 λ 如何取值", "close_read")]
    assert state["plan_tasks"][0]["kind"] == "close_read"


def test_pack_dialogue_memory_keeps_long_assistant_raw() -> None:
    history = [
        AskTurn(role="user", content="第一问"),
        AskTurn(role="assistant", content="B" * 1200),
    ]
    text = pack_dialogue_memory(history, "跟进", compress_fn=None)
    assert "B" * 1200 in text
    assert EARLIER_COMPRESSED_MARKER not in text


def test_pack_dialogue_memory_compresses_older_after_threshold() -> None:
    history = [
        AskTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn-{i}")
        for i in range(8)
    ]
    seen: list[tuple[str, str]] = []

    def compress_fn(older: str, query: str) -> str:
        seen.append((older, query))
        return "压缩笔记：turn-0 到 turn-3"

    text = pack_dialogue_memory(history, "当前问", compress_fn=compress_fn)
    assert seen and "turn-0" in seen[0][0]
    assert "turn-3" in seen[0][0]
    assert "turn-4" not in seen[0][0]
    assert EARLIER_COMPRESSED_MARKER in text
    assert "压缩笔记" in text
    assert "turn-7" in text
    assert "turn-0" not in text.split("[Recent turns]", 1)[-1]


def test_pack_dialogue_memory_keeps_older_raw_when_compress_fails() -> None:
    history = [AskTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn-{i}") for i in range(8)]

    def compress_fn(older: str, query: str) -> str:
        raise RuntimeError("boom")

    text = pack_dialogue_memory(history, "当前问", compress_fn=compress_fn)
    assert "[Earlier turns]" in text
    assert "turn-0" in text
    assert "turn-7" in text
    assert EARLIER_COMPRESSED_MARKER not in text


def test_long_history_runs_dialogue_compression(tmp_path: Path) -> None:
    model = ScriptedAskModel(QueryAnalysis(is_clear=True, questions=["λ 怎么取"]))
    worker = RecordingWorker()
    history = [
        AskTurn(role="user" if i % 2 == 0 else "assistant", content=f"msg-{i} λ" if i == 1 else f"msg-{i}")
        for i in range(8)
    ]
    state = run_ask(
        model,
        _settings(tmp_path),
        "paper-a",
        "那这个怎么取的？",
        history,
        worker_fn=worker,
    )
    assert model.compress_dialogue_calls == 1
    assert "msg-0" in model.last_older_turns
    assert EARLIER_COMPRESSED_MARKER in model.last_recent_turns
    assert "msg-7" in model.last_recent_turns
    assert worker.calls == [(0, "λ 怎么取", "close_read")]
    assert state["conversation_summary"].startswith(EARLIER_COMPRESSED_MARKER)