from pathlib import Path

from dl_agent.config import Settings
from dl_agent.domain.models import Section
from dl_agent.knowledge.adapters.vector.base import SparseEmbedding
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex
from dl_agent.knowledge.chunk import build_chunks, cap_chunks, chunk_text
from dl_agent.knowledge.service import KnowledgeService, PaperNotReadyError
from dl_agent.knowledge.store import FilePaperStore


def _section(**kwargs) -> Section:
    data = {
        "section_id": "s-method",
        "paper_id": "paper-a",
        "title": "4. Approach",
        "kind": "method",
        "level": 1,
        "page_start": 6,
        "page_end": 8,
        "text": "The total loss is L = lambda * L_con + L_ce.",
        "figure_ids": ["fig-1"],
    }
    data.update(kwargs)
    return Section(**data)


def test_short_section_is_one_chunk() -> None:
    chunks = build_chunks([_section()])
    assert len(chunks) == 1
    assert chunks[0].paper_id == "paper-a"
    assert chunks[0].section_id == "s-method"
    assert chunks[0].page_start == 6
    assert "lambda" in chunks[0].text


def test_long_section_overlaps() -> None:
    text = ("paragraph one talks about contrastive loss. " * 8 + "\n\n") + (
        "paragraph two talks about ablation on lambda. " * 8
    )
    parts = chunk_text(text, size=120, overlap=30)
    assert len(parts) >= 2


def test_references_and_ack_are_skipped() -> None:
    chunks = build_chunks(
        [
            _section(section_id="s-ref", title="References", kind="references", text="[1] Prior work."),
            _section(section_id="s-ack", title="Acknowledgements", kind="other", text="We thank everyone."),
            _section(section_id="s-empty", title="Empty", kind="method", text="  "),
        ]
    )
    assert chunks == []


def test_cap_drops_other_before_method() -> None:
    chunks = build_chunks(
        [
            _section(section_id="s-m", kind="method", text="method " * 40),
            _section(section_id="s-o", title="Appendix", kind="other", text="appendix " * 40),
        ],
        chunk_size=40,
        overlap=5,
    )
    assert len(chunks) > 2
    capped = cap_chunks(chunks, 2)
    assert len(capped) == 2
    assert all(item.section_kind == "method" for item in capped)


def test_index_then_query_does_not_cross_papers(tmp_path: Path) -> None:
    store = FilePaperStore(tmp_path)
    svc = KnowledgeService(store, Settings(data_dir=tmp_path, formula_vision_enabled=False))
    svc._encoder_attempted = True
    shared = "We train HRNet with contrastive loss on CIFAR-10."
    for paper_id in ("paper-a", "paper-b"):
        store.save_paper(
            _paper(paper_id, tmp_path),
        )
        store.save_sections(
            paper_id,
            [
                _section(paper_id=paper_id, section_id=f"{paper_id}-m", text=shared),
                _section(
                    paper_id=paper_id,
                    section_id=f"{paper_id}-r",
                    title="References",
                    kind="references",
                    text="[1] Same sentence contrastive loss.",
                ),
            ],
        )
        paper = store.get_paper(paper_id)
        assert paper is not None
        paper.status = "ready"
        store.save_paper(paper)
        indexed = svc.index_paper(paper_id, force=True)
        assert indexed.index_status == "ready"
        assert indexed.embedding_version == "lexical-v1"

    hits = svc.query("paper-a", "contrastive loss")
    assert hits
    assert all(item.section_id and item.section_id.startswith("paper-a") for item in hits)
    assert all("References" not in (item.section_title or "") for item in hits)
    other = store.get_chunks("paper-b")
    assert other and other[0].paper_id == "paper-b"


def test_delete_index_rebuilds_same_count(tmp_path: Path) -> None:
    store = FilePaperStore(tmp_path)
    svc = KnowledgeService(store, Settings(data_dir=tmp_path, formula_vision_enabled=False))
    svc._encoder_attempted = True
    store.save_paper(_paper("paper-a", tmp_path))
    store.save_sections("paper-a", [_section(text="loss " * 200)])
    paper = store.get_paper("paper-a")
    assert paper is not None
    paper.status = "ready"
    store.save_paper(paper)
    first = svc.index_paper("paper-a", force=True)
    count = len(store.get_chunks("paper-a"))
    assert first.index_status == "ready"
    assert count >= 1
    svc.delete_index("paper-a")
    assert store.get_chunks("paper-a") == []
    assert svc.get_paper("paper-a").index_status == "pending"
    svc.index_paper("paper-a", force=True)
    assert len(store.get_chunks("paper-a")) == count


def test_hybrid_query_uses_injected_vectors(tmp_path: Path) -> None:
    store = FilePaperStore(tmp_path)
    index = MemoryVectorIndex()

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0, 0.0, 0.0]
            low = text.lower()
            if "lambda" in low:
                vec[0] = 1.0
            if "hrnet" in low:
                vec[1] = 1.0
            if "cifar" in low:
                vec[2] = 1.0
            vectors.append(vec)
        return vectors

    def sparse(texts: list[str]) -> list[SparseEmbedding]:
        out: list[SparseEmbedding] = []
        for text in texts:
            indices: list[int] = []
            values: list[float] = []
            low = text.lower()
            if "lambda" in low:
                indices.append(11)
                values.append(1.0)
            if "imagenet" in low:
                indices.append(22)
                values.append(1.0)
            out.append(SparseEmbedding(indices, values))
        return out

    svc = KnowledgeService(
        store,
        Settings(data_dir=tmp_path, formula_vision_enabled=False),
        embed_fn=embed,
        sparse_fn=sparse,
        vector_index=index,
    )
    store.save_paper(_paper("paper-a", tmp_path))
    store.save_sections(
        "paper-a",
        [
            _section(section_id="s1", text="We set lambda = 0.5 in the loss."),
            _section(section_id="s2", title="3. Experiments", kind="experiment", text="Dataset is ImageNet."),
        ],
    )
    paper = store.get_paper("paper-a")
    assert paper is not None
    paper.status = "ready"
    store.save_paper(paper)
    svc.index_paper("paper-a", force=True)
    hits = svc.query("paper-a", "lambda")
    assert hits
    assert hits[0].section_id == "s1"


def test_query_rejects_unready_paper(tmp_path: Path) -> None:
    store = FilePaperStore(tmp_path)
    svc = KnowledgeService(store, Settings(data_dir=tmp_path, formula_vision_enabled=False))
    svc._encoder_attempted = True
    store.save_paper(_paper("paper-a", tmp_path, status="queued"))
    try:
        svc.query("paper-a", "loss")
        assert False, "expected PaperNotReadyError"
    except PaperNotReadyError as exc:
        assert exc.status == "queued"


def _paper(paper_id: str, tmp_path: Path, status: str = "ready"):
    from dl_agent.domain.models import Paper

    return Paper(
        paper_id=paper_id,
        sha256=paper_id,
        filename=f"{paper_id}.pdf",
        status=status,  # type: ignore[arg-type]
    )
