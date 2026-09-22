from pathlib import Path

from dl_agent.config import Settings
from dl_agent.domain.models import Figure, Paper, PaperTranslation, Section, SectionTranslation
from dl_agent.knowledge.adapters.vector.memory import MemoryVectorIndex
from dl_agent.knowledge.service import KnowledgeService
from dl_agent.knowledge.store import FilePaperStore
from dl_agent.understand.agent.tools import (
    NO_CAPTION_HITS,
    NO_FIGURE,
    NO_FIGURES,
    NO_PARENT_DOCUMENT,
    NO_RELEVANT_CHUNKS,
    NO_SECTIONS,
    NO_TRANSLATION,
    RetrievalTools,
)


def _embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        low = text.lower()
        vectors.append(
            [
                1.0 if "lambda" in low or "l_con" in low else 0.0,
                1.0 if "contrastive" in low or "hrnet" in low or "cifar" in low else 0.0,
                1.0 if "xyzzy" in low or "zebra" in low else 0.0,
            ]
        )
    return vectors


def _service(tmp_path: Path) -> KnowledgeService:
    return KnowledgeService(
        FilePaperStore(tmp_path),
        Settings(data_dir=tmp_path, formula_vision_enabled=False),
        embed_fn=_embed,
        vector_index=MemoryVectorIndex(),
    )


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


def _paper(paper_id: str, status: str = "ready") -> Paper:
    return Paper(
        paper_id=paper_id,
        sha256=paper_id,
        filename=f"{paper_id}.pdf",
        status=status,  # type: ignore[arg-type]
    )


def _index_two_papers(tmp_path: Path) -> KnowledgeService:
    svc = _service(tmp_path)
    store = svc.store
    shared = "We train HRNet with contrastive loss on CIFAR-10."
    for paper_id in ("paper-a", "paper-b"):
        store.save_paper(_paper(paper_id))
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
    return svc


def test_search_does_not_cross_papers(tmp_path: Path) -> None:
    svc = _index_two_papers(tmp_path)
    tools = RetrievalTools(svc, "paper-a")
    text = tools.search_child_chunks("contrastive loss")
    assert text != NO_RELEVANT_CHUNKS
    assert "paper-a-m" in text
    assert "paper-b-m" not in text
    assert "References" not in text
    assert all(item.section_id and item.section_id.startswith("paper-a") for item in tools.last_child_hits)


def test_unrelated_query_is_empty(tmp_path: Path) -> None:
    svc = _index_two_papers(tmp_path)
    tools = RetrievalTools(svc, "paper-a")
    assert tools.search_child_chunks("xyzzyplugh quantum zebra") == NO_RELEVANT_CHUNKS
    assert tools.last_child_hits == []
    assert tools.search_child_chunks("   ") == NO_RELEVANT_CHUNKS


def test_chinese_query_keeps_english_hits(tmp_path: Path) -> None:
    svc = _index_two_papers(tmp_path)
    tools = RetrievalTools(svc, "paper-a")
    text = tools.search_child_chunks("核心方法是什么？")
    assert text != NO_RELEVANT_CHUNKS
    assert tools.last_child_hits
    assert all(item.section_id and item.section_id.startswith("paper-a") for item in tools.last_child_hits)


def test_retrieve_parent_contains_child_span(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    store = svc.store
    store.save_paper(_paper("paper-a"))
    child_span = "The total loss is L = lambda * L_con + L_ce."
    store.save_sections("paper-a", [_section(text=child_span)])
    paper = store.get_paper("paper-a")
    assert paper is not None
    paper.status = "ready"
    store.save_paper(paper)
    svc.index_paper("paper-a", force=True)

    tools = RetrievalTools(svc, "paper-a")
    search = tools.search_child_chunks("lambda")
    assert "s-method" in search
    parent = tools.retrieve_parent_chunks("s-method")
    assert "lambda" in parent
    assert "L_con" in parent
    assert "page=6-8" in parent
    assert "section_id=s-method" in parent


def test_unknown_section_is_empty(tmp_path: Path) -> None:
    svc = _index_two_papers(tmp_path)
    tools = RetrievalTools(svc, "paper-a")
    assert tools.retrieve_parent_chunks("missing-section") == NO_PARENT_DOCUMENT
    assert tools.retrieve_parent_chunks("") == NO_PARENT_DOCUMENT


def test_parent_truncates_long_section(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    store = svc.store
    store.save_paper(_paper("paper-a"))
    store.save_sections("paper-a", [_section(text="lambda " + ("body " * 400))])
    paper = store.get_paper("paper-a")
    assert paper is not None
    paper.status = "ready"
    store.save_paper(paper)
    svc.index_paper("paper-a", force=True)

    tools = RetrievalTools(svc, "paper-a", parent_max_chars=80)
    parent = tools.retrieve_parent_chunks("s-method")
    assert parent.endswith("…")
    assert len(parent) < 200


def _save_figure(svc: KnowledgeService, paper_id: str = "paper-a") -> Figure:
    figure = Figure(
        figure_id="fig-1",
        paper_id=paper_id,
        section_id="s-method",
        page=6,
        kind="figure",
        label="Figure 3",
        caption="Architecture of the contrastive HRNet.",
        storage_key=f"papers/{paper_id}/figures/fig-1.png",
        width_px=100,
        height_px=80,
    )
    svc.store.save_figures(paper_id, [figure], {"fig-1": b"\x89PNG"})
    return figure


def test_list_sections_is_outline_without_body(tmp_path: Path) -> None:
    svc = _index_two_papers(tmp_path)
    tools = RetrievalTools(svc, "paper-a")
    text = tools.list_sections()
    assert "paper-a-m" in text
    assert "kind=method" in text
    assert "The total loss" not in text
    assert tools.last_hits == []
    method_only = tools.list_sections("method")
    assert "paper-a-m" in method_only
    assert "kind=references" not in method_only
    assert tools.list_sections("dataset") == NO_SECTIONS


def test_list_figures_and_caption(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    store = svc.store
    store.save_paper(_paper("paper-a"))
    store.save_sections("paper-a", [_section()])
    _save_figure(svc)
    paper = store.get_paper("paper-a")
    assert paper is not None
    paper.status = "ready"
    store.save_paper(paper)

    tools = RetrievalTools(svc, "paper-a")
    listed = tools.list_figures()
    assert listed != NO_FIGURES
    assert "figure_id=fig-1" in listed
    assert "Figure 3" in listed
    assert "contrastive HRNet" in listed
    by_section = tools.list_figures("s-method")
    assert "fig-1" in by_section
    assert tools.list_figures("missing") == NO_FIGURES

    caption = tools.get_figure_caption("fig-1")
    assert "caption: Architecture of the contrastive HRNet." in caption
    assert "nearby:" in caption
    assert tools.last_hits
    assert tools.last_hits[0].figure_ids == ["fig-1"]
    assert tools.get_figure_caption("missing") == NO_FIGURE
    assert tools.get_figure_caption("") == NO_FIGURE


def test_search_captions_matches_label(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    store = svc.store
    store.save_paper(_paper("paper-a"))
    store.save_sections("paper-a", [_section()])
    _save_figure(svc)
    paper = store.get_paper("paper-a")
    assert paper is not None
    paper.status = "ready"
    store.save_paper(paper)

    tools = RetrievalTools(svc, "paper-a")
    text = tools.search_captions("Figure 3")
    assert "fig-1" in text
    assert tools.last_hits[0].chunk_id == "fig-1:caption"
    assert tools.search_captions("xyzzy zebra") == NO_CAPTION_HITS
    assert tools.search_captions("  ") == NO_CAPTION_HITS


def test_get_translation_is_read_only(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    store = svc.store
    store.save_paper(_paper("paper-a"))
    store.save_sections("paper-a", [_section()])
    tools = RetrievalTools(svc, "paper-a")
    assert tools.get_translation("s-method") == NO_TRANSLATION
    store.save_translation(
        PaperTranslation(
            paper_id="paper-a",
            status="ready",
            model="x",
            prompt_version="t",
            sections=[
                SectionTranslation(
                    section_id="s-method",
                    title_zh="方法",
                    text_zh="对比损失里有 lambda。",
                )
            ],
        )
    )
    text = tools.get_translation("s-method")
    assert "对比损失里有 lambda。" in text
    assert "title_zh=方法" in text
    assert tools.last_hits == []
    assert tools.get_translation("other") == NO_TRANSLATION


def test_run_tools_dispatches_list_sections(tmp_path: Path) -> None:
    from dl_agent.understand.agent.nodes import run_tools

    svc = _index_two_papers(tmp_path)
    tools = RetrievalTools(svc, "paper-a")
    out = run_tools(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_outline", "name": "list_sections", "args": {"kind": "method"}}
                    ],
                }
            ],
            "retrieval_keys": [],
        },
        tools,
    )
    content = out["messages"][0]["content"]
    assert "UNKNOWN_TOOL" not in content
    assert "kind=method" in content
    assert "outline::method" in out["retrieval_keys"]
    assert out["evidence_bag"] == []
