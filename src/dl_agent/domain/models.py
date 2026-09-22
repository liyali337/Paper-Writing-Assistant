from typing import Literal

from pydantic import BaseModel, Field

PaperStatus = Literal[
    "queued", "parsing", "ready", "needs_ocr", "ingest_failed"
]
ParserName = Literal["docling", "pymupdf"]
ExplainStatus = Literal["pending", "ready", "failed", "skipped", "partial"]
TranslateStatus = Literal["pending", "ready", "failed", "partial", "cancelled"]
IndexStatus = Literal["pending", "ready", "failed", "skipped"]
SectionKind = Literal[
    "abstract",
    "intro",
    "related",
    "method",
    "experiment",
    "conclusion",
    "references",
    "other",
]
FigureKind = Literal["figure", "table_snapshot", "algorithm", "formula"]
FigureSource = Literal["docling_picture", "pymupdf_xref", "page_clip"]


class Paper(BaseModel):
    paper_id: str
    sha256: str
    filename: str
    status: PaperStatus
    parser: ParserName | None = None
    page_count: int = 0
    figure_count: int = 0
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    language: str = "eng"
    intro_status: ExplainStatus = "pending"
    method_status: ExplainStatus = "pending"
    translate_status: TranslateStatus = "pending"
    index_status: IndexStatus = "pending"
    index_error: str | None = None
    embedding_version: str | None = None


class Section(BaseModel):
    section_id: str
    paper_id: str
    title: str
    kind: SectionKind
    level: int
    page_start: int
    page_end: int
    text: str
    parent_id: str | None = None
    figure_ids: list[str] = Field(default_factory=list)


class ChildChunk(BaseModel):
    chunk_id: str
    paper_id: str
    section_id: str
    section_title: str
    section_kind: str
    page_start: int
    page_end: int
    text: str
    figure_ids: list[str] = Field(default_factory=list)
    order: int = 0
    source_hash: str = ""


class Figure(BaseModel):
    figure_id: str
    paper_id: str
    section_id: str | None = None
    page: int
    kind: FigureKind = "figure"
    label: str | None = None
    caption: str | None = None
    storage_key: str | None = None
    source: FigureSource = "docling_picture"
    width_px: int = 0
    height_px: int = 0
    bbox: tuple[float, float, float, float] | None = None


class Evidence(BaseModel):
    page: int
    section_title: str | None = None
    quote: str
    sourced: bool
    section_id: str | None = None
    score: float | None = None
    figure_ids: list[str] = Field(default_factory=list)
    chunk_id: str | None = None


class ExternalRef(BaseModel):
    source: Literal["arxiv", "asta", "web"]
    title: str
    url: str | None = None
    identifier: str | None = None
    snippet: str = ""
    year: int | None = None
    authors: list[str] = Field(default_factory=list)


class PaperIntro(BaseModel):
    paper_id: str
    title_zh: str
    one_sentence_zh: str
    problem_zh: str
    motivation_zh: str
    contributions: list[str]
    task_setting_zh: str
    reading_order_zh: list[str]
    overview_figure_id: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    partial: bool = False
    section_match: str = "matched"
    model: str
    prompt_version: str


class MethodStep(BaseModel):
    name: str
    what_zh: str
    page: int
    section_title: str
    quote: str | None = None
    sourced: bool = False
    figure_ids: list[str] = Field(default_factory=list)


class MethodExplain(BaseModel):
    paper_id: str
    overview_zh: str
    pipeline_steps: list[MethodStep]
    key_modules: list[MethodStep]
    vs_prior_zh: str
    assumptions_zh: list[str]
    open_to_read_zh: list[str]
    figure_ids: list[str] = Field(default_factory=list)
    partial: bool = False
    section_match: str = "matched"
    model: str
    prompt_version: str


class SectionTranslation(BaseModel):
    section_id: str
    title_zh: str
    text_zh: str
    partial: bool = False
    chunks_done: int = 0
    chunks_total: int = 0


class FigureTranslation(BaseModel):
    figure_id: str
    caption_zh: str


class PaperTranslation(BaseModel):
    paper_id: str
    status: TranslateStatus
    model: str
    prompt_version: str
    title_zh: str | None = None
    sections: list[SectionTranslation] = Field(default_factory=list)
    figures: list[FigureTranslation] = Field(default_factory=list)
    error: str | None = None


class AskTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[AskTurn] = Field(default_factory=list)


class LibraryHit(BaseModel):
    paper_id: str
    title: str | None = None
    filename: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    score: float | None = None
    why: str = ""
    section_title: str | None = None
    openable: bool = True
    index_status: IndexStatus = "pending"


class PaperAnswer(BaseModel):
    paper_id: str
    question: str
    answer_zh: str
    citations: list[Evidence] = Field(default_factory=list)
    figure_ids: list[str] = Field(default_factory=list)
    external_refs: list[ExternalRef] = Field(default_factory=list)
    library_hits: list[LibraryHit] = Field(default_factory=list)
    mode: Literal["close_read", "library", "arxiv"] = "close_read"
    no_evidence: bool = False
    partial: bool = False
    model: str
    prompt_version: str
    embedding_version: str = "lexical-v1"
