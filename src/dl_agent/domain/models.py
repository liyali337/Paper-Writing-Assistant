from typing import Literal

from pydantic import BaseModel, Field

PaperStatus = Literal[
    "queued", "parsing", "ready", "needs_ocr", "ingest_failed"
]
ParserName = Literal["docling", "pymupdf"]
ExplainStatus = Literal["pending", "ready", "failed", "skipped", "partial"]
TranslateStatus = Literal["pending", "ready", "failed", "partial"]
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


class PaperTranslation(BaseModel):
    paper_id: str
    status: TranslateStatus
    model: str
    prompt_version: str
    title_zh: str | None = None
    sections: list[SectionTranslation] = Field(default_factory=list)
    error: str | None = None
