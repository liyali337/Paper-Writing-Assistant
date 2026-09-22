export type PaperStatus =
  | "queued"
  | "parsing"
  | "ready"
  | "needs_ocr"
  | "ingest_failed";

export type ExplainStatus = "pending" | "ready" | "failed" | "skipped" | "partial" | "cancelled";

export type IndexStatus = "pending" | "ready" | "failed" | "skipped";

export type SectionKind =
  | "abstract"
  | "intro"
  | "related"
  | "method"
  | "experiment"
  | "conclusion"
  | "references"
  | "other";

export type Paper = {
  paper_id: string;
  sha256: string;
  filename: string;
  status: PaperStatus;
  parser: "docling" | "pymupdf" | null;
  page_count: number;
  figure_count: number;
  title: string | null;
  authors: string[];
  abstract: string | null;
  language: string;
  intro_status: ExplainStatus;
  method_status: ExplainStatus;
  translate_status: ExplainStatus;
  index_status?: IndexStatus;
  index_error?: string | null;
  embedding_version?: string | null;
};

export type Section = {
  section_id: string;
  paper_id: string;
  title: string;
  kind: SectionKind;
  level: number;
  page_start: number;
  page_end: number;
  text: string;
  parent_id: string | null;
  figure_ids: string[];
};

export type Figure = {
  figure_id: string;
  paper_id: string;
  section_id: string | null;
  page: number;
  kind: "figure" | "table_snapshot" | "algorithm" | "formula";
  label: string | null;
  caption: string | null;
  caption_zh?: string | null;
  storage_key: string | null;
  source: string;
  width_px: number;
  height_px: number;
};

export type Evidence = {
  page: number;
  section_title: string | null;
  quote: string;
  sourced: boolean;
  section_id?: string | null;
  score?: number | null;
  figure_ids?: string[];
  chunk_id?: string | null;
};

export type ExternalRef = {
  source: "arxiv" | "asta" | "web";
  title: string;
  url?: string | null;
  identifier?: string | null;
  snippet?: string;
  year?: number | null;
  authors?: string[];
};

export type LibraryHit = {
  paper_id: string;
  title?: string | null;
  filename?: string;
  authors?: string[];
  abstract?: string | null;
  score?: number | null;
  why?: string;
  section_title?: string | null;
  openable?: boolean;
  index_status?: IndexStatus;
};

export type PaperAnswer = {
  paper_id: string;
  question: string;
  answer_zh: string;
  citations: Evidence[];
  figure_ids: string[];
  external_refs?: ExternalRef[];
  library_hits?: LibraryHit[];
  mode?: "close_read" | "library" | "arxiv";
  no_evidence: boolean;
  partial: boolean;
  model: string;
  prompt_version: string;
  embedding_version?: string;
};

export type AskTurn = {
  role: "user" | "assistant";
  content: string;
};

export type AskRequest = {
  question: string;
  history?: AskTurn[];
};

export type PaperIntro = {
  paper_id: string;
  title_zh: string;
  one_sentence_zh: string;
  problem_zh: string;
  motivation_zh: string;
  contributions: string[];
  task_setting_zh: string;
  reading_order_zh: string[];
  overview_figure_id: string | null;
  evidence: Evidence[];
  partial: boolean;
  section_match: string;
  model: string;
  prompt_version: string;
};

export type MethodStep = {
  name: string;
  what_zh: string;
  page: number;
  section_title: string;
  quote: string | null;
  sourced: boolean;
  figure_ids: string[];
};

export type MethodExplain = {
  paper_id: string;
  overview_zh: string;
  pipeline_steps: MethodStep[];
  key_modules: MethodStep[];
  vs_prior_zh: string;
  assumptions_zh: string[];
  open_to_read_zh: string[];
  figure_ids: string[];
  partial: boolean;
  section_match: string;
  model: string;
  prompt_version: string;
};

export type ApiError = {
  code: string;
  stage: string;
  message: string;
};

export type Health = {
  status: string;
  service: string;
  version: string;
};

export type SectionTranslation = {
  section_id: string;
  title_zh: string;
  text_zh: string;
  partial?: boolean;
  chunks_done?: number;
  chunks_total?: number;
};

export type FigureTranslation = {
  figure_id: string;
  caption_zh: string;
};

export type PaperTranslation = {
  paper_id: string;
  status: ExplainStatus;
  model: string;
  prompt_version: string;
  title_zh?: string | null;
  sections: SectionTranslation[];
  figures?: FigureTranslation[];
  error?: string | null;
};

export class HttpError extends Error {
  status: number;
  api?: ApiError;

  constructor(status: number, message: string, api?: ApiError) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.api = api;
  }
}
