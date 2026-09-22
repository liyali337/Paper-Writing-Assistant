from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="DL_AGENT_",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Path("./data")
    max_pdf_bytes: int = 50 * 1024 * 1024
    docling_timeout_s: float = 300
    docling_formula_enrichment: bool = False
    images_scale: float = 2.0
    min_figure_px: int = 80
    min_chars_total: int = 800
    min_chars_per_page: int = 200
    clip_above_pt: float = 280
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "DL_AGENT_OPENAI_API_KEY"),
    )
    openai_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "DL_AGENT_OPENAI_BASE_URL"),
    )
    model_name: str = Field(
        default="deepseek-chat",
        validation_alias=AliasChoices("MODEL_NAME", "DL_AGENT_MODEL_NAME"),
    )
    translate_prompt_version: str = "translate-v10"
    translate_section_delay_s: float = 0.5
    translate_chunk_max_chars: int = Field(default=1200, validation_alias="TRANSLATE_CHUNK_MAX_CHARS")
    translate_parallelism: int = Field(default=1, validation_alias="TRANSLATE_PARALLELISM")
    llm_timeout_s: float = Field(default=180.0, validation_alias="LLM_TIMEOUT_S")
    llm_max_retries: int = 4
    formula_vision_enabled: bool = Field(default=True, validation_alias="FORMULA_VISION_ENABLED")
    formula_vision_model: str = Field(default="", validation_alias="FORMULA_VISION_MODEL")
    formula_vision_base_url: str = Field(default="", validation_alias="FORMULA_VISION_BASE_URL")
    formula_vision_api_key: str = Field(default="", validation_alias="FORMULA_VISION_API_KEY")
    formula_vision_max: int = Field(default=80, validation_alias="FORMULA_VISION_MAX")
    formula_vision_timeout_s: float = Field(default=60.0, validation_alias="FORMULA_VISION_TIMEOUT_S")
    retrieve_k: int = Field(default=8, validation_alias="RETRIEVE_K")
    chunk_size: int = Field(default=500, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, validation_alias="CHUNK_OVERLAP")
    max_chunks: int = Field(default=800, validation_alias="MAX_CHUNKS")
    embedding_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "DL_AGENT_EMBEDDING_API_KEY"),
    )
    embedding_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "DL_AGENT_EMBEDDING_BASE_URL"),
    )
    embedding_model: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_MODEL", "DL_AGENT_EMBEDDING_MODEL"),
    )
    dense_model: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        validation_alias=AliasChoices("DENSE_MODEL", "DL_AGENT_DENSE_MODEL"),
    )
    hf_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices("HF_ENDPOINT", "HF_HUB_ENDPOINT", "DL_AGENT_HF_ENDPOINT"),
    )
    sparse_model: str = Field(
        default="Qdrant/bm25",
        validation_alias=AliasChoices("SPARSE_MODEL", "DL_AGENT_SPARSE_MODEL"),
    )
    qdrant_path: str = Field(default="", validation_alias="QDRANT_PATH")
    qdrant_collection: str = Field(
        default="paper_child_chunks",
        validation_alias="QDRANT_COLLECTION",
    )
    ask_prompt_version: str = "ask-v1"
    ask_agent_prompt_version: str = Field(
        default="ask-agent-v1",
        validation_alias=AliasChoices("ASK_AGENT_PROMPT_VERSION", "DL_AGENT_ASK_AGENT_PROMPT_VERSION"),
    )
    ask_history_turns: int = 4
    ask_mode: Literal["agent", "simple"] = Field(
        default="agent",
        validation_alias=AliasChoices("ASK_MODE", "DL_AGENT_ASK_MODE"),
    )
    ask_max_tool_calls: int = Field(
        default=6,
        ge=1,
        validation_alias=AliasChoices("ASK_MAX_TOOL_CALLS", "DL_AGENT_ASK_MAX_TOOL_CALLS"),
    )
    ask_max_iterations: int = Field(
        default=8,
        ge=1,
        validation_alias=AliasChoices("ASK_MAX_ITERATIONS", "DL_AGENT_ASK_MAX_ITERATIONS"),
    )
    ask_max_subquestions: int = Field(
        default=3,
        ge=1,
        le=3,
        validation_alias=AliasChoices("ASK_MAX_SUBQUESTIONS", "DL_AGENT_ASK_MAX_SUBQUESTIONS"),
    )
    ask_compress_tokens: int = Field(
        default=2000,
        ge=1,
        validation_alias=AliasChoices("ASK_COMPRESS_TOKENS", "DL_AGENT_ASK_COMPRESS_TOKENS"),
    )
    ask_tool_protocol: Literal["native", "json"] = Field(
        default="native",
        validation_alias=AliasChoices("ASK_TOOL_PROTOCOL", "DL_AGENT_ASK_TOOL_PROTOCOL"),
    )
    ask_enable_external: bool = Field(
        default=False,
        validation_alias=AliasChoices("ASK_ENABLE_EXTERNAL", "DL_AGENT_ASK_ENABLE_EXTERNAL"),
    )
    arxiv_api_url: str = Field(
        default="https://export.arxiv.org/api/query",
        validation_alias=AliasChoices("ARXIV_API_URL", "DL_AGENT_ARXIV_API_URL"),
    )
    mcp_timeout_s: float = Field(
        default=30.0,
        ge=1,
        validation_alias=AliasChoices("MCP_TIMEOUT_S", "DL_AGENT_MCP_TIMEOUT_S"),
    )
    library_paper_k: int = Field(
        default=6,
        ge=1,
        le=16,
        validation_alias=AliasChoices("LIBRARY_PAPER_K", "DL_AGENT_LIBRARY_PAPER_K"),
    )
    library_chunks_per_paper: int = Field(
        default=2,
        ge=1,
        le=4,
        validation_alias=AliasChoices("LIBRARY_CHUNKS_PER_PAPER", "DL_AGENT_LIBRARY_CHUNKS_PER_PAPER"),
    )
    library_prompt_version: str = Field(
        default="library-ask-v1",
        validation_alias=AliasChoices("LIBRARY_PROMPT_VERSION", "DL_AGENT_LIBRARY_PROMPT_VERSION"),
    )
    arxiv_prompt_version: str = Field(
        default="arxiv-ask-v2",
        validation_alias=AliasChoices("ARXIV_PROMPT_VERSION", "DL_AGENT_ARXIV_PROMPT_VERSION"),
    )
    langfuse_public_key: str = Field(
        default="",
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "DL_AGENT_LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: str = Field(
        default="",
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "DL_AGENT_LANGFUSE_SECRET_KEY"),
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices(
            "LANGFUSE_BASE_URL",
            "LANGFUSE_HOST",
            "DL_AGENT_LANGFUSE_BASE_URL",
        ),
    )
    langfuse_tracing: bool = Field(
        default=True,
        validation_alias=AliasChoices("LANGFUSE_TRACING", "DL_AGENT_LANGFUSE_TRACING"),
    )

    def embedding_configured(self) -> bool:
        return bool(self.embedding_model.strip() and self.embedding_base_url.strip())

    def embedding_version(self) -> str:
        dense = self.embedding_model.strip() if self.embedding_configured() else self.dense_model.strip()
        sparse = self.sparse_model.strip()
        return f"{_slug(dense)}_{_slug(sparse)}"[:64]

    def resolved_qdrant_path(self) -> Path:
        if self.qdrant_path.strip():
            return Path(self.qdrant_path)
        return Path(self.data_dir) / "qdrant"


def _slug(value: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
    return slug[-32:] if slug else "default"


def get_settings() -> Settings:
    return Settings()
