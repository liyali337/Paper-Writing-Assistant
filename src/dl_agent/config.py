from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
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
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias="OPENAI_BASE_URL",
    )
    model_name: str = Field(default="qwen3.8-max", validation_alias="MODEL_NAME")
    translate_prompt_version: str = "translate-v8"
    translate_section_delay_s: float = 0.5
    translate_chunk_max_chars: int = Field(default=1200, validation_alias="TRANSLATE_CHUNK_MAX_CHARS")
    translate_parallelism: int = Field(default=1, validation_alias="TRANSLATE_PARALLELISM")
    llm_timeout_s: float = Field(default=180.0, validation_alias="LLM_TIMEOUT_S")
    llm_max_retries: int = 4


def get_settings() -> Settings:
    return Settings()
