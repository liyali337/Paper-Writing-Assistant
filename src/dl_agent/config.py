from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DL_AGENT_", extra="ignore")

    data_dir: Path = Path("./data")
    max_pdf_bytes: int = 50 * 1024 * 1024
    docling_timeout_s: float = 300
    docling_formula_enrichment: bool = False
    images_scale: float = 2.0
    min_figure_px: int = 80
    min_chars_total: int = 800
    min_chars_per_page: int = 200
    clip_above_pt: float = 280


def get_settings() -> Settings:
    return Settings()
