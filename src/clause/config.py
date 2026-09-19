from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every value here is an input, never a measurement."""

    model_config = SettingsConfigDict(env_prefix="CLAUSE_", env_file=".env", extra="ignore")

    database_url: str
    user_agent: str

    raw_cache_dir: Path = Path("data/raw")

    # Fetcher
    min_request_interval_s: float = Field(default=2.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    # Validation gate
    min_document_chars: int = Field(default=500, gt=0)

    # Chunking
    min_chunk_chars: int = Field(default=400, gt=0)
    max_chunk_chars: int = Field(default=2000, gt=0)
    window_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def _check_sizes(self) -> "Settings":
        if self.overlap_chars >= self.window_chars:
            raise ValueError("overlap_chars must be smaller than window_chars")
        if self.min_chunk_chars >= self.max_chunk_chars:
            raise ValueError("min_chunk_chars must be smaller than max_chunk_chars")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
