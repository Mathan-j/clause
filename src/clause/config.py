from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from clause.embed import DEFAULT_MODEL


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

    # Embedding + retrieval
    embedding_model: str = DEFAULT_MODEL
    qdrant_url: str = "http://localhost:6335"

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


class GateSettings(BaseSettings):
    """The narrow slice of configuration `clause gate` needs: a database to
    read chunk counts from, and the embedding model name the fingerprint
    compares.

    Deliberately not `Settings`: comparing two committed JSON files and a
    chunk count has nothing to do with a scraping identity, so requiring
    `CLAUSE_USER_AGENT` (as the full `Settings` does, with no default) made
    the gate refuse to even start wherever only `CLAUSE_DATABASE_URL` is
    set -- which is exactly CI's `gate` step. The coupling was the bug, not
    the missing value.

    `embedding_model` shares `Settings`' default via `clause.embed.DEFAULT_MODEL`
    rather than repeating the literal: two independently-typed copies of the
    same string, with nothing pinning them equal and CI setting neither, would
    let `run_eval` write one name into the fingerprint while `run_gate`
    rebuilds a different one after nothing but a one-sided edit -- a mismatch
    that re-running `make eval` cannot clear, because the gate's own default
    would still disagree with it.
    """

    model_config = SettingsConfigDict(env_prefix="CLAUSE_", env_file=".env", extra="ignore")

    database_url: str
    embedding_model: str = DEFAULT_MODEL


@lru_cache
def get_gate_settings() -> GateSettings:
    return GateSettings()  # type: ignore[call-arg]
