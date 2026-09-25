"""Application settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    """Runtime configuration. Every secret comes from the environment, never from code."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mcp_auth_token: str = Field(default="", repr=False)
    market_data_provider: Literal["yahoo", "synthetic"] = "yahoo"
    database_url: str = "sqlite:///./data/trading_mcp.db"
    model_dir: Path = Path("./data/models")
    cache_ttl_seconds: int = 300
    cors_origins: CsvList = ["https://claude.ai", "https://claude.com"]
    allowed_hosts: CsvList = []
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings()
