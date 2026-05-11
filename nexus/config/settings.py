from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Fail-fast application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    default_provider: Literal["anthropic", "openai"] = Field(alias="DEFAULT_PROVIDER")
    default_model: str = Field(alias="DEFAULT_MODEL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator(
        "anthropic_api_key",
        "openai_api_key",
        "default_model",
        "log_level",
    )
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        """Reject required string settings that are present but blank."""
        if not value.strip():
            raise ValueError("must not be empty")
        return value
