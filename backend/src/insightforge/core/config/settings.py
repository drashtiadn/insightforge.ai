"""Application settings from environment variables and optional `.env`."""

from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """App environment.

    - development: local work on a feature branch
    - production: code on main (merged / deployed)
    """

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Typed app config. Env vars override `.env`, which overrides defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "insightforge"
    app_version: str = "0.1.0"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_v1_prefix: str = "/api/v1"

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton for the process."""

    return Settings()
