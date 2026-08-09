"""Application settings from environment variables and optional `.env`."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def resolve_env_file(module_file: Path | None = None, cwd: Path | None = None) -> Path:
    """Locate the project ``.env`` for both editable and wheel installs.

    Editable / Docker src layout keeps ``settings.py`` under ``.../src/insightforge/...``,
    so walking parents finds the directory that contains ``pyproject.toml`` (e.g. ``backend/``
    or ``/app``).

    Non-editable installs place the module under ``site-packages``, where that walk cannot
    find the project root. Falling back to ``cwd / ".env"`` matches the image/workdir
    convention (``/app/.env``) instead of inventing a path under ``lib/pythonX.Y/``.
    """

    start = (module_file or Path(__file__)).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / ".env"
    return (cwd or Path.cwd()) / ".env"


_ENV_FILE = resolve_env_file()


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
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    app_name: str = "insightforge"
    app_version: str = "0.1.0"
    app_env: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_v1_prefix: str = "/api/v1"

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
    ]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    cors_expose_headers: list[str] = ["X-Request-ID", "X-Process-Time"]

    secret_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    @property
    def is_production(self) -> bool:
        return self.app_env == Environment.PRODUCTION

    @model_validator(mode="after")
    def check_production(self) -> Self:
        if self.is_production:
            if self.debug:
                raise ValueError("DEBUG must be false in production")
            if self.secret_key is None:
                raise ValueError("SECRET_KEY is required in production")
        return self


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton for the process."""

    return Settings()
