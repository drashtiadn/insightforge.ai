"""Unit tests for application configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from insightforge.core.config import Environment, Settings, get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.app_name == "insightforge"
    assert settings.app_env is Environment.DEVELOPMENT
    assert settings.debug is False
    assert settings.secret_key is None
    assert settings.is_production is False


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_NAME", "custom-app")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("API_PORT", "9000")

    settings = Settings(_env_file=None)

    assert settings.app_name == "custom-app"
    assert settings.debug is True
    assert settings.api_port == 9000


def test_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")

    settings = Settings(_env_file=None)

    assert isinstance(settings.openai_api_key, SecretStr)
    assert settings.openai_api_key.get_secret_value() == "sk-test-secret"
    assert "sk-test-secret" not in repr(settings)


def test_production_requires_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(_env_file=None)


def test_production_rejects_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("SECRET_KEY", "prod-secret")

    with pytest.raises(ValidationError, match="DEBUG"):
        Settings(_env_file=None)


def test_production_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("SECRET_KEY", "prod-secret")

    settings = Settings(_env_file=None)

    assert settings.is_production is True
    assert settings.secret_key is not None
    assert settings.secret_key.get_secret_value() == "prod-secret"


def test_get_settings_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")

    assert get_settings() is get_settings()
