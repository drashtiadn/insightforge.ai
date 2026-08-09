"""Unit tests for application configuration."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from insightforge.core.config import Environment, Settings, get_settings
from insightforge.core.config.settings import resolve_env_file


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_resolve_env_file_editable_src_layout(tmp_path: Path) -> None:
    project = tmp_path / "backend"
    module = project / "src" / "insightforge" / "core" / "config" / "settings.py"
    module.parent.mkdir(parents=True)
    module.write_text("# stub\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    assert resolve_env_file(module_file=module, cwd=tmp_path) == project / ".env"


def test_resolve_env_file_non_editable_falls_back_to_cwd(tmp_path: Path) -> None:
    site_pkg = (
        tmp_path
        / "venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "insightforge"
        / "core"
        / "config"
        / "settings.py"
    )
    site_pkg.parent.mkdir(parents=True)
    site_pkg.write_text("# stub\n", encoding="utf-8")
    workdir = tmp_path / "app"
    workdir.mkdir()

    assert resolve_env_file(module_file=site_pkg, cwd=workdir) == workdir / ".env"


def test_settings_reads_dotenv_from_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-editable layout must still load ``cwd/.env`` (not site-packages/.env)."""

    for key in ("APP_ENV", "DEBUG", "SECRET_KEY", "APP_NAME"):
        monkeypatch.delenv(key, raising=False)

    workdir = tmp_path / "app"
    workdir.mkdir()
    (workdir / ".env").write_text(
        "APP_ENV=production\nDEBUG=false\nSECRET_KEY=from-dotenv\nAPP_NAME=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(workdir)

    site_pkg = (
        tmp_path
        / "venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "insightforge"
        / "core"
        / "config"
        / "settings.py"
    )
    site_pkg.parent.mkdir(parents=True)
    site_pkg.write_text("# stub\n", encoding="utf-8")

    env_file = resolve_env_file(module_file=site_pkg, cwd=workdir)
    settings = Settings(_env_file=env_file)

    assert settings.app_env is Environment.PRODUCTION
    assert settings.app_name == "from-dotenv"
    assert settings.secret_key is not None
    assert settings.secret_key.get_secret_value() == "from-dotenv"


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
