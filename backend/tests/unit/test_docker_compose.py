"""Regression checks for production compose configuration."""

from pathlib import Path

_COMPOSE_PATH = Path(__file__).resolve().parents[3] / "infra" / "docker" / "docker-compose.yml"


def test_production_compose_does_not_default_secret_key() -> None:
    """`environment` overrides `env_file`; a weak SECRET_KEY default would win.

    Secrets must come from backend/.env (env_file) or an explicit host export,
    not from a known placeholder baked into compose.
    """

    text = _COMPOSE_PATH.read_text(encoding="utf-8")

    assert "change-me-in-production" not in text
    assert "SECRET_KEY:" not in text
