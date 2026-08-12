"""Logging tests."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from insightforge.core.config import get_settings
from insightforge.core.logging import JsonFormatter, configure_logging, get_logger
from insightforge.main import create_app


def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.method = "GET"
    record.path = "/health"

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["method"] == "GET"
    assert payload["path"] == "/health"


def test_configure_logging_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO", json_logs=True)
    get_logger("test.json").info("structured")

    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["message"] == "structured"
    assert payload["level"] == "INFO"


def test_unhandled_error_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    get_settings.cache_clear()
    app = create_app()

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    with (
        caplog.at_level(logging.ERROR),
        TestClient(app, raise_server_exceptions=False) as client,
    ):
        response = client.get("/boom")

    get_settings.cache_clear()

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["message"] == "Internal server error"
    assert "request_id" in body
    assert any(
        record.name == "insightforge.api" and "API GET /boom -> 500 (" in record.message
        for record in caplog.records
    )
