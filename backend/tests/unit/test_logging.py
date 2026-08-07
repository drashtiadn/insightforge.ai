"""Logging tests."""

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from insightforge.api.middleware import RequestLoggingMiddleware
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


def test_request_logging_adds_request_id() -> None:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app)
    response = client.get("/ping")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers


def test_request_logging_reuses_incoming_request_id() -> None:
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    client = TestClient(app)
    response = client.get("/ping", headers={"X-Request-ID": "req-123"})

    assert response.headers["X-Request-ID"] == "req-123"


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
    assert response.json() == {
        "code": "internal_error",
        "message": "Internal server error",
    }
    assert any("unhandled error" in record.message for record in caplog.records)
