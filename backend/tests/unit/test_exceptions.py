"""Exception handling tests."""

import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from insightforge.core.config import get_settings
from insightforge.core.exceptions import NotFoundError, ValidationFailedError
from insightforge.main import create_app


def _app_with_routes() -> TestClient:
    get_settings.cache_clear()
    app = create_app()

    @app.get("/raise-not-found")
    async def raise_not_found() -> None:
        raise NotFoundError("Report not found", details={"report_id": "abc"})

    @app.get("/raise-validation")
    async def raise_validation() -> None:
        raise ValidationFailedError(details={"field": "query"})

    @app.get("/raise-http")
    async def raise_http() -> None:
        raise HTTPException(status_code=403, detail="No access")

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    @app.post("/echo")
    async def echo(payload: dict[str, str]) -> dict[str, str]:
        return payload

    return TestClient(app, raise_server_exceptions=False)


def test_app_exception_returns_error_schema() -> None:
    client = _app_with_routes()
    response = client.get("/raise-not-found", headers={"X-Request-ID": "req-1"})

    assert response.status_code == 404
    assert response.json() == {
        "code": "not_found",
        "message": "Report not found",
        "details": {"report_id": "abc"},
        "request_id": "req-1",
    }
    get_settings.cache_clear()


def test_validation_failed_error() -> None:
    client = _app_with_routes()
    response = client.get("/raise-validation")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_failed"
    assert body["message"] == "Validation failed"
    assert body["details"] == {"field": "query"}
    get_settings.cache_clear()


def test_http_exception_normalized() -> None:
    client = _app_with_routes()
    response = client.get("/raise-http")

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden"
    assert response.json()["message"] == "No access"
    get_settings.cache_clear()


def test_request_validation_error_schema() -> None:
    client = _app_with_routes()
    response = client.post("/echo", json="not-an-object")

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_failed"
    assert body["message"] == "Request validation failed"
    assert isinstance(body["details"], list)
    get_settings.cache_clear()


def test_unhandled_error_schema_and_log(caplog: pytest.LogCaptureFixture) -> None:
    client = _app_with_routes()

    with caplog.at_level(logging.ERROR):
        response = client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "Internal server error",
    }
    assert "boom" not in response.text
    assert any("unhandled error" in record.message for record in caplog.records)
    get_settings.cache_clear()
