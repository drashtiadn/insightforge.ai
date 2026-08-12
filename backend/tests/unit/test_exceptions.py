"""Exception handling tests."""

import logging
from datetime import datetime
from uuid import UUID, uuid4

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

    @app.get("/raise-uuid-details")
    async def raise_uuid_details() -> None:
        raise NotFoundError(
            "Report not found",
            details={"report_id": UUID("12345678-1234-5678-1234-567812345678")},
        )

    @app.get("/raise-datetime-details")
    async def raise_datetime_details() -> None:
        raise NotFoundError(
            "Report not found",
            details={"created_at": datetime(2026, 8, 8, 12, 0, 0)},
        )

    @app.get("/raise-unserializable-details")
    async def raise_unserializable_details() -> None:
        raise NotFoundError("Report not found", details={"obj": object()})

    @app.get("/raise-http-uuid")
    async def raise_http_uuid() -> None:
        raise HTTPException(status_code=404, detail={"id": uuid4()})

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
    body = response.json()
    assert body["code"] == "internal_error"
    assert body["message"] == "Internal server error"
    assert "request_id" in body
    assert "boom" not in response.text
    error_records = [
        record
        for record in caplog.records
        if record.name == "insightforge.api" and "API GET /boom -> 500 (" in record.message
    ]
    assert error_records
    assert error_records[0].exc_info is not None
    get_settings.cache_clear()


def test_app_exception_serializes_uuid_details() -> None:
    client = _app_with_routes()
    response = client.get(
        "/raise-uuid-details",
        headers={"X-Request-ID": "req-uuid", "Origin": "http://localhost:3000"},
    )

    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == "req-uuid"
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.json() == {
        "code": "not_found",
        "message": "Report not found",
        "details": {"report_id": "12345678-1234-5678-1234-567812345678"},
        "request_id": "req-uuid",
    }
    get_settings.cache_clear()


def test_app_exception_serializes_datetime_details() -> None:
    client = _app_with_routes()
    response = client.get("/raise-datetime-details")

    assert response.status_code == 404
    assert response.json()["details"] == {"created_at": "2026-08-08T12:00:00"}
    get_settings.cache_clear()


def test_app_exception_drops_unserializable_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _app_with_routes()

    with caplog.at_level(logging.ERROR):
        response = client.get(
            "/raise-unserializable-details",
            headers={"X-Request-ID": "req-drop"},
        )

    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == "req-drop"
    assert response.json() == {
        "code": "not_found",
        "message": "Report not found",
        "request_id": "req-drop",
    }
    assert "details" not in response.json()
    assert any(
        "failed to serialize error details" in record.message for record in caplog.records
    )
    get_settings.cache_clear()


def test_http_exception_serializes_uuid_detail() -> None:
    client = _app_with_routes()
    response = client.get("/raise-http-uuid")

    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "not_found"
    assert body["message"] == "HTTP error"
    assert isinstance(body["details"]["detail"]["id"], str)
    UUID(body["details"]["detail"]["id"])
    get_settings.cache_clear()
