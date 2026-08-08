"""Middleware tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from insightforge.api.middleware.request_id import REQUEST_ID_HEADER, RequestIdMiddleware
from insightforge.api.middleware.timing import PROCESS_TIME_HEADER, RequestTimingMiddleware
from insightforge.core.config import get_settings
from insightforge.main import create_app


def test_request_id_generated() -> None:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    response = TestClient(app).get("/ping")

    assert response.status_code == 200
    assert REQUEST_ID_HEADER in response.headers


def test_request_id_reused() -> None:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    response = TestClient(app).get("/ping", headers={REQUEST_ID_HEADER: "req-123"})

    assert response.headers[REQUEST_ID_HEADER] == "req-123"


def test_process_time_header() -> None:
    app = FastAPI()
    app.add_middleware(RequestTimingMiddleware)

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    response = TestClient(app).get("/ping")

    assert PROCESS_TIME_HEADER in response.headers
    assert float(response.headers[PROCESS_TIME_HEADER]) >= 0


def test_cors_preflight_allows_configured_origin() -> None:
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    get_settings.cache_clear()


def test_app_exposes_request_id_and_timing() -> None:
    get_settings.cache_clear()
    client = TestClient(create_app())

    response = client.get("/api/v1/health", headers={REQUEST_ID_HEADER: "req-abc"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "req-abc"
    assert PROCESS_TIME_HEADER in response.headers
    get_settings.cache_clear()
