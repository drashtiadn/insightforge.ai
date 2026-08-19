"""Health and root endpoint tests."""

from fastapi.testclient import TestClient

from insightforge.core.config import get_settings


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_health_response_shape(client: TestClient) -> None:
    settings = get_settings()

    response = client.get("/api/v1/health")

    assert response.json() == {
        "status": "ok",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env.value,
    }


def test_root_returns_service_discovery(client: TestClient) -> None:
    settings = get_settings()

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.app_env.value,
        "docs_url": "/docs",
        "health_url": f"{settings.api_v1_prefix}/health",
        "research_url": f"{settings.api_v1_prefix}/research",
    }


def test_openapi_lists_versioned_health(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/api/v1/health" in paths
    assert "/api/v1/research" in paths
    assert "/" in paths
