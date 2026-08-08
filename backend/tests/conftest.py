"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from insightforge.core.config import get_settings
from insightforge.main import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Test client for a fresh app instance."""

    get_settings.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
