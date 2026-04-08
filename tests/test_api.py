"""Tests for the FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client. Model loading is slow so we test
    endpoint structure and validation, not actual generation."""
    from src.api.server import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "uptime_seconds" in data


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "parameters_human" in data
    assert "device" in data


def test_generate_endpoint(client):
    response = client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": 5,
        "temperature": 0.1,
    })
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "generated_text" in data
    assert "usage" in data
    assert "timing" in data
    assert data["usage"]["prompt_tokens"] > 0


def test_generate_validation_empty_prompt(client):
    response = client.post("/generate", json={
        "prompt": "",
        "max_tokens": 5,
    })
    assert response.status_code == 422


def test_generate_validation_bad_temperature(client):
    response = client.post("/generate", json={
        "prompt": "Hello",
        "temperature": 5.0,
    })
    assert response.status_code == 422


def test_engine_stats_endpoint(client):
    response = client.get("/engine/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "naive"
    assert "total_requests_served" in data
