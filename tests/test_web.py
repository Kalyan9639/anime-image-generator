"""
Unit tests for the Flask Web Application (web.app).
"""

import json
import pytest
from web.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"ASCII" in res.data


def test_samples_api(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_detect_api(client):
    sample = "+---+    +---+\n| A |--->| B |\n+---+    +---+"
    res = client.post("/api/detect", data=json.dumps({"text": sample}), content_type="application/json")
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["category"] == "diagram"
    assert data["confidence"] > 0.6


def test_convert_api_diagram(client):
    sample = "+---+    +---+\n| A |--->| B |\n+---+    +---+"
    res = client.post(
        "/api/convert",
        data=json.dumps({"text": sample, "mode": "diagram", "theme": "light", "scale": 1.0}),
        content_type="application/json"
    )
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["status"] == "success"
    assert "<svg" in data["svg"]
    assert len(data["png_b64"]) > 100
