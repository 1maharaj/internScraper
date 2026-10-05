"""
test_api.py
-----------
Tests for the scam_detector FastAPI endpoints.
"""

from __future__ import annotations

import pytest
from scam_detector.api import _FASTAPI_AVAILABLE, create_app

if _FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_health() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "shap_available" in data
    assert "decision_thresholds" in data


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_score_single() -> None:
    app = create_app()
    client = TestClient(app)
    payload = {
        "title": "Software Engineering Intern",
        "company": "Google",
        "applyLink": "https://careers.google.com/jobs/123",
        "stipend": {"amount": 50000, "period": "monthly"},
        "summary": "Join our engineering team to build scalable systems.",
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "scam_score" in data
    assert "decision" in data
    assert data["decision"] in ("clear", "review", "block")


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_score_batch() -> None:
    app = create_app()
    client = TestClient(app)
    payload = {
        "records": [
            {
                "_id": "rec_1",
                "name": "Backend Intern",
                "company": "Company A",
                "applyLink": "https://company-a.com/apply",
                "summary": "Develop python APIs",
            },
            {
                "_id": "rec_2",
                "name": "Frontend Intern",
                "company": "Company B",
                "applyLink": "https://company-b.com/apply",
                "summary": "Develop React components",
            },
        ]
    }
    response = client.post("/score/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert "results" in data


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_graph_analyze() -> None:
    app = create_app()
    client = TestClient(app)
    payload = {
        "records": [
            {"company": "Org 1", "applyLink": "https://sharedinfra.xyz/apply"},
            {"company": "Org 2", "applyLink": "https://sharedinfra.xyz/apply"},
            {"company": "Org 3", "applyLink": "https://sharedinfra.xyz/apply"},
            {"company": "Org 4", "applyLink": "https://sharedinfra.xyz/apply"},
        ],
        "target_company": "Org 1",
    }
    response = client.post("/graph/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert data["metrics"]["multi_company_networks_count"] == 1
    assert data["target_company_profile"]["shared_infrastructure"] is True


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_list_rules() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/rules")
    assert response.status_code == 200
    data = response.json()
    assert data["total_rules"] >= 11
    rule_ids = [r["rule_id"] for r in data["rules"]]
    assert "upfront_fee_and_pay_to_work" in rule_ids
    assert "suspicious_recruiter_contact" in rule_ids
    assert "urgency_psychological_pressure" in rule_ids


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_reputation_unknown_company() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/reputation/UnknownNonexistentCorp123")
    assert response.status_code == 200
    data = response.json()
    assert data["known"] is False
    assert data["reputation_score"] is None
    assert data["risk_level"] == "unknown"


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_stats() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_rules"] >= 11
    assert "decision_thresholds" in data
    assert "rule_weights" in data


@pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI is not installed")
def test_api_benchmark_sample() -> None:
    app = create_app()
    client = TestClient(app)
    payload = {
        "records": [
            {
                "title": "Data Entry Specialist",
                "company": "Fake Scam Co",
                "applyLink": "http://scam-pay.xyz/apply",
                "payment_required": 1,
                "registration_fee": 1000.0,
                "is_fake_posting": 1,
            },
            {
                "title": "Software Engineering Intern",
                "company": "Google LLC",
                "applyLink": "https://careers.google.com/jobs/123",
                "is_fake_posting": 0,
            },
        ],
        "label_field": "is_fake_posting",
        "decision_threshold": 50.0,
    }
    response = client.post("/benchmark/sample", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 2
    assert data["accuracy"] >= 0.50
    assert "precision" in data
    assert "recall" in data
    assert "f1_score" in data
    assert "latency_ms_per_record" in data

