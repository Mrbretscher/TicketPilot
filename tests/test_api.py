from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from test_orchestration import orchestration_service
from ticketpilot.api import create_app
from ticketpilot.orchestration import TicketPilotService


def api_client(tmp_path: Path, service: TicketPilotService | None = None) -> TestClient:
    app = create_app(
        service=service,
        load_artifacts=False,
        review_database_path=tmp_path / "reviews.sqlite",
    )
    return TestClient(app)


def test_health_reports_ready_with_injected_service(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service())

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert payload["status"] == "ok"


def test_health_reports_not_ready_without_artifacts(tmp_path: Path) -> None:
    client = api_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["status"] == "not_ready"


def test_model_info_requires_ready_service(tmp_path: Path) -> None:
    client = api_client(tmp_path)

    response = client.get("/model-info")

    assert response.status_code == 503
    assert "stack" not in str(response.json()).lower()


def test_classify_contract(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service())

    response = client.post(
        "/classify",
        json={"subject": "VPN login", "body": "Cannot authenticate."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_queue"] == "Technical Support"
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["top_queues"]


def test_retrieve_contract(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service())

    response = client.post(
        "/retrieve",
        json={"subject": "invoice", "body": "Need duplicate refund.", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["evidence"]) == 1
    assert payload["evidence"][0]["source_id"].startswith("TP-ticket-")
    assert 0.0 <= payload["evidence"][0]["similarity"] <= 1.0


def test_analyze_contract_uses_fake_generation_provider(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service())

    response = client.post(
        "/analyze",
        json={
            "subject": "VPN login fails",
            "body": "User cannot authenticate to VPN from laptop.",
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predicted_queue"] == "Technical Support"
    assert payload["predicted_priority"] is None
    assert payload["human_review_required"] is True
    assert payload["retrieval_scores"][0]["source_id"] == "TP-ticket-000001"
    assert payload["citations"] == ["TP-ticket-000001"]
    assert payload["model_info"]["priority_classifier"]["supported"] is False
    assert payload["model_info"]["retrieval"]["method"] == "tfidf_cosine_similarity"


def test_malformed_request_is_rejected(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service())

    response = client.post("/classify", json={"subject": "missing body"})

    assert response.status_code == 422


def test_max_text_length_is_rejected_without_stack_trace(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service(max_ticket_text_chars=10))

    response = client.post(
        "/classify",
        json={"subject": "short", "body": "x" * 20},
    )

    assert response.status_code == 400
    payload_text = str(response.json()).lower()
    assert "maximum length" in payload_text
    assert "traceback" not in payload_text


def test_review_create_and_get_contract(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service())

    create_response = client.post(
        "/reviews",
        json={
            "analysis_id": "analysis-api-001",
            "ticket_text": "VPN login fails User cannot authenticate.",
            "predicted_queue": "Technical Support",
            "queue_confidence": 0.84,
            "retrieved_evidence_ids": ["TP-ticket-000001"],
            "draft_response": "Draft for review. [TP-ticket-000001]",
            "model_provider_metadata": {"provider": "fake"},
        },
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["state"] == "pending"
    assert created["ticket_text_hash"]

    get_response = client.get("/reviews/analysis-api-001")

    assert get_response.status_code == 200
    assert get_response.json()["analysis_id"] == "analysis-api-001"


def test_missing_review_returns_404(tmp_path: Path) -> None:
    client = api_client(tmp_path, orchestration_service())

    response = client.get("/reviews/missing")

    assert response.status_code == 404
