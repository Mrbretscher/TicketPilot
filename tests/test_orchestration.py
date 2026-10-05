from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ticketpilot.drafting import FakeDraftGenerator
from ticketpilot.orchestration import (
    TicketPilotService,
    TicketValidationError,
    load_ticketpilot_service,
)
from ticketpilot.preparation import GROUP_ID_COLUMN, ROW_ID_COLUMN, SPLIT_COLUMN
from ticketpilot.retrieval import (
    RETRIEVAL_TEXT_COLUMN,
    SOURCE_ID_COLUMN,
    build_tfidf_retriever,
)


class FakeQueueClassifier:
    classes_ = np.asarray(["Billing and Payments", "Technical Support"])

    def predict(self, texts: pd.Series) -> np.ndarray[Any, Any]:
        values = texts.astype("string").tolist()
        if "invoice" in values[0].lower():
            return np.asarray(["Billing and Payments"])
        return np.asarray(["Technical Support"])

    def predict_proba(self, texts: pd.Series) -> np.ndarray[Any, Any]:
        values = texts.astype("string").tolist()
        if "low confidence" in values[0].lower():
            return np.asarray([[0.51, 0.49]])
        if "invoice" in values[0].lower():
            return np.asarray([[0.87, 0.13]])
        return np.asarray([[0.16, 0.84]])


def orchestration_service(*, max_ticket_text_chars: int = 6_000) -> TicketPilotService:
    corpus = pd.DataFrame(
        [
            {
                SOURCE_ID_COLUMN: "TP-ticket-000001",
                ROW_ID_COLUMN: "ticket-000001",
                GROUP_ID_COLUMN: "group-001",
                SPLIT_COLUMN: "train",
                "subject": "VPN login fails",
                "body": "User cannot authenticate to VPN from laptop.",
                "answer": "Reset MFA session and verify VPN profile.",
                "queue": "Technical Support",
                "priority": "high",
                RETRIEVAL_TEXT_COLUMN: (
                    "VPN login fails User cannot authenticate to VPN from laptop."
                ),
            },
            {
                SOURCE_ID_COLUMN: "TP-ticket-000002",
                ROW_ID_COLUMN: "ticket-000002",
                GROUP_ID_COLUMN: "group-002",
                SPLIT_COLUMN: "train",
                "subject": "Invoice refund request",
                "body": "Customer needs a billing refund for duplicate invoice.",
                "answer": "Review invoice ledger and issue refund if duplicate.",
                "queue": "Billing and Payments",
                "priority": "medium",
                RETRIEVAL_TEXT_COLUMN: (
                    "Invoice refund request Customer needs a billing refund "
                    "for duplicate invoice."
                ),
            },
        ]
    )
    return TicketPilotService(
        queue_classifier=FakeQueueClassifier(),
        retriever=build_tfidf_retriever(corpus),
        draft_generator=FakeDraftGenerator(),
        max_ticket_text_chars=max_ticket_text_chars,
        min_classifier_confidence=0.60,
        min_evidence_score=0.05,
    )


def test_service_constructs_classifier_text_from_subject_and_body_only() -> None:
    service = orchestration_service()

    ticket = service.validate_and_normalize_ticket(
        subject=" VPN access ",
        body=" User cannot connect. ",
    )

    assert ticket.classifier_text == "VPN access\n\nUser cannot connect."


def test_service_rejects_empty_or_oversized_ticket_text() -> None:
    service = orchestration_service(max_ticket_text_chars=10)

    with pytest.raises(TicketValidationError, match="cannot both be empty"):
        service.validate_and_normalize_ticket(subject=" ", body=" ")

    with pytest.raises(TicketValidationError, match="exceeds maximum length"):
        service.validate_and_normalize_ticket(subject="short", body="x" * 20)


def test_classification_output_schema_and_confidence_range() -> None:
    service = orchestration_service()

    prediction = service.classify(subject="VPN login", body="Cannot authenticate.")

    assert prediction.predicted_queue == "Technical Support"
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.top_queues[0]["queue"] == "Technical Support"


def test_analyze_returns_grounded_draft_with_citations() -> None:
    service = orchestration_service()

    result = service.analyze(
        subject="VPN login fails",
        body="User cannot authenticate to VPN from laptop.",
        top_k=1,
    )

    assert result.predicted_queue == "Technical Support"
    assert result.predicted_priority is None
    assert len(result.retrieved_evidence) == 1
    assert result.citations == ("TP-ticket-000001",)
    assert result.human_review_required is True
    assert "grounded_draft_ready_for_review" in result.reasons


def test_low_classifier_confidence_abstains_from_ready_draft() -> None:
    service = orchestration_service()

    result = service.analyze(
        subject="low confidence",
        body="Ambiguous issue with login and invoice.",
        top_k=1,
    )

    assert result.confidence_evidence_status == "low_classifier_confidence"
    assert result.abstention_reason is not None
    assert result.citations == ()
    assert "low_classifier_confidence" in result.reasons


def test_retrieval_uses_train_corpus_source_ids() -> None:
    service = orchestration_service()

    evidence = service.retrieve(
        subject="duplicate invoice", body="Need refund.", top_k=2
    )

    assert len(evidence) == 2
    assert {item.source_id for item in evidence} == {
        "TP-ticket-000001",
        "TP-ticket-000002",
    }


def test_missing_artifacts_fail_readiness_without_training(tmp_path: Any) -> None:
    with pytest.raises(Exception, match="Missing required artifacts"):
        load_ticketpilot_service(
            queue_model_path=tmp_path / "missing-model.joblib",
            retrieval_index_path=tmp_path / "missing-retriever.joblib",
        )
