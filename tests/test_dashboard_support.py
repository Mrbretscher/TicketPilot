from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ticketpilot.dashboard_support import (
    abbreviate_text,
    class_distribution_frame,
    classifier_metric_cards,
    confusion_matrix_frame,
    dataset_limitations,
    demo_tickets,
    load_report,
    model_comparison_rows,
    per_class_metrics_frame,
    retrieval_metric_rows,
    review_records_frame,
    selected_retrieval_method,
)
from ticketpilot.review import create_review_record, submit_review_action


def queue_report_fixture() -> dict[str, object]:
    return {
        "selected_model": {
            "name": "tfidf_linear_svc",
            "artifact_size_bytes": 123,
        },
        "abstention": {"test_result": {"review_rate": 0.1}},
        "class_distribution": {
            "test": {
                "Technical Support": {"count": 8, "rate": 0.8},
                "Billing and Payments": {"count": 2, "rate": 0.2},
            }
        },
        "test_metrics": {
            "accuracy": 0.7,
            "macro_f1": 0.6,
            "weighted_f1": 0.65,
            "top_k_accuracy": 0.9,
            "inference_latency": {"milliseconds_per_ticket": 0.5},
            "confusion_matrix": {
                "labels": ["Billing and Payments", "Technical Support"],
                "matrix": [[2, 0], [1, 7]],
            },
            "per_class": {
                "Billing and Payments": {
                    "precision": 0.67,
                    "recall": 1.0,
                    "f1": 0.8,
                    "support": 2,
                },
                "Technical Support": {
                    "precision": 1.0,
                    "recall": 0.875,
                    "f1": 0.93,
                    "support": 8,
                },
            },
        },
    }


def test_demo_tickets_are_safe_public_synthetic_examples() -> None:
    tickets = demo_tickets()

    assert len(tickets) >= 3
    assert all(ticket.subject and ticket.body for ticket in tickets)
    assert all("Synthetic" in ticket.note for ticket in tickets)


def test_load_report_reads_json_object(tmp_path: Path) -> None:
    report_path = tmp_path / "metrics.json"
    report_path.write_text(json.dumps({"task": "demo"}), encoding="utf-8")

    bundle = load_report(report_path)

    assert bundle.path == report_path
    assert bundle.report == {"task": "demo"}


def test_load_report_rejects_non_object_json(tmp_path: Path) -> None:
    report_path = tmp_path / "metrics.json"
    report_path.write_text(json.dumps(["bad"]), encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        load_report(report_path)


def test_classifier_report_helpers_parse_measured_metrics() -> None:
    report = queue_report_fixture()

    cards = classifier_metric_cards(report)
    matrix = confusion_matrix_frame(report)
    per_class = per_class_metrics_frame(report)
    distribution = class_distribution_frame(report, split="test")

    assert cards[1] == {"label": "Macro F1", "value": "0.600"}
    assert matrix.loc["Technical Support", "Technical Support"] == 7
    assert set(per_class["queue"]) == {"Billing and Payments", "Technical Support"}
    assert distribution["count"].sum() == 10


def test_model_comparison_rows_use_queue_and_tensorflow_reports() -> None:
    tensorflow_report = {
        "model_name": "tensorflow_textvectorization_embedding_conv1d",
        "test_metrics": {
            "macro_f1": 0.4,
            "weighted_f1": 0.45,
            "top_k_accuracy": 0.7,
        },
        "sklearn_comparison": {
            "inference_latency_ms_per_ticket": {"tensorflow": 0.3},
            "artifact_size_bytes": {"tensorflow": 456},
        },
    }

    rows = model_comparison_rows(queue_report_fixture(), tensorflow_report)

    assert rows[0]["model"] == "tfidf_linear_svc"
    assert rows[0]["selected_for_deployment"] is True
    assert rows[1]["model"] == "tensorflow_textvectorization_embedding_conv1d"
    assert rows[1]["selected_for_deployment"] is False


def test_retrieval_metric_rows_support_semantic_and_lexical_shapes() -> None:
    semantic_report = {
        "selection": {"selected_method": "hybrid"},
        "evaluation": {
            "hybrid": {
                "validation": {
                    "method": "hybrid",
                    "recall_at_k": {"recall@1": 0.7, "recall@3": 0.8, "recall@5": 0.9},
                    "mrr": 0.75,
                    "mean_latency_ms_per_query": 10,
                    "relevance_type": "silver_queue_match",
                    "human_labeled_relevance": False,
                }
            }
        },
    }
    lexical_report = {
        "retriever": "tfidf_cosine_similarity",
        "evaluation": {
            "test": {
                "recall_at_k": {"recall@1": 0.6, "recall@3": 0.7, "recall@5": 0.8},
                "mrr": 0.65,
            }
        },
    }

    semantic_rows = retrieval_metric_rows(semantic_report)
    lexical_rows = retrieval_metric_rows(lexical_report)

    assert selected_retrieval_method(semantic_report) == "hybrid"
    assert semantic_rows[0]["method"] == "hybrid"
    assert lexical_rows[0]["method"] == "tfidf_cosine_similarity"
    assert lexical_rows[0]["split"] == "test"


def test_review_records_frame_omits_full_ticket_text(tmp_path: Path) -> None:
    path = tmp_path / "reviews.sqlite"
    record = create_review_record(
        analysis_id="analysis-dashboard-001",
        ticket_text="Sensitive ticket body should not appear in queue table.",
        predicted_queue="Technical Support",
        queue_confidence=0.8,
        retrieved_evidence_ids=("TP-ticket-000001",),
        draft_response="Draft response for review.",
        model_provider_metadata={"provider": "fake"},
        path=path,
    )
    submit_review_action(record.analysis_id, action="accept", path=path)

    frame = review_records_frame(path=path)

    assert frame.loc[0, "analysis_id"] == "analysis-dashboard-001"
    assert frame.loc[0, "state"] == "approved"
    assert "ticket_text" not in frame.columns


def test_abbreviate_text_and_limitations() -> None:
    assert abbreviate_text("alpha beta gamma", max_chars=20) == "alpha beta gamma"
    assert abbreviate_text("x" * 30, max_chars=10) == "xxxxxxx..."
    assert any("human review" in item.lower() for item in dataset_limitations())


def test_confusion_matrix_empty_for_missing_payload() -> None:
    assert confusion_matrix_frame({}).empty
    assert isinstance(per_class_metrics_frame({}), pd.DataFrame)
