"""Support helpers for the recruiter-facing TicketPilot Streamlit dashboard."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ticketpilot.config import (
    DATASET_SUMMARY_PATH,
    QUEUE_BASELINE_REPORT_PATH,
    RETRIEVAL_REPORT_PATH,
    SEMANTIC_RETRIEVAL_REPORT_PATH,
    TENSORFLOW_REPORT_PATH,
)
from ticketpilot.review import ReviewRecord, ReviewState, list_review_records

PROJECT_ROOT = Path(os.getenv("TICKETPILOT_PROJECT_ROOT", Path.cwd())).resolve()


@dataclass(frozen=True)
class DemoTicket:
    """A public/synthetic demo ticket for the Streamlit console."""

    name: str
    subject: str
    body: str
    note: str


@dataclass(frozen=True)
class ReportBundle:
    """A loaded JSON report and its path."""

    path: Path
    report: dict[str, Any]


def demo_tickets() -> tuple[DemoTicket, ...]:
    """Return safe demo tickets derived from public/synthetic project examples."""
    return (
        DemoTicket(
            name="VPN access problem",
            subject="VPN login fails after MFA reset",
            body=(
                "I can sign in to the portal, but the VPN client rejects my login "
                "after I reset MFA on my work laptop."
            ),
            note="Synthetic IT-support ticket matching the public dataset style.",
        ),
        DemoTicket(
            name="Duplicate invoice charge",
            subject="Duplicate invoice charge needs review",
            body=(
                "Our account shows two charges for the same invoice number. "
                "Please review the billing ledger and advise next steps."
            ),
            note="Synthetic billing ticket matching the public dataset style.",
        ),
        DemoTicket(
            name="Mobile app crash",
            subject="Mobile app crashes on launch",
            body=(
                "The mobile app closes immediately after the latest update. "
                "I have already restarted the phone and reinstalled the app."
            ),
            note="Synthetic product-support ticket matching the public dataset style.",
        ),
    )


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a repository-relative path for local dashboard use."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_report(path: str | Path) -> ReportBundle:
    """Load one machine-readable JSON report."""
    report_path = resolve_project_path(path)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Report is not a JSON object: {report_path}")
    return ReportBundle(path=report_path, report=payload)


def load_report_if_available(path: str | Path) -> ReportBundle | None:
    """Load a report if present; otherwise return None for a helpful UI state."""
    try:
        return load_report(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return None


def default_report_bundles() -> dict[str, ReportBundle | None]:
    """Load dashboard reports from the repository's ignored artifact paths."""
    return {
        "queue": load_report_if_available(QUEUE_BASELINE_REPORT_PATH),
        "tensorflow": load_report_if_available(TENSORFLOW_REPORT_PATH),
        "retrieval": load_report_if_available(RETRIEVAL_REPORT_PATH),
        "semantic_retrieval": load_report_if_available(SEMANTIC_RETRIEVAL_REPORT_PATH),
        "dataset": load_report_if_available(DATASET_SUMMARY_PATH),
    }


def classifier_metric_cards(report: dict[str, Any]) -> list[dict[str, str]]:
    """Return headline classifier metrics from a queue baseline report."""
    metrics = _mapping(report.get("test_metrics"))
    abstention = _mapping(_mapping(report.get("abstention")).get("test_result"))
    return [
        {"label": "Accuracy", "value": format_metric(metrics.get("accuracy"))},
        {"label": "Macro F1", "value": format_metric(metrics.get("macro_f1"))},
        {"label": "Weighted F1", "value": format_metric(metrics.get("weighted_f1"))},
        {
            "label": "Top-3 Accuracy",
            "value": format_metric(metrics.get("top_k_accuracy")),
        },
        {
            "label": "Review Rate",
            "value": format_percent(abstention.get("review_rate")),
        },
    ]


def confusion_matrix_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Return the final-test confusion matrix as a labeled dataframe."""
    matrix_payload = _mapping(
        _mapping(report.get("test_metrics")).get("confusion_matrix")
    )
    labels = [str(label) for label in matrix_payload.get("labels", [])]
    matrix = matrix_payload.get("matrix", [])
    if not labels or not isinstance(matrix, list):
        return pd.DataFrame()
    return pd.DataFrame(matrix, index=labels, columns=labels)


def per_class_metrics_frame(report: dict[str, Any]) -> pd.DataFrame:
    """Return per-class precision, recall, F1, and support from a report."""
    per_class = _mapping(_mapping(report.get("test_metrics")).get("per_class"))
    rows = []
    for label, metrics in per_class.items():
        metric_map = _mapping(metrics)
        rows.append(
            {
                "queue": label,
                "precision": metric_map.get("precision"),
                "recall": metric_map.get("recall"),
                "f1": metric_map.get("f1"),
                "support": metric_map.get("support"),
            }
        )
    return pd.DataFrame(rows)


def class_distribution_frame(
    report: dict[str, Any], *, split: str = "test"
) -> pd.DataFrame:
    """Return class distribution rows for one split."""
    distribution = _mapping(_mapping(report.get("class_distribution")).get(split))
    rows = []
    for label, values in distribution.items():
        value_map = _mapping(values)
        rows.append(
            {
                "queue": label,
                "count": value_map.get("count"),
                "rate": value_map.get("rate"),
            }
        )
    return pd.DataFrame(rows)


def model_comparison_rows(
    queue_report: dict[str, Any] | None,
    tensorflow_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return model comparison rows from measured repository artifacts."""
    rows: list[dict[str, Any]] = []
    if queue_report:
        selected = _mapping(queue_report.get("selected_model"))
        metrics = _mapping(queue_report.get("test_metrics"))
        latency = _mapping(metrics.get("inference_latency"))
        rows.append(
            {
                "model": selected.get("name", "sklearn_queue_baseline"),
                "selected_for_deployment": True,
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "top_k_accuracy": metrics.get("top_k_accuracy"),
                "latency_ms": latency.get("milliseconds_per_ticket"),
                "artifact_size_bytes": selected.get("artifact_size_bytes"),
            }
        )
    if tensorflow_report:
        metrics = _mapping(tensorflow_report.get("test_metrics"))
        comparison = _mapping(tensorflow_report.get("sklearn_comparison"))
        latency = _mapping(comparison.get("inference_latency_ms_per_ticket"))
        artifact_size = _mapping(comparison.get("artifact_size_bytes"))
        rows.append(
            {
                "model": tensorflow_report.get(
                    "model_name",
                    "tensorflow_text_classifier",
                ),
                "selected_for_deployment": False,
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "top_k_accuracy": metrics.get("top_k_accuracy"),
                "latency_ms": latency.get("tensorflow"),
                "artifact_size_bytes": artifact_size.get("tensorflow"),
            }
        )
    return rows


def retrieval_metric_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return retrieval metrics from lexical or semantic retrieval reports."""
    if not report:
        return []
    evaluation = _mapping(report.get("evaluation"))
    rows: list[dict[str, Any]] = []
    for method, value in evaluation.items():
        if isinstance(value, dict) and {"validation", "test"}.intersection(value):
            for split, metrics in value.items():
                rows.append(_retrieval_row(str(method), str(split), _mapping(metrics)))
        else:
            rows.append(
                _retrieval_row(
                    str(report.get("retriever", "retrieval")),
                    str(method),
                    _mapping(value),
                )
            )
    return rows


def selected_retrieval_method(report: dict[str, Any] | None) -> str:
    """Return the measured selected retrieval method when the report records one."""
    if not report:
        return "Unavailable"
    selection = _mapping(report.get("selection"))
    return str(selection.get("selected_method", report.get("retriever", "Unavailable")))


def review_records_frame(
    *,
    path: Path,
    state: ReviewState | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Return review records as a dataframe for the Review Queue page."""
    records = list_review_records(path=path, state=state, limit=limit)
    rows = [review_record_summary(record) for record in records]
    return pd.DataFrame(rows)


def review_record_summary(record: ReviewRecord) -> dict[str, Any]:
    """Return a compact review summary without full ticket text."""
    return {
        "analysis_id": record.analysis_id,
        "state": record.state,
        "predicted_queue": record.predicted_queue,
        "queue_confidence": record.queue_confidence,
        "predicted_priority": record.predicted_priority,
        "evidence_count": len(record.retrieved_evidence_ids),
        "reviewer_action": record.reviewer_action,
        "final_queue": record.final_queue,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def dataset_limitations() -> tuple[str, ...]:
    """Return documented dashboard limitations for the About page."""
    return (
        "The source dataset is public and non-commercial; it is not private "
        "enterprise support data.",
        "Queue labels are dataset labels, not an organization's live routing policy.",
        "Retrieval relevance is measured with silver queue-match labels unless a "
        "future human-labeled gold set is added.",
        "Draft responses are for human review only and must not be sent automatically.",
        "No authentication or RBAC is included in this portfolio dashboard.",
    )


def abbreviate_text(text: object, *, max_chars: int = 180) -> str:
    """Return a compact one-line preview for issue and resolution text."""
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def format_metric(value: object) -> str:
    """Format a numeric metric for display."""
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "n/a"


def format_percent(value: object) -> str:
    """Format a numeric rate as a percentage."""
    if isinstance(value, int | float):
        return f"{float(value):.1%}"
    return "n/a"


def _retrieval_row(method: str, split: str, metrics: dict[str, Any]) -> dict[str, Any]:
    recall = _mapping(metrics.get("recall_at_k"))
    return {
        "method": metrics.get("method", method),
        "split": split,
        "recall@1": recall.get("recall@1"),
        "recall@3": recall.get("recall@3"),
        "recall@5": recall.get("recall@5"),
        "mrr": metrics.get("mrr"),
        "latency_ms": metrics.get("mean_latency_ms_per_query"),
        "relevance": metrics.get("relevance_type"),
        "human_labeled": metrics.get("human_labeled_relevance"),
    }


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
