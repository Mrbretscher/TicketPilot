"""Evaluation helpers for multiclass queue routing."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_queue_classifier(
    estimator: Any,
    x: pd.Series,
    y_true: pd.Series,
    *,
    labels: list[str],
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate a fitted multiclass queue classifier."""
    start = perf_counter()
    y_pred = pd.Series(estimator.predict(x), index=y_true.index)
    elapsed_seconds = perf_counter() - start
    scores = class_score_matrix(estimator, x, labels=labels)
    return evaluate_multiclass_predictions(
        y_true,
        y_pred,
        scores,
        labels=labels,
        elapsed_seconds=elapsed_seconds,
        top_k=top_k,
    )


def evaluate_multiclass_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    scores: np.ndarray,
    *,
    labels: list[str],
    elapsed_seconds: float,
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate multiclass predictions and aligned per-class scores."""

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.astype(int).tolist(),
        },
        "top_k_accuracy": top_k_accuracy(y_true, scores, labels=labels, k=top_k),
        "inference_latency": {
            "rows": int(len(y_true)),
            "total_seconds": float(elapsed_seconds),
            "milliseconds_per_ticket": float(
                (elapsed_seconds / max(len(y_true), 1)) * 1000
            ),
        },
    }


def predict_queue_routes(
    estimator: Any,
    x: pd.Series,
    *,
    labels: list[str],
    top_k: int = 3,
) -> pd.DataFrame:
    """Return queue predictions, confidence, and top-k candidates."""
    scores = class_score_matrix(estimator, x, labels=labels)
    predictions = np.asarray(estimator.predict(x))
    top_k = min(top_k, len(labels))
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
    top_labels = [[labels[index] for index in row] for row in top_indices]
    confidence = scores.max(axis=1)
    return pd.DataFrame(
        {
            "predicted_queue": predictions,
            "confidence": confidence.astype(float),
            "top_queues": top_labels,
        },
        index=x.index,
    )


def class_score_matrix(
    estimator: Any, x: pd.Series, *, labels: list[str]
) -> np.ndarray:
    """Return per-class scores aligned to ``labels``."""
    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(x), dtype=float)
        estimator_classes = [str(label) for label in estimator.classes_]
        return _align_score_columns(scores, estimator_classes, labels)

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x), dtype=float)
        estimator_classes = [str(label) for label in estimator.classes_]
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return _align_score_columns(scores, estimator_classes, labels)

    predictions = np.asarray(estimator.predict(x))
    scores = np.zeros((len(predictions), len(labels)), dtype=float)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for row_index, prediction in enumerate(predictions):
        scores[row_index, label_to_index[str(prediction)]] = 1.0
    return scores


def top_k_accuracy(
    y_true: pd.Series,
    scores: np.ndarray,
    *,
    labels: list[str],
    k: int,
) -> float:
    """Compute multiclass top-k accuracy from an aligned score matrix."""
    k = min(k, len(labels))
    label_to_index = {label: index for index, label in enumerate(labels)}
    truth_indices = np.asarray([label_to_index[str(label)] for label in y_true])
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    hits = [truth in row for truth, row in zip(truth_indices, top_indices, strict=True)]
    return float(np.mean(hits)) if hits else 0.0


def select_abstention_threshold(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    *,
    labels: list[str],
    minimum_coverage: float,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Select a validation-only confidence threshold for recommendation coverage."""
    if thresholds is None:
        thresholds = [round(value / 100, 2) for value in range(0, 101, 5)]

    rows = [
        evaluate_abstention(
            y_true,
            predictions["predicted_queue"],
            predictions["confidence"],
            threshold=threshold,
            labels=labels,
        )
        for threshold in thresholds
    ]
    eligible = [row for row in rows if row["coverage"] >= minimum_coverage]
    candidates = eligible if eligible else rows
    selected = max(
        candidates,
        key=lambda row: (
            row["auto_macro_f1"],
            row["auto_accuracy"],
            row["coverage"],
            -row["threshold"],
        ),
    )
    return {
        "selected_threshold": float(selected["threshold"]),
        "minimum_coverage": float(minimum_coverage),
        "selection_source": "validation_split",
        "selection_metric": "auto_macro_f1_with_minimum_coverage",
        "candidates": rows,
        "selected_validation_result": selected,
    }


def evaluate_abstention(
    y_true: pd.Series,
    y_pred: pd.Series,
    confidence: pd.Series,
    *,
    threshold: float,
    labels: list[str],
) -> dict[str, Any]:
    """Evaluate above-threshold recommendations for human-reviewed routing."""
    auto_mask = confidence.astype(float) >= threshold
    auto_count = int(auto_mask.sum())
    total = int(len(y_true))
    if auto_count == 0:
        auto_accuracy = 0.0
        auto_macro_f1 = 0.0
    else:
        auto_accuracy = float(accuracy_score(y_true[auto_mask], y_pred[auto_mask]))
        auto_macro_f1 = float(
            f1_score(
                y_true[auto_mask],
                y_pred[auto_mask],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )
    coverage = float(auto_count / total) if total else 0.0
    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "auto_routed_count": auto_count,
        "review_count": int(total - auto_count),
        "review_rate": float(1.0 - coverage),
        "auto_accuracy": auto_accuracy,
        "auto_macro_f1": auto_macro_f1,
    }


def save_confusion_matrix_csv(metrics: dict[str, Any], output_path: Path) -> Path:
    """Save a confusion matrix as a labeled CSV file."""
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path)
    return output_path


def save_confusion_matrix_svg(metrics: dict[str, Any], output_path: Path) -> Path:
    """Save a lightweight SVG confusion-matrix heatmap."""
    labels = metrics["confusion_matrix"]["labels"]
    matrix = np.asarray(metrics["confusion_matrix"]["matrix"], dtype=float)
    max_value = float(matrix.max()) if matrix.size else 1.0
    cell = 42
    label_width = 170
    top = 120
    width = label_width + cell * len(labels) + 40
    height = top + cell * len(labels) + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:10px}</style>",
        '<text x="10" y="24" font-size="16">Queue Confusion Matrix</text>',
    ]
    for index, label in enumerate(labels):
        x = label_width + index * cell + 6
        parts.append(
            f'<text x="{x}" y="106" transform="rotate(-35 {x} 106)">'
            f"{_escape(label)}</text>"
        )
    for row_index, label in enumerate(labels):
        y = top + row_index * cell + 25
        parts.append(f'<text x="10" y="{y}">{_escape(label)}</text>')
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(245 - (value / max(max_value, 1.0)) * 185)
            color = f"rgb({intensity},{intensity},{255})"
            x = label_width + col_index * cell
            y = top + row_index * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="#555"/>'
            )
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" '
                'text-anchor="middle">'
                f"{int(value)}</text>"
            )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def save_model_comparison_svg(
    metrics_by_model: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    metric_name: str = "macro_f1",
) -> Path:
    """Save a simple SVG bar plot comparing validation models."""
    names = list(metrics_by_model)
    values = [float(metrics_by_model[name][metric_name]) for name in names]
    width = 760
    height = 260
    left = 220
    bar_height = 28
    max_bar_width = 460
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}</style>",
        f'<text x="10" y="24" font-size="16">Validation {metric_name}</text>',
    ]
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        y = 52 + index * 56
        bar_width = max(2, value * max_bar_width)
        parts.append(f'<text x="10" y="{y + 19}">{_escape(name)}</text>')
        parts.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" '
            'fill="#3b82f6"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 8:.1f}" y="{y + 19}">{value:.3f}</text>'
        )
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path


def _align_score_columns(
    scores: np.ndarray,
    estimator_classes: list[str],
    labels: list[str],
) -> np.ndarray:
    class_to_index = {label: index for index, label in enumerate(estimator_classes)}
    aligned = np.zeros((scores.shape[0], len(labels)), dtype=float)
    for output_index, label in enumerate(labels):
        aligned[:, output_index] = scores[:, class_to_index[label]]
    return aligned


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
