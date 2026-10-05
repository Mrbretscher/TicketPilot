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

DEFAULT_CONFIDENCE_THRESHOLDS = [round(value / 100, 2) for value in range(0, 101, 5)]
DEFAULT_CALIBRATION_BINS = 10
LOW_SUPPORT_QUEUE_THRESHOLD = 100
LOW_RECALL_QUEUE_THRESHOLD = 0.60


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


def predict_queue_probabilities(
    estimator: Any,
    x: pd.Series,
    *,
    labels: list[str],
) -> np.ndarray:
    """Return per-class probabilities aligned to labels."""
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Estimator does not expose calibrated probabilities.")
    scores = np.asarray(estimator.predict_proba(x), dtype=float)
    estimator_classes = [str(label) for label in estimator.classes_]
    return _normalize_probability_rows(
        _align_score_columns(scores, estimator_classes, labels)
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


def multiclass_calibration_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    labels: list[str],
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> dict[str, Any]:
    """Return Brier score and top-label ECE for multiclass probabilities."""
    probabilities = _normalize_probability_rows(probabilities)
    if probabilities.shape[1] != len(labels):
        raise ValueError(
            "Probability column count does not match label count: "
            f"{probabilities.shape[1]} != {len(labels)}."
        )
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_indices = np.asarray([label_to_index[str(label)] for label in y_true])
    predicted_indices = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted_indices == y_indices
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_indices)), y_indices] = 1.0
    bins = calibration_bins(confidence, correct, n_bins=n_bins)
    total = max(len(y_indices), 1)
    ece = sum(
        (bin_row["count"] / total) * abs(bin_row["accuracy"] - bin_row["confidence"])
        for bin_row in bins
    )
    return {
        "method": "top_label_equal_width_bins",
        "bin_count": n_bins,
        "binning": (
            "Equal-width confidence bins over [0, 1]; bins are left-inclusive "
            "and right-exclusive except the final bin, which includes 1.0."
        ),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "expected_calibration_error": float(ece),
        "confidence_distribution": confidence_distribution(confidence),
        "bins": bins,
    }


def calibration_bins(
    confidence: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> list[dict[str, float | int]]:
    """Summarize confidence and accuracy in equal-width calibration bins."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []
    total = max(len(confidence), 1)
    for index, (left, right) in enumerate(
        zip(bin_edges[:-1], bin_edges[1:], strict=True)
    ):
        if index == n_bins - 1:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
        else:
            mean_confidence = 0.0
            accuracy = 0.0
        rows.append(
            {
                "bin_index": index,
                "left_edge": float(left),
                "right_edge": float(right),
                "count": count,
                "fraction": float(count / total),
                "confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": float(accuracy - mean_confidence),
            }
        )
    return rows


def confidence_distribution(confidence: np.ndarray | pd.Series) -> dict[str, Any]:
    """Return a compact distribution summary for confidence values."""
    values = np.asarray(confidence, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "mean": float(values.mean()),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def coverage_performance_curve(
    y_true: pd.Series,
    y_pred: pd.Series,
    confidence: pd.Series,
    *,
    labels: list[str],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate coverage and performance across confidence thresholds."""
    if thresholds is None:
        thresholds = DEFAULT_CONFIDENCE_THRESHOLDS
    return [
        evaluate_abstention(
            y_true,
            y_pred,
            confidence,
            threshold=threshold,
            labels=labels,
        )
        for threshold in thresholds
    ]


def per_queue_confidence_report(
    y_true: pd.Series,
    y_pred: pd.Series,
    confidence: pd.Series,
    *,
    threshold: float,
    labels: list[str],
    low_support_threshold: int = LOW_SUPPORT_QUEUE_THRESHOLD,
    low_recall_threshold: float = LOW_RECALL_QUEUE_THRESHOLD,
) -> dict[str, Any]:
    """Return per-queue quality, confidence, and review-rate diagnostics."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    confidence_values = confidence.astype(float)
    rows: dict[str, dict[str, Any]] = {}
    low_support_queues: list[str] = []
    low_recall_queues: list[str] = []
    for label in labels:
        mask = y_true.astype(str).eq(label)
        support = int(report[label]["support"])
        recall = float(report[label]["recall"])
        if support < low_support_threshold:
            low_support_queues.append(label)
        if recall < low_recall_threshold:
            low_recall_queues.append(label)
        queue_confidence = confidence_values[mask]
        review_rate = float((queue_confidence < threshold).mean()) if support else 0.0
        rows[label] = {
            "support": support,
            "precision": float(report[label]["precision"]),
            "recall": recall,
            "f1": float(report[label]["f1-score"]),
            "confidence_distribution": confidence_distribution(queue_confidence),
            "review_rate_at_selected_threshold": review_rate,
            "low_support": support < low_support_threshold,
            "low_recall": recall < low_recall_threshold,
        }
    return {
        "selected_threshold": float(threshold),
        "low_support_threshold": low_support_threshold,
        "low_recall_threshold": low_recall_threshold,
        "low_support_queues": low_support_queues,
        "low_recall_queues": low_recall_queues,
        "operational_note": (
            "Queue predictions remain recommendations subject to human review; "
            "no class-specific autonomous routing behavior is introduced."
        ),
        "queues": rows,
    }


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
        thresholds = DEFAULT_CONFIDENCE_THRESHOLDS

    rows = coverage_performance_curve(
        y_true,
        predictions["predicted_queue"],
        predictions["confidence"],
        thresholds=thresholds,
        labels=labels,
    )
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


def save_reliability_svg(
    calibration: dict[str, Any],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Save a lightweight reliability diagram from calibration bins."""
    bins = calibration["bins"]
    width = 560
    height = 420
    plot_left = 72
    plot_top = 52
    plot_size = 300
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}</style>",
        f'<text x="10" y="24" font-size="16">{_escape(title)}</text>',
        f'<line x1="{plot_left}" y1="{plot_top + plot_size}" '
        f'x2="{plot_left + plot_size}" y2="{plot_top}" '
        'stroke="#555" stroke-dasharray="4 4"/>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_size}" '
        f'height="{plot_size}" fill="none" stroke="#333"/>',
    ]
    for tick in range(0, 11):
        value = tick / 10
        x = plot_left + value * plot_size
        y = plot_top + plot_size - value * plot_size
        parts.append(
            f'<text x="{x - 8:.1f}" y="{plot_top + plot_size + 20}">{value:.1f}</text>'
        )
        parts.append(f'<text x="28" y="{y + 4:.1f}">{value:.1f}</text>')
    bar_width = plot_size / max(len(bins), 1) * 0.72
    for row in bins:
        confidence = float(row["confidence"])
        accuracy = float(row["accuracy"])
        count = int(row["count"])
        if count == 0:
            continue
        x = plot_left + confidence * plot_size - bar_width / 2
        y = plot_top + plot_size - accuracy * plot_size
        bar_height = accuracy * plot_size
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#2563eb" opacity="0.72"/>'
        )
    parts.extend(
        [
            f'<text x="{plot_left + 84}" y="{plot_top + plot_size + 46}">'
            "Mean confidence</text>",
            f'<text x="8" y="{plot_top + 145}" transform="rotate(-90 8 '
            f'{plot_top + 145})">Accuracy</text>',
            f'<text x="{plot_left}" y="{height - 28}">'
            f"ECE={float(calibration['expected_calibration_error']):.4f}; "
            f"Brier={float(calibration['brier_score']):.4f}</text>",
            "</svg>",
        ]
    )
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


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    normalized: np.ndarray = clipped / row_sums
    return normalized


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
