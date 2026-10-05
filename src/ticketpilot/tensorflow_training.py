"""TensorFlow queue classifier training and reporting workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import callbacks
from sklearn.metrics import log_loss

from ticketpilot.config import (
    PREPARED_DATASET_PATH,
    QUEUE_BASELINE_REPORT_PATH,
    TENSORFLOW_ARTIFACT_DIR,
    TENSORFLOW_BATCH_SIZE,
    TENSORFLOW_EARLY_STOPPING_PATIENCE,
    TENSORFLOW_MAX_EPOCHS,
    TENSORFLOW_RANDOM_SEED,
    TENSORFLOW_REPORT_DIR,
    TENSORFLOW_REPORT_PATH,
)
from ticketpilot.evaluation import (
    evaluate_multiclass_predictions,
    save_confusion_matrix_csv,
    save_confusion_matrix_svg,
)
from ticketpilot.preparation import CLASSIFIER_TEXT_COLUMN, SPLIT_COLUMN
from ticketpilot.tensorflow_modeling import (
    QueueLabelEncoder,
    adapt_text_vectorizer,
    build_conv1d_queue_model,
    build_text_vectorizer,
    compute_class_weights,
    set_tensorflow_seed,
)
from ticketpilot.training import QUEUE_LABEL_COLUMN, load_prepared_dataset


@dataclass(frozen=True)
class TensorFlowQueueRun:
    """Paths and report from a TensorFlow queue training run."""

    report: dict[str, Any]
    report_path: Path
    model_path: Path
    history_path: Path


def run_tensorflow_queue_training(
    *,
    prepared_dataset_path: Path = PREPARED_DATASET_PATH,
    sklearn_report_path: Path = QUEUE_BASELINE_REPORT_PATH,
    report_dir: Path = TENSORFLOW_REPORT_DIR,
    artifact_dir: Path = TENSORFLOW_ARTIFACT_DIR,
    random_seed: int = TENSORFLOW_RANDOM_SEED,
    max_epochs: int = TENSORFLOW_MAX_EPOCHS,
    batch_size: int = TENSORFLOW_BATCH_SIZE,
) -> TensorFlowQueueRun:
    """Train and evaluate a small TensorFlow queue classifier."""
    prepared = load_prepared_dataset(prepared_dataset_path)
    sklearn_report = _load_json(sklearn_report_path)
    report = train_and_evaluate_tensorflow_queue_model(
        prepared,
        sklearn_report=sklearn_report,
        report_dir=report_dir,
        artifact_dir=artifact_dir,
        random_seed=random_seed,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    report_path = save_tensorflow_report(report, TENSORFLOW_REPORT_PATH)
    return TensorFlowQueueRun(
        report=report,
        report_path=report_path,
        model_path=Path(report["artifacts"]["model"]),
        history_path=Path(report["artifacts"]["training_history_json"]),
    )


def train_and_evaluate_tensorflow_queue_model(
    prepared: pd.DataFrame,
    *,
    sklearn_report: dict[str, Any],
    report_dir: Path = TENSORFLOW_REPORT_DIR,
    artifact_dir: Path = TENSORFLOW_ARTIFACT_DIR,
    random_seed: int = TENSORFLOW_RANDOM_SEED,
    max_epochs: int = TENSORFLOW_MAX_EPOCHS,
    batch_size: int = TENSORFLOW_BATCH_SIZE,
) -> dict[str, Any]:
    """Fit TensorFlow text model, evaluate, and build report payload."""
    set_tensorflow_seed(random_seed)
    splits = _split_prepared_dataset(prepared)
    label_encoder = QueueLabelEncoder.fit(prepared[QUEUE_LABEL_COLUMN])
    labels = list(label_encoder.classes)
    y_train = label_encoder.transform(splits["train"][QUEUE_LABEL_COLUMN])
    y_validation = label_encoder.transform(splits["validation"][QUEUE_LABEL_COLUMN])
    y_test = label_encoder.transform(splits["test"][QUEUE_LABEL_COLUMN])

    vectorizer = build_text_vectorizer()
    adapt_text_vectorizer(vectorizer, splits["train"][CLASSIFIER_TEXT_COLUMN])
    model = build_conv1d_queue_model(
        vectorizer=vectorizer,
        num_classes=len(labels),
    )
    class_weights = compute_class_weights(y_train, num_classes=len(labels))

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=TENSORFLOW_EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )
    start = perf_counter()
    history = model.fit(
        splits["train"][CLASSIFIER_TEXT_COLUMN].astype(str).to_numpy(),
        y_train,
        validation_data=(
            splits["validation"][CLASSIFIER_TEXT_COLUMN].astype(str).to_numpy(),
            y_validation,
        ),
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=0,
    )
    training_seconds = perf_counter() - start

    validation_metrics = evaluate_tensorflow_model(
        model,
        splits["validation"][CLASSIFIER_TEXT_COLUMN],
        y_validation,
        labels=labels,
    )
    test_metrics = evaluate_tensorflow_model(
        model,
        splits["test"][CLASSIFIER_TEXT_COLUMN],
        y_test,
        labels=labels,
    )

    report_dir = Path(report_dir)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "tensorflow_queue_classifier.keras"
    model.save(model_path)
    history_path = report_dir / "training_history.json"
    history_csv_path = report_dir / "training_history.csv"
    save_training_history(history.history, history_path, history_csv_path)
    confusion_svg = save_confusion_matrix_svg(
        test_metrics,
        report_dir / "plots" / "tensorflow_test_confusion_matrix.svg",
    )
    confusion_csv = save_confusion_matrix_csv(
        test_metrics,
        report_dir / "tables" / "tensorflow_test_confusion_matrix.csv",
    )

    sklearn_comparison = build_sklearn_comparison(
        tensorflow_metrics=test_metrics,
        tensorflow_artifact_size=model_path.stat().st_size,
        tensorflow_training_seconds=training_seconds,
        sklearn_report=sklearn_report,
    )
    return {
        "task": "support_queue_routing",
        "model_name": "tensorflow_textvectorization_embedding_conv1d",
        "architecture": {
            "text_vectorization": "adapted_on_training_text_only",
            "embedding_dim": int(model.get_layer("token_embedding").output_dim),
            "sequence_model": "Conv1D + GlobalMaxPooling1D",
            "regularization": [
                "embedding_l2_1e-6",
                "conv_l2_1e-5",
                "spatial_dropout_0.20",
                "dropout_0.40",
                "early_stopping_val_loss",
            ],
        },
        "input_features": [CLASSIFIER_TEXT_COLUMN],
        "source_fields": ["subject", "body"],
        "label_column": QUEUE_LABEL_COLUMN,
        "random_seed": random_seed,
        "class_weights": {
            labels[index]: weight for index, weight in class_weights.items()
        },
        "training": {
            "epochs_requested": max_epochs,
            "epochs_run": len(history.history.get("loss", [])),
            "batch_size": batch_size,
            "training_seconds": float(training_seconds),
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": TENSORFLOW_EARLY_STOPPING_PATIENCE,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "sklearn_comparison": sklearn_comparison,
        "deployment_decision": deployment_decision(sklearn_comparison),
        "artifacts": {
            "model": str(model_path),
            "model_size_bytes": model_path.stat().st_size,
            "training_history_json": str(history_path),
            "training_history_csv": str(history_csv_path),
            "test_confusion_matrix_svg": str(confusion_svg),
            "test_confusion_matrix_csv": str(confusion_csv),
        },
    }


def evaluate_tensorflow_model(
    model: tf.keras.Model,
    text: pd.Series,
    encoded_labels: np.ndarray,
    *,
    labels: list[str],
) -> dict[str, Any]:
    """Evaluate TensorFlow model probabilities against encoded labels."""
    text_values = text.astype(str).to_numpy()
    start = perf_counter()
    probabilities = np.asarray(model.predict(text_values, verbose=0), dtype=float)
    elapsed_seconds = perf_counter() - start
    predicted_indices = probabilities.argmax(axis=1)
    y_true = pd.Series([labels[int(index)] for index in encoded_labels])
    y_pred = pd.Series([labels[int(index)] for index in predicted_indices])
    metrics = evaluate_multiclass_predictions(
        y_true,
        y_pred,
        probabilities,
        labels=labels,
        elapsed_seconds=elapsed_seconds,
    )
    metrics["calibration"] = calibration_metrics(y_true, probabilities, labels=labels)
    return metrics


def calibration_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    labels: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    """Return practical multiclass calibration diagnostics."""
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_indices = np.asarray([label_to_index[str(label)] for label in y_true])
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_indices
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for left, right in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
    return {
        "negative_log_loss": float(
            log_loss(y_indices, probabilities, labels=list(range(len(labels))))
        ),
        "expected_calibration_error": float(ece),
    }


def save_training_history(
    history: dict[str, list[float]],
    json_path: Path,
    csv_path: Path,
) -> None:
    """Persist Keras training history as JSON and CSV."""
    json_safe = {
        key: [float(value) for value in values] for key, values in history.items()
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(json_safe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(json_safe).to_csv(csv_path, index_label="epoch")


def build_sklearn_comparison(
    *,
    tensorflow_metrics: dict[str, Any],
    tensorflow_artifact_size: int,
    tensorflow_training_seconds: float,
    sklearn_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare TensorFlow metrics with the strongest sklearn baseline."""
    sklearn_test = sklearn_report["test_metrics"]
    sklearn_selected = sklearn_report["selected_model"]
    return {
        "sklearn_model": sklearn_selected["name"],
        "tensorflow_model": "tensorflow_textvectorization_embedding_conv1d",
        "macro_f1": {
            "sklearn": sklearn_test["macro_f1"],
            "tensorflow": tensorflow_metrics["macro_f1"],
            "delta_tensorflow_minus_sklearn": tensorflow_metrics["macro_f1"]
            - sklearn_test["macro_f1"],
        },
        "weighted_f1": {
            "sklearn": sklearn_test["weighted_f1"],
            "tensorflow": tensorflow_metrics["weighted_f1"],
            "delta_tensorflow_minus_sklearn": tensorflow_metrics["weighted_f1"]
            - sklearn_test["weighted_f1"],
        },
        "top_k_accuracy": {
            "sklearn": sklearn_test["top_k_accuracy"],
            "tensorflow": tensorflow_metrics["top_k_accuracy"],
            "delta_tensorflow_minus_sklearn": tensorflow_metrics["top_k_accuracy"]
            - sklearn_test["top_k_accuracy"],
        },
        "inference_latency_ms_per_ticket": {
            "sklearn": sklearn_test["inference_latency"]["milliseconds_per_ticket"],
            "tensorflow": tensorflow_metrics["inference_latency"][
                "milliseconds_per_ticket"
            ],
        },
        "artifact_size_bytes": {
            "sklearn": sklearn_selected["artifact_size_bytes"],
            "tensorflow": tensorflow_artifact_size,
        },
        "training_seconds": {
            "tensorflow": float(tensorflow_training_seconds),
            "sklearn": None,
        },
    }


def deployment_decision(comparison: dict[str, Any]) -> dict[str, str]:
    """Choose the current deployment candidate based on measured metrics."""
    if comparison["macro_f1"]["tensorflow"] > comparison["macro_f1"]["sklearn"]:
        return {
            "selected_model": "tensorflow_textvectorization_embedding_conv1d",
            "reason": (
                "TensorFlow has higher final-test macro F1 than the strongest "
                "sklearn baseline in the current run."
            ),
        }
    return {
        "selected_model": comparison["sklearn_model"],
        "reason": (
            "The sklearn baseline remains preferred because TensorFlow did not "
            "exceed its final-test macro F1."
        ),
    }


def save_tensorflow_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write TensorFlow report as deterministic JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def _split_prepared_dataset(prepared: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: prepared.loc[prepared[SPLIT_COLUMN].eq(split), :].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Required comparison report not found at {path}. "
            "Run `python scripts/train_queue_baseline.py` first."
        )
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded
