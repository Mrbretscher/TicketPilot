"""Queue-routing baseline training and reporting workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from ticketpilot.config import (
    ABSTENTION_MIN_VALIDATION_COVERAGE,
    PREPARED_DATASET_PATH,
    QUEUE_BASELINE_ARTIFACT_DIR,
    QUEUE_BASELINE_REPORT_DIR,
    QUEUE_BASELINE_REPORT_PATH,
    SPLIT_RANDOM_SEED,
)
from ticketpilot.evaluation import (
    coverage_performance_curve,
    evaluate_abstention,
    evaluate_queue_classifier,
    multiclass_calibration_metrics,
    per_queue_confidence_report,
    predict_queue_probabilities,
    predict_queue_routes,
    save_confusion_matrix_csv,
    save_confusion_matrix_svg,
    save_model_comparison_svg,
    save_reliability_svg,
    select_abstention_threshold,
)
from ticketpilot.modeling import (
    QueueModelName,
    build_queue_baseline_pipeline,
    build_queue_baseline_pipelines,
)
from ticketpilot.preparation import CLASSIFIER_TEXT_COLUMN, SPLIT_COLUMN

QUEUE_LABEL_COLUMN = "queue"
VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class QueueBaselineRun:
    """Paths and report from a queue-baseline run."""

    report: dict[str, Any]
    report_path: Path
    model_path: Path
    plot_paths: dict[str, Path]


def run_queue_baseline_training(
    *,
    prepared_dataset_path: Path = PREPARED_DATASET_PATH,
    report_dir: Path = QUEUE_BASELINE_REPORT_DIR,
    artifact_dir: Path = QUEUE_BASELINE_ARTIFACT_DIR,
    random_seed: int = SPLIT_RANDOM_SEED,
    minimum_abstention_coverage: float = ABSTENTION_MIN_VALIDATION_COVERAGE,
) -> QueueBaselineRun:
    """Train queue-routing baselines and persist metrics/artifacts."""
    prepared = load_prepared_dataset(prepared_dataset_path)
    report = train_and_evaluate_queue_baselines(
        prepared,
        random_seed=random_seed,
        minimum_abstention_coverage=minimum_abstention_coverage,
        artifact_dir=artifact_dir,
        report_dir=report_dir,
    )
    report_path = save_queue_baseline_report(report, QUEUE_BASELINE_REPORT_PATH)
    plot_paths = {
        "validation_model_comparison": Path(
            report["artifacts"]["validation_model_comparison_plot"]
        ),
        "test_confusion_matrix_svg": Path(
            report["artifacts"]["test_confusion_matrix_svg"]
        ),
        "test_confusion_matrix_csv": Path(
            report["artifacts"]["test_confusion_matrix_csv"]
        ),
    }
    return QueueBaselineRun(
        report=report,
        report_path=report_path,
        model_path=Path(report["selected_model"]["artifact_path"]),
        plot_paths=plot_paths,
    )


def load_prepared_dataset(path: Path = PREPARED_DATASET_PATH) -> pd.DataFrame:
    """Load a prepared queue-routing dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {path}. "
            "Run `python scripts/prepare_dataset.py` first."
        )
    frame = pd.read_csv(path, dtype="string")
    validate_prepared_dataset(frame)
    return frame


def validate_prepared_dataset(frame: pd.DataFrame) -> None:
    """Validate columns and split names for model training."""
    required_columns = {
        "ticket_row_id",
        "ticket_text_group_id",
        SPLIT_COLUMN,
        CLASSIFIER_TEXT_COLUMN,
        QUEUE_LABEL_COLUMN,
        "priority",
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError("Prepared dataset missing columns: " + ", ".join(missing))
    unexpected_splits = sorted(
        set(frame[SPLIT_COLUMN].dropna()).difference(VALID_SPLITS)
    )
    if unexpected_splits:
        raise ValueError(
            "Prepared dataset has unexpected splits: " + ", ".join(unexpected_splits)
        )
    empty_features = (
        frame[CLASSIFIER_TEXT_COLUMN].fillna("").astype("string").str.strip().eq("")
    )
    if bool(empty_features.any()):
        raise ValueError("Prepared dataset contains empty classifier_text rows.")


def train_and_evaluate_queue_baselines(
    prepared: pd.DataFrame,
    *,
    random_seed: int = SPLIT_RANDOM_SEED,
    minimum_abstention_coverage: float = ABSTENTION_MIN_VALIDATION_COVERAGE,
    artifact_dir: Path = QUEUE_BASELINE_ARTIFACT_DIR,
    report_dir: Path = QUEUE_BASELINE_REPORT_DIR,
) -> dict[str, Any]:
    """Train fixed queue baselines, select on validation, and report test metrics."""
    validate_prepared_dataset(prepared)
    splits = _split_prepared_dataset(prepared)
    labels = sorted(prepared[QUEUE_LABEL_COLUMN].dropna().unique().tolist())

    validation_metrics: dict[str, dict[str, Any]] = {}
    fitted_validation_models: dict[str, Any] = {}
    for model_name, pipeline in build_queue_baseline_pipelines(
        random_state=random_seed
    ).items():
        pipeline.fit(
            splits["train"][CLASSIFIER_TEXT_COLUMN], splits["train"][QUEUE_LABEL_COLUMN]
        )
        fitted_validation_models[model_name] = pipeline
        validation_metrics[model_name] = evaluate_queue_classifier(
            pipeline,
            splits["validation"][CLASSIFIER_TEXT_COLUMN],
            splits["validation"][QUEUE_LABEL_COLUMN],
            labels=labels,
        )

    selected_model_name = select_queue_model(validation_metrics)
    validation_confidence_model = build_confidence_model(
        selected_model_name,
        splits["train"],
        random_seed=random_seed,
    )
    selected_validation_metrics = evaluate_queue_classifier(
        validation_confidence_model,
        splits["validation"][CLASSIFIER_TEXT_COLUMN],
        splits["validation"][QUEUE_LABEL_COLUMN],
        labels=labels,
    )
    validation_predictions = predict_queue_routes(
        validation_confidence_model,
        splits["validation"][CLASSIFIER_TEXT_COLUMN],
        labels=labels,
    )
    validation_probabilities = predict_queue_probabilities(
        validation_confidence_model,
        splits["validation"][CLASSIFIER_TEXT_COLUMN],
        labels=labels,
    )
    validation_calibration = multiclass_calibration_metrics(
        splits["validation"][QUEUE_LABEL_COLUMN],
        validation_probabilities,
        labels=labels,
    )
    abstention_selection = select_abstention_threshold(
        splits["validation"][QUEUE_LABEL_COLUMN],
        validation_predictions,
        labels=labels,
        minimum_coverage=minimum_abstention_coverage,
    )

    train_validation = pd.concat(
        [splits["train"], splits["validation"]],
        axis=0,
        ignore_index=True,
    )
    final_model = build_final_model(
        selected_model_name,
        train_validation,
        random_seed=random_seed,
    )
    test_metrics = evaluate_queue_classifier(
        final_model,
        splits["test"][CLASSIFIER_TEXT_COLUMN],
        splits["test"][QUEUE_LABEL_COLUMN],
        labels=labels,
    )
    test_predictions = predict_queue_routes(
        final_model,
        splits["test"][CLASSIFIER_TEXT_COLUMN],
        labels=labels,
    )
    test_probabilities = predict_queue_probabilities(
        final_model,
        splits["test"][CLASSIFIER_TEXT_COLUMN],
        labels=labels,
    )
    test_calibration = multiclass_calibration_metrics(
        splits["test"][QUEUE_LABEL_COLUMN],
        test_probabilities,
        labels=labels,
    )
    threshold = float(abstention_selection["selected_threshold"])
    test_abstention = evaluate_abstention(
        splits["test"][QUEUE_LABEL_COLUMN],
        test_predictions["predicted_queue"],
        test_predictions["confidence"],
        threshold=threshold,
        labels=labels,
    )
    validation_threshold_curve = coverage_performance_curve(
        splits["validation"][QUEUE_LABEL_COLUMN],
        validation_predictions["predicted_queue"],
        validation_predictions["confidence"],
        labels=labels,
    )
    test_threshold_curve = coverage_performance_curve(
        splits["test"][QUEUE_LABEL_COLUMN],
        test_predictions["predicted_queue"],
        test_predictions["confidence"],
        labels=labels,
    )
    per_queue_analysis = {
        "validation": per_queue_confidence_report(
            splits["validation"][QUEUE_LABEL_COLUMN],
            validation_predictions["predicted_queue"],
            validation_predictions["confidence"],
            threshold=threshold,
            labels=labels,
        ),
        "test": per_queue_confidence_report(
            splits["test"][QUEUE_LABEL_COLUMN],
            test_predictions["predicted_queue"],
            test_predictions["confidence"],
            threshold=threshold,
            labels=labels,
        ),
    }

    report_dir = Path(report_dir)
    artifact_dir = Path(artifact_dir)
    plots_dir = report_dir / "plots"
    tables_dir = report_dir / "tables"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "selected_queue_router.joblib"
    joblib.dump(final_model, model_path)

    comparison_plot = save_model_comparison_svg(
        validation_metrics,
        plots_dir / "validation_model_comparison.svg",
    )
    confusion_svg = save_confusion_matrix_svg(
        test_metrics,
        plots_dir / "test_confusion_matrix.svg",
    )
    confusion_csv = save_confusion_matrix_csv(
        test_metrics,
        tables_dir / "test_confusion_matrix.csv",
    )
    validation_reliability_svg = save_reliability_svg(
        validation_calibration,
        plots_dir / "validation_reliability.svg",
        title="Validation Reliability Diagram",
    )
    test_reliability_svg = save_reliability_svg(
        test_calibration,
        plots_dir / "test_reliability.svg",
        title="Final Test Reliability Diagram",
    )

    return {
        "task": "support_queue_routing",
        "input_features": [CLASSIFIER_TEXT_COLUMN],
        "source_fields": ["subject", "body"],
        "label_column": QUEUE_LABEL_COLUMN,
        "excluded_fields": [
            "answer",
            "type",
            "language",
            "version",
            "tag_1",
            "tag_2",
            "tag_3",
            "tag_4",
            "tag_5",
            "tag_6",
            "tag_7",
            "tag_8",
            "queue",
            "priority",
        ],
        "random_seed": random_seed,
        "class_distribution": {
            split: _class_distribution(split_frame[QUEUE_LABEL_COLUMN])
            for split, split_frame in splits.items()
        },
        "model_selection": {
            "selection_split": "validation",
            "selection_metric": "macro_f1",
            "final_test_usage": (
                "Final test split is used only after model and abstention "
                "threshold selection are complete."
            ),
        },
        "validation_metrics": validation_metrics,
        "selected_model": {
            "name": selected_model_name,
            "validation_macro_f1": validation_metrics[selected_model_name]["macro_f1"],
            "calibrated_validation_macro_f1": selected_validation_metrics["macro_f1"],
            "confidence_model": _confidence_model_description(selected_model_name),
            "artifact_path": str(model_path),
            "artifact_size_bytes": model_path.stat().st_size,
        },
        "confidence_model_validation_metrics": selected_validation_metrics,
        "calibration": {
            "method": validation_calibration["method"],
            "binning": validation_calibration["binning"],
            "validation": validation_calibration,
            "test": {
                **test_calibration,
                "reporting_only": True,
            },
        },
        "confidence_analysis": {
            "validation": {
                "threshold_curve": validation_threshold_curve,
                "threshold_selection_source": "validation_split",
            },
            "test": {
                "threshold_curve": test_threshold_curve,
                "reporting_only": True,
            },
        },
        "per_queue_analysis": per_queue_analysis,
        "abstention": {
            "threshold_selection": abstention_selection,
            "test_result": {
                **test_abstention,
                "threshold_source": "validation_split",
                "reporting_only": True,
            },
        },
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str(model_path),
            "validation_model_comparison_plot": str(comparison_plot),
            "test_confusion_matrix_svg": str(confusion_svg),
            "test_confusion_matrix_csv": str(confusion_csv),
            "validation_reliability_svg": str(validation_reliability_svg),
            "test_reliability_svg": str(test_reliability_svg),
        },
    }


def select_queue_model(metrics_by_model: dict[str, dict[str, Any]]) -> QueueModelName:
    """Select the best queue model by validation macro F1."""
    selected = max(
        metrics_by_model,
        key=lambda name: (
            metrics_by_model[name]["macro_f1"],
            metrics_by_model[name]["weighted_f1"],
            metrics_by_model[name]["accuracy"],
            name,
        ),
    )
    return cast(QueueModelName, selected)


def build_confidence_model(
    model_name: QueueModelName,
    train_frame: pd.DataFrame,
    *,
    random_seed: int,
) -> Any:
    """Fit a validation-time confidence model without using the final test split."""
    pipeline = build_queue_baseline_pipeline(model_name, random_state=random_seed)
    if model_name == "tfidf_linear_svc":
        model = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
    else:
        model = pipeline
    model.fit(train_frame[CLASSIFIER_TEXT_COLUMN], train_frame[QUEUE_LABEL_COLUMN])
    return model


def build_final_model(
    model_name: QueueModelName,
    train_validation_frame: pd.DataFrame,
    *,
    random_seed: int,
) -> Any:
    """Fit the final selected model on train+validation only."""
    pipeline = build_queue_baseline_pipeline(model_name, random_state=random_seed)
    if model_name == "tfidf_linear_svc":
        model = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
    else:
        model = pipeline
    model.fit(
        train_validation_frame[CLASSIFIER_TEXT_COLUMN],
        train_validation_frame[QUEUE_LABEL_COLUMN],
    )
    return model


def save_queue_baseline_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write queue baseline metrics as deterministic JSON."""
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
        for split in VALID_SPLITS
    }


def _class_distribution(labels: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = labels.value_counts().sort_index()
    total = int(len(labels))
    return {
        str(label): {
            "count": int(count),
            "rate": float(count / total) if total else 0.0,
        }
        for label, count in counts.items()
    }


def _confidence_model_description(model_name: str) -> str:
    if model_name == "tfidf_linear_svc":
        return "sigmoid_calibrated_linear_svc_cv3_fit_without_test_data"
    return "native_predict_proba_fit_without_test_data"
