from typing import Any

import numpy as np
import pandas as pd

from ticketpilot.evaluation import (
    evaluate_abstention,
    multiclass_calibration_metrics,
    predict_queue_routes,
    select_abstention_threshold,
)
from ticketpilot.modeling import build_queue_baseline_pipelines
from ticketpilot.preparation import CLASSIFIER_TEXT_COLUMN, SPLIT_COLUMN
from ticketpilot.training import (
    QUEUE_LABEL_COLUMN,
    build_confidence_model,
    select_queue_model,
    train_and_evaluate_queue_baselines,
    validate_prepared_dataset,
)


def queue_training_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    queues = ["Technical Support", "Billing and Payments", "Product Support"]
    for index in range(18):
        queue = queues[index % len(queues)]
        if queue == "Technical Support":
            text = f"VPN login outage device network access {index}"
        elif queue == "Billing and Payments":
            text = f"invoice payment refund billing receipt {index}"
        else:
            text = f"mobile app crash product bug screen {index}"
        split = "train" if index < 12 else "validation"
        rows.append(
            {
                "ticket_row_id": f"ticket-{index:06d}",
                "ticket_text_group_id": f"group-{index:06d}",
                SPLIT_COLUMN: split,
                CLASSIFIER_TEXT_COLUMN: text,
                QUEUE_LABEL_COLUMN: queue,
                "priority": "medium",
            }
        )
    return pd.DataFrame(rows)


def complete_queue_training_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    queues = ["Technical Support", "Billing and Payments", "Product Support"]
    split_by_index = {
        **{index: "train" for index in range(30)},
        **{index: "validation" for index in range(30, 39)},
        **{index: "test" for index in range(39, 48)},
    }
    for index in range(48):
        queue = queues[index % len(queues)]
        if queue == "Technical Support":
            text = f"vpn login outage device network access {index}"
        elif queue == "Billing and Payments":
            text = f"invoice payment refund billing receipt {index}"
        else:
            text = f"mobile app crash product bug screen {index}"
        rows.append(
            {
                "ticket_row_id": f"ticket-{index:06d}",
                "ticket_text_group_id": f"group-{index:06d}",
                SPLIT_COLUMN: split_by_index[index],
                CLASSIFIER_TEXT_COLUMN: text,
                QUEUE_LABEL_COLUMN: queue,
                "priority": "medium",
            }
        )
    return pd.DataFrame(rows)


def test_prepared_dataset_rejects_missing_classifier_text() -> None:
    frame = queue_training_frame().drop(columns=[CLASSIFIER_TEXT_COLUMN])

    try:
        validate_prepared_dataset(frame)
    except ValueError as error:
        assert "classifier_text" in str(error)
    else:
        raise AssertionError("validate_prepared_dataset should reject missing features")


def test_pipeline_configuration_is_deterministic() -> None:
    pipelines = build_queue_baseline_pipelines(random_state=123)

    assert list(pipelines) == [
        "dummy_most_frequent",
        "tfidf_logistic_regression",
        "tfidf_linear_svc",
    ]
    assert (
        pipelines["tfidf_logistic_regression"].named_steps["model"].random_state == 123
    )
    assert pipelines["tfidf_linear_svc"].named_steps["model"].random_state == 123


def test_classifier_output_schema_and_confidence_range() -> None:
    frame = queue_training_frame()
    train = frame.loc[frame[SPLIT_COLUMN].eq("train"), :]
    validation = frame.loc[frame[SPLIT_COLUMN].eq("validation"), :]
    labels = sorted(frame[QUEUE_LABEL_COLUMN].unique().tolist())
    model = build_confidence_model(
        "tfidf_logistic_regression",
        train,
        random_seed=123,
    )

    predictions = predict_queue_routes(
        model,
        validation[CLASSIFIER_TEXT_COLUMN],
        labels=labels,
    )

    assert list(predictions.columns) == ["predicted_queue", "confidence", "top_queues"]
    assert predictions["confidence"].between(0.0, 1.0).all()
    assert predictions["top_queues"].map(len).eq(3).all()


def test_abstention_logic_reports_coverage_and_review_rate() -> None:
    y_true = pd.Series(["A", "B", "A", "B"])
    y_pred = pd.Series(["A", "B", "B", "B"])
    confidence = pd.Series([0.95, 0.8, 0.4, 0.2])

    result = evaluate_abstention(
        y_true,
        y_pred,
        confidence,
        threshold=0.75,
        labels=["A", "B"],
    )

    assert result["coverage"] == 0.5
    assert result["review_rate"] == 0.5
    assert result["auto_accuracy"] == 1.0


def test_threshold_selection_uses_validation_predictions() -> None:
    y_true = pd.Series(["A", "B", "A", "B"])
    predictions = pd.DataFrame(
        {
            "predicted_queue": ["A", "B", "B", "B"],
            "confidence": [0.95, 0.8, 0.4, 0.2],
        }
    )

    selection = select_abstention_threshold(
        y_true,
        predictions,
        labels=["A", "B"],
        minimum_coverage=0.5,
        thresholds=[0.0, 0.5, 0.75],
    )

    assert selection["selection_source"] == "validation_split"
    assert 0.0 <= selection["selected_threshold"] <= 1.0


def test_ece_uses_equal_width_top_label_bins() -> None:
    metrics = multiclass_calibration_metrics(
        pd.Series(["A", "B"]),
        np.asarray([[0.8, 0.2], [0.4, 0.6]]),
        labels=["A", "B"],
        n_bins=2,
    )

    assert metrics["brier_score"] == 0.2
    assert round(metrics["expected_calibration_error"], 6) == 0.3
    assert metrics["bins"][1]["count"] == 2
    assert metrics["bins"][1]["accuracy"] == 1.0


def test_queue_report_marks_test_threshold_results_as_reporting_only(
    tmp_path: Any,
) -> None:
    report = train_and_evaluate_queue_baselines(
        complete_queue_training_frame(),
        artifact_dir=tmp_path / "artifacts",
        report_dir=tmp_path / "reports",
    )

    threshold_selection = report["abstention"]["threshold_selection"]
    test_result = report["abstention"]["test_result"]

    assert threshold_selection["selection_source"] == "validation_split"
    assert test_result["threshold"] == threshold_selection["selected_threshold"]
    assert test_result["threshold_source"] == "validation_split"
    assert test_result["reporting_only"] is True
    assert report["confidence_analysis"]["test"]["reporting_only"] is True


def test_answer_text_cannot_enter_classifier_features() -> None:
    frame = queue_training_frame()
    frame["answer"] = "SECRET_RESOLUTION_TOKEN"

    assert "answer" not in frame[[CLASSIFIER_TEXT_COLUMN, QUEUE_LABEL_COLUMN]].columns
    assert (
        not frame[CLASSIFIER_TEXT_COLUMN].str.contains("SECRET_RESOLUTION_TOKEN").any()
    )


def test_model_selection_prefers_validation_macro_f1() -> None:
    metrics: dict[str, dict[str, Any]] = {
        "dummy_most_frequent": {"macro_f1": 0.1, "weighted_f1": 0.2, "accuracy": 0.3},
        "tfidf_logistic_regression": {
            "macro_f1": 0.5,
            "weighted_f1": 0.5,
            "accuracy": 0.5,
        },
        "tfidf_linear_svc": {"macro_f1": 0.6, "weighted_f1": 0.55, "accuracy": 0.5},
    }

    assert select_queue_model(metrics) == "tfidf_linear_svc"
