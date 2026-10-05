import numpy as np
import pandas as pd
import tensorflow as tf

from ticketpilot.preparation import CLASSIFIER_TEXT_COLUMN, SPLIT_COLUMN
from ticketpilot.tensorflow_modeling import (
    QueueLabelEncoder,
    adapt_text_vectorizer,
    build_conv1d_queue_model,
    build_text_vectorizer,
    compute_class_weights,
    set_tensorflow_seed,
)
from ticketpilot.tensorflow_training import (
    build_sklearn_comparison,
    deployment_decision,
)


def tiny_text_frame() -> pd.DataFrame:
    rows = []
    examples = [
        ("vpn login network", "Technical Support"),
        ("password reset account", "Technical Support"),
        ("invoice payment refund", "Billing and Payments"),
        ("receipt card charge", "Billing and Payments"),
        ("mobile crash app", "Product Support"),
        ("screen bug product", "Product Support"),
        ("validationonly token", "Technical Support"),
        ("billing validation case", "Billing and Payments"),
        ("product validation case", "Product Support"),
    ]
    for index, (text, queue) in enumerate(examples):
        rows.append(
            {
                "ticket_row_id": f"ticket-{index:06d}",
                "ticket_text_group_id": f"group-{index:06d}",
                SPLIT_COLUMN: "train" if index < 6 else "validation",
                CLASSIFIER_TEXT_COLUMN: text,
                "queue": queue,
                "priority": "medium",
                "answer": "SECRET_RESOLUTION_TOKEN",
            }
        )
    return pd.DataFrame(rows)


def test_label_encoder_is_deterministic() -> None:
    encoder = QueueLabelEncoder.fit(
        pd.Series(["Product Support", "Billing and Payments", "Technical Support"])
    )

    encoded = encoder.transform(
        pd.Series(["Billing and Payments", "Technical Support"])
    )

    assert encoder.classes == (
        "Billing and Payments",
        "Product Support",
        "Technical Support",
    )
    assert encoded.tolist() == [0, 2]
    assert encoder.inverse_transform(encoded).tolist() == [
        "Billing and Payments",
        "Technical Support",
    ]


def test_text_vectorizer_adapts_on_training_text_only() -> None:
    frame = tiny_text_frame()
    vectorizer = build_text_vectorizer(max_tokens=100, sequence_length=8)

    adapt_text_vectorizer(
        vectorizer,
        frame.loc[frame[SPLIT_COLUMN].eq("train"), CLASSIFIER_TEXT_COLUMN],
    )

    vocabulary = set(vectorizer.get_vocabulary())
    assert "vpn" in vocabulary
    assert "validationonly" not in vocabulary


def test_conv_model_contains_expected_lightweight_layers() -> None:
    vectorizer = build_text_vectorizer(max_tokens=100, sequence_length=8)
    adapt_text_vectorizer(vectorizer, pd.Series(["vpn login", "invoice payment"]))

    model = build_conv1d_queue_model(
        vectorizer=vectorizer,
        num_classes=3,
        max_tokens=100,
        embedding_dim=8,
        conv_filters=8,
    )
    layer_names = [layer.__class__.__name__ for layer in model.layers]

    assert "TextVectorization" in layer_names
    assert "Embedding" in layer_names
    assert "Conv1D" in layer_names
    assert "Dropout" in layer_names


def test_class_weights_use_inverse_frequency() -> None:
    weights = compute_class_weights(np.asarray([0, 0, 0, 1, 2]), num_classes=3)

    assert weights[0] < weights[1]
    assert weights[1] == weights[2]


def test_tensorflow_smoke_training_runs_one_epoch() -> None:
    set_tensorflow_seed(123)
    frame = tiny_text_frame()
    train = frame.loc[frame[SPLIT_COLUMN].eq("train"), :]
    validation = frame.loc[frame[SPLIT_COLUMN].eq("validation"), :]
    encoder = QueueLabelEncoder.fit(frame["queue"])
    vectorizer = build_text_vectorizer(max_tokens=100, sequence_length=8)
    adapt_text_vectorizer(vectorizer, train[CLASSIFIER_TEXT_COLUMN])
    model = build_conv1d_queue_model(
        vectorizer=vectorizer,
        num_classes=len(encoder.classes),
        max_tokens=100,
        embedding_dim=8,
        conv_filters=8,
    )

    history = model.fit(
        train[CLASSIFIER_TEXT_COLUMN].to_numpy(),
        encoder.transform(train["queue"]),
        validation_data=(
            validation[CLASSIFIER_TEXT_COLUMN].to_numpy(),
            encoder.transform(validation["queue"]),
        ),
        epochs=1,
        batch_size=3,
        verbose=0,
    )
    probabilities = model.predict(
        validation[CLASSIFIER_TEXT_COLUMN].to_numpy(), verbose=0
    )

    assert "loss" in history.history
    assert probabilities.shape == (3, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_tensorflow_feature_text_excludes_answer_field() -> None:
    frame = tiny_text_frame()

    assert (
        not frame[CLASSIFIER_TEXT_COLUMN].str.contains("SECRET_RESOLUTION_TOKEN").any()
    )


def test_deployment_decision_keeps_sklearn_when_tensorflow_does_not_win() -> None:
    decision = deployment_decision(
        {
            "sklearn_model": "tfidf_linear_svc",
            "validation": {"macro_f1": {"sklearn": 0.68, "tensorflow": 0.60}},
        }
    )

    assert decision["selected_model"] == "tfidf_linear_svc"
    assert decision["selection_metric_source"] == "validation"
    assert decision["final_test_usage"] == "reporting_only_after_family_selection"


def test_deployment_decision_uses_validation_metrics_not_final_test() -> None:
    comparison = build_sklearn_comparison(
        tensorflow_validation_metrics={
            "macro_f1": 0.40,
            "weighted_f1": 0.41,
            "top_k_accuracy": 0.70,
        },
        tensorflow_test_metrics={
            "macro_f1": 0.99,
            "weighted_f1": 0.99,
            "top_k_accuracy": 0.99,
            "inference_latency": {"milliseconds_per_ticket": 0.1},
        },
        tensorflow_artifact_size=456,
        tensorflow_training_seconds=1.2,
        sklearn_report={
            "selected_model": {
                "name": "tfidf_linear_svc",
                "artifact_size_bytes": 123,
            },
            "confidence_model_validation_metrics": {
                "macro_f1": 0.68,
                "weighted_f1": 0.66,
                "top_k_accuracy": 0.85,
            },
            "validation_metrics": {},
            "test_metrics": {
                "macro_f1": 0.60,
                "weighted_f1": 0.61,
                "top_k_accuracy": 0.80,
                "inference_latency": {"milliseconds_per_ticket": 0.5},
            },
        },
    )

    decision = deployment_decision(comparison)

    assert comparison["selection_metric_source"] == "validation"
    assert comparison["test"]["reporting_only"] is True
    assert decision["selected_model"] == "tfidf_linear_svc"


def test_tensorflow_runtime_is_available() -> None:
    assert tf.__version__
