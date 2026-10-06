import pandas as pd
import pytest

from ticketpilot.validation import (
    DataValidationError,
    suspicious_label_mention_report,
    validate_classifier_feature_columns,
    validate_ticket_frame,
)


def test_validate_ticket_frame_accepts_valid_source_frame(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    summary = validate_ticket_frame(valid_ticket_frame)

    assert summary.row_count == 4
    assert summary.language_counts == {"de": 1, "en": 3}
    assert summary.priority_counts == {"high": 1, "low": 1, "medium": 2}
    assert summary.null_counts["subject"] == 1
    assert summary.empty_subject_count == 1
    assert summary.empty_body_count == 0
    assert summary.duplicate_record_count == 0


def test_validate_ticket_frame_accepts_english_subset(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    english = valid_ticket_frame.loc[valid_ticket_frame["language"].eq("en"), :]

    summary = validate_ticket_frame(english, require_english_only=True)

    assert summary.language_counts == {"en": 3}


def test_validate_ticket_frame_rejects_missing_required_column(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    invalid = valid_ticket_frame.drop(columns=["answer"])

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_ticket_frame(invalid)


def test_validate_ticket_frame_rejects_unexpected_column(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    invalid = valid_ticket_frame.assign(extra_field="unexpected")

    with pytest.raises(DataValidationError, match="unexpected columns"):
        validate_ticket_frame(invalid)


def test_validate_ticket_frame_rejects_column_order_change(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    invalid = valid_ticket_frame.loc[:, list(reversed(valid_ticket_frame.columns))]

    with pytest.raises(DataValidationError, match="column order changed"):
        validate_ticket_frame(invalid)


def test_validate_ticket_frame_rejects_null_body(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    invalid = valid_ticket_frame.copy()
    invalid.loc[0, "body"] = pd.NA

    with pytest.raises(DataValidationError, match="null values"):
        validate_ticket_frame(invalid)


def test_validate_ticket_frame_rejects_empty_ticket_text(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    invalid = valid_ticket_frame.copy()
    invalid.loc[0, ["subject", "body"]] = ["  ", ""]

    with pytest.raises(DataValidationError, match="empty subject and body"):
        validate_ticket_frame(invalid)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("language", "fr"),
        ("queue", "Escalation Desk"),
        ("priority", "urgent"),
        ("type", "Question"),
    ],
)
def test_validate_ticket_frame_rejects_unexpected_label_values(
    valid_ticket_frame: pd.DataFrame,
    column: str,
    value: str,
) -> None:
    invalid = valid_ticket_frame.copy()
    invalid.loc[0, column] = value

    with pytest.raises(DataValidationError, match=f"Column {column!r}"):
        validate_ticket_frame(invalid)


def test_validate_ticket_frame_rejects_duplicate_rows(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    invalid = pd.concat(
        [valid_ticket_frame, valid_ticket_frame.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(DataValidationError, match="duplicate rows"):
        validate_ticket_frame(invalid)


def test_validate_ticket_frame_rejects_duplicate_subject_body(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    invalid = valid_ticket_frame.copy()
    invalid.loc[1, ["subject", "body"]] = invalid.loc[0, ["subject", "body"]]

    with pytest.raises(DataValidationError, match=r"duplicate subject\+body"):
        validate_ticket_frame(invalid)


def test_validate_ticket_frame_requires_english_only_when_requested(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    with pytest.raises(DataValidationError, match="only English records"):
        validate_ticket_frame(valid_ticket_frame, require_english_only=True)


def test_validate_ticket_frame_rejects_too_few_rows(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    with pytest.raises(DataValidationError, match="expected at least 5"):
        validate_ticket_frame(valid_ticket_frame, minimum_rows=5)


def test_validate_ticket_frame_reports_suspicious_label_mentions(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    suspicious = valid_ticket_frame.copy()
    suspicious.loc[0, "subject"] = "Technical Support routing request"
    suspicious.loc[1, "body"] = "This is a high priority app crash."

    summary = validate_ticket_frame(suspicious)

    assert summary.suspicious_label_leakage_counts["subject_contains_queue"] == 1
    assert summary.suspicious_label_leakage_counts["body_contains_priority"] == 1


def test_suspicious_label_mention_report_includes_counts_rates_and_examples(
    valid_ticket_frame: pd.DataFrame,
) -> None:
    suspicious = valid_ticket_frame.copy()
    suspicious.loc[0, "subject"] = "Technical Support routing request"
    suspicious.loc[1, "body"] = "This is a high priority app crash."

    report = suspicious_label_mention_report(suspicious, sample_size=2)

    assert report["queue_label_mentions"]["ticket_count"] == 1
    assert report["queue_label_mentions"]["rate"] == 0.25
    assert report["queue_label_mentions"]["affected_labels"] == ["Technical Support"]
    assert report["priority_label_mentions"]["ticket_count"] == 1
    assert report["priority_label_mentions"]["affected_labels"] == ["high"]
    assert report["any_label_mentions"]["ticket_count"] == 2
    assert len(report["safe_examples"]) == 2
    assert "answer" not in report["safe_examples"][0]
    assert report["action"] == "diagnostic_only_no_records_removed"


def test_validate_classifier_feature_columns_accepts_subject_and_body() -> None:
    validate_classifier_feature_columns(["subject", "body"])


@pytest.mark.parametrize("column", ["answer", "queue", "priority", "tag_1"])
def test_validate_classifier_feature_columns_rejects_leakage_columns(
    column: str,
) -> None:
    with pytest.raises(DataValidationError, match="prohibited leakage columns"):
        validate_classifier_feature_columns(["subject", column])
