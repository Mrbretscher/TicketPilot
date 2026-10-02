"""Schema and quality validation for TicketPilot support-ticket data."""

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from ticketpilot.config import (
    CLASSIFIER_INPUT_COLUMNS,
    EXPECTED_COLUMNS,
    LANGUAGE_VALUES,
    PRIORITY_VALUES,
    PROHIBITED_CLASSIFIER_INPUT_COLUMNS,
    QUEUE_VALUES,
    RECRUITER_READY_LANGUAGE,
    TYPE_VALUES,
)


class DataValidationError(ValueError):
    """Raised when a support-ticket dataframe violates the data contract."""


@dataclass(frozen=True)
class TicketDataValidationSummary:
    """Concise diagnostics for a validated support-ticket dataframe."""

    row_count: int
    column_count: int
    columns: tuple[str, ...]
    language_counts: dict[str, int]
    queue_counts: dict[str, int]
    priority_counts: dict[str, int]
    type_counts: dict[str, int]
    null_counts: dict[str, int]
    empty_subject_count: int
    empty_body_count: int
    empty_ticket_text_count: int
    duplicate_record_count: int
    duplicate_subject_body_count: int
    suspicious_label_leakage_counts: dict[str, int]

    def to_text(self) -> str:
        """Return a human-readable validation summary for CLI output."""
        return "\n".join(
            [
                "TicketPilot data validation summary",
                f"- Rows: {self.row_count}",
                f"- Columns: {self.column_count}",
                f"- Languages: {_format_counts(self.language_counts)}",
                f"- Queue distribution: {_format_counts(self.queue_counts)}",
                f"- Priority distribution: {_format_counts(self.priority_counts)}",
                f"- Null values: {_format_counts(self.null_counts)}",
                f"- Empty subjects: {self.empty_subject_count}",
                f"- Empty bodies: {self.empty_body_count}",
                f"- Empty subject+body records: {self.empty_ticket_text_count}",
                f"- Exact duplicate rows: {self.duplicate_record_count}",
                (
                    "- Exact duplicate subject+body records: "
                    f"{self.duplicate_subject_body_count}"
                ),
                (
                    "- Suspicious label mentions in classifier text: "
                    f"{_format_counts(self.suspicious_label_leakage_counts)}"
                ),
            ]
        )


def validate_ticket_frame(
    frame: pd.DataFrame,
    *,
    require_english_only: bool = False,
    minimum_rows: int = 1,
) -> TicketDataValidationSummary:
    """Validate support-ticket schema and return quality diagnostics."""
    if frame.empty:
        raise DataValidationError("Support-ticket dataframe is empty.")
    if len(frame) < minimum_rows:
        raise DataValidationError(
            f"Support-ticket dataframe has {len(frame)} rows; "
            f"expected at least {minimum_rows}."
        )

    _validate_schema(frame)
    null_counts = _null_counts(frame)
    _validate_non_nullable_columns(null_counts)

    empty_subject = _empty_text_mask(frame["subject"])
    empty_body = _empty_text_mask(frame["body"])
    empty_ticket_text = empty_subject & empty_body
    if bool(empty_ticket_text.any()):
        raise DataValidationError(
            "Support-ticket dataframe contains records with empty subject and body."
        )

    language_counts = _value_counts(frame["language"])
    _validate_known_values(language_counts, LANGUAGE_VALUES, "language")
    if require_english_only and set(language_counts) != {RECRUITER_READY_LANGUAGE}:
        raise DataValidationError(
            "Recruiter-ready dataset must contain only English records; found "
            + ", ".join(sorted(language_counts))
        )

    queue_counts = _value_counts(frame["queue"])
    priority_counts = _value_counts(frame["priority"])
    type_counts = _value_counts(frame["type"])
    _validate_known_values(queue_counts, QUEUE_VALUES, "queue")
    _validate_known_values(priority_counts, PRIORITY_VALUES, "priority")
    _validate_known_values(type_counts, TYPE_VALUES, "type")

    duplicate_record_count = int(frame.duplicated().sum())
    if duplicate_record_count:
        raise DataValidationError(
            "Support-ticket dataframe contains "
            f"{duplicate_record_count} duplicate rows."
        )

    duplicate_subject_body_count = int(frame.duplicated(["subject", "body"]).sum())
    if duplicate_subject_body_count:
        raise DataValidationError(
            "Support-ticket dataframe contains "
            f"{duplicate_subject_body_count} duplicate subject+body records."
        )

    suspicious_label_leakage_counts = _count_suspicious_label_mentions(frame)

    return TicketDataValidationSummary(
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=tuple(str(column) for column in frame.columns),
        language_counts=language_counts,
        queue_counts=queue_counts,
        priority_counts=priority_counts,
        type_counts=type_counts,
        null_counts=null_counts,
        empty_subject_count=int(empty_subject.sum()),
        empty_body_count=int(empty_body.sum()),
        empty_ticket_text_count=int(empty_ticket_text.sum()),
        duplicate_record_count=duplicate_record_count,
        duplicate_subject_body_count=duplicate_subject_body_count,
        suspicious_label_leakage_counts=suspicious_label_leakage_counts,
    )


def validate_classifier_feature_columns(columns: Iterable[str]) -> None:
    """Ensure classifier features use only pre-routing ticket text."""
    column_set = set(columns)
    unknown_columns = sorted(column_set.difference(EXPECTED_COLUMNS))
    if unknown_columns:
        raise DataValidationError(
            "Classifier feature columns are not in the raw schema: "
            + ", ".join(unknown_columns)
        )

    forbidden_columns = sorted(
        column_set.intersection(PROHIBITED_CLASSIFIER_INPUT_COLUMNS)
    )
    if forbidden_columns:
        raise DataValidationError(
            "Classifier features include prohibited leakage columns: "
            + ", ".join(forbidden_columns)
        )

    if not column_set.issubset(CLASSIFIER_INPUT_COLUMNS):
        raise DataValidationError(
            "Classifier features must be limited to: "
            + ", ".join(CLASSIFIER_INPUT_COLUMNS)
        )


def _validate_schema(frame: pd.DataFrame) -> None:
    columns = tuple(str(column) for column in frame.columns)
    missing_columns = sorted(set(EXPECTED_COLUMNS).difference(columns))
    extra_columns = sorted(set(columns).difference(EXPECTED_COLUMNS))
    if missing_columns:
        raise DataValidationError(
            "Support-ticket dataframe is missing required columns: "
            + ", ".join(missing_columns)
        )
    if extra_columns:
        raise DataValidationError(
            "Support-ticket dataframe has unexpected columns: "
            + ", ".join(extra_columns)
        )
    if columns != EXPECTED_COLUMNS:
        raise DataValidationError(
            "Support-ticket dataframe column order changed. Expected: "
            + ", ".join(EXPECTED_COLUMNS)
        )


def _validate_non_nullable_columns(null_counts: dict[str, int]) -> None:
    required_non_null = ("body", "type", "queue", "priority", "language", "version")
    columns_with_nulls = {
        column: null_counts[column]
        for column in required_non_null
        if null_counts[column] > 0
    }
    if columns_with_nulls:
        raise DataValidationError(
            "Support-ticket dataframe contains null values in required fields: "
            + _format_counts(columns_with_nulls)
        )


def _validate_known_values(
    observed_counts: dict[str, int],
    expected_values: Iterable[str],
    column_name: str,
) -> None:
    unexpected_values = sorted(set(observed_counts).difference(expected_values))
    if unexpected_values:
        raise DataValidationError(
            f"Column {column_name!r} contains unexpected values: "
            + ", ".join(unexpected_values)
        )


def _count_suspicious_label_mentions(frame: pd.DataFrame) -> dict[str, int]:
    subject = frame["subject"].fillna("").astype("string").str.lower()
    body = frame["body"].fillna("").astype("string").str.lower()
    queue = frame["queue"].astype("string").str.lower()
    priority = frame["priority"].astype("string").str.lower()

    return {
        "subject_contains_queue": _count_rowwise_contains(subject, queue),
        "body_contains_queue": _count_rowwise_contains(body, queue),
        "subject_contains_priority": _count_rowwise_contains(subject, priority),
        "body_contains_priority": _count_rowwise_contains(body, priority),
    }


def _count_rowwise_contains(text: pd.Series, labels: pd.Series) -> int:
    count = 0
    for value, label in zip(text.tolist(), labels.tolist(), strict=True):
        if str(label) in str(value):
            count += 1
    return count


def _null_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(column): int(count)
        for column, count in frame.isna().sum().to_dict().items()
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.to_dict().items()}


def _empty_text_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip().eq("")


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{label}={count}" for label, count in counts.items())
