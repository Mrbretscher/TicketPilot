"""Schema and quality validation for TicketPilot support-ticket data."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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


def suspicious_label_mention_report(
    frame: pd.DataFrame, *, sample_size: int = 5
) -> dict[str, Any]:
    """Report literal label mentions in subject/body using validation diagnostics."""
    required_columns = {"subject", "body", "queue", "priority"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Suspicious label mention report requires columns: "
            + ", ".join(missing_columns)
        )

    subject = frame["subject"].fillna("").astype("string").str.lower()
    body = frame["body"].fillna("").astype("string").str.lower()
    queue = frame["queue"].astype("string").str.lower()
    priority = frame["priority"].astype("string").str.lower()
    queue_subject_mask = _rowwise_contains_mask(subject, queue)
    queue_body_mask = _rowwise_contains_mask(body, queue)
    priority_subject_mask = _rowwise_contains_mask(subject, priority)
    priority_body_mask = _rowwise_contains_mask(body, priority)
    queue_mask = queue_subject_mask | queue_body_mask
    priority_mask = priority_subject_mask | priority_body_mask
    row_count = int(len(frame))

    return {
        "method": (
            "Literal row-wise substring check: a ticket is flagged when its own "
            "queue or priority label appears in its subject or body."
        ),
        "row_count": row_count,
        "field_counts": {
            "subject_contains_queue": int(queue_subject_mask.sum()),
            "body_contains_queue": int(queue_body_mask.sum()),
            "subject_contains_priority": int(priority_subject_mask.sum()),
            "body_contains_priority": int(priority_body_mask.sum()),
        },
        "queue_label_mentions": {
            "ticket_count": int(queue_mask.sum()),
            "rate": _rate(queue_mask, row_count),
            "affected_labels": _affected_labels(frame["queue"], queue_mask),
            "affected_label_counts": _affected_label_counts(frame["queue"], queue_mask),
        },
        "priority_label_mentions": {
            "ticket_count": int(priority_mask.sum()),
            "rate": _rate(priority_mask, row_count),
            "affected_labels": _affected_labels(frame["priority"], priority_mask),
            "affected_label_counts": _affected_label_counts(
                frame["priority"], priority_mask
            ),
        },
        "any_label_mentions": {
            "ticket_count": int((queue_mask | priority_mask).sum()),
            "rate": _rate(queue_mask | priority_mask, row_count),
        },
        "safe_examples": _safe_label_mention_examples(
            frame,
            queue_subject_mask=queue_subject_mask,
            queue_body_mask=queue_body_mask,
            priority_subject_mask=priority_subject_mask,
            priority_body_mask=priority_body_mask,
            sample_size=sample_size,
        ),
        "interpretation": (
            "Literal queue or priority words in user-authored ticket text can be "
            "natural support language, especially for words such as high, medium, "
            "or low. They are tracked as possible target-proxy risk and require "
            "review before model claims, but records are not automatically removed."
        ),
        "action": "diagnostic_only_no_records_removed",
    }


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
    return int(_rowwise_contains_mask(text, labels).sum())


def _rowwise_contains_mask(text: pd.Series, labels: pd.Series) -> pd.Series:
    values = []
    for value, label in zip(text.tolist(), labels.tolist(), strict=True):
        values.append(str(label) in str(value))
    return pd.Series(values, index=text.index, dtype=bool)


def _affected_labels(labels: pd.Series, mask: pd.Series) -> list[str]:
    return sorted(str(label) for label in labels.loc[mask].dropna().unique().tolist())


def _affected_label_counts(labels: pd.Series, mask: pd.Series) -> dict[str, int]:
    return _value_counts(labels.loc[mask])


def _rate(mask: pd.Series, row_count: int) -> float:
    return float(int(mask.sum()) / row_count) if row_count else 0.0


def _safe_label_mention_examples(
    frame: pd.DataFrame,
    *,
    queue_subject_mask: pd.Series,
    queue_body_mask: pd.Series,
    priority_subject_mask: pd.Series,
    priority_body_mask: pd.Series,
    sample_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    combined_mask = queue_subject_mask | queue_body_mask
    combined_mask = combined_mask | priority_subject_mask | priority_body_mask
    for index in frame.index[combined_mask].tolist()[:sample_size]:
        mention_fields: list[str] = []
        mention_types: list[str] = []
        if bool(queue_subject_mask.loc[index]):
            mention_fields.append("subject")
            mention_types.append("queue")
        if bool(queue_body_mask.loc[index]):
            mention_fields.append("body")
            mention_types.append("queue")
        if bool(priority_subject_mask.loc[index]):
            mention_fields.append("subject")
            mention_types.append("priority")
        if bool(priority_body_mask.loc[index]):
            mention_fields.append("body")
            mention_types.append("priority")
        ticket_id = (
            str(frame.loc[index, "ticket_row_id"])
            if "ticket_row_id" in frame.columns
            else f"row-{int(index)}"
        )
        rows.append(
            {
                "ticket_id": ticket_id,
                "queue": str(frame.loc[index, "queue"]),
                "priority": str(frame.loc[index, "priority"]),
                "mention_types": sorted(set(mention_types)),
                "mention_fields": sorted(set(mention_fields)),
                "subject_preview": _preview_text(frame.loc[index, "subject"]),
                "body_preview": _preview_text(frame.loc[index, "body"]),
            }
        )
    return rows


def _preview_text(value: object, *, max_chars: int = 180) -> str:
    if value is None or value is pd.NA:
        text = ""
    else:
        text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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
