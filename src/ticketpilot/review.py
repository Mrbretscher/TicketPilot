"""Local SQLite human-review workflow for TicketPilot draft decisions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ticketpilot.config import REVIEW_DATABASE_PATH

ReviewAction = Literal[
    "accept",
    "edit",
    "reject",
    "reroute",
    "mark_insufficient_evidence",
]
ReviewState = Literal["pending", "approved", "edited", "rejected"]

ALLOWED_REVIEW_ACTIONS: tuple[ReviewAction, ...] = (
    "accept",
    "edit",
    "reject",
    "reroute",
    "mark_insufficient_evidence",
)
FINAL_REVIEW_STATES: tuple[ReviewState, ...] = ("approved", "edited", "rejected")


class ReviewWorkflowError(ValueError):
    """Raised when a review record or state transition is invalid."""


@dataclass(frozen=True)
class ReviewRecord:
    """A persisted human-review decision record."""

    analysis_id: str
    state: ReviewState
    created_at: str
    updated_at: str
    ticket_text_hash: str
    predicted_queue: str
    queue_confidence: float
    predicted_priority: str | None
    retrieved_evidence_ids: tuple[str, ...]
    draft_response: str
    model_provider_metadata: dict[str, Any]
    reviewer_action: str | None
    edited_response: str | None
    final_queue: str | None
    review_note: str | None
    reviewed_at: str | None


def initialize_review_database(path: Path = REVIEW_DATABASE_PATH) -> Path:
    """Create the review database schema if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS review_records (
                analysis_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                ticket_text_hash TEXT NOT NULL,
                predicted_queue TEXT NOT NULL,
                queue_confidence REAL NOT NULL,
                predicted_priority TEXT,
                retrieved_evidence_ids_json TEXT NOT NULL,
                draft_response TEXT NOT NULL,
                model_provider_metadata_json TEXT NOT NULL,
                reviewer_action TEXT,
                edited_response TEXT,
                final_queue TEXT,
                review_note TEXT,
                reviewed_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_review_records_state
            ON review_records(state)
            """
        )
        connection.execute("PRAGMA user_version = 1")
    return path


def create_review_record(
    *,
    ticket_text: str,
    predicted_queue: str,
    queue_confidence: float,
    retrieved_evidence_ids: tuple[str, ...],
    draft_response: str,
    model_provider_metadata: dict[str, Any],
    predicted_priority: str | None = None,
    analysis_id: str | None = None,
    path: Path = REVIEW_DATABASE_PATH,
) -> ReviewRecord:
    """Create a pending local review record."""
    if not predicted_queue.strip():
        raise ReviewWorkflowError("predicted_queue is required.")
    if not 0.0 <= queue_confidence <= 1.0:
        raise ReviewWorkflowError("queue_confidence must be between 0 and 1.")
    if not draft_response.strip():
        raise ReviewWorkflowError("draft_response is required.")

    initialize_review_database(path)
    timestamp = utc_timestamp()
    record_id = analysis_id or str(uuid.uuid4())
    ticket_hash = hash_ticket_text(ticket_text)
    with sqlite3.connect(path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO review_records (
                    analysis_id,
                    state,
                    created_at,
                    updated_at,
                    ticket_text_hash,
                    predicted_queue,
                    queue_confidence,
                    predicted_priority,
                    retrieved_evidence_ids_json,
                    draft_response,
                    model_provider_metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    "pending",
                    timestamp,
                    timestamp,
                    ticket_hash,
                    predicted_queue,
                    queue_confidence,
                    predicted_priority,
                    json.dumps(list(retrieved_evidence_ids), sort_keys=True),
                    draft_response,
                    json.dumps(model_provider_metadata, sort_keys=True),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ReviewWorkflowError(
                f"Review analysis_id already exists: {record_id}"
            ) from error
    return get_review_record(record_id, path=path)


def submit_review_action(
    analysis_id: str,
    *,
    action: ReviewAction,
    edited_response: str | None = None,
    final_queue: str | None = None,
    review_note: str | None = None,
    path: Path = REVIEW_DATABASE_PATH,
) -> ReviewRecord:
    """Apply a reviewer action without sending responses or executing IT actions."""
    if action not in ALLOWED_REVIEW_ACTIONS:
        raise ReviewWorkflowError(f"Invalid reviewer action: {action}")

    record = get_review_record(analysis_id, path=path)
    if record.state != "pending":
        raise ReviewWorkflowError(
            f"Cannot apply reviewer action to a {record.state} review record."
        )

    new_state = state_for_action(action)
    normalized_edited_response = normalize_optional_text(edited_response)
    normalized_final_queue = normalize_optional_text(final_queue)
    normalized_note = normalize_optional_text(review_note)
    if action == "edit" and not normalized_edited_response:
        raise ReviewWorkflowError("edited_response is required for edit actions.")
    if action == "reroute" and not normalized_final_queue:
        raise ReviewWorkflowError("final_queue is required for reroute actions.")
    if action == "accept":
        normalized_final_queue = normalized_final_queue or record.predicted_queue

    timestamp = utc_timestamp()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE review_records
            SET
                state = ?,
                updated_at = ?,
                reviewer_action = ?,
                edited_response = ?,
                final_queue = ?,
                review_note = ?,
                reviewed_at = ?
            WHERE analysis_id = ?
            """,
            (
                new_state,
                timestamp,
                action,
                normalized_edited_response,
                normalized_final_queue,
                normalized_note,
                timestamp,
                analysis_id,
            ),
        )
    return get_review_record(analysis_id, path=path)


def get_review_record(
    analysis_id: str,
    *,
    path: Path = REVIEW_DATABASE_PATH,
) -> ReviewRecord:
    """Load one review record by analysis ID."""
    initialize_review_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM review_records WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
    if row is None:
        raise ReviewWorkflowError(f"Review analysis_id not found: {analysis_id}")
    return row_to_review_record(row)


def state_for_action(action: ReviewAction) -> ReviewState:
    """Map reviewer actions into persisted review states."""
    if action == "accept":
        return "approved"
    if action in {"edit", "reroute"}:
        return "edited"
    if action in {"reject", "mark_insufficient_evidence"}:
        return "rejected"
    raise ReviewWorkflowError(f"Invalid reviewer action: {action}")


def row_to_review_record(row: sqlite3.Row) -> ReviewRecord:
    """Convert a SQLite row into a typed record."""
    return ReviewRecord(
        analysis_id=str(row["analysis_id"]),
        state=str(row["state"]),  # type: ignore[arg-type]
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        ticket_text_hash=str(row["ticket_text_hash"]),
        predicted_queue=str(row["predicted_queue"]),
        queue_confidence=float(row["queue_confidence"]),
        predicted_priority=row["predicted_priority"],
        retrieved_evidence_ids=tuple(
            str(item) for item in json.loads(row["retrieved_evidence_ids_json"])
        ),
        draft_response=str(row["draft_response"]),
        model_provider_metadata=json.loads(row["model_provider_metadata_json"]),
        reviewer_action=row["reviewer_action"],
        edited_response=row["edited_response"],
        final_queue=row["final_queue"],
        review_note=row["review_note"],
        reviewed_at=row["reviewed_at"],
    )


def hash_ticket_text(ticket_text: str) -> str:
    """Return a stable local ticket text hash without storing full ticket text."""
    return hashlib.sha256(ticket_text.encode("utf-8")).hexdigest()


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_optional_text(value: str | None) -> str | None:
    """Trim optional reviewer text and preserve nulls for omitted fields."""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
