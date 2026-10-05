from datetime import datetime
from pathlib import Path

import pytest

from ticketpilot.review import (
    ReviewRecord,
    ReviewWorkflowError,
    create_review_record,
    get_review_record,
    hash_ticket_text,
    initialize_review_database,
    submit_review_action,
)


def db_path(tmp_path: Path) -> Path:
    return tmp_path / "reviews.sqlite"


def create_pending(path: Path) -> ReviewRecord:
    return create_review_record(
        path=path,
        analysis_id="analysis-001",
        ticket_text="Cannot connect to VPN after MFA approval.",
        predicted_queue="Technical Support",
        queue_confidence=0.84,
        predicted_priority="high",
        retrieved_evidence_ids=("TP-ticket-000001", "TP-ticket-000002"),
        draft_response="Please review the VPN profile. [TP-ticket-000001]",
        model_provider_metadata={
            "provider": "fake",
            "model": "deterministic-test-generator",
        },
    )


def test_database_initialization_and_safe_repeated_startup(tmp_path: Path) -> None:
    path = db_path(tmp_path)

    first = initialize_review_database(path)
    second = initialize_review_database(path)

    assert first == path
    assert second == path
    assert path.exists()


def test_create_review_record_persists_pending_state(tmp_path: Path) -> None:
    path = db_path(tmp_path)

    record = create_pending(path)
    loaded = get_review_record("analysis-001", path=path)

    assert record == loaded
    assert loaded.state == "pending"
    assert loaded.ticket_text_hash == hash_ticket_text(
        "Cannot connect to VPN after MFA approval."
    )
    assert loaded.retrieved_evidence_ids == (
        "TP-ticket-000001",
        "TP-ticket-000002",
    )
    assert loaded.model_provider_metadata["provider"] == "fake"


def test_accept_transition_approves_record(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)

    reviewed = submit_review_action(
        "analysis-001",
        action="accept",
        review_note="Looks good.",
        path=path,
    )

    assert reviewed.state == "approved"
    assert reviewed.reviewer_action == "accept"
    assert reviewed.final_queue == "Technical Support"
    assert reviewed.review_note == "Looks good."
    assert reviewed.reviewed_at is not None


def test_edit_transition_persists_edited_response(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)

    reviewed = submit_review_action(
        "analysis-001",
        action="edit",
        edited_response="Edited human-approved response.",
        path=path,
    )

    assert reviewed.state == "edited"
    assert reviewed.reviewer_action == "edit"
    assert reviewed.edited_response == "Edited human-approved response."


def test_reroute_transition_persists_final_queue(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)

    reviewed = submit_review_action(
        "analysis-001",
        action="reroute",
        final_queue="IT Support",
        review_note="Belongs to internal IT.",
        path=path,
    )

    assert reviewed.state == "edited"
    assert reviewed.reviewer_action == "reroute"
    assert reviewed.final_queue == "IT Support"
    assert reviewed.review_note == "Belongs to internal IT."


def test_reject_and_insufficient_evidence_use_rejected_state(tmp_path: Path) -> None:
    first_path = tmp_path / "reject.sqlite"
    second_path = tmp_path / "insufficient.sqlite"
    create_pending(first_path)
    create_review_record(
        path=second_path,
        analysis_id="analysis-002",
        ticket_text="Cannot connect to VPN.",
        predicted_queue="Technical Support",
        queue_confidence=0.44,
        predicted_priority=None,
        retrieved_evidence_ids=(),
        draft_response="Insufficient evidence.",
        model_provider_metadata={"provider": "fake"},
    )

    rejected = submit_review_action("analysis-001", action="reject", path=first_path)
    insufficient = submit_review_action(
        "analysis-002",
        action="mark_insufficient_evidence",
        review_note="No useful evidence.",
        path=second_path,
    )

    assert rejected.state == "rejected"
    assert insufficient.state == "rejected"
    assert insufficient.reviewer_action == "mark_insufficient_evidence"


def test_invalid_transition_after_final_state_is_rejected(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)
    submit_review_action("analysis-001", action="accept", path=path)

    with pytest.raises(ReviewWorkflowError, match="approved"):
        submit_review_action(
            "analysis-001",
            action="edit",
            edited_response="Too late.",
            path=path,
        )


def test_invalid_review_action_is_rejected(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)

    with pytest.raises(ReviewWorkflowError, match="Invalid reviewer action"):
        submit_review_action("analysis-001", action="send", path=path)  # type: ignore[arg-type]


def test_edit_requires_edited_response(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)

    with pytest.raises(ReviewWorkflowError, match="edited_response"):
        submit_review_action("analysis-001", action="edit", path=path)


def test_reroute_requires_final_queue(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)

    with pytest.raises(ReviewWorkflowError, match="final_queue"):
        submit_review_action("analysis-001", action="reroute", path=path)


def test_audit_timestamps_are_iso_and_updated(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    record = create_pending(path)
    reviewed = submit_review_action("analysis-001", action="accept", path=path)

    created_at = datetime.fromisoformat(record.created_at)
    updated_at = datetime.fromisoformat(reviewed.updated_at)
    reviewed_at = datetime.fromisoformat(str(reviewed.reviewed_at))

    assert updated_at >= created_at
    assert reviewed_at >= created_at
    assert reviewed.updated_at == reviewed.reviewed_at


def test_duplicate_analysis_id_is_rejected(tmp_path: Path) -> None:
    path = db_path(tmp_path)
    create_pending(path)

    with pytest.raises(ReviewWorkflowError, match="already exists"):
        create_pending(path)
