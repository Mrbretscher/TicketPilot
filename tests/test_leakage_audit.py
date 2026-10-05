import pandas as pd

from ticketpilot.leakage_audit import audit_near_duplicate_cross_split_leakage
from ticketpilot.preparation import CLASSIFIER_TEXT_COLUMN, ROW_ID_COLUMN, SPLIT_COLUMN


def _prepared_fixture() -> pd.DataFrame:
    rows = [
        {
            ROW_ID_COLUMN: "ticket-000000",
            SPLIT_COLUMN: "train",
            CLASSIFIER_TEXT_COLUMN: (
                "VPN login failure\n\nUser cannot authenticate to VPN after MFA reset."
            ),
            "queue": "IT Support",
            "priority": "high",
        },
        {
            ROW_ID_COLUMN: "ticket-000001",
            SPLIT_COLUMN: "validation",
            CLASSIFIER_TEXT_COLUMN: (
                "VPN login failure\n\nUser cannot authenticate to VPN after MFA reset."
            ),
            "queue": "IT Support",
            "priority": "high",
        },
        {
            ROW_ID_COLUMN: "ticket-000002",
            SPLIT_COLUMN: "test",
            CLASSIFIER_TEXT_COLUMN: (
                "VPN login failure\n\nUser cannot authenticate to VPN after MFA reset "
                "today."
            ),
            "queue": "Technical Support",
            "priority": "medium",
        },
        {
            ROW_ID_COLUMN: "ticket-000003",
            SPLIT_COLUMN: "train",
            CLASSIFIER_TEXT_COLUMN: (
                "Invoice question\n\nCustomer asks about a duplicate billing charge."
            ),
            "queue": "Billing and Payments",
            "priority": "low",
        },
        {
            ROW_ID_COLUMN: "ticket-000004",
            SPLIT_COLUMN: "validation",
            CLASSIFIER_TEXT_COLUMN: (
                "Laptop battery replacement\n\nEmployee needs a new battery for "
                "a travel laptop."
            ),
            "queue": "Product Support",
            "priority": "low",
        },
    ]
    return pd.DataFrame(rows)


def _raw_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [
                "VPN login failure",
                "VPN login failure",
                "VPN login failure",
                "Invoice question",
                "Laptop battery replacement",
            ],
            "body": [
                "User cannot authenticate to VPN after MFA reset.",
                "User cannot authenticate to VPN after MFA reset.",
                "User cannot authenticate to VPN after MFA reset today.",
                "Customer asks about a duplicate billing charge.",
                "Employee needs a new battery for a travel laptop.",
            ],
        }
    )


def test_audit_counts_cross_split_candidates_by_threshold() -> None:
    report = audit_near_duplicate_cross_split_leakage(
        _prepared_fixture(),
        raw_tickets=_raw_fixture(),
        thresholds=(0.90, 0.95, 0.98),
    )

    counts = report["candidate_pair_counts"]

    assert report["total_candidate_pair_count"] == 3
    assert counts["by_threshold"][">=0.90"] == 3
    assert counts["by_threshold"][">=0.95"] == 1
    assert counts["by_threshold"][">=0.98"] == 1
    assert counts["by_split_pair_and_threshold"]["train_vs_validation"][">=0.98"] == 1
    assert counts["by_split_pair_and_threshold"]["train_vs_test"][">=0.90"] == 1
    assert counts["by_split_pair_and_threshold"]["validation_vs_test"][">=0.90"] == 1
    assert report["affected_ticket_counts"]["by_threshold"][">=0.98"] == {
        "test": 0,
        "train": 1,
        "validation": 1,
        "total_unique_tickets": 2,
    }
    assert len(report["candidate_pairs"]) == 3


def test_audit_reports_same_and_different_queue_candidate_pairs() -> None:
    report = audit_near_duplicate_cross_split_leakage(
        _prepared_fixture(),
        raw_tickets=_raw_fixture(),
    )

    queue_summary = report["queue_label_summary"]

    assert queue_summary["same_queue_pair_count"] == 1
    assert queue_summary["different_queue_pair_count"] == 2
    assert queue_summary["queue_pair_counts"]["IT Support | IT Support"] == 1
    assert queue_summary["queue_pair_counts"]["IT Support | Technical Support"] == 2


def test_audit_reports_repeated_subjects_that_cross_splits() -> None:
    report = audit_near_duplicate_cross_split_leakage(
        _prepared_fixture(),
        raw_tickets=_raw_fixture(),
    )

    repeated_subjects = report["repeated_subject_cross_split_counts"]

    assert repeated_subjects["status"] == "computed"
    assert repeated_subjects["subject_value_count"] == 1
    assert repeated_subjects["cross_split_pair_count"] == 3
    assert repeated_subjects["pair_counts_by_split_pair"] == {
        "train_vs_test": 1,
        "train_vs_validation": 1,
        "validation_vs_test": 1,
    }
