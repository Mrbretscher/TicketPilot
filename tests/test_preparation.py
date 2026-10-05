from pathlib import Path

import pandas as pd

from ticketpilot.preparation import (
    CLASSIFIER_TEXT_COLUMN,
    GROUP_ID_COLUMN,
    SPLIT_COLUMN,
    build_classifier_text,
    prepare_ticket_dataset,
    write_prepared_dataset,
)


def _preparation_frame() -> pd.DataFrame:
    rows = []
    queues = [
        "Technical Support",
        "Product Support",
        "Billing and Payments",
        "Customer Service",
    ]
    priorities = ["high", "medium", "low"]
    for index in range(24):
        rows.append(
            {
                "subject": f"Ticket {index} access issue",
                "body": f"User reports device {index} cannot connect to VPN.",
                "answer": f"Resolution answer text {index} should stay out.",
                "type": "Incident" if index % 2 == 0 else "Request",
                "queue": queues[index % len(queues)],
                "priority": priorities[index % len(priorities)],
                "language": "en",
                "version": "51",
                "tag_1": "Connectivity",
                "tag_2": pd.NA,
                "tag_3": pd.NA,
                "tag_4": pd.NA,
                "tag_5": pd.NA,
                "tag_6": pd.NA,
                "tag_7": pd.NA,
                "tag_8": pd.NA,
            }
        )
    rows[5]["subject"] = rows[0]["subject"]
    rows[5]["body"] = rows[0]["body"]
    return pd.DataFrame(rows)


def test_build_classifier_text_uses_subject_and_body_only() -> None:
    text = build_classifier_text(" Printer error  ", "Line one\nLine two")

    assert text == "Printer error\n\nLine one Line two"


def test_prepare_ticket_dataset_is_deterministic() -> None:
    frame = _preparation_frame()

    first = prepare_ticket_dataset(frame, random_seed=42)
    second = prepare_ticket_dataset(frame, random_seed=42)

    pd.testing.assert_frame_equal(first.frame, second.frame)
    pd.testing.assert_frame_equal(first.split_manifest, second.split_manifest)
    assert first.summary == second.summary


def test_prepare_ticket_dataset_keeps_duplicate_groups_in_one_split() -> None:
    prepared = prepare_ticket_dataset(_preparation_frame(), random_seed=7)

    split_counts_by_group = prepared.split_manifest.groupby(GROUP_ID_COLUMN)[
        SPLIT_COLUMN
    ].nunique()

    assert int(split_counts_by_group.max()) == 1
    assert prepared.summary["exact_duplicate_ticket_text"]["duplicate_group_count"] == 1


def test_prepare_ticket_dataset_excludes_targets_from_classifier_features() -> None:
    prepared = prepare_ticket_dataset(_preparation_frame())

    assert prepared.classifier_feature_columns == (CLASSIFIER_TEXT_COLUMN,)
    assert "queue" not in prepared.classifier_feature_columns
    assert "priority" not in prepared.classifier_feature_columns
    assert "answer" not in prepared.classifier_feature_columns
    assert prepared.summary["classifier_text_sources"] == ["subject", "body"]


def test_prepare_ticket_dataset_does_not_include_answer_text_in_features() -> None:
    prepared = prepare_ticket_dataset(_preparation_frame())

    classifier_text = "\n".join(prepared.frame[CLASSIFIER_TEXT_COLUMN].tolist())

    assert "Resolution answer text" not in classifier_text


def test_write_prepared_dataset_outputs_machine_readable_artifacts(
    tmp_path: Path,
) -> None:
    prepared = prepare_ticket_dataset(_preparation_frame())
    prepared_path = tmp_path / "prepared.csv"
    manifest_path = tmp_path / "manifest.csv"
    summary_path = tmp_path / "summary.json"

    write_prepared_dataset(
        prepared,
        prepared_dataset_path=prepared_path,
        split_manifest_path=manifest_path,
        dataset_summary_path=summary_path,
    )

    assert prepared_path.exists()
    assert manifest_path.exists()
    assert summary_path.exists()
    assert '"final_test_set_policy"' in summary_path.read_text(encoding="utf-8")
