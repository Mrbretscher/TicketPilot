import pandas as pd
import pytest

from ticketpilot.gold_retrieval import (
    GOLD_LABEL_COLUMNS,
    build_threshold_analysis,
    score_gold_labels,
)


def gold_label_fixture(
    labels_by_query: dict[str, list[object]],
    *,
    split: str = "validation",
    scores_by_query: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    rows = []
    for query_id, labels in labels_by_query.items():
        for rank, label in enumerate(labels, start=1):
            score = (
                scores_by_query[query_id][rank - 1]
                if scores_by_query is not None
                else 1.0 / rank
            )
            rows.append(
                {
                    "evaluation_split": split,
                    "query_id": query_id,
                    "query_queue": "Technical Support",
                    "query_subject": f"Subject for {query_id}",
                    "query_body": f"Body for {query_id}",
                    "candidate_rank": rank,
                    "candidate_evidence_id": f"TP-ticket-{query_id}-{rank}",
                    "candidate_queue": "Technical Support",
                    "candidate_subject": f"Candidate {rank}",
                    "candidate_issue_excerpt": "Issue excerpt",
                    "candidate_resolution_excerpt": "Resolution excerpt",
                    "retrieval_score": score,
                    "relevance_label": label,
                    "reviewer_notes": "",
                }
            )
    return pd.DataFrame(rows, columns=GOLD_LABEL_COLUMNS)


def test_score_gold_labels_calculates_retrieval_metrics() -> None:
    labels = gold_label_fixture(
        {
            "query-1": [0, 2, 0, 1, 0],
            "query-2": [1, 0, 0, 0, 0],
        }
    )

    metrics = score_gold_labels(labels, expected_split="validation")

    assert metrics["query_count"] == 2
    assert metrics["candidate_count"] == 10
    assert metrics["human_labeled_relevance"] is True
    assert metrics["label_distribution"] == {"0": 7, "1": 2, "2": 1}
    assert metrics["recall_at_k"] == {
        "recall@1": 0.5,
        "recall@3": 1.0,
        "recall@5": 1.0,
    }
    assert metrics["mrr"] == 0.75
    assert metrics["ndcg@5"] == pytest.approx(0.819955, abs=1e-6)


def test_score_gold_labels_refuses_missing_human_labels() -> None:
    labels = gold_label_fixture({"query-1": [0, "", 1, 0, 0]})

    with pytest.raises(ValueError, match="Human relevance labels are required"):
        score_gold_labels(labels, expected_split="validation")


def test_score_gold_labels_refuses_invalid_label_values() -> None:
    labels = gold_label_fixture({"query-1": [0, 1, 3, 0, 0]})

    with pytest.raises(ValueError, match="Use only 0, 1, or 2"):
        score_gold_labels(labels, expected_split="validation")


def test_score_gold_labels_refuses_wrong_split() -> None:
    labels = gold_label_fixture({"query-1": [0, 1, 2, 0, 0]})
    labels.loc[:, "evaluation_split"] = "test"

    with pytest.raises(ValueError, match="Expected only split"):
        score_gold_labels(labels, expected_split="validation")


def test_threshold_analysis_selects_validation_supported_threshold() -> None:
    validation = gold_label_fixture(
        {
            "query-1": [0, 2, 1, 0, 0],
            "query-2": [2, 0, 0, 0, 0],
            "query-3": [2, 1, 0, 0, 0],
            "query-4": [2, 0, 0, 0, 0],
        },
        scores_by_query={
            "query-1": [0.38, 0.20, 0.10, 0.05, 0.01],
            "query-2": [0.39, 0.20, 0.10, 0.05, 0.01],
            "query-3": [0.50, 0.20, 0.10, 0.05, 0.01],
            "query-4": [0.20, 0.10, 0.05, 0.02, 0.01],
        },
    )
    test = gold_label_fixture(
        {
            "query-5": [2, 0, 0, 0, 0],
            "query-6": [1, 0, 0, 0, 0],
        },
        split="test",
        scores_by_query={
            "query-5": [0.40, 0.20, 0.10, 0.05, 0.01],
            "query-6": [0.30, 0.20, 0.10, 0.05, 0.01],
        },
    )

    analysis = build_threshold_analysis(
        validation_labels=validation,
        test_labels=test,
        thresholds=(0.0, 0.35, 0.39, 0.50),
    )

    assert analysis["selection"]["selected_threshold"] == 0.39
    assert analysis["selection"]["validation_allowed_query_rate"] == 0.5
    assert analysis["selection"]["validation_precision_allowed_queries"] == 1.0
    assert analysis["test_behavior_at_selected_threshold"]["allowed_query_rate"] == 0.5
