"""Human-labeled gold retrieval workflow for deployed TicketPilot retrieval."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from ticketpilot.config import (
    GOLD_RETRIEVAL_METRICS_PATH,
    GOLD_RETRIEVAL_TEST_LABEL_PATH,
    GOLD_RETRIEVAL_VALIDATION_LABEL_PATH,
    PREPARED_DATASET_PATH,
    RAW_ENGLISH_DATA_PATH,
    RETRIEVAL_INDEX_PATH,
    RETRIEVAL_REPORT_PATH,
)
from ticketpilot.data import load_ticket_csv
from ticketpilot.preparation import ROW_ID_COLUMN, SPLIT_COLUMN
from ticketpilot.retrieval import (
    RETRIEVAL_TEXT_COLUMN,
    SOURCE_ID_COLUMN,
    TfidfTicketRetriever,
    build_query_frame,
    retrieve_similar_tickets,
)
from ticketpilot.training import load_prepared_dataset

GOLD_VALIDATION_SPLIT = "validation"
GOLD_TEST_SPLIT = "test"
GOLD_QUERY_COUNT = 20
GOLD_TOP_K = 5
RELEVANCE_LABEL_COLUMN = "relevance_label"
REVIEWER_NOTES_COLUMN = "reviewer_notes"
MIN_THRESHOLD_ALLOWED_QUERY_RATE = 0.50
DEFAULT_EVIDENCE_THRESHOLDS = tuple(
    sorted({round(value / 100, 2) for value in range(0, 76, 5)} | {0.39})
)

GOLD_LABEL_COLUMNS = [
    "evaluation_split",
    "query_id",
    "query_queue",
    "query_subject",
    "query_body",
    "candidate_rank",
    "candidate_evidence_id",
    "candidate_queue",
    "candidate_subject",
    "candidate_issue_excerpt",
    "candidate_resolution_excerpt",
    "retrieval_score",
    RELEVANCE_LABEL_COLUMN,
    REVIEWER_NOTES_COLUMN,
]


@dataclass(frozen=True)
class GoldLabelingRun:
    """Paths written by the gold retrieval labeling workflow."""

    validation_path: Path
    test_path: Path
    validation_query_count: int
    test_query_count: int


def load_deployed_tfidf_retriever(
    index_path: Path = RETRIEVAL_INDEX_PATH,
) -> TfidfTicketRetriever:
    """Load the retriever artifact used by the deployed local service."""
    if not Path(index_path).exists():
        raise FileNotFoundError(
            "Deployed retrieval artifact is missing. Build it first with "
            f"scripts/build_retrieval_baseline.py: {index_path}"
        )
    retriever = joblib.load(index_path)
    if not isinstance(retriever, TfidfTicketRetriever):
        raise TypeError(
            "The deployed retrieval artifact must be a TfidfTicketRetriever. "
            f"Found {type(retriever).__name__} at {index_path}."
        )
    return retriever


def build_gold_labeling_files(
    *,
    raw_data_path: Path = RAW_ENGLISH_DATA_PATH,
    prepared_dataset_path: Path = PREPARED_DATASET_PATH,
    retriever_index_path: Path = RETRIEVAL_INDEX_PATH,
    validation_output_path: Path = GOLD_RETRIEVAL_VALIDATION_LABEL_PATH,
    test_output_path: Path = GOLD_RETRIEVAL_TEST_LABEL_PATH,
    query_count: int = GOLD_QUERY_COUNT,
    top_k: int = GOLD_TOP_K,
) -> GoldLabelingRun:
    """Create blank human-labeling CSVs for validation and final test queries."""
    raw_tickets = load_ticket_csv(raw_data_path)
    prepared = load_prepared_dataset(prepared_dataset_path)
    retriever = load_deployed_tfidf_retriever(retriever_index_path)
    queries = build_query_frame(raw_tickets, prepared)

    validation_frame = build_gold_label_frame(
        retriever,
        queries,
        split=GOLD_VALIDATION_SPLIT,
        query_count=query_count,
        top_k=top_k,
    )
    test_frame = build_gold_label_frame(
        retriever,
        queries,
        split=GOLD_TEST_SPLIT,
        query_count=query_count,
        top_k=top_k,
    )
    _write_gold_label_frame(validation_frame, validation_output_path)
    _write_gold_label_frame(test_frame, test_output_path)
    return GoldLabelingRun(
        validation_path=Path(validation_output_path),
        test_path=Path(test_output_path),
        validation_query_count=validation_frame["query_id"].nunique(),
        test_query_count=test_frame["query_id"].nunique(),
    )


def build_gold_label_frame(
    retriever: TfidfTicketRetriever,
    queries: pd.DataFrame,
    *,
    split: str,
    query_count: int = GOLD_QUERY_COUNT,
    top_k: int = GOLD_TOP_K,
) -> pd.DataFrame:
    """Return unlabeled candidate rows for one gold retrieval split."""
    if query_count <= 0:
        raise ValueError("query_count must be positive.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    split_queries = queries.loc[queries[SPLIT_COLUMN].eq(split), :].copy()
    if len(split_queries) < query_count:
        raise ValueError(
            f"Split {split!r} has only {len(split_queries)} queries; "
            f"{query_count} are required."
        )

    sampled_queries = select_stratified_gold_queries(
        split_queries,
        retriever=retriever,
        query_count=query_count,
    )
    rows: list[dict[str, Any]] = []
    for query in sampled_queries.to_dict("records"):
        results = retrieve_similar_tickets(
            retriever,
            query_text=query[RETRIEVAL_TEXT_COLUMN],
            top_k=top_k,
        )
        for rank, result in enumerate(results.to_dict("records"), start=1):
            rows.append(
                {
                    "evaluation_split": split,
                    "query_id": str(query[ROW_ID_COLUMN]),
                    "query_queue": _clean_text(query["queue"]),
                    "query_subject": _clean_text(query["subject"]),
                    "query_body": _clean_text(query["body"]),
                    "candidate_rank": rank,
                    "candidate_evidence_id": str(result[SOURCE_ID_COLUMN]),
                    "candidate_queue": _clean_text(result["queue"]),
                    "candidate_subject": _clean_text(result["subject"]),
                    "candidate_issue_excerpt": _excerpt(result["body"]),
                    "candidate_resolution_excerpt": _excerpt(result["answer"]),
                    "retrieval_score": float(result["similarity"]),
                    RELEVANCE_LABEL_COLUMN: "",
                    REVIEWER_NOTES_COLUMN: "",
                }
            )
    return pd.DataFrame(rows, columns=GOLD_LABEL_COLUMNS)


def select_stratified_gold_queries(
    split_queries: pd.DataFrame,
    *,
    retriever: TfidfTicketRetriever,
    query_count: int = GOLD_QUERY_COUNT,
) -> pd.DataFrame:
    """Select a deterministic small query set with practical stratification."""
    if len(split_queries) < query_count:
        raise ValueError(
            f"Cannot select {query_count} queries from {len(split_queries)} rows."
        )
    candidates = split_queries.reset_index(drop=True).copy()
    candidates["text_token_count"] = candidates[RETRIEVAL_TEXT_COLUMN].map(
        lambda value: len(_clean_text(value).split())
    )
    candidates["length_group"] = _length_groups(candidates["text_token_count"])
    candidates["queue_support_group"] = _queue_support_groups(candidates["queue"])
    candidates = _stratification_candidate_pool(
        candidates,
        minimum_size=query_count,
    )
    top_scores, margins = _top_retrieval_score_features(candidates, retriever)
    candidates["top_retrieval_score"] = top_scores
    candidates["top_score_margin"] = margins
    candidates["difficulty_group"] = _difficulty_groups(candidates)

    selected_indices: list[int] = []
    for support_group in ("lower_support", "common"):
        queue_counts = (
            candidates.loc[candidates["queue_support_group"].eq(support_group), "queue"]
            .value_counts()
            .sort_values(kind="mergesort")
        )
        for queue in queue_counts.index.astype("string").tolist():
            if len(selected_indices) >= query_count:
                break
            pool = candidates.loc[
                candidates["queue"].eq(queue)
                & ~candidates.index.isin(selected_indices),
                :,
            ]
            if not pool.empty:
                selected_indices.append(
                    _best_query_index(pool, candidates, selected_indices)
                )

    while len(selected_indices) < query_count:
        pool = candidates.loc[~candidates.index.isin(selected_indices), :]
        selected_indices.append(_best_query_index(pool, candidates, selected_indices))

    selected = candidates.loc[selected_indices, :].sort_values(
        ROW_ID_COLUMN,
        kind="mergesort",
    )
    return selected.reset_index(drop=True)


def score_gold_label_files(
    *,
    validation_label_path: Path = GOLD_RETRIEVAL_VALIDATION_LABEL_PATH,
    test_label_path: Path = GOLD_RETRIEVAL_TEST_LABEL_PATH,
    output_path: Path | None = GOLD_RETRIEVAL_METRICS_PATH,
    silver_report_path: Path = RETRIEVAL_REPORT_PATH,
) -> dict[str, Any]:
    """Score completed human labels for validation and test gold retrieval sets."""
    validation_labels = pd.read_csv(validation_label_path, dtype="string")
    test_labels = pd.read_csv(test_label_path, dtype="string")
    validation = score_gold_labels(
        validation_labels,
        expected_split=GOLD_VALIDATION_SPLIT,
    )
    test = score_gold_labels(
        test_labels,
        expected_split=GOLD_TEST_SPLIT,
    )
    threshold_analysis = build_threshold_analysis(
        validation_labels=validation_labels,
        test_labels=test_labels,
    )
    report: dict[str, Any] = {
        "gold_validation": validation,
        "gold_test": test,
        "silver_queue_match_comparison": load_silver_queue_match_metrics(
            silver_report_path
        ),
        "evidence_threshold_analysis": threshold_analysis,
        "policy": {
            "validation_usage": (
                "Validation labels may be used for retrieval/evidence-threshold "
                "selection."
            ),
            "test_usage": (
                "Test labels are reporting-only and must not be used to choose "
                "thresholds, retriever parameters, prompts, or hybrid weights."
            ),
        },
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def score_gold_labels(
    labels: pd.DataFrame,
    *,
    expected_split: str,
) -> dict[str, Any]:
    """Calculate Recall@K, MRR, and nDCG@5 from completed human labels."""
    _validate_gold_label_frame(labels, expected_split=expected_split)
    scored = labels.copy()
    scored["_label"] = [
        _parse_relevance_label(value, row_number=index + 2)
        for index, value in enumerate(scored[RELEVANCE_LABEL_COLUMN].tolist())
    ]
    scored["candidate_rank"] = scored["candidate_rank"].astype(int)
    query_count = int(scored["query_id"].nunique())
    recall_hits = {1: 0, 3: 0, 5: 0}
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []

    for _, group in scored.groupby("query_id", sort=True):
        ranked = group.sort_values("candidate_rank", kind="mergesort")
        labels_for_query = [int(value) for value in ranked["_label"].tolist()]
        first_rank = _first_relevant_rank(labels_for_query)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for k in recall_hits:
            if any(label > 0 for label in labels_for_query[:k]):
                recall_hits[k] += 1
        ndcg_scores.append(_ndcg_at_k(labels_for_query, k=5))

    return {
        "evaluation_split": expected_split,
        "query_count": query_count,
        "candidate_count": int(len(scored)),
        "human_labeled_relevance": True,
        "label_distribution": _label_distribution(scored["_label"]),
        "recall_at_k": {
            f"recall@{k}": _safe_rate(recall_hits[k], query_count)
            for k in sorted(recall_hits)
        },
        "mrr": _mean(reciprocal_ranks),
        "ndcg@5": _mean(ndcg_scores),
    }


def build_threshold_analysis(
    *,
    validation_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_EVIDENCE_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate evidence-score threshold candidates using validation for selection."""
    validation_frame = _scored_label_frame(
        validation_labels,
        expected_split=GOLD_VALIDATION_SPLIT,
    )
    test_frame = _scored_label_frame(
        test_labels,
        expected_split=GOLD_TEST_SPLIT,
    )
    validation_metrics = evaluate_evidence_thresholds(
        validation_frame,
        thresholds=thresholds,
    )
    selection = select_evidence_threshold(validation_metrics)
    selected_threshold = selection.get("selected_threshold")
    test_at_selected = None
    if selected_threshold is not None:
        test_at_selected = evaluate_single_evidence_threshold(
            test_frame,
            threshold=float(selected_threshold),
        )
    return {
        "selection_split": GOLD_VALIDATION_SPLIT,
        "selection_policy": (
            "Choose the lowest candidate threshold with perfect validation "
            "top-evidence precision while allowing at least "
            f"{MIN_THRESHOLD_ALLOWED_QUERY_RATE:.0%} of validation queries. "
            "If no candidate meets that bar, do not replace the conservative "
            "drafting threshold."
        ),
        "relevance_definition": (
            "For threshold selection, a query allowed through is counted precise "
            "when its top-ranked retrieved evidence has human relevance_label > 0."
        ),
        "validation_thresholds": validation_metrics,
        "score_distributions": {
            "validation": score_distributions(validation_frame),
            "test": score_distributions(test_frame),
        },
        "selection": selection,
        "test_behavior_at_selected_threshold": test_at_selected,
    }


def evaluate_evidence_thresholds(
    labels: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = DEFAULT_EVIDENCE_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Evaluate query-level evidence gates for candidate score thresholds."""
    return [
        evaluate_single_evidence_threshold(labels, threshold=threshold)
        for threshold in thresholds
    ]


def evaluate_single_evidence_threshold(
    labels: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate one best-score evidence threshold on labeled retrieval rows."""
    top = _top_ranked_rows(labels)
    query_count = int(len(top))
    allowed = top.loc[top["retrieval_score"].ge(threshold), :]
    allowed_count = int(len(allowed))
    relevant_allowed = int(allowed["_is_relevant"].sum()) if allowed_count else 0
    strongly_relevant_allowed = (
        int(allowed["_is_strongly_relevant"].sum()) if allowed_count else 0
    )
    false_allowed = allowed_count - relevant_allowed
    return {
        "threshold": float(threshold),
        "query_count": query_count,
        "allowed_query_count": allowed_count,
        "allowed_query_rate": _safe_rate(allowed_count, query_count),
        "abstained_query_count": query_count - allowed_count,
        "abstained_query_rate": _safe_rate(query_count - allowed_count, query_count),
        "relevance_precision_allowed_queries": (
            None if allowed_count == 0 else _safe_rate(relevant_allowed, allowed_count)
        ),
        "strong_relevance_rate_allowed_queries": (
            None
            if allowed_count == 0
            else _safe_rate(strongly_relevant_allowed, allowed_count)
        ),
        "false_allowed_query_count": false_allowed,
    }


def select_evidence_threshold(
    threshold_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select a validation-supported evidence threshold when one is defensible."""
    eligible = [
        item
        for item in threshold_metrics
        if item["allowed_query_count"] > 0
        and item["allowed_query_rate"] >= MIN_THRESHOLD_ALLOWED_QUERY_RATE
        and item["relevance_precision_allowed_queries"] == 1.0
    ]
    if not eligible:
        return {
            "selected_threshold": None,
            "decision": "no_defensible_single_threshold",
            "reason": (
                "No candidate threshold achieved perfect validation precision "
                f"while allowing at least {MIN_THRESHOLD_ALLOWED_QUERY_RATE:.0%} "
                "of queries."
            ),
        }
    selected = min(eligible, key=lambda item: float(item["threshold"]))
    return {
        "selected_threshold": float(selected["threshold"]),
        "decision": "replace_default_threshold",
        "validation_allowed_query_rate": selected["allowed_query_rate"],
        "validation_precision_allowed_queries": selected[
            "relevance_precision_allowed_queries"
        ],
        "validation_abstained_query_rate": selected["abstained_query_rate"],
        "reason": (
            "The selected threshold is the lowest validation candidate that "
            "excludes all top-ranked irrelevant evidence in the labeled "
            "validation sample while still allowing at least half of queries."
        ),
    }


def score_distributions(labels: pd.DataFrame) -> dict[str, Any]:
    """Return retrieval-score distributions for relevant and irrelevant evidence."""
    return {
        "candidate_level": {
            "irrelevant_label_0": _score_summary(
                labels.loc[labels["_label"].eq(0), "retrieval_score"]
            ),
            "relevant_labels_1_or_2": _score_summary(
                labels.loc[labels["_is_relevant"], "retrieval_score"]
            ),
            "plausibly_useful_label_1": _score_summary(
                labels.loc[labels["_label"].eq(1), "retrieval_score"]
            ),
            "strongly_relevant_label_2": _score_summary(
                labels.loc[labels["_label"].eq(2), "retrieval_score"]
            ),
        },
        "top_ranked_query_level": {
            "irrelevant_label_0": _score_summary(
                _top_ranked_rows(labels).loc[
                    _top_ranked_rows(labels)["_label"].eq(0), "retrieval_score"
                ]
            ),
            "relevant_labels_1_or_2": _score_summary(
                _top_ranked_rows(labels).loc[
                    _top_ranked_rows(labels)["_is_relevant"], "retrieval_score"
                ]
            ),
        },
    }


def load_silver_queue_match_metrics(report_path: Path) -> dict[str, Any]:
    """Load existing silver queue-match metrics for comparison, not selection."""
    path = Path(report_path)
    if not path.exists():
        return {
            "available": False,
            "reason": f"Silver retrieval report not found: {path}",
        }
    report = json.loads(path.read_text(encoding="utf-8"))
    evaluation = report.get("evaluation", {})
    return {
        "available": True,
        "source": str(path),
        "relevance_type": "silver_queue_match",
        "human_labeled_relevance": False,
        "deployed_retriever": report.get("retriever", "unknown"),
        "validation": evaluation.get(GOLD_VALIDATION_SPLIT),
        "test": evaluation.get(GOLD_TEST_SPLIT),
        "policy": (
            "Silver queue-match metrics are useful sanity checks but are not "
            "combined with human gold metrics."
        ),
    }


def _write_gold_label_frame(frame: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def _scored_label_frame(labels: pd.DataFrame, *, expected_split: str) -> pd.DataFrame:
    _validate_gold_label_frame(labels, expected_split=expected_split)
    scored = labels.copy()
    scored["_label"] = [
        _parse_relevance_label(value, row_number=index + 2)
        for index, value in enumerate(scored[RELEVANCE_LABEL_COLUMN].tolist())
    ]
    scored["candidate_rank"] = scored["candidate_rank"].astype(int)
    scored["retrieval_score"] = scored["retrieval_score"].astype(float)
    scored["_is_relevant"] = scored["_label"].gt(0)
    scored["_is_strongly_relevant"] = scored["_label"].eq(2)
    return scored


def _validate_gold_label_frame(labels: pd.DataFrame, *, expected_split: str) -> None:
    missing_columns = sorted(set(GOLD_LABEL_COLUMNS).difference(labels.columns))
    if missing_columns:
        raise ValueError(
            "Gold label CSV missing required columns: " + ", ".join(missing_columns)
        )
    unexpected_splits = sorted(
        set(labels["evaluation_split"].dropna().astype("string").tolist())
        - {expected_split}
    )
    if unexpected_splits:
        raise ValueError(
            f"Expected only split {expected_split!r}, found: "
            + ", ".join(unexpected_splits)
        )
    if labels[RELEVANCE_LABEL_COLUMN].isna().any():
        missing_rows = _missing_label_rows(labels)
        raise ValueError(
            "Human relevance labels are required before scoring. Missing rows: "
            + ", ".join(str(row) for row in missing_rows[:10])
        )
    blank_labels = labels[RELEVANCE_LABEL_COLUMN].astype("string").str.strip().eq("")
    if bool(blank_labels.any()):
        missing_rows = [index + 2 for index in labels.index[blank_labels].tolist()]
        raise ValueError(
            "Human relevance labels are required before scoring. Missing rows: "
            + ", ".join(str(row) for row in missing_rows[:10])
        )
    duplicate_ranks = labels.duplicated(subset=["query_id", "candidate_rank"])
    if bool(duplicate_ranks.any()):
        raise ValueError("Each query_id/candidate_rank pair must be unique.")


def _parse_relevance_label(value: object, *, row_number: int) -> int:
    text = _clean_text(value)
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid relevance_label at CSV row {row_number}: {text!r}. "
            "Use only 0, 1, or 2."
        ) from exc
    if not number.is_integer() or int(number) not in {0, 1, 2}:
        raise ValueError(
            f"Invalid relevance_label at CSV row {row_number}: {text!r}. "
            "Use only 0, 1, or 2."
        )
    return int(number)


def _missing_label_rows(labels: pd.DataFrame) -> list[int]:
    return [index + 2 for index in labels.index[labels[RELEVANCE_LABEL_COLUMN].isna()]]


def _length_groups(token_counts: pd.Series) -> pd.Series:
    lower = float(token_counts.quantile(0.25))
    upper = float(token_counts.quantile(0.75))
    groups = []
    for value in token_counts.tolist():
        count = int(value)
        if count <= lower:
            groups.append("short")
        elif count >= upper:
            groups.append("long")
        else:
            groups.append("medium")
    return pd.Series(groups, index=token_counts.index)


def _queue_support_groups(queues: pd.Series) -> pd.Series:
    counts = queues.value_counts()
    count_by_queue = {
        str(label): int(count) for label, count in counts.astype(int).items()
    }
    median_count = float(counts.median())
    groups = [
        "lower_support" if count_by_queue[str(queue)] <= median_count else "common"
        for queue in queues.astype("string").tolist()
    ]
    return pd.Series(groups, index=queues.index)


def _difficulty_groups(candidates: pd.DataFrame) -> pd.Series:
    score_median = float(candidates["top_retrieval_score"].median())
    margin_median = float(candidates["top_score_margin"].median())
    groups = []
    for row in candidates.to_dict("records"):
        easy = (
            float(row["top_retrieval_score"]) >= score_median
            and float(row["top_score_margin"]) >= margin_median
        )
        groups.append("easy" if easy else "ambiguous")
    return pd.Series(groups, index=candidates.index)


def _stratification_candidate_pool(
    candidates: pd.DataFrame,
    *,
    minimum_size: int,
    per_queue_limit: int = 12,
) -> pd.DataFrame:
    selected_indices: set[int] = set()
    queue_counts = candidates["queue"].value_counts().sort_values(kind="mergesort")
    for queue in queue_counts.index.astype("string").tolist():
        queue_rows = candidates.loc[candidates["queue"].eq(queue), :]
        for index in (
            queue_rows.sort_values(
                ["text_token_count", ROW_ID_COLUMN],
                ascending=[True, True],
                kind="mergesort",
            )
            .head(per_queue_limit // 3)
            .index.tolist()
        ):
            selected_indices.add(int(cast(Any, index)))
        for index in (
            queue_rows.sort_values(
                ["text_token_count", ROW_ID_COLUMN],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(per_queue_limit // 3)
            .index.tolist()
        ):
            selected_indices.add(int(cast(Any, index)))
        for index in _evenly_spaced_indices(
            queue_rows.sort_values(ROW_ID_COLUMN, kind="mergesort"),
            count=per_queue_limit - 2 * (per_queue_limit // 3),
        ):
            selected_indices.add(index)

    if len(selected_indices) < minimum_size:
        for index in candidates.sort_values(ROW_ID_COLUMN, kind="mergesort").index:
            selected_indices.add(int(cast(Any, index)))
            if len(selected_indices) >= minimum_size:
                break
    return candidates.loc[sorted(selected_indices), :].copy()


def _evenly_spaced_indices(frame: pd.DataFrame, *, count: int) -> list[int]:
    if frame.empty or count <= 0:
        return []
    if len(frame) <= count:
        return [int(cast(Any, index)) for index in frame.index.tolist()]
    positions = [
        round(position * (len(frame) - 1) / (count - 1)) for position in range(count)
    ]
    return [int(cast(Any, frame.index[position])) for position in positions]


def _best_query_index(
    pool: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_indices: list[int],
) -> int:
    selected = (
        candidates.loc[selected_indices, :] if selected_indices else candidates.head(0)
    )
    current_counts = _selected_group_counts(selected)
    target_per_group = max(3, min(6, math.ceil((len(selected_indices) + 1) / 4)))
    scored_pool = []
    for index, row in pool.iterrows():
        groups = [
            _clean_text(row["length_group"]),
            _clean_text(row["difficulty_group"]),
            _clean_text(row["queue_support_group"]),
        ]
        improvement = sum(
            1 for group in groups if current_counts.get(group, 0) < target_per_group
        )
        queue_count = (
            int(selected["queue"].eq(row["queue"]).sum()) if not selected.empty else 0
        )
        scored_pool.append(
            (
                -improvement,
                queue_count,
                _clean_text(row["queue"]),
                _clean_text(row[ROW_ID_COLUMN]),
                int(cast(Any, index)),
            )
        )
    scored_pool.sort()
    return int(scored_pool[0][-1])


def _selected_group_counts(selected: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in ("length_group", "difficulty_group", "queue_support_group"):
        values = (
            selected[column].astype("string").tolist() if column in selected else []
        )
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _top_retrieval_score_features(
    candidates: pd.DataFrame,
    retriever: TfidfTicketRetriever,
) -> tuple[list[float], list[float]]:
    query_texts = candidates[RETRIEVAL_TEXT_COLUMN].fillna("").astype("string").tolist()
    query_matrix = retriever.vectorizer.transform(query_texts)
    score_matrix = query_matrix @ retriever.corpus_matrix.T
    top_scores: list[float] = []
    margins: list[float] = []
    for row_index in range(score_matrix.shape[0]):
        scores = np.asarray(score_matrix.getrow(row_index).toarray()).ravel()
        if scores.size == 0:
            top_scores.append(0.0)
            margins.append(0.0)
            continue
        if scores.size == 1:
            top_score = float(scores[0])
            top_scores.append(top_score)
            margins.append(top_score)
            continue
        top_two = np.partition(scores, -2)[-2:]
        top_two.sort()
        second_score = float(top_two[0])
        top_score = float(top_two[1])
        top_scores.append(top_score)
        margins.append(top_score - second_score)
    return top_scores, margins


def _first_relevant_rank(labels: list[int]) -> int | None:
    for rank, label in enumerate(labels, start=1):
        if label > 0:
            return rank
    return None


def _top_ranked_rows(labels: pd.DataFrame) -> pd.DataFrame:
    return (
        labels.sort_values(["query_id", "candidate_rank"], kind="mergesort")
        .groupby("query_id", sort=True)
        .first()
        .reset_index()
    )


def _score_summary(scores: pd.Series) -> dict[str, float | int | None]:
    clean_scores = scores.dropna().astype(float)
    if clean_scores.empty:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "count": int(len(clean_scores)),
        "mean": float(clean_scores.mean()),
        "min": float(clean_scores.min()),
        "p25": float(clean_scores.quantile(0.25)),
        "median": float(clean_scores.median()),
        "p75": float(clean_scores.quantile(0.75)),
        "max": float(clean_scores.max()),
    }


def _ndcg_at_k(labels: list[int], *, k: int) -> float:
    gains = labels[:k]
    ideal = sorted(labels, reverse=True)[:k]
    denominator = _dcg(ideal)
    if denominator == 0.0:
        return 0.0
    return _dcg(gains) / denominator


def _dcg(labels: list[int]) -> float:
    return float(
        sum(
            (2.0**label - 1.0) / math.log2(rank + 1)
            for rank, label in enumerate(labels, start=1)
        )
    )


def _label_distribution(labels: pd.Series) -> dict[str, int]:
    counts = labels.value_counts().sort_index()
    return {str(int(cast(Any, label))): int(count) for label, count in counts.items()}


def _excerpt(value: object, *, max_chars: int = 700) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _clean_text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return " ".join(str(value).split()).strip()


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
