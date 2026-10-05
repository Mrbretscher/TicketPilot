"""Near-duplicate cross-split leakage audit utilities."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ticketpilot.preparation import CLASSIFIER_TEXT_COLUMN, ROW_ID_COLUMN, SPLIT_COLUMN

DEFAULT_SIMILARITY_THRESHOLDS = (0.90, 0.95, 0.98)
SPLIT_PAIRS = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)


@dataclass(frozen=True)
class CandidatePair:
    """One near-duplicate candidate that crosses split boundaries."""

    left_ticket_id: str
    right_ticket_id: str
    left_split: str
    right_split: str
    similarity: float
    left_queue: str
    right_queue: str
    same_queue: bool
    left_priority: str
    right_priority: str
    same_priority: bool
    left_text_preview: str
    right_text_preview: str


def audit_near_duplicate_cross_split_leakage(
    prepared: pd.DataFrame,
    *,
    raw_tickets: pd.DataFrame | None = None,
    thresholds: tuple[float, ...] = DEFAULT_SIMILARITY_THRESHOLDS,
    sample_size: int = 25,
) -> dict[str, Any]:
    """Audit cross-split near-duplicate text candidates without changing splits."""
    _validate_thresholds(thresholds)
    audit_frame = _audit_frame(prepared)
    candidates = _find_cross_split_candidates(audit_frame, thresholds=thresholds)
    repeated_subjects = (
        _repeated_subject_cross_split_summary(prepared, raw_tickets)
        if raw_tickets is not None
        else {"status": "not_computed", "reason": "raw tickets were not supplied"}
    )

    return {
        "method": {
            "name": "tfidf_cosine_cross_split_nearest_neighbors",
            "ticket_representation": (
                "Normalized subject + body text from prepared classifier_text."
            ),
            "normalization": (
                "Whitespace collapsed, leading/trailing whitespace stripped, and "
                "Unicode casefold applied before vectorization."
            ),
            "vectorizer": {
                "class": "sklearn.feature_extraction.text.TfidfVectorizer",
                "ngram_range": [1, 2],
                "lowercase": False,
                "norm": "l2",
            },
            "neighbor_search": {
                "class": "sklearn.neighbors.NearestNeighbors",
                "metric": "cosine",
                "algorithm": "brute",
                "split_pairs": [list(pair) for pair in SPLIT_PAIRS],
                "minimum_similarity": min(thresholds),
            },
            "interpretation_limits": [
                "High lexical similarity is a candidate signal, not proof of leakage.",
                (
                    "TF-IDF cosine can miss paraphrases and can over-score "
                    "repeated templates."
                ),
            ],
        },
        "thresholds": list(thresholds),
        "row_counts_by_split": _counts(audit_frame[SPLIT_COLUMN]),
        "total_candidate_pair_count": len(candidates),
        "candidate_pair_counts": _candidate_pair_counts(candidates, thresholds),
        "affected_ticket_counts": _affected_ticket_counts(candidates, thresholds),
        "score_distribution": _score_distribution(
            [candidate.similarity for candidate in candidates]
        ),
        "queue_label_summary": _queue_label_summary(candidates),
        "priority_label_summary": _priority_label_summary(candidates),
        "repeated_subject_cross_split_counts": repeated_subjects,
        "candidate_pairs": [_candidate_to_dict(candidate) for candidate in candidates],
        "representative_high_similarity_pairs": [
            _candidate_to_dict(candidate)
            for candidate in sorted(
                candidates, key=lambda item: item.similarity, reverse=True
            )[:sample_size]
        ],
    }


def _find_cross_split_candidates(
    frame: pd.DataFrame, *, thresholds: tuple[float, ...]
) -> list[CandidatePair]:
    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1, 2),
        norm="l2",
    )
    vectors = vectorizer.fit_transform(frame["normalized_ticket_text"].tolist())
    min_similarity = min(thresholds)
    max_distance = 1.0 - min_similarity
    candidates: list[CandidatePair] = []

    for left_split, right_split in SPLIT_PAIRS:
        left_positions = frame.index[frame[SPLIT_COLUMN].eq(left_split)].tolist()
        right_positions = frame.index[frame[SPLIT_COLUMN].eq(right_split)].tolist()
        if not left_positions or not right_positions:
            continue

        right_vectors = vectors[right_positions]
        neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        neighbors.fit(right_vectors)
        distances, indices = neighbors.radius_neighbors(
            vectors[left_positions],
            radius=max_distance,
            return_distance=True,
            sort_results=True,
        )

        for left_offset, (row_distances, row_indices) in enumerate(
            zip(distances, indices, strict=True)
        ):
            left_position = left_positions[left_offset]
            left_row = frame.loc[left_position]
            for distance, right_offset in zip(row_distances, row_indices, strict=True):
                similarity = 1.0 - float(distance)
                if similarity + 1e-12 < min_similarity:
                    continue
                right_position = right_positions[int(right_offset)]
                right_row = frame.loc[right_position]
                candidates.append(
                    CandidatePair(
                        left_ticket_id=str(left_row[ROW_ID_COLUMN]),
                        right_ticket_id=str(right_row[ROW_ID_COLUMN]),
                        left_split=left_split,
                        right_split=right_split,
                        similarity=similarity,
                        left_queue=str(left_row["queue"]),
                        right_queue=str(right_row["queue"]),
                        same_queue=bool(left_row["queue"] == right_row["queue"]),
                        left_priority=str(left_row["priority"]),
                        right_priority=str(right_row["priority"]),
                        same_priority=bool(
                            left_row["priority"] == right_row["priority"]
                        ),
                        left_text_preview=_preview(left_row[CLASSIFIER_TEXT_COLUMN]),
                        right_text_preview=_preview(right_row[CLASSIFIER_TEXT_COLUMN]),
                    )
                )

    return sorted(
        candidates,
        key=lambda item: (
            item.left_split,
            item.right_split,
            -item.similarity,
            item.left_ticket_id,
            item.right_ticket_id,
        ),
    )


def _audit_frame(prepared: pd.DataFrame) -> pd.DataFrame:
    required = {
        ROW_ID_COLUMN,
        SPLIT_COLUMN,
        CLASSIFIER_TEXT_COLUMN,
        "queue",
        "priority",
    }
    missing = sorted(required.difference(prepared.columns))
    if missing:
        raise ValueError("Prepared dataset is missing columns: " + ", ".join(missing))
    frame = prepared.loc[:, sorted(required)].copy()
    frame["normalized_ticket_text"] = frame[CLASSIFIER_TEXT_COLUMN].map(
        normalize_ticket_text
    )
    if frame["normalized_ticket_text"].str.len().eq(0).any():
        raise ValueError("Prepared dataset contains empty normalized ticket text.")
    return frame.reset_index(drop=True)


def normalize_ticket_text(value: object) -> str:
    """Normalize ticket text for cross-split near-duplicate comparison."""
    return re.sub(r"\s+", " ", _clean_text(value).casefold()).strip()


def _clean_text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _repeated_subject_cross_split_summary(
    prepared: pd.DataFrame, raw_tickets: pd.DataFrame
) -> dict[str, Any]:
    if len(prepared) != len(raw_tickets):
        raise ValueError(
            "Prepared and raw ticket row counts differ: "
            f"{len(prepared)} != {len(raw_tickets)}."
        )
    if "subject" not in raw_tickets.columns:
        raise ValueError("Raw tickets are missing the subject column.")

    subjects = raw_tickets["subject"].map(normalize_ticket_text)
    subject_frame = prepared.loc[:, [ROW_ID_COLUMN, SPLIT_COLUMN, "queue"]].copy()
    subject_frame["normalized_subject"] = subjects.tolist()
    subject_frame = subject_frame.loc[
        subject_frame["normalized_subject"].astype(str).str.len() > 0, :
    ]

    cross_split_subjects: list[dict[str, Any]] = []
    pair_counts_by_split_pair: Counter[str] = Counter()
    same_queue_pairs = 0
    different_queue_pairs = 0

    for subject, group in subject_frame.groupby("normalized_subject", sort=True):
        splits = sorted(group[SPLIT_COLUMN].unique().tolist())
        if len(splits) < 2:
            continue
        pair_count = 0
        records = group.reset_index(drop=True)
        for left_index in range(len(records)):
            left = records.loc[left_index]
            for right_index in range(left_index + 1, len(records)):
                right = records.loc[right_index]
                if left[SPLIT_COLUMN] == right[SPLIT_COLUMN]:
                    continue
                split_pair = _split_pair_key(
                    str(left[SPLIT_COLUMN]), str(right[SPLIT_COLUMN])
                )
                pair_counts_by_split_pair[split_pair] += 1
                pair_count += 1
                if left["queue"] == right["queue"]:
                    same_queue_pairs += 1
                else:
                    different_queue_pairs += 1
        cross_split_subjects.append(
            {
                "normalized_subject": str(subject),
                "row_count": int(len(group)),
                "splits": splits,
                "cross_split_pair_count": pair_count,
                "queues": _counts(group["queue"]),
            }
        )

    top_subjects = sorted(
        cross_split_subjects,
        key=lambda item: (
            int(item["cross_split_pair_count"]),
            int(item["row_count"]),
            str(item["normalized_subject"]),
        ),
        reverse=True,
    )[:25]
    return {
        "status": "computed",
        "subject_value_count": len(cross_split_subjects),
        "cross_split_pair_count": int(sum(pair_counts_by_split_pair.values())),
        "pair_counts_by_split_pair": dict(sorted(pair_counts_by_split_pair.items())),
        "same_queue_pair_count": same_queue_pairs,
        "different_queue_pair_count": different_queue_pairs,
        "top_cross_split_subjects": top_subjects,
    }


def _candidate_pair_counts(
    candidates: list[CandidatePair], thresholds: tuple[float, ...]
) -> dict[str, Any]:
    by_threshold = {
        _threshold_key(threshold): sum(
            candidate.similarity + 1e-12 >= threshold for candidate in candidates
        )
        for threshold in thresholds
    }
    by_split_pair: dict[str, dict[str, int]] = {}
    for left_split, right_split in SPLIT_PAIRS:
        key = _split_pair_key(left_split, right_split)
        split_candidates = [
            candidate
            for candidate in candidates
            if candidate.left_split == left_split
            and candidate.right_split == right_split
        ]
        by_split_pair[key] = {
            _threshold_key(threshold): sum(
                candidate.similarity + 1e-12 >= threshold
                for candidate in split_candidates
            )
            for threshold in thresholds
        }
    return {
        "by_threshold": by_threshold,
        "by_split_pair_and_threshold": by_split_pair,
    }


def _affected_ticket_counts(
    candidates: list[CandidatePair], thresholds: tuple[float, ...]
) -> dict[str, Any]:
    by_threshold: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        split_to_ticket_ids: dict[str, set[str]] = {
            "train": set(),
            "validation": set(),
            "test": set(),
        }
        for candidate in candidates:
            if candidate.similarity + 1e-12 < threshold:
                continue
            split_to_ticket_ids[candidate.left_split].add(candidate.left_ticket_id)
            split_to_ticket_ids[candidate.right_split].add(candidate.right_ticket_id)
        by_threshold[_threshold_key(threshold)] = {
            **{
                split: len(ticket_ids)
                for split, ticket_ids in split_to_ticket_ids.items()
            },
            "total_unique_tickets": len(set().union(*split_to_ticket_ids.values())),
        }
    return {"by_threshold": by_threshold}


def _queue_label_summary(candidates: list[CandidatePair]) -> dict[str, Any]:
    queue_pairs = Counter(
        f"{candidate.left_queue} | {candidate.right_queue}" for candidate in candidates
    )
    return {
        "same_queue_pair_count": sum(candidate.same_queue for candidate in candidates),
        "different_queue_pair_count": sum(
            not candidate.same_queue for candidate in candidates
        ),
        "queue_pair_counts": dict(sorted(queue_pairs.items())),
    }


def _priority_label_summary(candidates: list[CandidatePair]) -> dict[str, Any]:
    priority_pairs = Counter(
        f"{candidate.left_priority} | {candidate.right_priority}"
        for candidate in candidates
    )
    return {
        "same_priority_pair_count": sum(
            candidate.same_priority for candidate in candidates
        ),
        "different_priority_pair_count": sum(
            not candidate.same_priority for candidate in candidates
        ),
        "priority_pair_counts": dict(sorted(priority_pairs.items())),
    }


def _score_distribution(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p98": None,
            "p99": None,
            "max": None,
        }
    values = np.array(scores, dtype=float)
    return {
        "count": int(values.shape[0]),
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p98": float(np.quantile(values, 0.98)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _candidate_to_dict(candidate: CandidatePair) -> dict[str, Any]:
    return {
        "left_ticket_id": candidate.left_ticket_id,
        "right_ticket_id": candidate.right_ticket_id,
        "split_pair": _split_pair_key(candidate.left_split, candidate.right_split),
        "similarity": candidate.similarity,
        "left_queue": candidate.left_queue,
        "right_queue": candidate.right_queue,
        "same_queue": candidate.same_queue,
        "left_priority": candidate.left_priority,
        "right_priority": candidate.right_priority,
        "same_priority": candidate.same_priority,
        "left_text_preview": candidate.left_text_preview,
        "right_text_preview": candidate.right_text_preview,
    }


def _preview(value: object, *, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.to_dict().items()}


def _split_pair_key(left_split: str, right_split: str) -> str:
    split_order = {
        split: index for index, split in enumerate(("train", "validation", "test"))
    }
    ordered = sorted([left_split, right_split], key=lambda split: split_order[split])
    return f"{ordered[0]}_vs_{ordered[1]}"


def _threshold_key(threshold: float) -> str:
    return f">={threshold:.2f}"


def _validate_thresholds(thresholds: tuple[float, ...]) -> None:
    if not thresholds:
        raise ValueError("At least one similarity threshold is required.")
    if any(threshold <= 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("Similarity thresholds must be in the range (0, 1].")
