"""Leakage-safe classifier dataset preparation for TicketPilot."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from ticketpilot.config import (
    CLASSIFIER_INPUT_COLUMNS,
    CLASSIFIER_LABEL_COLUMNS,
    SPLIT_RANDOM_SEED,
    TEST_SPLIT_FRACTION,
    TRAIN_SPLIT_FRACTION,
    VALIDATION_SPLIT_FRACTION,
)
from ticketpilot.validation import (
    suspicious_label_mention_report,
    validate_classifier_feature_columns,
)

CLASSIFIER_TEXT_COLUMN = "classifier_text"
GROUP_ID_COLUMN = "ticket_text_group_id"
SPLIT_COLUMN = "split"
ROW_ID_COLUMN = "ticket_row_id"
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class PreparedTicketDataset:
    """Prepared classifier dataframe and machine-readable diagnostics."""

    frame: pd.DataFrame
    split_manifest: pd.DataFrame
    summary: dict[str, Any]
    classifier_feature_columns: tuple[str, ...]


def prepare_ticket_dataset(
    frame: pd.DataFrame,
    *,
    random_seed: int = SPLIT_RANDOM_SEED,
    train_fraction: float = TRAIN_SPLIT_FRACTION,
    validation_fraction: float = VALIDATION_SPLIT_FRACTION,
    test_fraction: float = TEST_SPLIT_FRACTION,
) -> PreparedTicketDataset:
    """Construct leakage-safe classifier text and deterministic data splits."""
    _validate_split_fractions(train_fraction, validation_fraction, test_fraction)
    validate_classifier_feature_columns(CLASSIFIER_INPUT_COLUMNS)

    prepared = frame.reset_index(drop=True).copy()
    prepared[ROW_ID_COLUMN] = [f"ticket-{index:06d}" for index in prepared.index]
    prepared[CLASSIFIER_TEXT_COLUMN] = [
        build_classifier_text(subject, body)
        for subject, body in zip(
            prepared["subject"].tolist(),
            prepared["body"].tolist(),
            strict=True,
        )
    ]
    prepared[GROUP_ID_COLUMN] = prepared[CLASSIFIER_TEXT_COLUMN].map(
        ticket_text_fingerprint
    )
    prepared["ticket_text_char_length"] = prepared[CLASSIFIER_TEXT_COLUMN].str.len()
    prepared["ticket_text_word_count"] = prepared[CLASSIFIER_TEXT_COLUMN].map(
        _word_count
    )

    split_by_group, split_strategy = _assign_group_splits(
        prepared,
        random_seed=random_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    prepared[SPLIT_COLUMN] = prepared[GROUP_ID_COLUMN].map(split_by_group)

    classifier_feature_columns = (CLASSIFIER_TEXT_COLUMN,)
    manifest_columns = (
        ROW_ID_COLUMN,
        GROUP_ID_COLUMN,
        SPLIT_COLUMN,
        "queue",
        "priority",
        "ticket_text_char_length",
        "ticket_text_word_count",
    )
    split_manifest = prepared.loc[:, list(manifest_columns)].copy()
    summary = build_dataset_summary(
        prepared,
        classifier_feature_columns=classifier_feature_columns,
        split_strategy=split_strategy,
        random_seed=random_seed,
        split_fractions={
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
    )

    output_columns = (
        ROW_ID_COLUMN,
        GROUP_ID_COLUMN,
        SPLIT_COLUMN,
        "subject",
        "body",
        CLASSIFIER_TEXT_COLUMN,
        "queue",
        "priority",
    )
    return PreparedTicketDataset(
        frame=prepared.loc[:, list(output_columns)].copy(),
        split_manifest=split_manifest,
        summary=summary,
        classifier_feature_columns=classifier_feature_columns,
    )


def build_classifier_text(subject: object, body: object) -> str:
    """Combine subject and body with minimal whitespace normalization."""
    pieces = [_clean_input_text(subject), _clean_input_text(body)]
    return "\n\n".join(piece for piece in pieces if piece)


def ticket_text_fingerprint(text: object) -> str:
    """Return a stable fingerprint for duplicate-aware split grouping."""
    normalized = _normalize_for_fingerprint(text)
    return sha256(normalized.encode("utf-8")).hexdigest()


def write_prepared_dataset(
    prepared: PreparedTicketDataset,
    *,
    prepared_dataset_path: Path,
    split_manifest_path: Path,
    dataset_summary_path: Path,
) -> None:
    """Write prepared data, split manifest, and JSON summary artifacts."""
    prepared_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_summary_path.parent.mkdir(parents=True, exist_ok=True)

    prepared.frame.to_csv(prepared_dataset_path, index=False)
    prepared.split_manifest.to_csv(split_manifest_path, index=False)
    dataset_summary_path.write_text(
        json.dumps(prepared.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_dataset_summary(
    prepared: pd.DataFrame,
    *,
    classifier_feature_columns: tuple[str, ...],
    split_strategy: dict[str, Any],
    random_seed: int,
    split_fractions: dict[str, float],
) -> dict[str, Any]:
    """Build machine-readable diagnostics for prepared ticket data."""
    group_counts = prepared[GROUP_ID_COLUMN].value_counts()
    duplicate_group_sizes = group_counts[group_counts > 1]
    label_conflicts = _label_conflict_summary(prepared)

    return {
        "row_count": int(len(prepared)),
        "group_count": int(group_counts.shape[0]),
        "classifier_text_sources": list(CLASSIFIER_INPUT_COLUMNS),
        "classifier_feature_columns": list(classifier_feature_columns),
        "classifier_label_columns": list(CLASSIFIER_LABEL_COLUMNS),
        "forbidden_feature_columns": [
            "answer",
            "type",
            "language",
            "version",
            "tag_1",
            "tag_2",
            "tag_3",
            "tag_4",
            "tag_5",
            "tag_6",
            "tag_7",
            "tag_8",
            "queue",
            "priority",
        ],
        "split_counts": _counts(prepared[SPLIT_COLUMN]),
        "split_group_counts": _counts(
            prepared.drop_duplicates(GROUP_ID_COLUMN)[SPLIT_COLUMN]
        ),
        "queue_distribution": _counts(prepared["queue"]),
        "priority_distribution": _counts(prepared["priority"]),
        "split_queue_distribution": _nested_split_counts(prepared, "queue"),
        "split_priority_distribution": _nested_split_counts(prepared, "priority"),
        "ticket_length": _ticket_length_summary(prepared),
        "missing_values": _missing_value_summary(prepared),
        "exact_duplicate_ticket_text": {
            "duplicate_group_count": int(duplicate_group_sizes.shape[0]),
            "duplicate_record_count": int(
                duplicate_group_sizes.sum() - duplicate_group_sizes.shape[0]
            ),
            "max_group_size": int(duplicate_group_sizes.max())
            if not duplicate_group_sizes.empty
            else 1,
        },
        "label_conflicts_within_duplicate_groups": label_conflicts,
        "suspicious_label_mentions": suspicious_label_mention_report(prepared),
        "repeated_or_highly_similar_subject_body": _similarity_summary(prepared),
        "split_strategy": split_strategy,
        "random_seed": random_seed,
        "split_fractions": split_fractions,
        "final_test_set_policy": (
            "The test split is reserved for final reporting only and must not be "
            "used for hyperparameter selection, threshold selection, or classifier "
            "selection."
        ),
    }


def _assign_group_splits(
    prepared: pd.DataFrame,
    *,
    random_seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups = _group_label_frame(prepared)
    stratify_column, warnings = _choose_stratification_column(groups)
    split_by_group: dict[str, str] = {}

    if stratify_column is None:
        ordered_groups = _deterministic_order(
            groups[GROUP_ID_COLUMN].tolist(), random_seed
        )
        split_by_group.update(
            _split_ordered_group_ids(
                ordered_groups,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
            )
        )
    else:
        for _, stratum in groups.groupby(stratify_column, sort=True):
            ordered_groups = _deterministic_order(
                stratum[GROUP_ID_COLUMN].tolist(),
                random_seed,
            )
            split_by_group.update(
                _split_ordered_group_ids(
                    ordered_groups,
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                    test_fraction=test_fraction,
                )
            )

    return split_by_group, {
        "unit_of_observation": "ticket row",
        "unit_of_splitting": "normalized subject+body fingerprint",
        "stratification": stratify_column or "none",
        "warnings": warnings,
    }


def _group_label_frame(prepared: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for group_id, group in prepared.groupby(GROUP_ID_COLUMN, sort=True):
        queue = str(group["queue"].mode(dropna=False).iloc[0])
        priority = str(group["priority"].mode(dropna=False).iloc[0])
        rows.append(
            {
                GROUP_ID_COLUMN: str(group_id),
                "queue": queue,
                "priority": priority,
                "queue_priority": f"{queue} | {priority}",
            }
        )
    return pd.DataFrame(rows)


def _choose_stratification_column(groups: pd.DataFrame) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    for candidate in ("queue_priority", "queue"):
        counts = groups[candidate].value_counts()
        too_small = counts[counts < len(SPLIT_NAMES)]
        if too_small.empty:
            return candidate, warnings
        warnings.append(
            f"{candidate} has strata with fewer than {len(SPLIT_NAMES)} groups: "
            + ", ".join(f"{label}={count}" for label, count in too_small.items())
        )
    warnings.append(
        "Falling back to deterministic unstratified grouped split because some "
        "classes are too small for reliable three-way stratification."
    )
    return None, warnings


def _split_ordered_group_ids(
    group_ids: list[str],
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    group_count = len(group_ids)
    if group_count == 0:
        return {}
    if group_count >= len(SPLIT_NAMES):
        test_count = max(1, round(group_count * test_fraction))
        validation_count = max(1, round(group_count * validation_fraction))
        if test_count + validation_count >= group_count:
            test_count = 1
            validation_count = 1
    else:
        test_count = round(group_count * test_fraction)
        validation_count = round(group_count * validation_fraction)
    train_count = group_count - validation_count - test_count
    if train_count <= 0 and group_count > 0:
        train_count = 1
        if validation_count > test_count and validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1

    split_by_group: dict[str, str] = {}
    train_end = train_count
    validation_end = train_count + validation_count
    for index, group_id in enumerate(group_ids):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        split_by_group[group_id] = split
    return split_by_group


def _deterministic_order(group_ids: list[str], random_seed: int) -> list[str]:
    return sorted(
        group_ids,
        key=lambda group_id: sha256(f"{random_seed}:{group_id}".encode()).hexdigest(),
    )


def _similarity_summary(prepared: pd.DataFrame) -> dict[str, Any]:
    subject_counts = Counter(
        _normalize_for_fingerprint(value)
        for value in prepared["subject"].tolist()
        if _normalize_for_fingerprint(value)
    )
    body_counts = Counter(
        _normalize_for_fingerprint(value)
        for value in prepared["body"].tolist()
        if _normalize_for_fingerprint(value)
    )
    prefix_counts = Counter(
        _prefix_signature(value) for value in prepared[CLASSIFIER_TEXT_COLUMN].tolist()
    )
    repeated_prefixes = {
        prefix: count for prefix, count in prefix_counts.items() if prefix and count > 1
    }
    return {
        "repeated_subject_count": sum(
            1 for count in subject_counts.values() if count > 1
        ),
        "repeated_body_count": sum(1 for count in body_counts.values() if count > 1),
        "similar_prefix_group_count": len(repeated_prefixes),
        "high_similarity_pair_count": _count_high_similarity_pairs(prepared),
    }


def _count_high_similarity_pairs(prepared: pd.DataFrame) -> int:
    pair_count = 0
    text_by_prefix: dict[str, list[str]] = {}
    for text in prepared[CLASSIFIER_TEXT_COLUMN].tolist():
        prefix = _prefix_signature(text, token_count=8)
        if prefix:
            text_by_prefix.setdefault(prefix, []).append(
                _normalize_for_fingerprint(text)
            )

    for texts in text_by_prefix.values():
        unique_texts = sorted(set(texts))
        if len(unique_texts) < 2 or len(unique_texts) > 25:
            continue
        for left_index, left in enumerate(unique_texts):
            for right in unique_texts[left_index + 1 :]:
                if SequenceMatcher(a=left, b=right).ratio() >= 0.92:
                    pair_count += 1
    return pair_count


def _label_conflict_summary(prepared: pd.DataFrame) -> dict[str, int]:
    conflict_counts = {"queue": 0, "priority": 0}
    for _, group in prepared.groupby(GROUP_ID_COLUMN):
        if group["queue"].nunique(dropna=False) > 1:
            conflict_counts["queue"] += 1
        if group["priority"].nunique(dropna=False) > 1:
            conflict_counts["priority"] += 1
    return conflict_counts


def _ticket_length_summary(prepared: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "characters": _describe_numeric(prepared["ticket_text_char_length"]),
        "words": _describe_numeric(prepared["ticket_text_word_count"]),
    }


def _missing_value_summary(prepared: pd.DataFrame) -> dict[str, int]:
    return {
        "subject": int(prepared["subject"].isna().sum()),
        "body": int(prepared["body"].isna().sum()),
        "classifier_text": int(
            prepared[CLASSIFIER_TEXT_COLUMN]
            .fillna("")
            .astype("string")
            .str.strip()
            .eq("")
            .sum()
        ),
    }


def _nested_split_counts(
    prepared: pd.DataFrame, column: str
) -> dict[str, dict[str, int]]:
    return {
        str(split): _counts(split_frame[column])
        for split, split_frame in prepared.groupby(SPLIT_COLUMN, sort=True)
    }


def _counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.to_dict().items()}


def _describe_numeric(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def _clean_input_text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_for_fingerprint(value: object) -> str:
    return re.sub(r"\s+", " ", _clean_input_text(value).casefold()).strip()


def _prefix_signature(value: object, *, token_count: int = 12) -> str:
    normalized = _normalize_for_fingerprint(value)
    return " ".join(normalized.split()[:token_count])


def _word_count(value: object) -> int:
    return len(_clean_input_text(value).split())


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0; received {total}.")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Split fractions must all be positive.")
