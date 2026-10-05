"""Lexical similar-ticket retrieval for resolved TicketPilot evidence."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ticketpilot.config import (
    PREPARED_DATASET_PATH,
    RAW_ENGLISH_DATA_PATH,
    RETRIEVAL_ARTIFACT_DIR,
    RETRIEVAL_INDEX_PATH,
    RETRIEVAL_LABEL_TEMPLATE_PATH,
    RETRIEVAL_REPORT_DIR,
    RETRIEVAL_REPORT_PATH,
)
from ticketpilot.data import load_ticket_csv
from ticketpilot.preparation import (
    CLASSIFIER_TEXT_COLUMN,
    GROUP_ID_COLUMN,
    ROW_ID_COLUMN,
    SPLIT_COLUMN,
    build_classifier_text,
)
from ticketpilot.training import load_prepared_dataset, validate_prepared_dataset

SOURCE_ID_COLUMN = "source_id"
RETRIEVAL_TEXT_COLUMN = "retrieval_text"
RESOLVED_ANSWER_COLUMN = "answer"
TRAIN_SPLIT = "train"
EVALUATION_SPLITS = ("validation", "test")
DEFAULT_TOP_K_VALUES = (1, 3, 5)


@dataclass(frozen=True)
class TfidfTicketRetriever:
    """TF-IDF retrieval index with corpus metadata kept alongside vectors."""

    vectorizer: TfidfVectorizer
    corpus_matrix: Any
    corpus: pd.DataFrame
    retrieval_text_column: str = RETRIEVAL_TEXT_COLUMN


@dataclass(frozen=True)
class RetrievalRun:
    """Paths and machine-readable report from a lexical retrieval run."""

    report: dict[str, Any]
    report_path: Path
    index_path: Path
    label_template_path: Path


def run_lexical_retrieval_baseline(
    *,
    raw_data_path: Path = RAW_ENGLISH_DATA_PATH,
    prepared_dataset_path: Path = PREPARED_DATASET_PATH,
    report_dir: Path = RETRIEVAL_REPORT_DIR,
    artifact_dir: Path = RETRIEVAL_ARTIFACT_DIR,
    index_path: Path = RETRIEVAL_INDEX_PATH,
    label_template_path: Path = RETRIEVAL_LABEL_TEMPLATE_PATH,
) -> RetrievalRun:
    """Build the train-only lexical retrieval index and persist evaluation output."""
    raw_tickets = load_ticket_csv(raw_data_path)
    prepared = load_prepared_dataset(prepared_dataset_path)
    corpus = build_resolved_training_corpus(raw_tickets, prepared)
    retriever = build_tfidf_retriever(corpus)

    report_dir = Path(report_dir)
    artifact_dir = Path(artifact_dir)
    index_path = Path(index_path)
    label_template_path = Path(label_template_path)
    if index_path == RETRIEVAL_INDEX_PATH and artifact_dir != RETRIEVAL_ARTIFACT_DIR:
        index_path = artifact_dir / RETRIEVAL_INDEX_PATH.name
    if (
        label_template_path == RETRIEVAL_LABEL_TEMPLATE_PATH
        and report_dir != RETRIEVAL_REPORT_DIR
    ):
        label_template_path = report_dir / RETRIEVAL_LABEL_TEMPLATE_PATH.name
    report_path = report_dir / RETRIEVAL_REPORT_PATH.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(retriever, index_path)
    report = build_retrieval_report(retriever, raw_tickets, prepared)
    report["artifacts"] = {
        "index": str(index_path),
        "metrics": str(report_path),
        "manual_relevance_template": str(label_template_path),
    }
    report["index"]["artifact_size_bytes"] = index_path.stat().st_size
    report_path = save_retrieval_report(report, report_path)
    write_manual_relevance_template(
        retriever,
        raw_tickets,
        prepared,
        output_path=label_template_path,
    )
    return RetrievalRun(
        report=report,
        report_path=report_path,
        index_path=index_path,
        label_template_path=label_template_path,
    )


def build_resolved_training_corpus(
    raw_tickets: pd.DataFrame,
    prepared: pd.DataFrame,
) -> pd.DataFrame:
    """Return resolved training tickets with original evidence fields preserved."""
    _validate_raw_ticket_columns(raw_tickets)
    validate_prepared_dataset(prepared)
    _validate_raw_prepared_alignment(raw_tickets, prepared)

    joined = raw_tickets.reset_index(drop=True).copy()
    prepared_reset = prepared.reset_index(drop=True)
    joined[ROW_ID_COLUMN] = prepared_reset[ROW_ID_COLUMN]
    joined[GROUP_ID_COLUMN] = prepared_reset[GROUP_ID_COLUMN]
    joined[SPLIT_COLUMN] = prepared_reset[SPLIT_COLUMN]
    joined[RETRIEVAL_TEXT_COLUMN] = [
        preprocess_retrieval_query(subject, body)
        for subject, body in zip(
            joined["subject"].tolist(),
            joined["body"].tolist(),
            strict=True,
        )
    ]
    joined[SOURCE_ID_COLUMN] = joined[ROW_ID_COLUMN].map(make_source_id)

    resolved_answer = joined[RESOLVED_ANSWER_COLUMN].map(_clean_text)
    corpus = joined.loc[
        joined[SPLIT_COLUMN].eq(TRAIN_SPLIT) & resolved_answer.ne(""),
        [
            SOURCE_ID_COLUMN,
            ROW_ID_COLUMN,
            GROUP_ID_COLUMN,
            SPLIT_COLUMN,
            "subject",
            "body",
            RESOLVED_ANSWER_COLUMN,
            "queue",
            "priority",
            RETRIEVAL_TEXT_COLUMN,
        ],
    ].copy()
    if corpus.empty:
        raise ValueError(
            "Retrieval corpus is empty; no resolved training tickets found."
        )
    return corpus.reset_index(drop=True)


def build_query_frame(
    raw_tickets: pd.DataFrame,
    prepared: pd.DataFrame,
) -> pd.DataFrame:
    """Build retrieval queries from held-out subject/body text and labels."""
    _validate_raw_ticket_columns(raw_tickets)
    validate_prepared_dataset(prepared)
    _validate_raw_prepared_alignment(raw_tickets, prepared)

    joined = raw_tickets.reset_index(drop=True).copy()
    prepared_reset = prepared.reset_index(drop=True)
    joined[ROW_ID_COLUMN] = prepared_reset[ROW_ID_COLUMN]
    joined[GROUP_ID_COLUMN] = prepared_reset[GROUP_ID_COLUMN]
    joined[SPLIT_COLUMN] = prepared_reset[SPLIT_COLUMN]
    joined[CLASSIFIER_TEXT_COLUMN] = prepared_reset[CLASSIFIER_TEXT_COLUMN]
    joined[RETRIEVAL_TEXT_COLUMN] = [
        preprocess_retrieval_query(subject, body)
        for subject, body in zip(
            joined["subject"].tolist(),
            joined["body"].tolist(),
            strict=True,
        )
    ]
    return joined.loc[
        :,
        [
            ROW_ID_COLUMN,
            GROUP_ID_COLUMN,
            SPLIT_COLUMN,
            "subject",
            "body",
            "queue",
            "priority",
            CLASSIFIER_TEXT_COLUMN,
            RETRIEVAL_TEXT_COLUMN,
        ],
    ].copy()


def build_tfidf_retriever(
    corpus: pd.DataFrame,
    *,
    max_features: int = 100_000,
) -> TfidfTicketRetriever:
    """Fit a deterministic TF-IDF vector index over train-ticket subject/body text."""
    _validate_corpus(corpus)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(corpus[RETRIEVAL_TEXT_COLUMN].tolist())
    return TfidfTicketRetriever(
        vectorizer=vectorizer,
        corpus_matrix=matrix,
        corpus=corpus.reset_index(drop=True).copy(),
    )


def preprocess_retrieval_query(subject: object, body: object = "") -> str:
    """Normalize only whitespace while preserving punctuation and technical tokens."""
    return build_classifier_text(subject, body)


def retrieve_similar_tickets(
    retriever: TfidfTicketRetriever,
    *,
    subject: object = "",
    body: object = "",
    query_text: object | None = None,
    top_k: int = 5,
    metadata_filter: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return top-k similar resolved tickets with stable citation IDs and scores."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    query = (
        _clean_text(query_text)
        if query_text is not None
        else _clean_text(preprocess_retrieval_query(subject, body))
    )
    if not query:
        return _empty_results_frame()

    corpus = _apply_metadata_filter(retriever.corpus, metadata_filter)
    if corpus.empty:
        return _empty_results_frame()

    row_positions = corpus.index.to_list()
    filtered_matrix = retriever.corpus_matrix[row_positions]
    query_matrix = retriever.vectorizer.transform([query])
    scores = cosine_similarity(query_matrix, filtered_matrix).ravel()
    ranked = pd.DataFrame(
        {
            "corpus_position": row_positions,
            "similarity": scores,
        }
    )
    ranked[SOURCE_ID_COLUMN] = corpus[SOURCE_ID_COLUMN].to_list()
    ranked = ranked.sort_values(
        by=["similarity", SOURCE_ID_COLUMN],
        ascending=[False, True],
        kind="mergesort",
    ).head(top_k)
    result = corpus.loc[ranked["corpus_position"].tolist(), :].copy()
    result["similarity"] = ranked["similarity"].to_list()
    return result.loc[:, _result_columns()].reset_index(drop=True)


def evaluate_retrieval_split(
    retriever: TfidfTicketRetriever,
    queries: pd.DataFrame,
    *,
    split: str,
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K_VALUES,
    relevance_column: str = "queue",
) -> dict[str, Any]:
    """Evaluate retrieval with silver relevance from a matching metadata column."""
    if relevance_column not in {"queue", "priority"}:
        raise ValueError("relevance_column must be 'queue' or 'priority'.")
    split_queries = queries.loc[queries[SPLIT_COLUMN].eq(split), :].reset_index(
        drop=True
    )
    query_count = int(len(split_queries))
    if query_count == 0:
        return {
            "split": split,
            "query_count": 0,
            "relevance_type": f"silver_{relevance_column}_match",
            "human_labeled_relevance": False,
            "recall_at_k": {f"recall@{k}": 0.0 for k in top_k_values},
            "mrr": 0.0,
            "mean_retrieved_count": 0.0,
            "mean_latency_ms_per_query": 0.0,
        }

    max_k = max(top_k_values)
    effective_max_k = min(max_k, len(retriever.corpus))
    start = time.perf_counter()
    query_matrix = retriever.vectorizer.transform(
        split_queries[RETRIEVAL_TEXT_COLUMN].fillna("").astype("string").tolist()
    )
    score_matrix = cosine_similarity(query_matrix, retriever.corpus_matrix)
    latency_ms_per_query = ((time.perf_counter() - start) * 1000) / query_count
    corpus_source_ids = np.asarray(retriever.corpus[SOURCE_ID_COLUMN].tolist())
    corpus_labels = np.asarray(retriever.corpus[relevance_column].tolist())
    query_labels = split_queries[relevance_column].tolist()
    reciprocal_ranks: list[float] = []
    recall_hits = {k: 0 for k in top_k_values}

    for row_index, query_label in enumerate(query_labels):
        scores = score_matrix[row_index]
        if effective_max_k == len(scores):
            candidate_indices = np.arange(len(scores))
        else:
            candidate_indices = np.argpartition(scores, -effective_max_k)[
                -effective_max_k:
            ]
        order = np.lexsort(
            (corpus_source_ids[candidate_indices], -scores[candidate_indices])
        )
        top_indices = candidate_indices[order]
        relevant = (corpus_labels[top_indices] == query_label).tolist()
        first_rank = _first_relevant_rank(relevant)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for k in top_k_values:
            if any(relevant[:k]):
                recall_hits[k] += 1

    return {
        "split": split,
        "query_count": query_count,
        "relevance_type": f"silver_{relevance_column}_match",
        "human_labeled_relevance": False,
        "recall_at_k": {
            f"recall@{k}": _safe_rate(recall_hits[k], query_count) for k in top_k_values
        },
        "mrr": _mean(reciprocal_ranks),
        "mean_retrieved_count": float(effective_max_k),
        "mean_latency_ms_per_query": float(latency_ms_per_query),
    }


def build_retrieval_report(
    retriever: TfidfTicketRetriever,
    raw_tickets: pd.DataFrame,
    prepared: pd.DataFrame,
) -> dict[str, Any]:
    """Build machine-readable retrieval diagnostics and silver evaluation metrics."""
    queries = build_query_frame(raw_tickets, prepared)
    held_out_source_ids = set(
        queries.loc[queries[SPLIT_COLUMN].ne(TRAIN_SPLIT), ROW_ID_COLUMN].map(
            make_source_id
        )
    )
    corpus_source_ids = set(retriever.corpus[SOURCE_ID_COLUMN].tolist())
    overlap = sorted(corpus_source_ids.intersection(held_out_source_ids))

    return {
        "task": "similar_resolved_ticket_retrieval",
        "retriever": "tfidf_cosine_similarity",
        "query_text_sources": ["subject", "body"],
        "corpus_text_indexed": ["subject", "body"],
        "evidence_fields_returned": [
            "source_id",
            "ticket_row_id",
            "subject",
            "body",
            "answer",
            "queue",
            "priority",
            "similarity",
        ],
        "excluded_from_query": [
            "answer",
            "queue",
            "priority",
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
        ],
        "corpus_policy": (
            "Only resolved tickets from the training split are indexed. "
            "Validation and test tickets are query-only held-out records."
        ),
        "index": {
            "corpus_count": int(len(retriever.corpus)),
            "vocabulary_size": int(len(retriever.vectorizer.vocabulary_)),
            "source_id_pattern": "TP-ticket-######",
            "held_out_source_id_overlap_count": len(overlap),
            "held_out_source_id_overlap_examples": overlap[:10],
        },
        "metadata_distribution": {
            "queue": _distribution(retriever.corpus["queue"]),
            "priority": _distribution(retriever.corpus["priority"]),
        },
        "evaluation": {
            split: evaluate_retrieval_split(retriever, queries, split=split)
            for split in EVALUATION_SPLITS
        },
        "manual_gold_template": {
            "human_labeled_relevance": False,
            "policy": (
                "Template rows are candidate query/result pairs. Human relevance "
                "labels and notes are intentionally blank until manually reviewed."
            ),
        },
    }


def write_manual_relevance_template(
    retriever: TfidfTicketRetriever,
    raw_tickets: pd.DataFrame,
    prepared: pd.DataFrame,
    *,
    output_path: Path,
    split: str = "validation",
    max_queries: int = 50,
    top_k: int = 5,
) -> Path:
    """Write an unlabeled CSV template for future human retrieval judgments."""
    queries = build_query_frame(raw_tickets, prepared)
    split_queries = (
        queries.loc[queries[SPLIT_COLUMN].eq(split), :]
        .sort_values(ROW_ID_COLUMN)
        .head(max_queries)
    )
    rows: list[dict[str, Any]] = []
    for _, query in split_queries.iterrows():
        results = retrieve_similar_tickets(
            retriever,
            query_text=query[RETRIEVAL_TEXT_COLUMN],
            top_k=top_k,
        )
        for rank, result in enumerate(results.to_dict("records"), start=1):
            rows.append(
                {
                    "query_ticket_id": query[ROW_ID_COLUMN],
                    "query_split": query[SPLIT_COLUMN],
                    "query_queue": query["queue"],
                    "query_priority": query["priority"],
                    "query_text": query[RETRIEVAL_TEXT_COLUMN],
                    "rank": rank,
                    "retrieved_source_id": result[SOURCE_ID_COLUMN],
                    "retrieved_ticket_id": result[ROW_ID_COLUMN],
                    "retrieved_queue": result["queue"],
                    "retrieved_priority": result["priority"],
                    "similarity": result["similarity"],
                    "silver_queue_match": result["queue"] == query["queue"],
                    "human_relevance_label": "",
                    "human_notes": "",
                }
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def save_retrieval_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write retrieval metrics as deterministic JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def make_source_id(ticket_id: object) -> str:
    """Return a stable citation ID derived from the prepared ticket row ID."""
    return f"TP-{_clean_text(ticket_id)}"


def _validate_raw_ticket_columns(raw_tickets: pd.DataFrame) -> None:
    required = {"subject", "body", "answer", "queue", "priority"}
    missing = sorted(required.difference(raw_tickets.columns))
    if missing:
        raise ValueError("Raw ticket data missing columns: " + ", ".join(missing))


def _validate_raw_prepared_alignment(
    raw_tickets: pd.DataFrame,
    prepared: pd.DataFrame,
) -> None:
    if len(raw_tickets) != len(prepared):
        raise ValueError(
            "Raw ticket and prepared dataset row counts differ: "
            f"{len(raw_tickets)} != {len(prepared)}."
        )
    expected_ids = [f"ticket-{index:06d}" for index in range(len(prepared))]
    actual_ids = prepared[ROW_ID_COLUMN].astype("string").tolist()
    if actual_ids != expected_ids:
        raise ValueError(
            "Prepared ticket_row_id values do not align with raw English row order."
        )


def _validate_corpus(corpus: pd.DataFrame) -> None:
    required = {
        SOURCE_ID_COLUMN,
        ROW_ID_COLUMN,
        GROUP_ID_COLUMN,
        SPLIT_COLUMN,
        "subject",
        "body",
        "answer",
        "queue",
        "priority",
        RETRIEVAL_TEXT_COLUMN,
    }
    missing = sorted(required.difference(corpus.columns))
    if missing:
        raise ValueError("Retrieval corpus missing columns: " + ", ".join(missing))
    if bool(corpus[SPLIT_COLUMN].ne(TRAIN_SPLIT).any()):
        raise ValueError("Retrieval corpus may only contain training split tickets.")
    empty_text = corpus[RETRIEVAL_TEXT_COLUMN].map(_clean_text).eq("")
    if bool(empty_text.any()):
        raise ValueError("Retrieval corpus contains empty subject/body text.")
    empty_answer = corpus[RESOLVED_ANSWER_COLUMN].map(_clean_text).eq("")
    if bool(empty_answer.any()):
        raise ValueError(
            "Retrieval corpus contains unresolved tickets without answers."
        )


def _apply_metadata_filter(
    corpus: pd.DataFrame,
    metadata_filter: dict[str, str] | None,
) -> pd.DataFrame:
    if not metadata_filter:
        return corpus
    filtered = corpus
    allowed_columns = {"queue", "priority"}
    unsupported = sorted(set(metadata_filter).difference(allowed_columns))
    if unsupported:
        raise ValueError(
            "Unsupported retrieval metadata filters: " + ", ".join(unsupported)
        )
    for column, value in metadata_filter.items():
        filtered = filtered.loc[filtered[column].eq(value), :]
    return filtered


def _result_columns() -> list[str]:
    return [
        SOURCE_ID_COLUMN,
        ROW_ID_COLUMN,
        "subject",
        "body",
        RESOLVED_ANSWER_COLUMN,
        "queue",
        "priority",
        "similarity",
    ]


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_result_columns())


def _first_relevant_rank(relevant: list[bool]) -> int | None:
    for index, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return index
    return None


def _distribution(series: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = series.value_counts().sort_index()
    total = int(len(series))
    return {
        str(label): {
            "count": int(count),
            "rate": _safe_rate(int(count), total),
        }
        for label, count in counts.items()
    }


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
