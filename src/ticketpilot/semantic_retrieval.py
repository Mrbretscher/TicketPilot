"""Semantic and hybrid similar-ticket retrieval for TicketPilot."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from ticketpilot.config import (
    HYBRID_SEMANTIC_WEIGHT,
    PREPARED_DATASET_PATH,
    RAW_ENGLISH_DATA_PATH,
    RETRIEVAL_ARTIFACT_DIR,
    RETRIEVAL_REPORT_DIR,
    SEMANTIC_RETRIEVAL_ARTIFACT_DIR,
    SEMANTIC_RETRIEVAL_BATCH_SIZE,
    SEMANTIC_RETRIEVAL_CONFIG_PATH,
    SEMANTIC_RETRIEVAL_EMBEDDINGS_PATH,
    SEMANTIC_RETRIEVAL_LABEL_TEMPLATE_PATH,
    SEMANTIC_RETRIEVAL_METADATA_PATH,
    SEMANTIC_RETRIEVAL_MODEL_NAME,
    SEMANTIC_RETRIEVAL_NORMALIZE_EMBEDDINGS,
    SEMANTIC_RETRIEVAL_REPORT_DIR,
    SEMANTIC_RETRIEVAL_REPORT_PATH,
)
from ticketpilot.data import load_ticket_csv
from ticketpilot.preparation import ROW_ID_COLUMN, SPLIT_COLUMN
from ticketpilot.retrieval import (
    DEFAULT_TOP_K_VALUES,
    EVALUATION_SPLITS,
    RETRIEVAL_TEXT_COLUMN,
    SOURCE_ID_COLUMN,
    TRAIN_SPLIT,
    TfidfTicketRetriever,
    build_query_frame,
    build_resolved_training_corpus,
    build_tfidf_retriever,
    make_source_id,
    retrieve_similar_tickets,
    save_retrieval_report,
)
from ticketpilot.training import load_prepared_dataset

EmbeddingArray = npt.NDArray[np.float32]


class TextEmbedder(Protocol):
    """Minimal embedding interface implemented by SentenceTransformer and tests."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        """Encode text into a two-dimensional embedding array."""


@dataclass(frozen=True)
class SemanticTicketRetriever:
    """Dense semantic retrieval index with corpus metadata."""

    corpus: pd.DataFrame
    embeddings: EmbeddingArray
    model_name: str
    model_config: dict[str, Any]
    embedding_model: TextEmbedder | None = None


@dataclass(frozen=True)
class SemanticRetrievalRun:
    """Paths and report from a semantic retrieval comparison run."""

    report: dict[str, Any]
    report_path: Path
    embeddings_path: Path
    metadata_path: Path
    config_path: Path
    label_template_path: Path


def load_sentence_transformer(model_name: str) -> TextEmbedder:
    """Load the configured sentence-transformers model."""
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    from sentence_transformers import SentenceTransformer

    return cast(TextEmbedder, SentenceTransformer(model_name))


def run_semantic_retrieval_comparison(
    *,
    raw_data_path: Path = RAW_ENGLISH_DATA_PATH,
    prepared_dataset_path: Path = PREPARED_DATASET_PATH,
    semantic_report_dir: Path = SEMANTIC_RETRIEVAL_REPORT_DIR,
    semantic_artifact_dir: Path = SEMANTIC_RETRIEVAL_ARTIFACT_DIR,
    lexical_report_dir: Path = RETRIEVAL_REPORT_DIR,
    lexical_artifact_dir: Path = RETRIEVAL_ARTIFACT_DIR,
    model_name: str = SEMANTIC_RETRIEVAL_MODEL_NAME,
    batch_size: int = SEMANTIC_RETRIEVAL_BATCH_SIZE,
    normalize_embeddings: bool = SEMANTIC_RETRIEVAL_NORMALIZE_EMBEDDINGS,
    hybrid_semantic_weight: float = HYBRID_SEMANTIC_WEIGHT,
    embedding_model: TextEmbedder | None = None,
) -> SemanticRetrievalRun:
    """Build semantic artifacts and compare lexical, semantic, and hybrid retrieval."""
    raw_tickets = load_ticket_csv(raw_data_path)
    prepared = load_prepared_dataset(prepared_dataset_path)
    corpus = build_resolved_training_corpus(raw_tickets, prepared)
    queries = build_query_frame(raw_tickets, prepared)
    lexical_retriever = build_tfidf_retriever(corpus)

    model = embedding_model or load_sentence_transformer(model_name)
    semantic_retriever = build_semantic_retriever(
        corpus,
        embedding_model=model,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    semantic_report_dir = Path(semantic_report_dir)
    semantic_artifact_dir = Path(semantic_artifact_dir)
    lexical_report_dir = Path(lexical_report_dir)
    lexical_artifact_dir = Path(lexical_artifact_dir)
    semantic_report_dir.mkdir(parents=True, exist_ok=True)
    semantic_artifact_dir.mkdir(parents=True, exist_ok=True)
    lexical_report_dir.mkdir(parents=True, exist_ok=True)
    lexical_artifact_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = semantic_artifact_dir / SEMANTIC_RETRIEVAL_EMBEDDINGS_PATH.name
    metadata_path = semantic_artifact_dir / SEMANTIC_RETRIEVAL_METADATA_PATH.name
    config_path = semantic_artifact_dir / SEMANTIC_RETRIEVAL_CONFIG_PATH.name
    report_path = semantic_report_dir / SEMANTIC_RETRIEVAL_REPORT_PATH.name
    label_template_path = (
        semantic_report_dir / SEMANTIC_RETRIEVAL_LABEL_TEMPLATE_PATH.name
    )

    save_semantic_artifacts(
        semantic_retriever,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        config_path=config_path,
    )
    report = build_semantic_retrieval_report(
        lexical_retriever,
        semantic_retriever,
        queries,
        hybrid_semantic_weight=hybrid_semantic_weight,
    )
    report["artifacts"] = {
        "semantic_embeddings": str(embeddings_path),
        "semantic_metadata": str(metadata_path),
        "semantic_config": str(config_path),
        "metrics": str(report_path),
        "manual_relevance_template": str(label_template_path),
    }
    report["semantic_index"]["embedding_artifact_size_bytes"] = (
        embeddings_path.stat().st_size
    )
    report["semantic_index"]["metadata_artifact_size_bytes"] = (
        metadata_path.stat().st_size
    )
    report_path = save_retrieval_report(report, report_path)
    write_comparison_label_template(
        lexical_retriever,
        semantic_retriever,
        queries,
        selected_method=str(report["selection"]["selected_method"]),
        output_path=label_template_path,
        embedding_model=model,
        hybrid_semantic_weight=hybrid_semantic_weight,
    )
    return SemanticRetrievalRun(
        report=report,
        report_path=report_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        config_path=config_path,
        label_template_path=label_template_path,
    )


def build_semantic_retriever(
    corpus: pd.DataFrame,
    *,
    embedding_model: TextEmbedder,
    model_name: str,
    batch_size: int = SEMANTIC_RETRIEVAL_BATCH_SIZE,
    normalize_embeddings: bool = SEMANTIC_RETRIEVAL_NORMALIZE_EMBEDDINGS,
) -> SemanticTicketRetriever:
    """Embed the train-only retrieval corpus and build a semantic retriever."""
    texts = corpus[RETRIEVAL_TEXT_COLUMN].fillna("").astype("string").tolist()
    embeddings = encode_texts(
        embedding_model,
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    if len(corpus) != embeddings.shape[0]:
        raise ValueError(
            "Embedding row count does not match corpus metadata: "
            f"{embeddings.shape[0]} != {len(corpus)}."
        )
    model_config = build_model_config(
        embedding_model,
        model_name=model_name,
        embedding_dimension=int(embeddings.shape[1]),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    return SemanticTicketRetriever(
        corpus=corpus.reset_index(drop=True).copy(),
        embeddings=embeddings,
        model_name=model_name,
        model_config=model_config,
        embedding_model=embedding_model,
    )


def encode_texts(
    embedding_model: TextEmbedder,
    texts: list[str],
    *,
    batch_size: int = SEMANTIC_RETRIEVAL_BATCH_SIZE,
    normalize_embeddings: bool = SEMANTIC_RETRIEVAL_NORMALIZE_EMBEDDINGS,
) -> EmbeddingArray:
    """Encode text and return a float32 two-dimensional matrix."""
    embeddings = embedding_model.encode(
        [_clean_text(text) for text in texts],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, found shape {array.shape}.")
    if normalize_embeddings:
        array = normalize_embedding_matrix(array)
    return array


def normalize_embedding_matrix(embeddings: EmbeddingArray) -> EmbeddingArray:
    """L2-normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return np.asarray(embeddings / safe_norms, dtype=np.float32)


def retrieve_semantic_tickets(
    retriever: SemanticTicketRetriever,
    *,
    query_text: object,
    top_k: int = 5,
    embedding_model: TextEmbedder | None = None,
    metadata_filter: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return top-k semantic nearest neighbors with stable source IDs."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    query = _clean_text(query_text)
    if not query:
        return _empty_results_frame()
    model = embedding_model or retriever.embedding_model
    if model is None:
        raise ValueError("An embedding model is required to encode semantic queries.")

    corpus = _apply_metadata_filter(retriever.corpus, metadata_filter)
    if corpus.empty:
        return _empty_results_frame()
    query_embedding = encode_texts(
        model,
        [query],
        batch_size=int(retriever.model_config["batch_size"]),
        normalize_embeddings=bool(retriever.model_config["normalize_embeddings"]),
    )[0]
    scores = retriever.embeddings[corpus.index.to_numpy()] @ query_embedding
    return _rank_results(corpus, scores, top_k=top_k)


def retrieve_hybrid_tickets(
    lexical_retriever: TfidfTicketRetriever,
    semantic_retriever: SemanticTicketRetriever,
    *,
    query_text: object,
    top_k: int = 5,
    embedding_model: TextEmbedder | None = None,
    semantic_weight: float = HYBRID_SEMANTIC_WEIGHT,
) -> pd.DataFrame:
    """Blend normalized lexical and semantic scores for hybrid retrieval."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    query = _clean_text(query_text)
    if not query:
        return _empty_results_frame()
    _validate_aligned_corpora(lexical_retriever.corpus, semantic_retriever.corpus)
    model = embedding_model or semantic_retriever.embedding_model
    if model is None:
        raise ValueError("An embedding model is required to encode hybrid queries.")

    lexical_scores, _ = _lexical_score_matrix(lexical_retriever, [query])
    semantic_query = encode_texts(
        model,
        [query],
        batch_size=int(semantic_retriever.model_config["batch_size"]),
        normalize_embeddings=bool(
            semantic_retriever.model_config["normalize_embeddings"]
        ),
    )
    semantic_scores = semantic_query @ semantic_retriever.embeddings.T
    combined_scores = combine_lexical_semantic_scores(
        lexical_scores.reshape(1, -1),
        semantic_scores,
        semantic_weight=semantic_weight,
    )[0]
    return _rank_results(
        semantic_retriever.corpus,
        combined_scores,
        top_k=top_k,
        score_column="hybrid_score",
    )


def build_semantic_retrieval_report(
    lexical_retriever: TfidfTicketRetriever,
    semantic_retriever: SemanticTicketRetriever,
    queries: pd.DataFrame,
    *,
    hybrid_semantic_weight: float = HYBRID_SEMANTIC_WEIGHT,
) -> dict[str, Any]:
    """Compare TF-IDF, semantic, and hybrid retrieval on held-out queries."""
    _validate_aligned_corpora(lexical_retriever.corpus, semantic_retriever.corpus)
    held_out_source_ids = set(
        queries.loc[queries[SPLIT_COLUMN].ne(TRAIN_SPLIT), ROW_ID_COLUMN].map(
            make_source_id
        )
    )
    corpus_source_ids = set(semantic_retriever.corpus[SOURCE_ID_COLUMN].tolist())
    overlap = sorted(corpus_source_ids.intersection(held_out_source_ids))

    method_metrics: dict[str, dict[str, Any]] = {
        "tfidf": {},
        "semantic": {},
        "hybrid_tfidf_semantic_50_50": {},
    }
    for split in EVALUATION_SPLITS:
        query_texts = _split_query_texts(queries, split)
        lexical_scores, lexical_latency = _lexical_score_matrix(
            lexical_retriever,
            query_texts,
        )
        method_metrics["tfidf"][split] = evaluate_matrix_retrieval_split(
            lexical_scores,
            lexical_retriever.corpus,
            queries,
            split=split,
            method_name="tfidf_cosine_similarity",
            latency_ms_per_query=lexical_latency,
        )
        semantic_scores, semantic_latency = _semantic_score_matrix(
            semantic_retriever,
            query_texts,
        )
        method_metrics["semantic"][split] = evaluate_matrix_retrieval_split(
            semantic_scores,
            semantic_retriever.corpus,
            queries,
            split=split,
            method_name="semantic_cosine_similarity",
            latency_ms_per_query=semantic_latency,
        )
        hybrid_scores = combine_lexical_semantic_scores(
            lexical_scores,
            semantic_scores,
            semantic_weight=hybrid_semantic_weight,
        )
        method_metrics["hybrid_tfidf_semantic_50_50"][split] = (
            evaluate_matrix_retrieval_split(
                hybrid_scores,
                semantic_retriever.corpus,
                queries,
                split=split,
                method_name="hybrid_tfidf_semantic_minmax_50_50",
                latency_ms_per_query=(
                    method_metrics["tfidf"][split]["mean_latency_ms_per_query"]
                    + semantic_latency
                ),
            )
        )

    selected = select_retrieval_method(method_metrics)
    return {
        "task": "similar_resolved_ticket_retrieval",
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
        "corpus_policy": (
            "Only resolved tickets from the training split are embedded or indexed. "
            "Validation and test tickets remain query-only held-out records."
        ),
        "semantic_index": {
            "corpus_count": int(len(semantic_retriever.corpus)),
            "embedding_shape": list(semantic_retriever.embeddings.shape),
            "source_id_pattern": "TP-ticket-######",
            "held_out_source_id_overlap_count": len(overlap),
            "held_out_source_id_overlap_examples": overlap[:10],
        },
        "model": semantic_retriever.model_config,
        "hybrid": {
            "method": "query_minmax_normalized_weighted_average",
            "semantic_weight": hybrid_semantic_weight,
            "lexical_weight": 1.0 - hybrid_semantic_weight,
            "justification": (
                "A simple fixed-weight hybrid is included as an understandable "
                "baseline for combining exact lexical matches with semantic "
                "similarity. It is selected only if validation metrics support it."
            ),
        },
        "evaluation": method_metrics,
        "selection": selected,
        "manual_gold_template": {
            "human_labeled_relevance": False,
            "schema": "same columns as Milestone 5 manual relevance template",
        },
    }


def evaluate_matrix_retrieval_split(
    score_matrix: npt.NDArray[np.floating[Any]],
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    split: str,
    method_name: str,
    top_k_values: tuple[int, ...] = DEFAULT_TOP_K_VALUES,
    relevance_column: str = "queue",
    latency_ms_per_query: float | None = None,
) -> dict[str, Any]:
    """Evaluate a precomputed query-by-corpus score matrix."""
    split_queries = queries.loc[queries[SPLIT_COLUMN].eq(split), :].reset_index(
        drop=True
    )
    query_count = int(len(split_queries))
    if score_matrix.shape[0] != query_count:
        raise ValueError(
            "Score matrix query rows do not match split query count: "
            f"{score_matrix.shape[0]} != {query_count}."
        )
    if score_matrix.shape[1] != len(corpus):
        raise ValueError(
            "Score matrix corpus columns do not match corpus count: "
            f"{score_matrix.shape[1]} != {len(corpus)}."
        )
    if query_count == 0:
        return _empty_metric_result(split, method_name, top_k_values)

    start = time.perf_counter()
    effective_max_k = min(max(top_k_values), len(corpus))
    corpus_source_ids = np.asarray(corpus[SOURCE_ID_COLUMN].tolist())
    corpus_labels = np.asarray(corpus[relevance_column].tolist())
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

    ranking_latency = ((time.perf_counter() - start) * 1000) / query_count
    total_latency = (
        ranking_latency
        if latency_ms_per_query is None
        else (latency_ms_per_query + ranking_latency)
    )
    return {
        "split": split,
        "method": method_name,
        "query_count": query_count,
        "relevance_type": f"silver_{relevance_column}_match",
        "human_labeled_relevance": False,
        "recall_at_k": {
            f"recall@{k}": _safe_rate(recall_hits[k], query_count) for k in top_k_values
        },
        "mrr": _mean(reciprocal_ranks),
        "mean_retrieved_count": float(effective_max_k),
        "mean_latency_ms_per_query": float(total_latency),
    }


def combine_lexical_semantic_scores(
    lexical_scores: npt.NDArray[np.floating[Any]],
    semantic_scores: npt.NDArray[np.floating[Any]],
    *,
    semantic_weight: float = HYBRID_SEMANTIC_WEIGHT,
) -> npt.NDArray[np.float32]:
    """Combine score matrices after per-query min-max normalization."""
    if lexical_scores.shape != semantic_scores.shape:
        raise ValueError(
            "Lexical and semantic score matrices must have the same shape."
        )
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0 and 1.")
    lexical_normalized = _minmax_normalize_rows(lexical_scores)
    semantic_normalized = _minmax_normalize_rows(semantic_scores)
    combined = (
        1.0 - semantic_weight
    ) * lexical_normalized + semantic_weight * semantic_normalized
    return np.asarray(combined, dtype=np.float32)


def select_retrieval_method(
    metrics_by_method: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Select a method by validation Recall@5, then MRR, then Recall@1."""
    selected = max(
        metrics_by_method,
        key=lambda method: (
            metrics_by_method[method]["validation"]["recall_at_k"]["recall@5"],
            metrics_by_method[method]["validation"]["mrr"],
            metrics_by_method[method]["validation"]["recall_at_k"]["recall@1"],
            method,
        ),
    )
    return {
        "selection_split": "validation",
        "selection_metric": "recall@5_then_mrr_then_recall@1",
        "selected_method": selected,
        "validation_metrics": metrics_by_method[selected]["validation"],
        "test_metrics_for_selected_method": metrics_by_method[selected]["test"],
        "policy": (
            "Semantic retrieval is selected only if validation metrics beat the "
            "lexical baseline under the same train-only corpus and proxy relevance."
        ),
    }


def save_semantic_artifacts(
    retriever: SemanticTicketRetriever,
    *,
    embeddings_path: Path = SEMANTIC_RETRIEVAL_EMBEDDINGS_PATH,
    metadata_path: Path = SEMANTIC_RETRIEVAL_METADATA_PATH,
    config_path: Path = SEMANTIC_RETRIEVAL_CONFIG_PATH,
) -> None:
    """Persist embeddings, corpus metadata, and model configuration."""
    embeddings_path = Path(embeddings_path)
    metadata_path = Path(metadata_path)
    config_path = Path(config_path)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, embeddings=retriever.embeddings)
    retriever.corpus.to_csv(metadata_path, index=False)
    config_path.write_text(
        json.dumps(retriever.model_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_semantic_retriever(
    *,
    embeddings_path: Path = SEMANTIC_RETRIEVAL_EMBEDDINGS_PATH,
    metadata_path: Path = SEMANTIC_RETRIEVAL_METADATA_PATH,
    config_path: Path = SEMANTIC_RETRIEVAL_CONFIG_PATH,
    embedding_model: TextEmbedder | None = None,
) -> SemanticTicketRetriever:
    """Load persisted semantic embeddings and metadata."""
    embeddings = np.load(embeddings_path)["embeddings"].astype(np.float32)
    metadata = pd.read_csv(metadata_path, dtype="string")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return SemanticTicketRetriever(
        corpus=metadata.reset_index(drop=True),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        model_name=str(config["model_name"]),
        model_config=config,
        embedding_model=embedding_model,
    )


def build_model_config(
    embedding_model: TextEmbedder,
    *,
    model_name: str,
    embedding_dimension: int,
    batch_size: int,
    normalize_embeddings: bool,
) -> dict[str, Any]:
    """Record semantic model and package configuration for reproducibility."""
    return {
        "model_name": model_name,
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "embedding_dimension": embedding_dimension,
        "max_seq_length": getattr(embedding_model, "max_seq_length", None),
        "batch_size": batch_size,
        "normalize_embeddings": normalize_embeddings,
        "corpus_text_indexed": ["subject", "body"],
        "query_text_sources": ["subject", "body"],
    }


def write_comparison_label_template(
    lexical_retriever: TfidfTicketRetriever,
    semantic_retriever: SemanticTicketRetriever,
    queries: pd.DataFrame,
    *,
    selected_method: str,
    output_path: Path,
    embedding_model: TextEmbedder,
    hybrid_semantic_weight: float = HYBRID_SEMANTIC_WEIGHT,
    split: str = "validation",
    max_queries: int = 50,
    top_k: int = 5,
) -> Path:
    """Write the Milestone 5-compatible manual relevance template."""
    split_queries = (
        queries.loc[queries[SPLIT_COLUMN].eq(split), :]
        .sort_values(ROW_ID_COLUMN)
        .head(max_queries)
    )
    rows: list[dict[str, Any]] = []
    for _, query in split_queries.iterrows():
        query_text = query[RETRIEVAL_TEXT_COLUMN]
        if selected_method == "semantic":
            results = retrieve_semantic_tickets(
                semantic_retriever,
                query_text=query_text,
                top_k=top_k,
                embedding_model=embedding_model,
            )
        elif selected_method == "hybrid_tfidf_semantic_50_50":
            results = retrieve_hybrid_tickets(
                lexical_retriever,
                semantic_retriever,
                query_text=query_text,
                top_k=top_k,
                embedding_model=embedding_model,
                semantic_weight=hybrid_semantic_weight,
            )
        else:
            results = retrieve_similar_tickets(
                lexical_retriever,
                query_text=query_text,
                top_k=top_k,
            )
        for rank, result in enumerate(results.to_dict("records"), start=1):
            rows.append(
                {
                    "query_ticket_id": query[ROW_ID_COLUMN],
                    "query_split": query[SPLIT_COLUMN],
                    "query_queue": query["queue"],
                    "query_priority": query["priority"],
                    "query_text": query_text,
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


def _semantic_score_matrix(
    retriever: SemanticTicketRetriever,
    query_texts: list[str],
) -> tuple[npt.NDArray[np.float32], float]:
    model = retriever.embedding_model
    if model is None:
        raise ValueError("Semantic score evaluation requires an embedding model.")
    start = time.perf_counter()
    query_embeddings = encode_texts(
        model,
        query_texts,
        batch_size=int(retriever.model_config["batch_size"]),
        normalize_embeddings=bool(retriever.model_config["normalize_embeddings"]),
    )
    scores = query_embeddings @ retriever.embeddings.T
    latency = ((time.perf_counter() - start) * 1000) / max(len(query_texts), 1)
    return np.asarray(scores, dtype=np.float32), latency


def _lexical_score_matrix(
    retriever: TfidfTicketRetriever,
    query_texts: list[str],
) -> tuple[npt.NDArray[np.float32], float]:
    start = time.perf_counter()
    query_matrix = retriever.vectorizer.transform(query_texts)
    scores = query_matrix @ retriever.corpus_matrix.T
    latency = ((time.perf_counter() - start) * 1000) / max(1, len(query_texts))
    dense_scores = np.asarray(scores.toarray(), dtype=np.float32)
    return dense_scores, latency


def _split_query_texts(queries: pd.DataFrame, split: str) -> list[str]:
    return (
        queries.loc[queries[SPLIT_COLUMN].eq(split), RETRIEVAL_TEXT_COLUMN]
        .fillna("")
        .astype("string")
        .tolist()
    )


def _rank_results(
    corpus: pd.DataFrame,
    scores: npt.NDArray[np.floating[Any]],
    *,
    top_k: int,
    score_column: str = "similarity",
) -> pd.DataFrame:
    ranked = pd.DataFrame(
        {
            "corpus_position": corpus.index.to_numpy(),
            score_column: scores,
            SOURCE_ID_COLUMN: corpus[SOURCE_ID_COLUMN].to_list(),
        }
    ).sort_values(
        by=[score_column, SOURCE_ID_COLUMN],
        ascending=[False, True],
        kind="mergesort",
    )
    result = corpus.loc[ranked["corpus_position"].head(top_k).tolist(), :].copy()
    result["similarity"] = ranked[score_column].head(top_k).to_list()
    return result.loc[:, _result_columns()].reset_index(drop=True)


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


def _validate_aligned_corpora(left: pd.DataFrame, right: pd.DataFrame) -> None:
    left_ids = left[SOURCE_ID_COLUMN].astype("string").tolist()
    right_ids = right[SOURCE_ID_COLUMN].astype("string").tolist()
    if left_ids != right_ids:
        raise ValueError("Lexical and semantic corpora are not row-aligned.")


def _minmax_normalize_rows(
    scores: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]:
    row_min = np.min(scores, axis=1, keepdims=True)
    row_max = np.max(scores, axis=1, keepdims=True)
    denominator = np.where((row_max - row_min) == 0.0, 1.0, row_max - row_min)
    return np.asarray((scores - row_min) / denominator, dtype=np.float32)


def _empty_metric_result(
    split: str,
    method_name: str,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "split": split,
        "method": method_name,
        "query_count": 0,
        "relevance_type": "silver_queue_match",
        "human_labeled_relevance": False,
        "recall_at_k": {f"recall@{k}": 0.0 for k in top_k_values},
        "mrr": 0.0,
        "mean_retrieved_count": 0.0,
        "mean_latency_ms_per_query": 0.0,
    }


def _first_relevant_rank(relevant: list[bool]) -> int | None:
    for index, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            return index
    return None


def _result_columns() -> list[str]:
    return [
        SOURCE_ID_COLUMN,
        ROW_ID_COLUMN,
        "subject",
        "body",
        "answer",
        "queue",
        "priority",
        "similarity",
    ]


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_result_columns())


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


def _package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
