#!/usr/bin/env python
"""Build semantic retrieval artifacts and compare retrieval methods."""

import argparse
from pathlib import Path

from ticketpilot.config import (
    HYBRID_SEMANTIC_WEIGHT,
    PREPARED_DATASET_PATH,
    RAW_ENGLISH_DATA_PATH,
    RETRIEVAL_ARTIFACT_DIR,
    RETRIEVAL_REPORT_DIR,
    SEMANTIC_RETRIEVAL_ARTIFACT_DIR,
    SEMANTIC_RETRIEVAL_BATCH_SIZE,
    SEMANTIC_RETRIEVAL_MODEL_NAME,
    SEMANTIC_RETRIEVAL_NORMALIZE_EMBEDDINGS,
    SEMANTIC_RETRIEVAL_REPORT_DIR,
)
from ticketpilot.semantic_retrieval import run_semantic_retrieval_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sentence-transformers semantic retrieval index and compare "
            "TF-IDF, semantic, and simple hybrid retrieval on held-out queries."
        )
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=RAW_ENGLISH_DATA_PATH,
        help="Validated English ticket CSV with original answer evidence fields.",
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=PREPARED_DATASET_PATH,
        help="Prepared dataset CSV with deterministic train/validation/test splits.",
    )
    parser.add_argument(
        "--semantic-report-dir",
        type=Path,
        default=SEMANTIC_RETRIEVAL_REPORT_DIR,
        help="Ignored directory for semantic retrieval metrics.",
    )
    parser.add_argument(
        "--semantic-artifact-dir",
        type=Path,
        default=SEMANTIC_RETRIEVAL_ARTIFACT_DIR,
        help="Ignored directory for semantic embeddings and metadata.",
    )
    parser.add_argument(
        "--lexical-report-dir",
        type=Path,
        default=RETRIEVAL_REPORT_DIR,
        help="Ignored directory for lexical comparison artifacts.",
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        type=Path,
        default=RETRIEVAL_ARTIFACT_DIR,
        help="Ignored directory for lexical comparison artifacts.",
    )
    parser.add_argument(
        "--model-name",
        default=SEMANTIC_RETRIEVAL_MODEL_NAME,
        help="sentence-transformers model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=SEMANTIC_RETRIEVAL_BATCH_SIZE,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable L2 normalization before cosine retrieval.",
    )
    parser.add_argument(
        "--hybrid-semantic-weight",
        type=float,
        default=HYBRID_SEMANTIC_WEIGHT,
        help="Semantic score weight for the fixed hybrid experiment.",
    )
    args = parser.parse_args()

    result = run_semantic_retrieval_comparison(
        raw_data_path=args.raw_data,
        prepared_dataset_path=args.prepared_data,
        semantic_report_dir=args.semantic_report_dir,
        semantic_artifact_dir=args.semantic_artifact_dir,
        lexical_report_dir=args.lexical_report_dir,
        lexical_artifact_dir=args.lexical_artifact_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=(
            SEMANTIC_RETRIEVAL_NORMALIZE_EMBEDDINGS and not args.no_normalize_embeddings
        ),
        hybrid_semantic_weight=args.hybrid_semantic_weight,
    )
    selection = result.report["selection"]
    print(f"Semantic retrieval metrics: {result.report_path}")
    print(f"Embeddings: {result.embeddings_path}")
    print(f"Metadata: {result.metadata_path}")
    print(f"Config: {result.config_path}")
    print(f"Manual label template: {result.label_template_path}")
    print(f"Selected retrieval method: {selection['selected_method']}")
    for method, metrics in result.report["evaluation"].items():
        validation = metrics["validation"]
        test = metrics["test"]
        print(
            f"{method}: "
            f"validation Recall@5={validation['recall_at_k']['recall@5']:.4f}, "
            f"validation MRR={validation['mrr']:.4f}, "
            f"test Recall@5={test['recall_at_k']['recall@5']:.4f}, "
            f"test MRR={test['mrr']:.4f}"
        )


if __name__ == "__main__":
    main()
