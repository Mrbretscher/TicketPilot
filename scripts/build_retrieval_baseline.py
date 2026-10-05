#!/usr/bin/env python
"""Build and evaluate the TicketPilot lexical similar-ticket retriever."""

import argparse
from pathlib import Path

from ticketpilot.config import (
    PREPARED_DATASET_PATH,
    RAW_ENGLISH_DATA_PATH,
    RETRIEVAL_ARTIFACT_DIR,
    RETRIEVAL_INDEX_PATH,
    RETRIEVAL_LABEL_TEMPLATE_PATH,
    RETRIEVAL_REPORT_DIR,
)
from ticketpilot.retrieval import run_lexical_retrieval_baseline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only TF-IDF similar-ticket retrieval index and evaluate "
            "held-out queries with silver queue-match relevance."
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
        "--report-dir",
        type=Path,
        default=RETRIEVAL_REPORT_DIR,
        help="Ignored directory for retrieval metrics and label templates.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=RETRIEVAL_ARTIFACT_DIR,
        help="Ignored directory for retrieval index artifacts.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=RETRIEVAL_INDEX_PATH,
        help="Joblib path for the TF-IDF retrieval index.",
    )
    parser.add_argument(
        "--label-template",
        type=Path,
        default=RETRIEVAL_LABEL_TEMPLATE_PATH,
        help="CSV path for future manual relevance labeling.",
    )
    args = parser.parse_args()

    result = run_lexical_retrieval_baseline(
        raw_data_path=args.raw_data,
        prepared_dataset_path=args.prepared_data,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir,
        index_path=args.index_path,
        label_template_path=args.label_template,
    )
    validation = result.report["evaluation"]["validation"]
    test = result.report["evaluation"]["test"]
    print(f"Retrieval metrics: {result.report_path}")
    print(f"Retrieval index: {result.index_path}")
    print(f"Manual label template: {result.label_template_path}")
    print(f"Corpus tickets: {result.report['index']['corpus_count']}")
    print(
        "Validation silver metrics: "
        f"Recall@1={validation['recall_at_k']['recall@1']:.4f}, "
        f"Recall@3={validation['recall_at_k']['recall@3']:.4f}, "
        f"Recall@5={validation['recall_at_k']['recall@5']:.4f}, "
        f"MRR={validation['mrr']:.4f}"
    )
    print(
        "Test silver metrics: "
        f"Recall@1={test['recall_at_k']['recall@1']:.4f}, "
        f"Recall@3={test['recall_at_k']['recall@3']:.4f}, "
        f"Recall@5={test['recall_at_k']['recall@5']:.4f}, "
        f"MRR={test['mrr']:.4f}"
    )


if __name__ == "__main__":
    main()
