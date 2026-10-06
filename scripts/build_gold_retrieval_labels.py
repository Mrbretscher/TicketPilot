#!/usr/bin/env python
"""Create blank human-labeling CSVs for gold retrieval evaluation."""

import argparse
from pathlib import Path

from ticketpilot.config import (
    GOLD_RETRIEVAL_TEST_LABEL_PATH,
    GOLD_RETRIEVAL_VALIDATION_LABEL_PATH,
    PREPARED_DATASET_PATH,
    RAW_ENGLISH_DATA_PATH,
    RETRIEVAL_INDEX_PATH,
)
from ticketpilot.gold_retrieval import (
    GOLD_QUERY_COUNT,
    GOLD_TOP_K,
    build_gold_labeling_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create blank gold retrieval labeling CSVs using the deployed "
            "TicketPilot TF-IDF retriever artifact."
        )
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=RAW_ENGLISH_DATA_PATH,
        help="Validated English ticket CSV.",
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=PREPARED_DATASET_PATH,
        help="Prepared dataset CSV with train/validation/test split assignments.",
    )
    parser.add_argument(
        "--retriever-index",
        type=Path,
        default=RETRIEVAL_INDEX_PATH,
        help="Deployed TF-IDF retriever joblib artifact.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=GOLD_RETRIEVAL_VALIDATION_LABEL_PATH,
        help="CSV to write for GOLD VALIDATION labels.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=GOLD_RETRIEVAL_TEST_LABEL_PATH,
        help="CSV to write for GOLD TEST labels.",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=GOLD_QUERY_COUNT,
        help="Number of queries to sample from each evaluation split.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=GOLD_TOP_K,
        help="Number of evidence candidates to retrieve per query.",
    )
    args = parser.parse_args()

    result = build_gold_labeling_files(
        raw_data_path=args.raw_data,
        prepared_dataset_path=args.prepared_data,
        retriever_index_path=args.retriever_index,
        validation_output_path=args.validation_output,
        test_output_path=args.test_output,
        query_count=args.query_count,
        top_k=args.top_k,
    )
    print(
        "GOLD VALIDATION: "
        f"{result.validation_query_count} queries -> {result.validation_path}"
    )
    print(f"GOLD TEST: {result.test_query_count} queries -> {result.test_path}")
    print("Human relevance labels and reviewer notes were intentionally left blank.")


if __name__ == "__main__":
    main()
