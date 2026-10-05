#!/usr/bin/env python
"""Prepare leakage-safe TicketPilot classifier splits."""

import argparse
from pathlib import Path

from ticketpilot.config import (
    DATASET_SUMMARY_PATH,
    PREPARED_DATASET_PATH,
    RAW_ENGLISH_DATA_PATH,
    SPLIT_MANIFEST_PATH,
)
from ticketpilot.data import load_ticket_csv
from ticketpilot.preparation import prepare_ticket_dataset, write_prepared_dataset
from ticketpilot.validation import validate_ticket_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Construct classifier text from subject+body, create grouped "
            "train/validation/test splits, and write preparation artifacts."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_ENGLISH_DATA_PATH,
        help="Validated English ticket CSV produced by the acquisition command.",
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=PREPARED_DATASET_PATH,
        help="CSV path for prepared classifier rows.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=SPLIT_MANIFEST_PATH,
        help="CSV path for split membership and group IDs.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DATASET_SUMMARY_PATH,
        help="JSON path for dataset preparation diagnostics.",
    )
    args = parser.parse_args()

    frame = load_ticket_csv(args.input)
    validate_ticket_frame(frame, require_english_only=True)
    prepared = prepare_ticket_dataset(frame)
    write_prepared_dataset(
        prepared,
        prepared_dataset_path=args.prepared_output,
        split_manifest_path=args.split_manifest,
        dataset_summary_path=args.summary,
    )

    print(f"Prepared dataset: {args.prepared_output}")
    print(f"Split manifest: {args.split_manifest}")
    print(f"Dataset summary: {args.summary}")
    print(
        "Split counts: "
        + ", ".join(
            f"{split}={count}"
            for split, count in prepared.summary["split_counts"].items()
        )
    )


if __name__ == "__main__":
    main()
