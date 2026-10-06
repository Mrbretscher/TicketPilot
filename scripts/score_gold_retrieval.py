#!/usr/bin/env python
"""Score completed human labels for gold retrieval evaluation."""

import argparse
import json
from pathlib import Path

from ticketpilot.config import (
    GOLD_RETRIEVAL_METRICS_PATH,
    GOLD_RETRIEVAL_TEST_LABEL_PATH,
    GOLD_RETRIEVAL_VALIDATION_LABEL_PATH,
)
from ticketpilot.gold_retrieval import score_gold_label_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score completed GOLD VALIDATION and GOLD TEST retrieval labels. "
            "The command refuses to run when any human relevance labels are missing."
        )
    )
    parser.add_argument(
        "--validation-labels",
        type=Path,
        default=GOLD_RETRIEVAL_VALIDATION_LABEL_PATH,
        help="Completed GOLD VALIDATION labeling CSV.",
    )
    parser.add_argument(
        "--test-labels",
        type=Path,
        default=GOLD_RETRIEVAL_TEST_LABEL_PATH,
        help="Completed GOLD TEST labeling CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=GOLD_RETRIEVAL_METRICS_PATH,
        help="JSON metrics report to write.",
    )
    args = parser.parse_args()

    report = score_gold_label_files(
        validation_label_path=args.validation_labels,
        test_label_path=args.test_labels,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Gold retrieval metrics written to {args.output}")


if __name__ == "__main__":
    main()
