#!/usr/bin/env python
"""Train and evaluate TicketPilot queue-routing baselines."""

import argparse
from pathlib import Path

from ticketpilot.config import (
    PREPARED_DATASET_PATH,
    QUEUE_BASELINE_ARTIFACT_DIR,
    QUEUE_BASELINE_REPORT_DIR,
)
from ticketpilot.training import run_queue_baseline_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train scikit-learn queue-routing baselines."
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=PREPARED_DATASET_PATH,
        help="Prepared dataset CSV with train/validation/test splits.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=QUEUE_BASELINE_REPORT_DIR,
        help="Ignored directory for metrics and plots.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=QUEUE_BASELINE_ARTIFACT_DIR,
        help="Ignored directory for selected model artifacts.",
    )
    args = parser.parse_args()

    result = run_queue_baseline_training(
        prepared_dataset_path=args.prepared_data,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir,
    )
    selected = result.report["selected_model"]
    test_metrics = result.report["test_metrics"]
    abstention = result.report["abstention"]["test_result"]
    print(f"Metrics report: {result.report_path}")
    print(f"Selected model: {selected['name']}")
    print(f"Validation macro F1: {selected['validation_macro_f1']:.4f}")
    print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(
        "Abstention: "
        f"threshold={abstention['threshold']:.2f}, "
        f"coverage={abstention['coverage']:.2%}, "
        f"review_rate={abstention['review_rate']:.2%}"
    )


if __name__ == "__main__":
    main()
