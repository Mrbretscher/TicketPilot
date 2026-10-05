#!/usr/bin/env python
"""Train and evaluate the TensorFlow queue text classifier."""

import argparse
from pathlib import Path

from ticketpilot.config import (
    PREPARED_DATASET_PATH,
    QUEUE_BASELINE_REPORT_PATH,
    TENSORFLOW_ARTIFACT_DIR,
    TENSORFLOW_REPORT_DIR,
)
from ticketpilot.tensorflow_training import run_tensorflow_queue_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TensorFlow TextVectorization queue classifier."
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=PREPARED_DATASET_PATH,
        help="Prepared dataset CSV with train/validation/test splits.",
    )
    parser.add_argument(
        "--sklearn-report",
        type=Path,
        default=QUEUE_BASELINE_REPORT_PATH,
        help="Queue sklearn baseline report for comparison.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=TENSORFLOW_REPORT_DIR,
        help="Ignored directory for TensorFlow metrics and history.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=TENSORFLOW_ARTIFACT_DIR,
        help="Ignored directory for TensorFlow model artifacts.",
    )
    args = parser.parse_args()

    result = run_tensorflow_queue_training(
        prepared_dataset_path=args.prepared_data,
        sklearn_report_path=args.sklearn_report,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir,
    )
    test_metrics = result.report["test_metrics"]
    comparison = result.report["sklearn_comparison"]
    decision = result.report["deployment_decision"]
    print(f"Metrics report: {result.report_path}")
    print(f"TensorFlow test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"TensorFlow test weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"TensorFlow top-3 accuracy: {test_metrics['top_k_accuracy']:.4f}")
    print(
        "Delta macro F1 vs sklearn: "
        f"{comparison['macro_f1']['delta_tensorflow_minus_sklearn']:.4f}"
    )
    print(f"Deployment choice: {decision['selected_model']}")


if __name__ == "__main__":
    main()
