#!/usr/bin/env python
"""Run the TicketPilot near-duplicate cross-split leakage audit."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ticketpilot.config import PREPARED_DATASET_PATH, RAW_ENGLISH_DATA_PATH
from ticketpilot.leakage_audit import audit_near_duplicate_cross_split_leakage
from ticketpilot.training import load_prepared_dataset

DEFAULT_REPORT_PATH = Path("reports/near_duplicate_leakage/audit_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit near-duplicate subject/body ticket text crossing "
            "train/validation/test split boundaries."
        )
    )
    parser.add_argument(
        "--prepared-data",
        type=Path,
        default=PREPARED_DATASET_PATH,
        help="Prepared dataset CSV with existing train/validation/test assignments.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=RAW_ENGLISH_DATA_PATH,
        help="Validated public English ticket CSV used to compute repeated subjects.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Machine-readable JSON report path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of highest-similarity candidate pairs to include for inspection.",
    )
    args = parser.parse_args()

    prepared = load_prepared_dataset(args.prepared_data)
    raw_tickets = pd.read_csv(args.raw_data)
    report = audit_near_duplicate_cross_split_leakage(
        prepared,
        raw_tickets=raw_tickets,
        sample_size=args.sample_size,
    )
    report["generated_at_utc"] = datetime.now(UTC).isoformat()
    report["inputs"] = {
        "prepared_dataset_path": str(args.prepared_data),
        "raw_dataset_path": str(args.raw_data),
    }
    report["split_policy"] = (
        "Audit only: existing train/validation/test assignments were not changed."
    )
    report["training_policy"] = (
        "Audit only: no models or retrieval indexes were trained."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = report["candidate_pair_counts"]["by_threshold"]
    split_counts = report["candidate_pair_counts"]["by_split_pair_and_threshold"]
    print(f"Near-duplicate audit report: {args.output}")
    print(f"Total candidate pairs >= 0.90: {report['total_candidate_pair_count']}")
    print("Counts by threshold: " + json.dumps(counts, sort_keys=True))
    print("Counts by split pair: " + json.dumps(split_counts, sort_keys=True))


if __name__ == "__main__":
    main()
