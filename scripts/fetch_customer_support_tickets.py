#!/usr/bin/env python
"""Fetch and validate the TicketPilot public support-ticket dataset."""

import argparse
from pathlib import Path

from ticketpilot.data import acquire_customer_support_tickets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the pinned Hugging Face customer-support ticket CSV, "
            "filter English records, validate, and save under data/raw/."
        )
    )
    parser.add_argument(
        "--source-output",
        type=Path,
        default=Path("data/raw/customer_support_tickets_source.csv"),
        help="Local path for the downloaded source CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/customer_support_tickets_en.csv"),
        help="Local path for the validated English CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even if the local source CSV already exists.",
    )
    args = parser.parse_args()

    result = acquire_customer_support_tickets(
        source_path=args.source_output,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print(f"Saved source dataset to {result.source_path}")
    print(f"Saved English dataset to {result.output_path}")
    print(result.english_summary.to_text())


if __name__ == "__main__":
    main()
