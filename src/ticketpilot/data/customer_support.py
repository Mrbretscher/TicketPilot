"""Acquisition helpers for the public customer-support ticket dataset."""

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from ticketpilot.config import (
    HF_DATASET_SHA256,
    HF_DATASET_URL,
    MIN_ENGLISH_ROWS,
    MIN_SOURCE_ROWS,
    RAW_ENGLISH_DATA_PATH,
    RAW_SOURCE_DATA_PATH,
    RECRUITER_READY_LANGUAGE,
)
from ticketpilot.validation import (
    DataValidationError,
    TicketDataValidationSummary,
    validate_ticket_frame,
)

DownloadCsv = Callable[[str, Path], None]


@dataclass(frozen=True)
class TicketDatasetAcquisitionResult:
    """Result of acquiring and validating the local TicketPilot dataset."""

    source_path: Path
    output_path: Path
    source_summary: TicketDataValidationSummary
    english_summary: TicketDataValidationSummary


def _download_csv(url: str, output_path: Path) -> None:
    with urlopen(url, timeout=60) as response:
        output_path.write_bytes(response.read())


def acquire_customer_support_tickets(
    *,
    source_path: Path = RAW_SOURCE_DATA_PATH,
    output_path: Path = RAW_ENGLISH_DATA_PATH,
    source_url: str = HF_DATASET_URL,
    expected_sha256: str | None = HF_DATASET_SHA256,
    overwrite: bool = False,
    downloader: DownloadCsv = _download_csv,
    minimum_source_rows: int = MIN_SOURCE_ROWS,
    minimum_english_rows: int = MIN_ENGLISH_ROWS,
) -> TicketDatasetAcquisitionResult:
    """Download, filter, validate, and save English support-ticket records."""
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists() or overwrite:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        downloader(source_url, source_path)

    if expected_sha256 is not None:
        _validate_file_sha256(source_path, expected_sha256)

    source_frame = load_ticket_csv(source_path)
    source_summary = validate_ticket_frame(
        source_frame,
        minimum_rows=minimum_source_rows,
    )

    english_frame = source_frame.loc[
        source_frame["language"].eq(RECRUITER_READY_LANGUAGE),
        :,
    ].copy()
    english_summary = validate_ticket_frame(
        english_frame,
        require_english_only=True,
        minimum_rows=minimum_english_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    english_frame.to_csv(output_path, index=False)

    return TicketDatasetAcquisitionResult(
        source_path=source_path,
        output_path=output_path,
        source_summary=source_summary,
        english_summary=english_summary,
    )


def load_ticket_csv(path: Path = RAW_ENGLISH_DATA_PATH) -> pd.DataFrame:
    """Load a support-ticket CSV using stable string dtypes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TicketPilot dataset not found at {path}. "
            "Run `python scripts/fetch_customer_support_tickets.py` first."
        )
    return pd.read_csv(path, dtype="string")


def _validate_file_sha256(path: Path, expected_sha256: str) -> None:
    digest = sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected_sha256.lower():
        raise DataValidationError(
            f"Downloaded dataset SHA256 mismatch for {path}: "
            f"expected {expected_sha256.lower()}, found {digest.lower()}."
        )
