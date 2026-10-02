from pathlib import Path

import pandas as pd
import pytest

from ticketpilot.config import HF_DATASET_URL
from ticketpilot.data import acquire_customer_support_tickets, load_ticket_csv


def test_acquire_customer_support_tickets_downloads_filters_validates_and_saves(
    tmp_path: Path,
    valid_ticket_frame: pd.DataFrame,
) -> None:
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "english.csv"
    fixture_path = tmp_path / "fixture.csv"
    valid_ticket_frame.to_csv(fixture_path, index=False)
    calls: list[tuple[str, Path]] = []

    def fake_downloader(url: str, path: Path) -> None:
        calls.append((url, path))
        path.write_bytes(fixture_path.read_bytes())

    result = acquire_customer_support_tickets(
        source_path=source_path,
        output_path=output_path,
        expected_sha256=None,
        downloader=fake_downloader,
        minimum_source_rows=1,
        minimum_english_rows=1,
    )

    saved = pd.read_csv(output_path, dtype="string")

    assert calls == [(HF_DATASET_URL, source_path)]
    assert result.source_path == source_path
    assert result.output_path == output_path
    assert result.source_summary.row_count == 4
    assert result.english_summary.row_count == 3
    assert output_path.exists()
    assert set(saved["language"]) == {"en"}
    assert list(saved.columns) == list(valid_ticket_frame.columns)


def test_acquire_customer_support_tickets_reuses_existing_source_without_download(
    tmp_path: Path,
    valid_ticket_frame: pd.DataFrame,
) -> None:
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "english.csv"
    valid_ticket_frame.to_csv(source_path, index=False)

    def fail_downloader(url: str, path: Path) -> None:
        raise AssertionError(f"unit test should not fetch {url} to {path}")

    result = acquire_customer_support_tickets(
        source_path=source_path,
        output_path=output_path,
        expected_sha256=None,
        downloader=fail_downloader,
        minimum_source_rows=1,
        minimum_english_rows=1,
    )

    assert result.english_summary.language_counts == {"en": 3}


def test_load_ticket_csv_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="TicketPilot dataset not found"):
        load_ticket_csv(tmp_path / "missing.csv")
