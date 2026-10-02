import os
from pathlib import Path

import pytest

from ticketpilot.data import acquire_customer_support_tickets


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("TICKETPILOT_RUN_INTEGRATION") != "1",
    reason="set TICKETPILOT_RUN_INTEGRATION=1 to fetch the real Hugging Face dataset",
)
def test_fetch_customer_support_tickets_from_hugging_face(tmp_path: Path) -> None:
    result = acquire_customer_support_tickets(
        source_path=tmp_path / "source.csv",
        output_path=tmp_path / "english.csv",
        overwrite=True,
    )

    assert result.source_path.exists()
    assert result.output_path.exists()
    assert result.source_summary.language_counts == {"de": 12249, "en": 16338}
    assert result.english_summary.language_counts == {"en": 16338}
