from __future__ import annotations

import importlib


def test_streamlit_app_imports_without_rendering_ui() -> None:
    module = importlib.import_module("ticketpilot.streamlit_app")

    assert module.PAGE_NAMES == (
        "Analyze Ticket",
        "Review Queue",
        "Evaluation",
        "System / Model Information",
        "About / Limitations",
    )


def test_streamlit_app_exposes_main_entrypoint() -> None:
    module = importlib.import_module("ticketpilot.streamlit_app")

    assert callable(module.main)
