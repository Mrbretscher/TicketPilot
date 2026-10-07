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


def test_streamlit_wording_does_not_claim_autonomous_routing() -> None:
    module = importlib.import_module("ticketpilot.streamlit_app")
    source = module.Path(module.__file__).read_text(encoding="utf-8")

    assert "Auto-route" not in source
    assert "Review recommendation ready" in source


def test_streamlit_runtime_does_not_require_matplotlib_styling() -> None:
    module = importlib.import_module("ticketpilot.streamlit_app")
    source = module.Path(module.__file__).read_text(encoding="utf-8")

    assert "background_gradient" not in source
    assert "use_container_width" not in source
