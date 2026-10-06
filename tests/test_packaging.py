from __future__ import annotations

import json
import tomllib
from pathlib import Path

import joblib

from test_orchestration import FakeQueueClassifier, orchestration_service
from ticketpilot.orchestration import load_ticketpilot_service

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_dependencies_exclude_training_only_packages() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    runtime_dependencies = set(pyproject["project"]["dependencies"])
    training_dependencies = set(
        pyproject["project"]["optional-dependencies"]["training"]
    )

    assert "uvicorn>=0.30,<1" in runtime_dependencies
    assert not any(item.startswith("tensorflow") for item in runtime_dependencies)
    assert not any(
        item.startswith("sentence-transformers") for item in runtime_dependencies
    )
    assert any(item.startswith("tensorflow") for item in training_dependencies)
    assert any(
        item.startswith("sentence-transformers") for item in training_dependencies
    )


def test_dockerignore_excludes_generated_and_secret_material() -> None:
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for pattern in (
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "data/raw",
        "data/processed",
        "reports",
        "artifacts",
        ".env.*",
        "*.sqlite",
        "*.joblib",
        "*.keras",
        "*.npz",
    ):
        assert pattern in dockerignore


def test_gitignore_excludes_generated_artifacts_and_secret_material() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    for pattern in (
        ".env",
        ".env.*",
        "data/raw/",
        "data/processed/",
        "artifacts/",
        "reports/",
        "models/",
        "vector_indexes/",
        "*.sqlite",
        "*.joblib",
        "*.keras",
        "*.h5",
        "*.npz",
        "screenshots/",
    ):
        assert pattern in gitignore


def test_dockerfile_uses_runtime_install_without_training_commands() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    lower = dockerfile.lower()

    assert "python:3.11-slim" in dockerfile
    assert "pip install --no-build-isolation ." in dockerfile
    assert "user ticketpilot" in lower
    assert "train_queue_baseline.py" not in dockerfile
    assert "build_retrieval_baseline.py" not in dockerfile
    assert "fetch_data.ps1" not in dockerfile
    assert "huggingface" not in lower


def test_artifact_loading_uses_local_classifier_retriever_and_report(
    tmp_path: Path,
) -> None:
    service = orchestration_service()
    queue_model_path = tmp_path / "selected_queue_router.joblib"
    retrieval_index_path = tmp_path / "tfidf_ticket_retriever.joblib"
    queue_report_path = tmp_path / "queue_baseline_metrics.json"
    joblib.dump(FakeQueueClassifier(), queue_model_path)
    joblib.dump(service.retriever, retrieval_index_path)
    queue_report_path.write_text(
        json.dumps(
            {
                "selected_model": {
                    "name": "tfidf_linear_svc",
                    "confidence_model": "sigmoid_calibrated_linear_svc",
                },
                "abstention": {"threshold_selection": {"selected_threshold": 0.30}},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_ticketpilot_service(
        queue_model_path=queue_model_path,
        retrieval_index_path=retrieval_index_path,
        queue_report_path=queue_report_path,
    )

    info = loaded.model_info()
    assert info["queue_classifier"]["name"] == "tfidf_linear_svc"
    assert info["queue_classifier"]["abstention_threshold"] == 0.30
    assert info["priority_classifier"]["supported"] is False
    assert info["retrieval"]["method"] == "tfidf_cosine_similarity"
