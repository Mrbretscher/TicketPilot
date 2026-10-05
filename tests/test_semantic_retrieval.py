from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ticketpilot.preparation import (
    CLASSIFIER_TEXT_COLUMN,
    GROUP_ID_COLUMN,
    SPLIT_COLUMN,
)
from ticketpilot.retrieval import SOURCE_ID_COLUMN, build_resolved_training_corpus
from ticketpilot.semantic_retrieval import (
    TextEmbedder,
    build_semantic_retriever,
    load_semantic_retriever,
    retrieve_semantic_tickets,
    save_semantic_artifacts,
)


class KeywordEmbedder:
    max_seq_length = 128

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> Any:
        del batch_size, show_progress_bar, convert_to_numpy
        vectors = []
        for sentence in sentences:
            text = sentence.casefold()
            vector = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if any(token in text for token in ("vpn", "password", "authenticate")):
                vector[0] = 1.0
            if any(token in text for token in ("invoice", "billing", "refund")):
                vector[1] = 1.0
            if any(token in text for token in ("app", "crash", "product")):
                vector[2] = 1.0
            vector[3] = float(len(text.split()) % 3) / 3.0
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


def semantic_fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = [
        {
            "subject": "VPN login fails",
            "body": "User cannot authenticate to VPN from laptop.",
            "answer": "Reset MFA session and verify VPN profile.",
            "queue": "Technical Support",
            "priority": "high",
        },
        {
            "subject": "Invoice refund request",
            "body": "Customer needs a billing refund for duplicate invoice.",
            "answer": "Review invoice ledger and issue refund if duplicate.",
            "queue": "Billing and Payments",
            "priority": "medium",
        },
        {
            "subject": "Mobile app crash",
            "body": "Application crashes after product update on launch.",
            "answer": "Collect app logs and roll back the faulty update.",
            "queue": "Product Support",
            "priority": "medium",
        },
        {
            "subject": "Held out billing question",
            "body": "Question about a duplicate invoice charge.",
            "answer": "Do not index this validation answer.",
            "queue": "Billing and Payments",
            "priority": "medium",
        },
        {
            "subject": "Held out app question",
            "body": "Application crashes on the splash screen.",
            "answer": "Do not index this test answer.",
            "queue": "Product Support",
            "priority": "high",
        },
    ]
    raw = pd.DataFrame(raw_rows)
    prepared_rows = []
    for index, row in raw.iterrows():
        split = "train"
        if index == 3:
            split = "validation"
        if index == 4:
            split = "test"
        prepared_rows.append(
            {
                "ticket_row_id": f"ticket-{index:06d}",
                GROUP_ID_COLUMN: f"group-{index}",
                SPLIT_COLUMN: split,
                CLASSIFIER_TEXT_COLUMN: f"{row['subject']}\n\n{row['body']}",
                "queue": row["queue"],
                "priority": row["priority"],
            }
        )
    return raw, pd.DataFrame(prepared_rows)


def fitted_semantic_retriever() -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    raw, prepared = semantic_fixture_frames()
    corpus = build_resolved_training_corpus(raw, prepared)
    retriever = build_semantic_retriever(
        corpus,
        embedding_model=KeywordEmbedder(),
        model_name="fake-keyword-embedder",
        batch_size=2,
    )
    return raw, prepared, retriever


def test_semantic_embedding_shape() -> None:
    _, _, retriever = fitted_semantic_retriever()

    assert retriever.embeddings.shape == (3, 4)
    assert retriever.model_config["embedding_dimension"] == 4


def test_semantic_metadata_mapping_is_deterministic() -> None:
    _, _, first = fitted_semantic_retriever()
    _, _, second = fitted_semantic_retriever()

    assert (
        first.corpus[SOURCE_ID_COLUMN].tolist()
        == second.corpus[SOURCE_ID_COLUMN].tolist()
    )
    np.testing.assert_allclose(first.embeddings, second.embeddings)


def test_semantic_retrieval_returns_top_k_and_source_ids() -> None:
    _, _, retriever = fitted_semantic_retriever()

    results = retrieve_semantic_tickets(
        retriever,
        query_text="billing invoice refund",
        top_k=2,
    )

    assert len(results) == 2
    assert results.iloc[0]["queue"] == "Billing and Payments"
    assert results[SOURCE_ID_COLUMN].str.startswith("TP-ticket-").all()


def test_semantic_empty_input_returns_no_results() -> None:
    _, _, retriever = fitted_semantic_retriever()

    results = retrieve_semantic_tickets(retriever, query_text="   ", top_k=5)

    assert results.empty


def test_semantic_artifact_loading_round_trips(tmp_path: Path) -> None:
    _, _, retriever = fitted_semantic_retriever()
    embeddings_path = tmp_path / "embeddings.npz"
    metadata_path = tmp_path / "metadata.csv"
    config_path = tmp_path / "config.json"

    save_semantic_artifacts(
        retriever,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        config_path=config_path,
    )
    loaded = load_semantic_retriever(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        config_path=config_path,
        embedding_model=KeywordEmbedder(),
    )

    assert loaded.model_name == "fake-keyword-embedder"
    pd.testing.assert_frame_equal(
        loaded.corpus.reset_index(drop=True),
        retriever.corpus.reset_index(drop=True).astype("string"),
    )
    np.testing.assert_allclose(loaded.embeddings, retriever.embeddings)


def test_semantic_query_corpus_isolation_excludes_held_out_sources() -> None:
    raw, prepared, retriever = fitted_semantic_retriever()
    held_out_ids = {
        f"TP-ticket-{index:06d}"
        for index, split in enumerate(prepared[SPLIT_COLUMN].tolist())
        if split != "train"
    }

    results = retrieve_semantic_tickets(
        retriever,
        query_text="application crash product",
        top_k=3,
    )

    assert set(retriever.corpus[SPLIT_COLUMN]) == {"train"}
    assert held_out_ids.isdisjoint(set(results[SOURCE_ID_COLUMN]))
    assert "Do not index" not in "\n".join(results["answer"].tolist())
    assert len(raw) == len(prepared)


def test_keyword_embedder_satisfies_protocol() -> None:
    embedder: TextEmbedder = KeywordEmbedder()

    vectors = embedder.encode(
        ["vpn login", "invoice refund"],
        batch_size=2,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    assert np.asarray(vectors).shape == (2, 4)
