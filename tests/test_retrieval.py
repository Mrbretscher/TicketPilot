import pandas as pd

from ticketpilot.preparation import (
    CLASSIFIER_TEXT_COLUMN,
    GROUP_ID_COLUMN,
    SPLIT_COLUMN,
)
from ticketpilot.retrieval import (
    SOURCE_ID_COLUMN,
    TfidfTicketRetriever,
    build_query_frame,
    build_resolved_training_corpus,
    build_tfidf_retriever,
    evaluate_retrieval_split,
    make_source_id,
    retrieve_similar_tickets,
)


def retrieval_fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = [
        {
            "subject": "VPN login fails",
            "body": "User cannot authenticate to VPN from laptop.",
            "answer": "Reset MFA session and verify VPN profile.",
            "queue": "Technical Support",
            "priority": "high",
        },
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
            "subject": "Password reset needed",
            "body": "Employee cannot reset account password.",
            "answer": "Use self-service reset and confirm identity.",
            "queue": "Technical Support",
            "priority": "low",
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
        if index == 5:
            split = "validation"
        if index == 6:
            split = "test"
        prepared_rows.append(
            {
                "ticket_row_id": f"ticket-{index:06d}",
                GROUP_ID_COLUMN: (
                    "vpn-duplicate" if index in {0, 1} else f"group-{index}"
                ),
                SPLIT_COLUMN: split,
                CLASSIFIER_TEXT_COLUMN: f"{row['subject']}\n\n{row['body']}",
                "queue": row["queue"],
                "priority": row["priority"],
            }
        )
    return raw, pd.DataFrame(prepared_rows)


def fitted_retriever() -> tuple[pd.DataFrame, pd.DataFrame, TfidfTicketRetriever]:
    raw, prepared = retrieval_fixture_frames()
    corpus = build_resolved_training_corpus(raw, prepared)
    return raw, prepared, build_tfidf_retriever(corpus)


def test_retrieval_returns_requested_top_k_size() -> None:
    _, _, retriever = fitted_retriever()

    results = retrieve_similar_tickets(
        retriever,
        subject="duplicate invoice",
        body="Need help with billing refund invoice",
        top_k=3,
    )

    assert len(results) == 3
    assert results["similarity"].between(0.0, 1.0).all()


def test_retrieval_is_deterministic() -> None:
    _, _, retriever = fitted_retriever()

    first = retrieve_similar_tickets(retriever, query_text="vpn login laptop", top_k=4)
    second = retrieve_similar_tickets(retriever, query_text="vpn login laptop", top_k=4)

    pd.testing.assert_frame_equal(first, second)


def test_corpus_contains_only_training_resolved_tickets() -> None:
    raw, prepared = retrieval_fixture_frames()

    corpus = build_resolved_training_corpus(raw, prepared)

    assert set(corpus[SPLIT_COLUMN]) == {"train"}
    assert corpus["answer"].str.strip().ne("").all()
    assert "ticket-000005" not in set(corpus["ticket_row_id"])
    assert "ticket-000006" not in set(corpus["ticket_row_id"])


def test_held_out_tickets_are_excluded_from_retrieval_sources() -> None:
    raw, prepared, retriever = fitted_retriever()
    query_frame = build_query_frame(raw, prepared)
    held_out_ids = set(
        query_frame.loc[query_frame[SPLIT_COLUMN].ne("train"), "ticket_row_id"].map(
            make_source_id
        )
    )

    results = retrieve_similar_tickets(
        retriever,
        query_text="duplicate invoice charge",
        top_k=5,
    )

    assert held_out_ids.isdisjoint(set(results[SOURCE_ID_COLUMN]))


def test_results_include_stable_citation_identifiers() -> None:
    _, _, retriever = fitted_retriever()

    results = retrieve_similar_tickets(retriever, query_text="vpn login", top_k=2)

    assert results[SOURCE_ID_COLUMN].str.startswith("TP-ticket-").all()
    assert results[SOURCE_ID_COLUMN].is_unique


def test_empty_query_returns_no_results() -> None:
    _, _, retriever = fitted_retriever()

    results = retrieve_similar_tickets(retriever, query_text="   ", top_k=5)

    assert results.empty
    assert list(results.columns) == [
        SOURCE_ID_COLUMN,
        "ticket_row_id",
        "subject",
        "body",
        "answer",
        "queue",
        "priority",
        "similarity",
    ]


def test_duplicate_tickets_keep_distinct_citation_ids() -> None:
    _, _, retriever = fitted_retriever()

    results = retrieve_similar_tickets(
        retriever,
        query_text="vpn login laptop authenticate",
        top_k=3,
    )
    duplicate_results = results.loc[
        results["subject"].eq("VPN login fails")
        & results["body"].eq("User cannot authenticate to VPN from laptop."),
        :,
    ]

    assert len(duplicate_results) == 2
    assert duplicate_results[SOURCE_ID_COLUMN].is_unique


def test_retrieval_metrics_use_silver_queue_relevance() -> None:
    raw, prepared, retriever = fitted_retriever()
    queries = build_query_frame(raw, prepared)

    metrics = evaluate_retrieval_split(retriever, queries, split="validation")

    assert metrics["relevance_type"] == "silver_queue_match"
    assert metrics["human_labeled_relevance"] is False
    assert set(metrics["recall_at_k"]) == {"recall@1", "recall@3", "recall@5"}
