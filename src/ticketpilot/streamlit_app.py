"""Recruiter-facing Streamlit dashboard for TicketPilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ticketpilot.config import QUEUE_VALUES, REVIEW_DATABASE_PATH
from ticketpilot.dashboard_support import (
    PROJECT_ROOT,
    abbreviate_text,
    class_distribution_frame,
    classifier_metric_cards,
    confusion_matrix_frame,
    dataset_limitations,
    default_report_bundles,
    demo_tickets,
    format_percent,
    model_comparison_rows,
    per_class_metrics_frame,
    resolve_project_path,
    retrieval_metric_rows,
    review_records_frame,
    selected_retrieval_method,
)
from ticketpilot.drafting import FakeDraftGenerator
from ticketpilot.orchestration import (
    ArtifactLoadError,
    TicketAnalysisResult,
    TicketPilotService,
    TicketValidationError,
    load_ticketpilot_service,
)
from ticketpilot.preparation import build_classifier_text
from ticketpilot.review import (
    ReviewAction,
    ReviewWorkflowError,
    create_review_record,
    submit_review_action,
)

PAGE_NAMES = (
    "Analyze Ticket",
    "Review Queue",
    "Evaluation",
    "System / Model Information",
    "About / Limitations",
)


def main() -> None:
    """Render the Streamlit dashboard."""
    st = _load_streamlit()
    st.set_page_config(
        page_title="TicketPilot",
        page_icon="TP",
        layout="wide",
    )
    _apply_style(st)
    st.sidebar.title("TicketPilot")
    page = st.sidebar.radio("Workspace", PAGE_NAMES, index=0)
    review_database_path = resolve_project_path(
        st.sidebar.text_input(
            "Review database",
            value=str(REVIEW_DATABASE_PATH),
            help="Local SQLite review store. It is ignored by Git.",
        )
    )
    st.sidebar.caption("Decision support only. No email or IT actions are sent.")

    if page == "Analyze Ticket":
        _render_analyze_ticket(st, review_database_path)
    elif page == "Review Queue":
        _render_review_queue(st, review_database_path)
    elif page == "Evaluation":
        _render_evaluation(st)
    elif page == "System / Model Information":
        _render_system_info(st)
    else:
        _render_about(st)


def _load_streamlit() -> Any:
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Streamlit is not installed. Run `pip install -e .` in the project "
            "environment, then use `scripts/run_streamlit_app.ps1`."
        ) from error
    return st


def _render_analyze_ticket(st: Any, review_database_path: Path) -> None:
    st.title("Analyze Ticket")
    st.caption(
        "Route a support ticket, retrieve similar solved tickets, and draft a "
        "human-reviewed response using local artifacts."
    )

    _render_demo_ticket_selector(st)

    with st.form("analyze-ticket-form"):
        subject = st.text_input(
            "Subject",
            value=str(st.session_state.get("demo_subject", "")),
            placeholder="VPN login fails after MFA reset",
        )
        body = st.text_area(
            "Ticket body",
            value=str(st.session_state.get("demo_body", "")),
            height=180,
            placeholder="Describe the requester issue here.",
        )
        submitted = st.form_submit_button("Analyze Ticket", type="primary")

    if submitted:
        service = _load_service_for_ui(st)
        if service is None:
            return
        try:
            result = service.analyze(subject=subject, body=body, top_k=5)
        except TicketValidationError as error:
            st.error(str(error))
            return
        st.session_state["analysis_result"] = result
        st.session_state["analysis_subject"] = subject
        st.session_state["analysis_body"] = body
        st.session_state.pop("review_record_id", None)

    result = st.session_state.get("analysis_result")
    if not isinstance(result, TicketAnalysisResult):
        st.info("Choose a demo ticket or enter a subject and body, then analyze.")
        return

    _render_analysis_result(st, result)
    _render_reviewer_controls(st, result, review_database_path)


def _render_demo_ticket_selector(st: Any) -> None:
    demos = demo_tickets()
    labels = ["Custom ticket"] + [ticket.name for ticket in demos]
    selected = st.selectbox("Demo ticket", labels, index=0)
    if selected == "Custom ticket":
        return
    ticket = demos[labels.index(selected) - 1]
    st.caption(ticket.note)
    if st.button("Use Demo Ticket"):
        st.session_state["demo_subject"] = ticket.subject
        st.session_state["demo_body"] = ticket.body
        st.rerun()


def _render_analysis_result(st: Any, result: TicketAnalysisResult) -> None:
    routing, evidence = st.columns([1, 2])
    with routing:
        st.subheader("Routing")
        st.metric("Predicted queue", result.predicted_queue)
        st.metric("Confidence", format_percent(result.confidence))
        st.metric("Predicted priority", result.predicted_priority or "Not available")
        review_label = (
            "Human review required"
            if result.human_review_required
            else "Review recommendation ready"
        )
        st.warning(review_label)
        if result.reasons:
            st.caption("Reasons: " + ", ".join(result.reasons))

    with evidence:
        st.subheader("Similar Resolved Tickets")
        if not result.retrieved_evidence:
            st.info("No similar resolved tickets were retrieved.")
        else:
            rows = [
                {
                    "evidence_id": item.source_id,
                    "similarity": item.similarity,
                    "issue": abbreviate_text(
                        f"{item.subject}. {item.body}",
                        max_chars=130,
                    ),
                    "resolution": abbreviate_text(item.answer, max_chars=150),
                }
                for item in result.retrieved_evidence
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Draft Response")
    if result.confidence_evidence_status != "ready_for_review":
        st.warning("The draft is not ready because evidence or confidence is weak.")
    if result.abstention_reason:
        st.info(result.abstention_reason)
    st.text_area(
        "Generated response",
        value=result.draft_response,
        height=180,
        disabled=True,
    )
    if result.citations:
        st.caption("Sources: " + " ".join(f"[{source}]" for source in result.citations))
    else:
        st.caption("Sources: none cited")


def _render_reviewer_controls(
    st: Any,
    result: TicketAnalysisResult,
    review_database_path: Path,
) -> None:
    st.subheader("Reviewer Controls")
    st.caption("These controls record local review decisions only.")
    left, middle, right = st.columns(3)
    with left:
        if st.button("Accept"):
            _persist_review_action(st, result, "accept", review_database_path)
    with middle:
        if st.button("Reject"):
            _persist_review_action(st, result, "reject", review_database_path)
    with right:
        if st.button("Insufficient Evidence"):
            _persist_review_action(
                st,
                result,
                "mark_insufficient_evidence",
                review_database_path,
            )

    with st.expander("Edit draft"):
        edited_response = st.text_area(
            "Edited response",
            value=result.draft_response,
            height=160,
        )
        if st.button("Save Edit"):
            _persist_review_action(
                st,
                result,
                "edit",
                review_database_path,
                edited_response=edited_response,
            )

    with st.expander("Reroute"):
        options = list(QUEUE_VALUES)
        index = (
            options.index(result.predicted_queue)
            if result.predicted_queue in options
            else 0
        )
        final_queue = st.selectbox("Final queue", options=options, index=index)
        if st.button("Save Reroute"):
            _persist_review_action(
                st,
                result,
                "reroute",
                review_database_path,
                final_queue=final_queue,
            )


def _persist_review_action(
    st: Any,
    result: TicketAnalysisResult,
    action: ReviewAction,
    review_database_path: Path,
    *,
    edited_response: str | None = None,
    final_queue: str | None = None,
) -> None:
    subject = str(st.session_state.get("analysis_subject", ""))
    body = str(st.session_state.get("analysis_body", ""))
    ticket_text = build_classifier_text(subject, body)
    metadata = {
        "provider": result.model_info.get("drafting", {}).get("provider", "unknown"),
        "queue_model": result.model_info.get("queue_classifier", {}).get(
            "name",
            "unknown",
        ),
        "retrieval_method": result.model_info.get("retrieval", {}).get(
            "method",
            "unknown",
        ),
    }
    try:
        if "review_record_id" not in st.session_state:
            record = create_review_record(
                analysis_id=result.analysis_id,
                ticket_text=ticket_text,
                predicted_queue=result.predicted_queue,
                queue_confidence=result.confidence,
                predicted_priority=result.predicted_priority,
                retrieved_evidence_ids=tuple(
                    item.source_id for item in result.retrieved_evidence
                ),
                draft_response=result.draft_response,
                model_provider_metadata=metadata,
                path=review_database_path,
            )
            st.session_state["review_record_id"] = record.analysis_id
        reviewed = submit_review_action(
            str(st.session_state["review_record_id"]),
            action=action,
            edited_response=edited_response,
            final_queue=final_queue,
            path=review_database_path,
        )
    except ReviewWorkflowError as error:
        st.error(str(error))
        return
    st.success(f"Review saved as `{reviewed.state}`.")


def _render_review_queue(st: Any, review_database_path: Path) -> None:
    st.title("Review Queue")
    st.caption("Local SQLite review records. Full ticket text is not stored.")
    state_filter = st.selectbox(
        "State filter",
        options=["all", "pending", "approved", "edited", "rejected"],
        index=0,
    )
    state = None if state_filter == "all" else state_filter
    try:
        frame = review_records_frame(
            path=review_database_path,
            state=state,
            limit=200,
        )
    except ReviewWorkflowError as error:
        st.error(str(error))
        return
    if frame.empty:
        st.info("No review records found yet.")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _render_evaluation(st: Any) -> None:
    st.title("Evaluation")
    reports = default_report_bundles()
    queue_report = reports["queue"].report if reports["queue"] else None
    tensorflow_report = reports["tensorflow"].report if reports["tensorflow"] else None
    retrieval_report = (
        reports["semantic_retrieval"].report
        if reports["semantic_retrieval"]
        else reports["retrieval"].report
        if reports["retrieval"]
        else None
    )

    if queue_report is None:
        st.info("Queue baseline metrics were not found. Run the training workflow.")
    else:
        st.subheader("Classifier Metrics")
        metric_columns = st.columns(len(classifier_metric_cards(queue_report)))
        for column, card in zip(
            metric_columns,
            classifier_metric_cards(queue_report),
            strict=True,
        ):
            column.metric(card["label"], card["value"])

        st.subheader("Model Comparison")
        comparison = pd.DataFrame(
            model_comparison_rows(queue_report, tensorflow_report)
        )
        if comparison.empty:
            st.info("No model comparison rows were found.")
        else:
            st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.subheader("Confusion Matrix")
        matrix = confusion_matrix_frame(queue_report)
        if matrix.empty:
            st.info("No confusion matrix was found in the report.")
        else:
            st.dataframe(
                matrix.style.background_gradient(axis=None), use_container_width=True
            )

        st.subheader("Per-Class Metrics")
        per_class = per_class_metrics_frame(queue_report)
        if not per_class.empty:
            st.dataframe(per_class, use_container_width=True, hide_index=True)

        st.subheader("Class Distribution")
        distribution = class_distribution_frame(queue_report, split="test")
        if not distribution.empty:
            st.bar_chart(distribution, x="queue", y="count", use_container_width=True)

    st.subheader("Retrieval Metrics")
    selected_method = selected_retrieval_method(retrieval_report)
    st.caption(f"Current measured retrieval method: `{selected_method}`")
    retrieval_rows = pd.DataFrame(retrieval_metric_rows(retrieval_report))
    if retrieval_rows.empty:
        st.info("Retrieval metrics were not found. Run the retrieval workflow.")
    else:
        st.dataframe(retrieval_rows, use_container_width=True, hide_index=True)

    st.subheader("Dataset Limitations")
    for limitation in dataset_limitations():
        st.markdown(f"- {limitation}")


def _render_system_info(st: Any) -> None:
    st.title("System / Model Information")
    service = _load_service_for_ui(st, show_success=True)
    if service is None:
        st.info(
            "The dashboard still loads without artifacts, but analysis requires "
            "the ignored local model and retrieval files."
        )
        return
    info = service.model_info()
    st.json(info)
    st.caption(
        "Secrets are not displayed. OpenAI generation is optional; the dashboard "
        "uses a deterministic local draft generator by default."
    )


def _render_about(st: Any) -> None:
    st.title("About / Limitations")
    st.markdown(
        """
TicketPilot is a portfolio-scale support-agent console for decision support.
It predicts a support queue from ticket subject and body, retrieves similar
resolved tickets, and prepares a cited draft response for human review.

It does not send messages, close tickets, reset passwords, change permissions,
or perform any IT operation.
"""
    )
    st.subheader("Limitations")
    for limitation in dataset_limitations():
        st.markdown(f"- {limitation}")
    st.subheader("Relevant Documentation")
    links = [
        ("Architecture", PROJECT_ROOT / "docs" / "ARCHITECTURE.md"),
        ("Model card", PROJECT_ROOT / "docs" / "MODEL_CARD.md"),
        ("RAG evaluation", PROJECT_ROOT / "docs" / "RAG_EVALUATION.md"),
        ("Human review policy", PROJECT_ROOT / "docs" / "HUMAN_REVIEW_POLICY.md"),
    ]
    for label, path in links:
        if path.exists():
            st.markdown(f"- [{label}]({path.as_uri()})")


def _load_service_for_ui(
    st: Any,
    *,
    show_success: bool = False,
) -> TicketPilotService | None:
    if "ticketpilot_service" in st.session_state:
        service = st.session_state["ticketpilot_service"]
        return service if isinstance(service, TicketPilotService) else None
    try:
        service = load_ticketpilot_service(draft_generator=FakeDraftGenerator())
    except ArtifactLoadError as error:
        st.error(str(error))
        return None
    st.session_state["ticketpilot_service"] = service
    if show_success:
        st.success("Local model and retrieval artifacts are loaded.")
    return service


def _apply_style(st: Any) -> None:
    st.markdown(
        """
<style>
.stButton > button { border-radius: 6px; }
</style>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
