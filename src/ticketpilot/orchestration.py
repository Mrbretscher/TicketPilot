"""Core TicketPilot workflow orchestration independent of FastAPI."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from ticketpilot.config import (
    API_DEFAULT_RETRIEVAL_TOP_K,
    API_MAX_TICKET_TEXT_CHARS,
    DRAFT_MIN_CLASSIFIER_CONFIDENCE,
    DRAFT_MIN_EVIDENCE_SCORE,
    QUEUE_BASELINE_REPORT_PATH,
    QUEUE_ROUTER_MODEL_PATH,
    RETRIEVAL_INDEX_PATH,
)
from ticketpilot.drafting import (
    DraftGenerationRequest,
    DraftGenerator,
    DraftResponse,
    EvidenceItem,
    FakeDraftGenerator,
    generate_ticket_draft,
)
from ticketpilot.preparation import build_classifier_text
from ticketpilot.retrieval import TfidfTicketRetriever, retrieve_similar_tickets


class TicketPilotServiceError(RuntimeError):
    """Base class for orchestration failures that should be sanitized by callers."""


class TicketValidationError(ValueError):
    """Raised when an incoming ticket request is malformed."""


class ArtifactLoadError(TicketPilotServiceError):
    """Raised when required local artifacts cannot be loaded."""


@dataclass(frozen=True)
class TicketText:
    """Validated ticket text assembled only from subject and body."""

    subject: str
    body: str
    classifier_text: str


@dataclass(frozen=True)
class QueuePrediction:
    """Queue classifier result."""

    predicted_queue: str
    confidence: float
    top_queues: tuple[dict[str, float | str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable queue prediction."""
        return {
            "predicted_queue": self.predicted_queue,
            "confidence": self.confidence,
            "top_queues": list(self.top_queues),
        }


@dataclass(frozen=True)
class PriorityPrediction:
    """Optional priority classifier result."""

    predicted_priority: str
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable priority prediction."""
        return {
            "predicted_priority": self.predicted_priority,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RetrievedEvidence:
    """Retrieved solved-ticket evidence exposed by orchestration and the API."""

    source_id: str
    subject: str
    body: str
    answer: str
    queue: str
    priority: str
    similarity: float

    def to_draft_evidence(self) -> EvidenceItem:
        """Convert retrieval output to the drafting interface schema."""
        return EvidenceItem(
            source_id=self.source_id,
            subject=self.subject,
            body=self.body,
            answer=self.answer,
            queue=self.queue,
            priority=self.priority,
            similarity=self.similarity,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence item."""
        return {
            "source_id": self.source_id,
            "subject": self.subject,
            "body": self.body,
            "answer": self.answer,
            "queue": self.queue,
            "priority": self.priority,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class TicketAnalysisResult:
    """Complete TicketPilot analysis result for human review."""

    analysis_id: str
    predicted_queue: str
    confidence: float
    predicted_priority: str | None
    retrieved_evidence: tuple[RetrievedEvidence, ...]
    draft_response: str
    citations: tuple[str, ...]
    confidence_evidence_status: str
    abstention_reason: str | None
    human_review_required: bool
    reasons: tuple[str, ...]
    model_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable analysis result."""
        return {
            "analysis_id": self.analysis_id,
            "predicted_queue": self.predicted_queue,
            "confidence": self.confidence,
            "predicted_priority": self.predicted_priority,
            "retrieved_evidence": [item.to_dict() for item in self.retrieved_evidence],
            "retrieval_scores": [
                {"source_id": item.source_id, "similarity": item.similarity}
                for item in self.retrieved_evidence
            ],
            "draft_response": self.draft_response,
            "citations": list(self.citations),
            "confidence_evidence_status": self.confidence_evidence_status,
            "abstention_reason": self.abstention_reason,
            "human_review_required": self.human_review_required,
            "reasons": list(self.reasons),
            "model_info": self.model_info,
        }


@dataclass(frozen=True)
class TicketPilotArtifacts:
    """Loaded artifacts required by the local TicketPilot service."""

    queue_classifier: Any
    retriever: TfidfTicketRetriever
    queue_model_report: dict[str, Any]
    queue_model_path: Path
    retrieval_index_path: Path


class TicketPilotService:
    """Run TicketPilot classification, retrieval, drafting, and review routing."""

    def __init__(
        self,
        *,
        queue_classifier: Any,
        retriever: TfidfTicketRetriever,
        draft_generator: DraftGenerator | None = None,
        queue_model_report: dict[str, Any] | None = None,
        queue_model_path: Path = QUEUE_ROUTER_MODEL_PATH,
        retrieval_index_path: Path = RETRIEVAL_INDEX_PATH,
        max_ticket_text_chars: int = API_MAX_TICKET_TEXT_CHARS,
        default_top_k: int = API_DEFAULT_RETRIEVAL_TOP_K,
        min_classifier_confidence: float = DRAFT_MIN_CLASSIFIER_CONFIDENCE,
        min_evidence_score: float = DRAFT_MIN_EVIDENCE_SCORE,
    ) -> None:
        self.queue_classifier = queue_classifier
        self.retriever = retriever
        self.draft_generator = draft_generator or FakeDraftGenerator()
        self.queue_model_report = queue_model_report or {}
        self.queue_model_path = Path(queue_model_path)
        self.retrieval_index_path = Path(retrieval_index_path)
        self.max_ticket_text_chars = max_ticket_text_chars
        self.default_top_k = default_top_k
        self.min_classifier_confidence = min_classifier_confidence
        self.min_evidence_score = min_evidence_score

    def validate_and_normalize_ticket(
        self,
        *,
        subject: object = "",
        body: object = "",
    ) -> TicketText:
        """Validate request text and assemble leakage-safe classifier input."""
        subject_text = normalize_ticket_component(subject)
        body_text = normalize_ticket_component(body)
        classifier_text = build_classifier_text(subject_text, body_text)
        if not classifier_text:
            raise TicketValidationError("Ticket subject and body cannot both be empty.")
        if len(classifier_text) > self.max_ticket_text_chars:
            raise TicketValidationError(
                "Ticket text exceeds maximum length "
                f"({len(classifier_text)} > {self.max_ticket_text_chars})."
            )
        return TicketText(
            subject=subject_text,
            body=body_text,
            classifier_text=classifier_text,
        )

    def classify(self, *, subject: object = "", body: object) -> QueuePrediction:
        """Predict the destination queue from subject and body only."""
        ticket = self.validate_and_normalize_ticket(subject=subject, body=body)
        return predict_queue(
            self.queue_classifier,
            ticket.classifier_text,
            top_k=3,
        )

    def retrieve(
        self,
        *,
        subject: object = "",
        body: object = "",
        top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> tuple[RetrievedEvidence, ...]:
        """Retrieve similar solved training tickets for a new ticket."""
        ticket = self.validate_and_normalize_ticket(subject=subject, body=body)
        requested_top_k = top_k or self.default_top_k
        if requested_top_k <= 0:
            raise TicketValidationError("top_k must be positive.")
        results = retrieve_similar_tickets(
            self.retriever,
            query_text=ticket.classifier_text,
            top_k=requested_top_k,
            metadata_filter=metadata_filter,
        )
        rows = cast(list[dict[str, Any]], results.to_dict("records"))
        return tuple(retrieval_row_to_evidence(row) for row in rows)

    def analyze(
        self,
        *,
        subject: object = "",
        body: object,
        top_k: int | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> TicketAnalysisResult:
        """Run the full decision-support workflow and return one structured result."""
        ticket = self.validate_and_normalize_ticket(subject=subject, body=body)
        queue_prediction = predict_queue(
            self.queue_classifier,
            ticket.classifier_text,
            top_k=3,
        )
        evidence = self.retrieve(
            subject=ticket.subject,
            body=ticket.body,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        draft_request = DraftGenerationRequest(
            incoming_ticket_text=ticket.classifier_text,
            predicted_queue=queue_prediction.predicted_queue,
            classifier_confidence=queue_prediction.confidence,
            retrieved_evidence=tuple(item.to_draft_evidence() for item in evidence),
        )
        draft = generate_ticket_draft(
            draft_request,
            generator=self.draft_generator,
            min_classifier_confidence=self.min_classifier_confidence,
            min_evidence_score=self.min_evidence_score,
        )
        reasons = analysis_reasons(
            queue_prediction=queue_prediction,
            evidence=evidence,
            draft=draft,
            min_classifier_confidence=self.min_classifier_confidence,
            min_evidence_score=self.min_evidence_score,
        )
        return TicketAnalysisResult(
            analysis_id=str(uuid.uuid4()),
            predicted_queue=queue_prediction.predicted_queue,
            confidence=queue_prediction.confidence,
            predicted_priority=None,
            retrieved_evidence=evidence,
            draft_response=draft.draft_response,
            citations=draft.cited_evidence_ids,
            confidence_evidence_status=draft.confidence_evidence_status,
            abstention_reason=draft.abstention_reason,
            human_review_required=True,
            reasons=reasons,
            model_info=self.model_info(),
        )

    def model_info(self) -> dict[str, Any]:
        """Return loaded artifact metadata without exposing secrets or ticket text."""
        selected_model = self.queue_model_report.get("selected_model", {})
        abstention = self.queue_model_report.get("abstention", {})
        threshold_selection = abstention.get("threshold_selection", {})
        return {
            "service": "ticketpilot",
            "queue_classifier": {
                "supported": True,
                "name": selected_model.get(
                    "name", type(self.queue_classifier).__name__
                ),
                "artifact_path": str(self.queue_model_path),
                "confidence_model": selected_model.get("confidence_model", "unknown"),
                "abstention_threshold": threshold_selection.get(
                    "selected_threshold",
                    self.min_classifier_confidence,
                ),
                "classes": classifier_classes(self.queue_classifier),
            },
            "priority_classifier": {
                "supported": False,
                "reason": (
                    "Automated priority prediction is unsupported and out of scope "
                    "for recruiter-ready v1. Source priority metadata may appear "
                    "only on retrieved evidence where available."
                ),
            },
            "retrieval": {
                "supported": True,
                "method": retriever_method(self.retriever),
                "artifact_path": str(self.retrieval_index_path),
                "corpus_size": int(len(self.retriever.corpus)),
                "default_top_k": self.default_top_k,
                "min_evidence_score": self.min_evidence_score,
            },
            "drafting": {
                "supported": True,
                "provider": type(self.draft_generator).__name__,
                "min_classifier_confidence": self.min_classifier_confidence,
                "human_review_required": True,
            },
        }

    def health(self) -> dict[str, Any]:
        """Return service readiness for API health checks."""
        return {
            "status": "ok",
            "ready": True,
            "errors": [],
            "model_info": self.model_info(),
        }


def load_ticketpilot_service(
    *,
    queue_model_path: Path = QUEUE_ROUTER_MODEL_PATH,
    retrieval_index_path: Path = RETRIEVAL_INDEX_PATH,
    queue_report_path: Path = QUEUE_BASELINE_REPORT_PATH,
    draft_generator: DraftGenerator | None = None,
) -> TicketPilotService:
    """Explicitly load persisted artifacts without retraining models."""
    missing = [
        str(path)
        for path in (queue_model_path, retrieval_index_path)
        if not Path(path).exists()
    ]
    if missing:
        raise ArtifactLoadError("Missing required artifacts: " + ", ".join(missing))
    queue_classifier = joblib.load(queue_model_path)
    retriever = joblib.load(retrieval_index_path)
    if not isinstance(retriever, TfidfTicketRetriever):
        raise ArtifactLoadError(
            "Retrieval artifact is not a TfidfTicketRetriever instance."
        )
    report = load_json_if_present(queue_report_path)
    return TicketPilotService(
        queue_classifier=queue_classifier,
        retriever=retriever,
        draft_generator=draft_generator,
        queue_model_report=report,
        queue_model_path=queue_model_path,
        retrieval_index_path=retrieval_index_path,
    )


def predict_queue(
    classifier: Any,
    classifier_text: str,
    *,
    top_k: int = 3,
) -> QueuePrediction:
    """Predict queue labels and confidence from a fitted sklearn-style estimator."""
    input_series = pd.Series([classifier_text])
    predicted_queue = str(classifier.predict(input_series)[0])
    classes = classifier_classes(classifier)
    if hasattr(classifier, "predict_proba") and classes:
        scores = np.asarray(classifier.predict_proba(input_series)[0], dtype=float)
    elif hasattr(classifier, "decision_function") and classes:
        decision_scores = np.asarray(
            classifier.decision_function(input_series)[0],
            dtype=float,
        )
        scores = softmax(decision_scores)
    else:
        classes = [predicted_queue]
        scores = np.asarray([1.0], dtype=float)

    if len(classes) != len(scores):
        classes = [predicted_queue]
        scores = np.asarray([1.0], dtype=float)
    order = sorted(
        range(len(classes)),
        key=lambda index: (-float(scores[index]), classes[index]),
    )
    top_items: list[dict[str, float | str]] = []
    for index in order[:top_k]:
        top_items.append(
            {
                "queue": classes[index],
                "score": clamp_probability(float(scores[index])),
            }
        )
    confidence = clamp_probability(
        next(
            (
                float(scores[index])
                for index, label in enumerate(classes)
                if label == predicted_queue
            ),
            float(scores[order[0]]) if len(order) else 0.0,
        )
    )
    return QueuePrediction(
        predicted_queue=predicted_queue,
        confidence=confidence,
        top_queues=tuple(top_items),
    )


def classifier_classes(classifier: Any) -> list[str]:
    """Return sklearn classifier classes as strings when available."""
    classes = getattr(classifier, "classes_", None)
    if classes is None:
        return []
    return [str(item) for item in list(classes)]


def retriever_method(retriever: TfidfTicketRetriever) -> str:
    """Return the retrieval method exposed by the loaded retriever artifact."""
    method_name = getattr(retriever, "method_name", None)
    if isinstance(method_name, str) and method_name:
        return method_name
    return type(retriever).__name__


def softmax(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a stable softmax for non-probabilistic classifiers."""
    if scores.ndim == 0:
        return np.asarray([1.0], dtype=float)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    denominator = np.sum(exp_scores)
    if denominator == 0 or math.isnan(float(denominator)):
        return np.full_like(exp_scores, 1.0 / len(exp_scores), dtype=float)
    return np.asarray(exp_scores / denominator, dtype=float)


def retrieval_row_to_evidence(row: dict[str, Any]) -> RetrievedEvidence:
    """Convert a retrieval dataframe row into service evidence."""
    return RetrievedEvidence(
        source_id=str(row["source_id"]),
        subject=str(row["subject"]),
        body=str(row["body"]),
        answer=str(row["answer"]),
        queue=str(row["queue"]),
        priority=str(row["priority"]),
        similarity=clamp_probability(float(row["similarity"])),
    )


def analysis_reasons(
    *,
    queue_prediction: QueuePrediction,
    evidence: tuple[RetrievedEvidence, ...],
    draft: DraftResponse,
    min_classifier_confidence: float,
    min_evidence_score: float,
) -> tuple[str, ...]:
    """Explain conservative routing decisions for the reviewer."""
    reasons = ["human_review_required"]
    if queue_prediction.confidence < min_classifier_confidence:
        reasons.append("low_classifier_confidence")
    if not evidence:
        reasons.append("no_retrieved_evidence")
    elif max(item.similarity for item in evidence) < min_evidence_score:
        reasons.append("weak_retrieved_evidence")
    if draft.abstention_reason:
        reasons.append(draft.confidence_evidence_status)
    if draft.confidence_evidence_status == "ready_for_review":
        reasons.append("grounded_draft_ready_for_review")
    return tuple(dict.fromkeys(reasons))


def normalize_ticket_component(value: object) -> str:
    """Normalize user-supplied ticket text while preserving useful punctuation."""
    if value is None or value is pd.NA:
        return ""
    return " ".join(str(value).split()).strip()


def clamp_probability(value: float) -> float:
    """Clamp a numeric score into the public confidence range."""
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def load_json_if_present(path: Path) -> dict[str, Any]:
    """Load optional JSON metadata from disk."""
    if not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ArtifactLoadError(f"JSON report is not an object: {path}")
    return payload
