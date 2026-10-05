"""FastAPI adapter for the local TicketPilot decision-support service."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi import Path as ApiPath
from pydantic import BaseModel, Field, field_validator

from ticketpilot.config import (
    API_DEFAULT_RETRIEVAL_TOP_K,
    API_MAX_TICKET_TEXT_CHARS,
    REVIEW_DATABASE_PATH,
)
from ticketpilot.orchestration import (
    ArtifactLoadError,
    RetrievedEvidence,
    TicketPilotService,
    TicketPilotServiceError,
    TicketValidationError,
    load_ticketpilot_service,
)
from ticketpilot.review import (
    ReviewRecord,
    ReviewWorkflowError,
    create_review_record,
    get_review_record,
)


class TicketRequest(BaseModel):
    """Request body for subject/body TicketPilot analysis."""

    subject: str = Field(default="", max_length=1_000)
    body: str = Field(..., min_length=1, max_length=API_MAX_TICKET_TEXT_CHARS)

    @field_validator("subject", "body")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        """Trim request text without changing punctuation or terminology."""
        return " ".join(value.split()).strip()


class RetrievalRequest(TicketRequest):
    """Request body for retrieval endpoint."""

    top_k: int = Field(default=API_DEFAULT_RETRIEVAL_TOP_K, ge=1, le=20)
    metadata_filter: dict[str, str] | None = None


class ClassificationResponse(BaseModel):
    """Queue classification response schema."""

    predicted_queue: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_queues: list[dict[str, str | float]]


class EvidenceResponse(BaseModel):
    """Retrieved evidence response schema."""

    source_id: str
    subject: str
    body: str
    answer: str
    queue: str
    priority: str
    similarity: float = Field(ge=0.0, le=1.0)


class RetrievalResponse(BaseModel):
    """Top-k retrieval response schema."""

    evidence: list[EvidenceResponse]


class AnalysisResponse(BaseModel):
    """Full TicketPilot workflow response schema."""

    analysis_id: str
    predicted_queue: str
    confidence: float = Field(ge=0.0, le=1.0)
    predicted_priority: str | None
    retrieved_evidence: list[EvidenceResponse]
    retrieval_scores: list[dict[str, str | float]]
    draft_response: str
    citations: list[str]
    confidence_evidence_status: str
    abstention_reason: str | None
    human_review_required: bool
    reasons: list[str]
    model_info: dict[str, Any]


class HealthResponse(BaseModel):
    """API health/readiness response schema."""

    status: Literal["ok", "not_ready"]
    ready: bool
    errors: list[str]
    model_info: dict[str, Any] | None = None


class ReviewCreateRequest(BaseModel):
    """Create a pending human-review record from an analysis result."""

    analysis_id: str | None = None
    ticket_text: str = Field(..., min_length=1, max_length=API_MAX_TICKET_TEXT_CHARS)
    predicted_queue: str = Field(..., min_length=1)
    queue_confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_priority: str | None = None
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    draft_response: str = Field(..., min_length=1)
    model_provider_metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewResponse(BaseModel):
    """Human-review persistence response schema."""

    analysis_id: str
    state: str
    created_at: str
    updated_at: str
    ticket_text_hash: str
    predicted_queue: str
    queue_confidence: float
    predicted_priority: str | None
    retrieved_evidence_ids: list[str]
    draft_response: str
    model_provider_metadata: dict[str, Any]
    reviewer_action: str | None
    edited_response: str | None
    final_queue: str | None
    review_note: str | None
    reviewed_at: str | None


def create_app(
    *,
    service: TicketPilotService | None = None,
    load_artifacts: bool = True,
    review_database_path: Path = REVIEW_DATABASE_PATH,
) -> FastAPI:
    """Create the TicketPilot API with explicit, non-training artifact loading."""
    app = FastAPI(
        title="TicketPilot API",
        version="0.1.0",
        description="Human-reviewed support-ticket decision-support API.",
    )
    app.state.service = service
    app.state.readiness_error = None
    app.state.review_database_path = Path(review_database_path)

    if service is None and load_artifacts:
        try:
            app.state.service = load_ticketpilot_service()
        except ArtifactLoadError as error:
            app.state.readiness_error = str(error)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.service
        if loaded is None:
            return HealthResponse(
                status="not_ready",
                ready=False,
                errors=[
                    app.state.readiness_error or "TicketPilot artifacts are not loaded."
                ],
                model_info=None,
            )
        return HealthResponse(**loaded.health())

    @app.get("/model-info")
    def model_info(
        service_dependency: Annotated[
            TicketPilotService,
            Depends(get_service_from_request),
        ],
    ) -> dict[str, Any]:
        return service_dependency.model_info()

    @app.post("/classify", response_model=ClassificationResponse)
    def classify(
        request: TicketRequest,
        service_dependency: Annotated[
            TicketPilotService,
            Depends(get_service_from_request),
        ],
    ) -> ClassificationResponse:
        try:
            prediction = service_dependency.classify(
                subject=request.subject,
                body=request.body,
            )
        except TicketValidationError as error:
            raise bad_request(str(error)) from error
        except TicketPilotServiceError as error:
            raise service_unavailable(str(error)) from error
        return ClassificationResponse(**prediction.to_dict())

    @app.post("/retrieve", response_model=RetrievalResponse)
    def retrieve(
        request: RetrievalRequest,
        service_dependency: Annotated[
            TicketPilotService,
            Depends(get_service_from_request),
        ],
    ) -> RetrievalResponse:
        try:
            evidence = service_dependency.retrieve(
                subject=request.subject,
                body=request.body,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
        except TicketValidationError as error:
            raise bad_request(str(error)) from error
        except ValueError as error:
            raise bad_request(str(error)) from error
        return RetrievalResponse(
            evidence=[to_evidence_response(item) for item in evidence]
        )

    @app.post("/analyze", response_model=AnalysisResponse)
    def analyze(
        request: RetrievalRequest,
        service_dependency: Annotated[
            TicketPilotService,
            Depends(get_service_from_request),
        ],
    ) -> AnalysisResponse:
        try:
            result = service_dependency.analyze(
                subject=request.subject,
                body=request.body,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
        except TicketValidationError as error:
            raise bad_request(str(error)) from error
        except ValueError as error:
            raise bad_request(str(error)) from error
        except TicketPilotServiceError as error:
            raise service_unavailable(str(error)) from error
        return AnalysisResponse(**result.to_dict())

    @app.post(
        "/reviews",
        response_model=ReviewResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_review(request: ReviewCreateRequest) -> ReviewResponse:
        try:
            record = create_review_record(
                analysis_id=request.analysis_id,
                ticket_text=request.ticket_text,
                predicted_queue=request.predicted_queue,
                queue_confidence=request.queue_confidence,
                predicted_priority=request.predicted_priority,
                retrieved_evidence_ids=tuple(request.retrieved_evidence_ids),
                draft_response=request.draft_response,
                model_provider_metadata=request.model_provider_metadata,
                path=app.state.review_database_path,
            )
        except ReviewWorkflowError as error:
            raise bad_request(str(error)) from error
        return review_record_response(record)

    @app.get("/reviews/{analysis_id}", response_model=ReviewResponse)
    def get_review(
        analysis_id: str = ApiPath(..., min_length=1),
    ) -> ReviewResponse:
        try:
            record = get_review_record(
                analysis_id,
                path=app.state.review_database_path,
            )
        except ReviewWorkflowError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "message": "Review record was not found.",
                    "errors": [str(error)],
                },
            ) from error
        return review_record_response(record)

    return app


def get_service_from_request(request: Request) -> TicketPilotService:
    """Load the app-scoped TicketPilot service or fail with sanitized readiness."""
    loaded = request.app.state.service
    if loaded is None:
        fallback = "TicketPilot artifacts are not loaded."
        detail = request.app.state.readiness_error or fallback
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "TicketPilot service is not ready.",
                "errors": [detail],
            },
        )
    return cast(TicketPilotService, loaded)


def to_evidence_response(evidence: RetrievedEvidence) -> EvidenceResponse:
    """Convert core evidence to the API schema."""
    return EvidenceResponse(**evidence.to_dict())


def review_record_response(record: ReviewRecord) -> ReviewResponse:
    """Convert a SQLite review record to the API schema."""
    return ReviewResponse(
        analysis_id=record.analysis_id,
        state=record.state,
        created_at=record.created_at,
        updated_at=record.updated_at,
        ticket_text_hash=record.ticket_text_hash,
        predicted_queue=record.predicted_queue,
        queue_confidence=record.queue_confidence,
        predicted_priority=record.predicted_priority,
        retrieved_evidence_ids=list(record.retrieved_evidence_ids),
        draft_response=record.draft_response,
        model_provider_metadata=record.model_provider_metadata,
        reviewer_action=record.reviewer_action,
        edited_response=record.edited_response,
        final_queue=record.final_queue,
        review_note=record.review_note,
        reviewed_at=record.reviewed_at,
    )


def bad_request(message: str) -> HTTPException:
    """Return a sanitized 400 error."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"message": message, "errors": [message]},
    )


def service_unavailable(message: str) -> HTTPException:
    """Return a sanitized 503 error."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"message": "TicketPilot service failed.", "errors": [message]},
    )


app = create_app()
