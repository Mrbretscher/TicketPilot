"""Evidence-grounded draft response generation for human review."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ticketpilot.config import (
    DRAFT_MAX_EVIDENCE_FIELD_CHARS,
    DRAFT_MAX_TICKET_CHARS,
    DRAFT_MIN_CLASSIFIER_CONFIDENCE,
    DRAFT_MIN_EVIDENCE_SCORE,
    OPENAI_DRAFT_MODEL_DEFAULT,
    OPENAI_DRAFT_MODEL_ENV_VAR,
)
from ticketpilot.rag_prompts import RAG_DRAFT_SYSTEM_PROMPT, build_rag_user_prompt

DraftStatus = Literal[
    "ready_for_review",
    "insufficient_evidence",
    "low_classifier_confidence",
    "provider_error",
]


class DraftGenerationError(RuntimeError):
    """Raised when a draft provider cannot produce valid structured output."""


class DraftGenerator(Protocol):
    """Provider-neutral draft generator interface."""

    def generate(self, request: DraftGenerationRequest) -> DraftResponse:
        """Generate a structured draft response for human review."""


@dataclass(frozen=True)
class EvidenceItem:
    """Retrieved resolved-ticket evidence exposed to the draft generator."""

    source_id: str
    subject: str
    body: str
    answer: str
    queue: str
    priority: str
    similarity: float


@dataclass(frozen=True)
class DraftGenerationRequest:
    """Inputs required for evidence-grounded draft generation."""

    incoming_ticket_text: str
    predicted_queue: str
    classifier_confidence: float
    retrieved_evidence: tuple[EvidenceItem, ...]


@dataclass(frozen=True)
class DraftResponse:
    """Structured draft result for a human reviewer."""

    draft_response: str
    cited_evidence_ids: tuple[str, ...]
    confidence_evidence_status: DraftStatus
    abstention_reason: str | None
    human_review_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable schema for downstream callers."""
        return {
            "draft_response": self.draft_response,
            "cited_evidence_ids": list(self.cited_evidence_ids),
            "confidence_evidence_status": self.confidence_evidence_status,
            "abstention_reason": self.abstention_reason,
            "human_review_required": self.human_review_required,
        }


def generate_ticket_draft(
    request: DraftGenerationRequest,
    *,
    generator: DraftGenerator,
    min_evidence_score: float = DRAFT_MIN_EVIDENCE_SCORE,
    min_classifier_confidence: float = DRAFT_MIN_CLASSIFIER_CONFIDENCE,
) -> DraftResponse:
    """Apply evidence gates and generate a structured draft for human review."""
    gate_response = evidence_gate_response(
        request,
        min_evidence_score=min_evidence_score,
        min_classifier_confidence=min_classifier_confidence,
    )
    if gate_response is not None:
        return gate_response

    try:
        response = generator.generate(request)
    except DraftGenerationError as error:
        return provider_error_response(str(error))
    except (OSError, ValueError, RuntimeError) as error:
        return provider_error_response(f"Draft provider failed: {error}")

    validate_draft_response(response, evidence_ids=evidence_ids(request))
    return response


def evidence_gate_response(
    request: DraftGenerationRequest,
    *,
    min_evidence_score: float = DRAFT_MIN_EVIDENCE_SCORE,
    min_classifier_confidence: float = DRAFT_MIN_CLASSIFIER_CONFIDENCE,
) -> DraftResponse | None:
    """Return an abstention response when classifier or retrieval evidence is weak."""
    if request.classifier_confidence < min_classifier_confidence:
        return abstention_response(
            status="low_classifier_confidence",
            reason=(
                "Classifier confidence is below the configured drafting threshold "
                f"({request.classifier_confidence:.3f} < "
                f"{min_classifier_confidence:.3f})."
            ),
        )
    if not request.retrieved_evidence:
        return abstention_response(
            status="insufficient_evidence",
            reason="No retrieved evidence was provided for grounding.",
        )
    best_score = max(item.similarity for item in request.retrieved_evidence)
    if best_score < min_evidence_score:
        return abstention_response(
            status="insufficient_evidence",
            reason=(
                "Retrieved evidence similarity is below the configured threshold "
                f"({best_score:.3f} < {min_evidence_score:.3f})."
            ),
        )
    return None


def abstention_response(*, status: DraftStatus, reason: str) -> DraftResponse:
    """Build a conservative no-draft response that still requires human review."""
    return DraftResponse(
        draft_response=(
            "TicketPilot did not generate a grounded draft because the evidence "
            "or classifier confidence was insufficient. A human reviewer should "
            "inspect the ticket and retrieved evidence before responding."
        ),
        cited_evidence_ids=(),
        confidence_evidence_status=status,
        abstention_reason=reason,
        human_review_required=True,
    )


def provider_error_response(reason: str) -> DraftResponse:
    """Build a structured response when a provider fails."""
    return DraftResponse(
        draft_response=(
            "TicketPilot could not generate a draft because the configured draft "
            "provider failed. A human reviewer should respond manually."
        ),
        cited_evidence_ids=(),
        confidence_evidence_status="provider_error",
        abstention_reason=reason,
        human_review_required=True,
    )


class FakeDraftGenerator:
    """Deterministic local generator for unit tests and offline demos."""

    def generate(self, request: DraftGenerationRequest) -> DraftResponse:
        """Generate a conservative cited draft without external API calls."""
        ids = evidence_ids(request)
        if not ids:
            return abstention_response(
                status="insufficient_evidence",
                reason="No retrieved evidence was provided for grounding.",
            )
        cited = ids[:2]
        evidence_sentence = " ".join(f"[{source_id}]" for source_id in cited)
        return DraftResponse(
            draft_response=(
                "Draft for human review: Thanks for contacting support. Based on "
                f"similar resolved tickets, this appears related to "
                f"{request.predicted_queue}. Please review the original ticket and "
                f"the cited evidence before sending. {evidence_sentence}"
            ),
            cited_evidence_ids=cited,
            confidence_evidence_status="ready_for_review",
            abstention_reason=None,
            human_review_required=True,
        )


class OpenAIResponsesDraftGenerator:
    """Optional OpenAI Responses API implementation of DraftGenerator."""

    endpoint = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model or os.getenv(
            OPENAI_DRAFT_MODEL_ENV_VAR,
            OPENAI_DRAFT_MODEL_DEFAULT,
        )
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def generate(self, request: DraftGenerationRequest) -> DraftResponse:
        """Call the Responses API and parse the required JSON draft schema."""
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise DraftGenerationError(
                "OPENAI_API_KEY is required only when OpenAI drafting is invoked."
            )

        prompt = build_prompt_payload(request)
        body = {
            "model": self.model,
            "store": False,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": RAG_DRAFT_SYSTEM_PROMPT}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
        }
        payload = json.dumps(body).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", errors="replace")
            raise DraftGenerationError(
                f"OpenAI Responses API returned HTTP {error.code}: {message[:500]}"
            ) from error
        except urllib.error.URLError as error:
            raise DraftGenerationError(
                f"OpenAI Responses API request failed: {error.reason}"
            ) from error

        output_text = extract_openai_output_text(response_body)
        parsed = parse_draft_response_json(output_text)
        validate_draft_response(parsed, evidence_ids=evidence_ids(request))
        return parsed


def build_prompt_payload(request: DraftGenerationRequest) -> str:
    """Build the provider prompt without logging full ticket text."""
    return build_rag_user_prompt(
        incoming_ticket_text=truncate_text(
            request.incoming_ticket_text,
            DRAFT_MAX_TICKET_CHARS,
        ),
        predicted_queue=request.predicted_queue,
        classifier_confidence=request.classifier_confidence,
        evidence_block=format_evidence_block(request.retrieved_evidence),
    )


def format_evidence_block(evidence: tuple[EvidenceItem, ...]) -> str:
    """Format retrieved evidence as data with stable source IDs."""
    blocks: list[str] = []
    for item in evidence:
        blocks.append(
            "\n".join(
                [
                    f"Evidence ID: {item.source_id}",
                    f"Similarity: {item.similarity:.4f}",
                    f"Queue: {item.queue}",
                    f"Priority: {item.priority}",
                    "Subject:",
                    truncate_text(item.subject, DRAFT_MAX_EVIDENCE_FIELD_CHARS),
                    "Body:",
                    truncate_text(item.body, DRAFT_MAX_EVIDENCE_FIELD_CHARS),
                    "Resolved answer:",
                    truncate_text(item.answer, DRAFT_MAX_EVIDENCE_FIELD_CHARS),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def parse_draft_response_json(text: str) -> DraftResponse:
    """Parse provider JSON into DraftResponse and fail closed on invalid schema."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise DraftGenerationError(
            "Draft provider returned non-JSON output."
        ) from error
    if not isinstance(payload, dict):
        raise DraftGenerationError("Draft provider returned a non-object JSON value.")
    try:
        return DraftResponse(
            draft_response=str(payload["draft_response"]),
            cited_evidence_ids=tuple(
                str(item) for item in payload["cited_evidence_ids"]
            ),
            confidence_evidence_status=str(payload["confidence_evidence_status"]),  # type: ignore[arg-type]
            abstention_reason=(
                None
                if payload.get("abstention_reason") is None
                else str(payload["abstention_reason"])
            ),
            human_review_required=bool(payload["human_review_required"]),
        )
    except KeyError as error:
        raise DraftGenerationError(
            f"Draft provider output missing required field: {error.args[0]}"
        ) from error


def validate_draft_response(
    response: DraftResponse,
    *,
    evidence_ids: tuple[str, ...],
) -> None:
    """Validate the structured draft schema and citation policy."""
    allowed_statuses: set[DraftStatus] = {
        "ready_for_review",
        "insufficient_evidence",
        "low_classifier_confidence",
        "provider_error",
    }
    if response.confidence_evidence_status not in allowed_statuses:
        raise DraftGenerationError(
            f"Invalid confidence_evidence_status: {response.confidence_evidence_status}"
        )
    if not response.human_review_required:
        raise DraftGenerationError("Draft responses must require human review.")
    if not isinstance(response.draft_response, str) or not response.draft_response:
        raise DraftGenerationError("draft_response must be a non-empty string.")
    cited_ids = set(response.cited_evidence_ids)
    unknown = sorted(cited_ids.difference(evidence_ids))
    if unknown:
        raise DraftGenerationError(
            "Draft cited unknown evidence IDs: " + ", ".join(unknown)
        )
    if response.confidence_evidence_status == "ready_for_review":
        if not response.cited_evidence_ids:
            raise DraftGenerationError("Ready drafts must cite retrieved evidence.")
        missing_mentions = [
            source_id
            for source_id in response.cited_evidence_ids
            if f"[{source_id}]" not in response.draft_response
        ]
        if missing_mentions:
            raise DraftGenerationError(
                "Ready draft does not include citation markers for: "
                + ", ".join(missing_mentions)
            )
        if response.abstention_reason is not None:
            raise DraftGenerationError(
                "Ready drafts must not include abstention_reason."
            )
    elif response.abstention_reason is None:
        raise DraftGenerationError("Abstention/provider-error responses need a reason.")


def evidence_ids(request: DraftGenerationRequest) -> tuple[str, ...]:
    """Return stable evidence IDs in retrieval order."""
    return tuple(item.source_id for item in request.retrieved_evidence)


def truncate_text(text: str, max_chars: int) -> str:
    """Limit prompt text without exposing extra ticket content to providers."""
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 15].rstrip() + " [truncated]"


def extract_openai_output_text(response_body: dict[str, Any]) -> str:
    """Extract text from a Responses API response object."""
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    pieces: list[str] = []
    for item in response_body.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                pieces.append(content["text"])
    if not pieces:
        raise DraftGenerationError("OpenAI response did not include output text.")
    return "\n".join(pieces)
