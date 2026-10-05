# Architecture

## Current Architecture

TicketPilot currently contains local, testable components for public-data
ingestion, leakage-safe preparation, queue classification, retrieval,
evidence-grounded draft generation, human-review persistence, core workflow
orchestration, a FastAPI service, and a local Streamlit dashboard.

There is no hosted production deployment, no email sender, and no autonomous
support-action executor.

## Planned Components

1. Data ingestion and validation
   - Load public or synthetic support-ticket data.
   - Validate required fields, types, labels, and provenance.
   - Reject fields that would not be available before routing.

2. Preprocessing
   - Normalize ticket title and body text.
   - Keep training and inference preprocessing consistent.
   - Fit preprocessing only on training data.

3. Baseline classification
   - Use scikit-learn TF-IDF features and simple classifiers.
   - Predict support queue and priority.
   - Report calibrated confidence where appropriate.

4. Advanced text modeling
   - Add TensorFlow text modeling only after the baseline is verified.
   - Compare against the baseline using the same validation protocol.

5. Retrieval
   - Retrieve similar resolved tickets from approved public or synthetic data.
   - Keep retrieved evidence separate from generated draft text.

6. RAG draft generation
   - Produce response drafts grounded in retrieved evidence.
   - Preserve citations to source evidence.
   - Keep production prompts in version-controlled source code.

7. Abstention and human review
   - Flag low-confidence predictions.
   - Flag weak-evidence retrieval results.
   - Require human review before any response is sent or action is taken.
   - Persist local review decisions in SQLite for portfolio v1.

8. Application surfaces
   - FastAPI service for local inference and review-record persistence.
   - Streamlit support-agent console for portfolio demonstration.
   - Production deployments must add authentication and RBAC.

9. Evaluation
   - Classification metrics.
   - Leakage checks.
   - Retrieval metrics.
   - RAG citation and groundedness checks.
   - Human-review policy tests.

## Data Boundaries

Classifier features must never include resolved-answer text, resolution notes, final response text, or any field unavailable before initial routing. Retrieved evidence may include resolved-ticket material, but it must not be mixed into classifier inputs.

## Safety Boundaries

TicketPilot is decision support software. It must not directly change account state, reset credentials, modify permissions, close tickets, or send replies.

## Core Orchestration And API

Milestone 9 adds `ticketpilot.orchestration.TicketPilotService` as the core
application boundary. It is independent of FastAPI and runs the complete local
workflow:

1. validate request text and maximum length
2. normalize ticket text with minimal whitespace cleanup
3. construct classifier input from `subject + body`
4. predict support queue with the persisted scikit-learn queue router
5. expose priority only when a future priority artifact is available
6. apply classifier-confidence logic
7. retrieve similar solved tickets from the train-only retrieval index
8. apply evidence-quality logic
9. generate a grounded response draft only when gates pass
10. return one structured result that always requires human review

The FastAPI adapter in `ticketpilot.api` is intentionally thin. It exposes:

- `GET /health`
- `GET /model-info`
- `POST /classify`
- `POST /retrieve`
- `POST /analyze`
- `POST /reviews`
- `GET /reviews/{id}`

API startup loads persisted artifacts explicitly and never retrains models.
If the ignored model or retrieval artifacts are missing, `/health` returns a
not-ready response and inference endpoints return `503` with sanitized error
details. Request schemas validate malformed payloads and maximum text length.
The API does not log API keys, expose stack traces intentionally, send email,
or execute support actions.

## Streamlit Dashboard

Milestone 10 adds `ticketpilot.streamlit_app`, a local recruiter-facing support
agent console. It uses the core orchestration service directly rather than
calling the FastAPI app, so the UI remains testable and does not require an API
server process.

Dashboard pages:

- Analyze Ticket
- Review Queue
- Evaluation
- System / Model Information
- About / Limitations

The Analyze Ticket page accepts `subject` and `ticket body`, then shows routing,
similar resolved tickets, draft response, citations, warnings, abstention
reasons, and reviewer controls. Reviewer controls write to the local SQLite
review workflow only; they do not send responses or execute IT actions.

The Evaluation page reads measured repository artifacts from ignored `reports/`
paths. It does not hard-code fake metrics. If artifacts are missing, it shows
empty states that explain which workflow should be run. The dashboard uses a
deterministic local draft generator by default, so classification, retrieval,
and demo drafting work without `OPENAI_API_KEY`.

## Local Human-Review Store

Milestone 8 uses SQLite through the Python standard library. The default local
database path is `artifacts/review/reviews.sqlite`, which is ignored by Git.

Each review record stores:

- unique analysis ID
- created/updated/reviewed timestamps
- ticket text hash
- predicted queue and queue confidence
- optional predicted priority
- retrieved evidence IDs
- draft response
- model/provider metadata
- reviewer action
- edited response
- reviewer-selected final queue
- optional review note

Allowed reviewer actions are `accept`, `edit`, `reject`, `reroute`, and
`mark_insufficient_evidence`. Persisted states are `pending`, `approved`,
`edited`, and `rejected`.

The review layer records human decisions only. It never sends email, changes a
ticket, performs account operations, or invokes support tools.
