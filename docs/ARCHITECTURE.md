# Architecture

## Current Architecture

TicketPilot currently contains local, testable components for public-data
ingestion, leakage-safe preparation, queue classification, retrieval,
evidence-grounded draft generation, and human-review persistence.

There is no production API, no hosted UI, no email sender, and no autonomous
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
   - Future FastAPI service for inference.
   - Future Streamlit review interface for portfolio demonstration.
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
