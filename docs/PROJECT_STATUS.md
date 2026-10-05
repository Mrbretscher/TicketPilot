# Project Status

Last updated: 2026-10-05

## Status Legend

- COMPLETE: Implemented, documented, and verified.
- IN PROGRESS: Started but not fully verified.
- NOT STARTED: Planned but no implementation yet.

## Milestones

| Status | Milestone | Evidence | Next Step |
| --- | --- | --- | --- |
| COMPLETE | Initial repository scaffold | Documentation, pyproject, package shell, scripts, and import smoke test | Begin data contract |
| COMPLETE | Dataset acquisition and validation | Hugging Face acquisition script, validation utilities, offline fixtures, integration test gate, and updated data documentation | Prepare leakage-safe splits |
| COMPLETE | Leakage-safe dataset preparation | Subject+body classifier text, grouped deterministic splits, preparation summaries, split manifest, and split-safety tests | Build scikit-learn baseline without using the final test set for selection |
| COMPLETE | scikit-learn queue-routing baseline | Dummy, TF-IDF logistic regression, TF-IDF LinearSVC, validation-selected calibrated LinearSVC, test metrics, reports, plots, and model artifact | Add priority baseline or improve queue error analysis |
| COMPLETE | Confidence and abstention experiment | Validation-selected threshold 0.30, final-test coverage/review metrics, calibrated confidence scores | Define operational review policy only after broader validation |
| COMPLETE | TensorFlow text model | TextVectorization + embedding + Conv1D model, class weighting, early stopping, training history, test metrics, calibration diagnostics, and sklearn comparison | Keep sklearn LinearSVC as deployment candidate; investigate priority classification or error analysis |
| COMPLETE | Retrieval | Train-only resolved-ticket TF-IDF index, top-k cosine retrieval, stable source IDs, silver queue-match metrics, manual relevance template, and leakage tests | Label a small gold retrieval set before adding RAG drafting |
| COMPLETE | Semantic retrieval | sentence-transformers embeddings, train-only semantic index, artifact persistence, TF-IDF/semantic/hybrid comparison, and offline fake-embedder tests | Human-label retrieval candidates before tuning hybrid weights or adding generation |
| COMPLETE | RAG drafting | Provider-neutral draft generator protocol, fake test generator, optional OpenAI Responses provider, version-controlled prompts, evidence gating, structured draft schema, and prompt-injection tests | Add an app/API review surface only after preserving human approval |
| COMPLETE | Human-review workflow | Local SQLite review records, allowed reviewer actions, state transitions, audit timestamps, edited response/reroute persistence, and safe startup tests | Add a portfolio review UI without send/execute capabilities |
| COMPLETE | FastAPI service | Core orchestration service, explicit artifact loading, health/model-info/classify/retrieve/analyze/review endpoints, Pydantic schemas, readiness failures, and fake-provider API tests | Add UI or deployment packaging without send/execute capabilities |
| NOT STARTED | Streamlit human-review app | None | Add after review policy and API contract |
| NOT STARTED | Docker and CI | None | Add after core commands stabilize |

## Current Focus

Core orchestration and FastAPI inference endpoints are implemented. TicketPilot
remains decision-support software: it records reviewer decisions but cannot send
responses or execute support actions.

## Immediate Next Task

Add a portfolio review UI or deployment packaging that uses the API and SQLite
workflow without send/execute capabilities.

## Known Constraints

- Use Python 3.11.
- Do not add Streamlit, Docker, or a vector database yet.
- Tests must not require paid API calls.
- Do not use private employer, university, customer, or support-ticket data.
- Do not commit `data/raw/` downloads.
- The selected Hugging Face dataset is licensed `cc-by-nc-4.0`; keep portfolio use non-commercial unless separately approved.
