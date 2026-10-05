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
| COMPLETE | Near-duplicate leakage audit | TF-IDF cosine cross-split audit over normalized subject+body text, thresholded candidate counts, repeated-subject counts, representative inspection, machine-readable report, and synthetic audit tests | Preserve the existing split; monitor high-similarity candidates before future release claims |
| COMPLETE | scikit-learn queue-routing baseline | Dummy, TF-IDF logistic regression, TF-IDF LinearSVC, validation-selected calibrated LinearSVC, test metrics, reports, plots, and model artifact | Improve queue error analysis; automated priority prediction is out of scope for recruiter-ready v1 |
| COMPLETE | Confidence and abstention experiment | Validation-selected threshold 0.30, final-test coverage/review metrics, calibrated confidence scores | Define operational review policy only after broader validation |
| COMPLETE | Classifier evaluation methodology remediation | sklearn calibration Brier/ECE/reliability plots, confidence-threshold curves, per-queue rare-class report, validation-sourced sklearn-vs-TensorFlow deployment decision, and regression tests | Treat final test as reporting-only and document historical test-inspection limitations |
| COMPLETE | TensorFlow text model | TextVectorization + embedding + Conv1D model, class weighting, early stopping, training history, test metrics, calibration diagnostics, and sklearn comparison | Keep sklearn LinearSVC as deployment candidate; investigate queue error analysis |
| COMPLETE | Retrieval | Train-only resolved-ticket TF-IDF index, top-k cosine retrieval, stable source IDs, silver queue-match metrics, manual relevance template, and leakage tests | Label a small gold retrieval set before adding RAG drafting |
| COMPLETE | Semantic retrieval | sentence-transformers embeddings, train-only semantic index, artifact persistence, TF-IDF/semantic/hybrid comparison, and offline fake-embedder tests | Human-label retrieval candidates before tuning hybrid weights or adding generation |
| COMPLETE | RAG drafting | Provider-neutral draft generator protocol, fake test generator, optional OpenAI Responses provider, version-controlled prompts, evidence gating, structured draft schema, and prompt-injection tests | Add an app/API review surface only after preserving human approval |
| COMPLETE | Human-review workflow | Local SQLite review records, allowed reviewer actions, state transitions, audit timestamps, edited response/reroute persistence, and safe startup tests | Add a portfolio review UI without send/execute capabilities |
| COMPLETE | FastAPI service | Core orchestration service, explicit artifact loading, health/model-info/classify/retrieve/analyze/review endpoints, Pydantic schemas, readiness failures, and fake-provider API tests. The active retriever is reported truthfully as TF-IDF; the validation-selected hybrid candidate is not deployed until its semantic query encoder is locally reproducible. | Add deployment packaging without send/execute capabilities |
| COMPLETE | Streamlit dashboard | Recruiter-facing Analyze Ticket console, Review Queue, artifact-backed Evaluation page, System/Model Information, About/Limitations, sample demo tickets, and smoke tests | Add deployment packaging without send/execute capabilities |
| NOT STARTED | Docker and CI | None | Add after core commands stabilize |

## Current Focus

The local Streamlit dashboard is implemented on top of the core orchestration
service. TicketPilot remains decision-support software: it records reviewer
decisions but cannot send responses, route autonomously, or execute support
actions. Queue routing is the implemented classifier capability for
recruiter-ready v1; automated priority prediction is unsupported.

The 2026-10-05 near-duplicate leakage audit found no material cross-split
leakage requiring a split rebuild. The existing train/validation/final-test
split and reported metrics remain preserved, with a small residual optimism risk
noted for a handful of high-similarity synthetic template variants.

The classifier evaluation remediation now reports calibration and per-queue
confidence evidence for the sklearn confidence model and bases
sklearn-vs-TensorFlow deployment-family selection on validation metrics. Final
test metrics remain in the reports for post-selection reporting, but historical
test inspection is documented as a methodology limitation.

## Immediate Next Task

Add deployment packaging that preserves the no-send, no-execute safety boundary.

## Known Constraints

- Use Python 3.11.
- Do not add Docker or a vector database yet.
- Tests must not require paid API calls.
- Do not use private employer, university, customer, or support-ticket data.
- Do not commit `data/raw/` downloads.
- The selected Hugging Face dataset is licensed `cc-by-nc-4.0`; keep portfolio use non-commercial unless separately approved.
