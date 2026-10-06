# Project Status

Last updated: 2026-10-06

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
| COMPLETE | Retrieval | Train-only resolved-ticket TF-IDF index, top-k cosine retrieval, stable source IDs, silver queue-match metrics, human-labeled 20-query validation/test gold retrieval sets, and leakage tests | Use gold labels for validation-only evidence thresholding; keep final test reporting-only |
| COMPLETE | Semantic retrieval | sentence-transformers embeddings, train-only semantic index, artifact persistence, TF-IDF/semantic/hybrid comparison, and offline fake-embedder tests | Human-label retrieval candidates before tuning hybrid weights or adding generation |
| COMPLETE | RAG drafting | Provider-neutral draft generator protocol, fake test generator, optional OpenAI Responses provider, version-controlled prompts, validation-supported evidence threshold `0.39`, structured draft schema, and prompt-injection tests | Keep abstention conservative and revisit the evidence threshold only with additional validation labels |
| COMPLETE | Human-review workflow | Local SQLite review records, allowed reviewer actions, state transitions, audit timestamps, edited response/reroute persistence, and safe startup tests | Add a portfolio review UI without send/execute capabilities |
| COMPLETE | FastAPI service | Core orchestration service, explicit artifact loading, health/model-info/classify/retrieve/analyze/review endpoints, Pydantic schemas, readiness failures, and fake-provider API tests. The active retriever is reported truthfully as TF-IDF; the validation-selected hybrid candidate is not deployed until its semantic query encoder is locally reproducible. | Add deployment packaging without send/execute capabilities |
| COMPLETE | Streamlit dashboard | Recruiter-facing Analyze Ticket console, Review Queue, artifact-backed Evaluation page, System/Model Information, About/Limitations, sample demo tickets, and smoke tests | Add deployment packaging without send/execute capabilities |
| IN PROGRESS | Docker and CI | Dockerfile, Docker Compose, .dockerignore, dependency split, GitHub Actions workflow, packaging tests, local FastAPI health/model-info/analyze checks, and local Streamlit health check. Docker host verification is still pending because Docker is unavailable in the current environment. | Run Docker build and Compose health/analyze checks on a machine with Docker installed before marking complete |
| IN PROGRESS | Recruiter-facing release documentation | README rewrite, architecture diagram, updated data/model/security docs, demo script, recruiter summary, resume bullets, LinkedIn description, and a committed dashboard placeholder image. Live demo, video, and real screenshot links remain unfilled by design for repo-only recruiter review. | Add verified screenshot/demo links after Docker runtime verification |

## Current Focus

The local Streamlit dashboard and FastAPI API are implemented on top of the
core orchestration service. TicketPilot remains decision-support software: it
records reviewer decisions but cannot send responses, route autonomously, or
execute support actions. Queue routing is the implemented classifier capability
for recruiter-ready v1; automated priority prediction is unsupported.

The human-labeled gold retrieval workflow now reports GOLD VALIDATION and GOLD
TEST separately. GOLD VALIDATION selected an evidence score threshold of `0.39`
for RAG drafting gates under a conservative precision rule. GOLD TEST was used
only once afterward for reporting behavior at that fixed threshold.

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

Before publishing a hosted recruiter demo, run Docker build and Docker Compose
verification on a machine with Docker installed, replace the placeholder image
with a real dashboard screenshot, add demo/video links, and mark the packaging
and recruiter-documentation milestones complete if those checks pass.

## Known Constraints

- Use Python 3.11.
- Docker packaging is configured, but Docker runtime verification is pending.
- Current release posture is repo-only recruiter review until Docker runtime
  verification and demo assets are complete.
- Do not add a vector database yet.
- Tests must not require paid API calls.
- Do not use private employer, university, customer, or support-ticket data.
- Do not commit `data/raw/` downloads.
- The selected Hugging Face dataset is licensed `cc-by-nc-4.0`; keep portfolio use non-commercial unless separately approved.
