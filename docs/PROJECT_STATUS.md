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
| NOT STARTED | Retrieval | None | Choose retrieval approach after approved data exists |
| NOT STARTED | RAG drafting | None | Add version-controlled prompts and cited draft contract |
| NOT STARTED | TensorFlow text model | None | Add only after baseline evaluation |
| NOT STARTED | FastAPI service | None | Add after inference contract is stable |
| NOT STARTED | Streamlit human-review app | None | Add after review policy and API contract |
| NOT STARTED | Docker and CI | None | Add after core commands stabilize |

## Current Focus

Queue-routing baseline complete; next focus is either priority classification or deeper queue error analysis.

## Immediate Next Task

Extend baseline coverage to priority classification or analyze queue-routing errors before adding retrieval.

## Known Constraints

- Use Python 3.11.
- Do not add TensorFlow, sentence-transformers, OpenAI, FastAPI, Streamlit, Docker, or a vector database yet.
- Tests must not require paid API calls.
- Do not use private employer, university, customer, or support-ticket data.
- Do not commit `data/raw/` downloads.
- The selected Hugging Face dataset is licensed `cc-by-nc-4.0`; keep portfolio use non-commercial unless separately approved.
