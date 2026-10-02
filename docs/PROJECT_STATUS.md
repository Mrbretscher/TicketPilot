# Project Status

Last updated: 2026-10-02

## Status Legend

- COMPLETE: Implemented, documented, and verified.
- IN PROGRESS: Started but not fully verified.
- NOT STARTED: Planned but no implementation yet.

## Milestones

| Status | Milestone | Evidence | Next Step |
| --- | --- | --- | --- |
| COMPLETE | Initial repository scaffold | Documentation, pyproject, package shell, scripts, and import smoke test | Begin data contract |
| NOT STARTED | Data contract and synthetic seed data | None | Define ticket schema and validation |
| NOT STARTED | scikit-learn baseline | None | Build TF-IDF baseline after data contract |
| NOT STARTED | Confidence and abstention | None | Define confidence thresholds and review routing |
| NOT STARTED | Retrieval | None | Choose retrieval approach after approved data exists |
| NOT STARTED | RAG drafting | None | Add version-controlled prompts and cited draft contract |
| NOT STARTED | TensorFlow text model | None | Add only after baseline evaluation |
| NOT STARTED | FastAPI service | None | Add after inference contract is stable |
| NOT STARTED | Streamlit human-review app | None | Add after review policy and API contract |
| NOT STARTED | Docker and CI | None | Add after core commands stabilize |

## Current Focus

Initial scaffold.

## Immediate Next Task

Define the ticket data contract, including fields, labels, prohibited leakage fields, and a small synthetic fixture for tests.

## Known Constraints

- Use Python 3.11.
- Do not add TensorFlow, sentence-transformers, OpenAI, FastAPI, Streamlit, Docker, or a vector database yet.
- Tests must not require paid API calls.
- Do not use private employer, university, customer, or support-ticket data.
