# TicketPilot

TicketPilot is a human-reviewed IT support copilot for portfolio-scale AI engineering. It receives a support ticket, predicts the appropriate support queue and priority, retrieves similar resolved tickets, drafts a grounded response with citations, and routes uncertain cases to human review.

TicketPilot does not autonomously perform IT actions and does not automatically send generated responses.

## Intended Portfolio Coverage

- scikit-learn NLP
- TensorFlow text modeling
- retrieval
- RAG
- confidence and abstention
- human-in-the-loop AI
- FastAPI
- Streamlit
- testing
- Docker
- CI
- reproducibility
- evaluation

The initial scaffold intentionally includes only the Python package, documentation, tests, and development tooling. TensorFlow, sentence-transformers, OpenAI, FastAPI, Streamlit, Docker, and vector databases are future milestones.

## Quick Start

```powershell
.\scripts\setup.ps1
.\scripts\verify.ps1
```

The project targets Python 3.11 and uses a src-based package layout.

Fetch the public support-ticket dataset locally:

```powershell
.\scripts\fetch_data.ps1
```

The command downloads the pinned Hugging Face source CSV, filters English
records, validates schema and quality checks, and writes ignored files under
`data/raw/`.

## Current State

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for milestone status and the next planned work.

## Documentation

- [Project brief](docs/PROJECT_BRIEF.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Data card](docs/DATA_CARD.md)
- [Model card](docs/MODEL_CARD.md)
- [Evaluation plan](docs/EVALUATION_PLAN.md)
- [RAG evaluation](docs/RAG_EVALUATION.md)
- [Human review policy](docs/HUMAN_REVIEW_POLICY.md)
- [Security](docs/SECURITY.md)

## Repository Policy

Do not commit private support tickets, downloaded datasets, generated model artifacts, vector indexes, caches, experiment databases, logs, or secrets. Keep runtime prompts in version-controlled source code when generation features are introduced.
