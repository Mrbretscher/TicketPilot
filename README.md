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

The project now includes dataset acquisition, leakage-safe preparation,
scikit-learn queue routing, TensorFlow comparison modeling, and train-only
similar-ticket retrieval. OpenAI integration, FastAPI, Streamlit, Docker, and
vector databases remain future milestones.

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

Prepare leakage-safe classifier splits:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py
```

The preparation command constructs classifier text from `subject` and `body`,
keeps duplicate text groups within one split, and writes ignored artifacts under
`reports/dataset_preparation/`.

Train and evaluate the first queue-routing baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_queue_baseline.py
```

Current measured result: TF-IDF + LinearSVC was selected on validation macro F1
and evaluated once on the final test split. Test macro F1 is `0.6829`, test
accuracy is `0.6673`, and top-3 routing accuracy is `0.8988`. With the
validation-selected confidence threshold `0.30`, `90.49%` of test tickets route
automatically and `9.51%` go to human review.

Build and compare similar-ticket retrieval methods:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_baseline.py
.\.venv\Scripts\python.exe scripts\build_semantic_retrieval.py
```

Current retrieval result: the 50/50 TF-IDF/semantic hybrid was selected on
validation Recall@5 and MRR. Validation Recall@5 is `0.8792`; final test
Recall@5 is `0.8771` and test MRR is `0.8018`. These are silver queue-match
proxy metrics, not human-labeled relevance judgments.

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
