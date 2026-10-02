# Project Brief

## Problem

IT support teams receive tickets that vary in urgency, routing destination, and completeness. Junior analysts and triage teams often need to identify the right queue, priority, similar past cases, and a safe response draft before a human decides what to do next.

## Intended Solution

TicketPilot is a human-reviewed IT support copilot. Given a support ticket, it will:

1. Predict the likely support queue.
2. Predict the likely priority.
3. Retrieve similar resolved tickets.
4. Draft a response grounded in retrieved evidence.
5. Abstain or route to human review when confidence is low or evidence is weak.

The system must not autonomously perform IT actions and must not automatically send responses.

## Portfolio Goals

TicketPilot is designed to demonstrate practical AI-engineering skills across classical NLP, later neural text modeling, retrieval, RAG, evaluation, application development, reproducibility, and human-in-the-loop design.

## Initial Constraints

- Python 3.11.
- src-based package layout.
- Initial runtime dependencies: numpy, pandas, scikit-learn.
- Initial development dependencies: pytest, pytest-cov, ruff, mypy, pandas-stubs, jupyterlab.
- No TensorFlow, sentence-transformers, OpenAI, FastAPI, Streamlit, Docker, or vector database in the initial scaffold.
