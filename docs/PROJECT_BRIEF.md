# Project Brief

## Problem

IT support teams receive tickets that vary in urgency, routing destination, and
completeness. Junior analysts and triage teams often need to identify the right
queue, inspect similar past cases, and prepare a safe response draft before a
human decides what to do next.

## Intended Solution

TicketPilot is a human-reviewed IT support copilot prototype. Given a support
ticket, it will:

1. Predict the likely support queue.
2. Retrieve similar resolved tickets.
3. Draft a response grounded in retrieved evidence when gates pass.
4. Abstain or route to human review when confidence is low or evidence is weak.

Automated priority prediction is explicitly unsupported for recruiter-ready v1.
Runtime schemas keep optional priority fields for compatibility, but analysis
returns `null` rather than fabricating a priority.

The system must not autonomously perform IT actions and must not automatically
send responses.

## Portfolio Goals

TicketPilot is designed to demonstrate practical AI-engineering skills across
classical NLP, neural text modeling comparison, retrieval, RAG, evaluation,
application development, reproducibility, and human-in-the-loop design.

It should be described as a portfolio prototype, not as a production helpdesk
product or deployed enterprise system.

## Initial Constraints

- Python 3.11.
- src-based package layout.
- Initial runtime dependencies: numpy, pandas, scikit-learn.
- Initial development dependencies: pytest, pytest-cov, ruff, mypy, pandas-stubs, jupyterlab.
- No TensorFlow, sentence-transformers, OpenAI, FastAPI, Streamlit, Docker, or vector database in the initial scaffold.
