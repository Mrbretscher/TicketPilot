# Recruiter Summary

## Project Summary

TicketPilot is a portfolio prototype for human-reviewed IT support triage. It
takes a support ticket, predicts the likely support queue, retrieves similar
resolved tickets, and drafts or abstains based on classifier confidence and
retrieval evidence quality. The system records human-review decisions locally
and does not send responses or perform IT actions.

The project demonstrates practical AI-engineering work across data validation,
leakage-safe splitting, scikit-learn baseline modeling, TensorFlow comparison
modeling, retrieval, RAG-style prompt design, prompt-injection defenses,
FastAPI, Streamlit, SQLite persistence, Docker packaging, and CI checks.

Verified highlights:

- selected queue model: TF-IDF + calibrated LinearSVC
- final-test queue macro F1: `0.6829`
- final-test top-3 routing accuracy: `0.8988`
- deployed retriever: train-only TF-IDF cosine similarity
- final-test TF-IDF retrieval Recall@5: `0.8755` with silver queue-match
  relevance
- priority prediction: unsupported in v1; outputs remain null rather than
  fabricated
- OpenAI drafting: optional, not required for local classification/retrieval

## Resume Bullet Options

1. Built TicketPilot, a human-reviewed IT support triage prototype using
   Python, scikit-learn, FastAPI, Streamlit, and SQLite; implemented leakage-safe
   dataset preparation, TF-IDF queue classification, train-only retrieval,
   evidence-gated drafting, and tests covering API, retrieval, RAG safety, and
   review workflows.

2. Evaluated classical and neural text models for support-queue routing,
   selecting a calibrated TF-IDF LinearSVC baseline with `0.6829` final-test
   macro F1 and `0.8988` top-3 accuracy; documented TensorFlow comparison,
   retrieval metrics, confidence thresholds, abstention behavior, and
   prototype limitations.

## LinkedIn Project Description

TicketPilot is my end-to-end AI-engineering portfolio project for
human-reviewed IT support triage. The system predicts a support queue from
ticket subject/body text, retrieves similar resolved tickets, and creates a
cited response draft only when confidence and evidence gates pass. It keeps
human review at the center: no auto-send, no ticket-closing, and no IT actions.

The project includes public-data validation, duplicate-aware splits,
scikit-learn TF-IDF baselines, a TensorFlow comparison model, TF-IDF and
semantic retrieval experiments, RAG prompt-injection defenses, FastAPI,
Streamlit, SQLite review persistence, Docker packaging, and CI checks.

Current measured result: the selected calibrated TF-IDF LinearSVC queue router
reaches `0.6829` final-test macro F1 and `0.8988` final-test top-3 routing
accuracy. The deployed v1 retriever is TF-IDF; hybrid semantic retrieval is
offline-evaluated but not deployed.

## Claims Intentionally Avoided

- no production deployment claim
- no real enterprise user claim
- no cost-savings or productivity claim
- no autonomous support-action claim
- no priority-prediction claim
- no production security-hardening claim
