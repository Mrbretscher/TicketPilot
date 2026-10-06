# TicketPilot Demo Script

Use this for a two-to-three-minute recruiter or technical-screen demo. For the
current repo-only release posture, run it locally from the Streamlit dashboard.
Replace the live demo and video placeholders after the Docker workflow is
verified and recorded.

## 1. Opening Problem

"TicketPilot is a portfolio prototype for IT support triage. Support tickets are
often incomplete or noisy, so a reviewer needs help identifying the right queue,
finding related solved cases, and drafting a safe response without letting AI
send or execute anything autonomously."

## 2. Application Input

Open the Streamlit dashboard and choose a synthetic demo ticket such as:

- Subject: `VPN login fails after MFA reset`
- Body: `I can sign in to the portal, but the VPN client rejects my login after
  I reset MFA on my work laptop.`

Explain that the classifier input is built only from `subject + body`; resolved
answers and labels are excluded from classifier features.

## 3. Model Output

Click **Analyze Ticket**.

Show:

- predicted queue
- confidence score
- predicted priority shown as unavailable/null
- similar resolved tickets
- draft or abstention status
- human-review reasons

Say: "Queue routing is implemented; automated priority prediction is
unsupported in v1, so the app does not fabricate a priority."

## 4. Supporting Evidence

Point to the retrieved evidence table. Note that the deployed retriever is a
train-only TF-IDF cosine index with stable source IDs such as
`TP-ticket-000272`.

If the draft abstains because evidence score is below threshold, explain that
this is expected conservative behavior. TicketPilot prefers human review over a
weakly grounded draft.

## 5. Operational Control

Show the reviewer controls:

- Accept
- Reject
- Insufficient Evidence
- Edit draft
- Reroute

Say: "These controls write local SQLite review records only. They do not send a
response, close a ticket, reset a password, or perform an IT action."

## 6. Architecture Summary

Summarize the pipeline:

1. pinned public dataset acquisition
2. leakage-safe split preparation
3. scikit-learn TF-IDF queue classifier
4. TensorFlow comparison model
5. TF-IDF retrieval over train-only resolved tickets
6. evidence-gated drafting
7. FastAPI and Streamlit surfaces
8. local human-review persistence

## 7. Measured Results

Use only verified repository results:

- selected model: TF-IDF + calibrated LinearSVC
- final-test queue macro F1: `0.6829`
- final-test top-3 routing accuracy: `0.8988`
- validation-selected confidence threshold: `0.30`
- final-test review rate at that threshold: `9.51%`
- deployed TF-IDF retrieval final-test Recall@5: `0.8755` with silver
  queue-match relevance
- gold retrieval test Recall@5: `1.0000` on 20 labeled queries

## 8. Limitation

"This is not a production helpdesk product. It has no hosted deployment,
authentication, RBAC, production monitoring, or autonomous action execution.
This repository is ready for local, repo-based review; Docker packaging exists,
but Docker runtime verification is still pending in the current environment."

## 9. Closing

"The project is meant to show end-to-end AI engineering: careful data
boundaries, baseline-first modeling, comparison against a neural text model,
retrieval and RAG safety controls, API/UI implementation, tests, and honest
documentation of what is and is not deployed."
