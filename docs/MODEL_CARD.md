# Model Card

## Model Status

No model has been trained yet.

## Intended Use

TicketPilot models are intended to assist human support triage by predicting likely support queue, likely priority, and whether a ticket needs human review. Models are not intended to take IT actions or send responses.

## Inputs

Approved pre-resolution ticket fields only. Resolved-answer text and any post-resolution fields are prohibited classifier inputs.

## Outputs

Planned outputs:

- predicted support queue
- queue confidence
- predicted priority
- priority confidence
- abstention or human-review flag
- optional explanation metadata

## Baseline Model

The first required model is a scikit-learn NLP baseline using TF-IDF features and simple classifiers.

## Advanced Model

A TensorFlow text model may be added later only after the baseline, validation strategy, leakage checks, and evaluation reporting are in place.

## Evaluation

Model evaluation must report more than accuracy. Planned metrics include F1, precision, recall, confusion matrices, calibration behavior, abstention coverage, and human-review routing outcomes.

## Limitations

No performance, fairness, calibration, or operational-readiness claims have been established yet.

## Safety

Low-confidence predictions and weak-evidence cases must support abstention and human review. Human approval is required before any response is sent or action is taken.
