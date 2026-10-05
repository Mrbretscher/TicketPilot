# Evaluation Plan

## Evaluation Status

Milestone 3 queue-routing baseline evaluation has been run. The current measured
task is support queue prediction from `classifier_text`, where
`classifier_text` is built only from ticket `subject` and `body`.

## Classification Tasks

TicketPilot will evaluate:

- support queue classification
- priority classification
- abstention and human-review routing

Current completed task:

- support queue classification

Planned later tasks:

- priority classification
- retrieval
- RAG drafting

## Validation Strategy

TicketPilot uses deterministic duplicate-aware three-way splits for classifier
experiments:

- Training split: fit preprocessing and models only.
- Validation split: select model families, hyperparameters, confidence
  thresholds, and abstention thresholds.
- Final test split: use once for final reporting after model and threshold
  choices are fixed.

The unit of observation is one ticket row. The unit of splitting is a normalized
`subject` + `body` text fingerprint, so exact duplicate ticket text cannot cross
train, validation, and test splits.

The splitter first attempts group-level stratification by combined
`queue | priority`. If any combined stratum has too few groups for a reliable
three-way split, it falls back to queue-level stratification. If queue-level
stratification is also too sparse, the preparation command records the problem
and uses deterministic grouped splitting rather than silently dropping classes.

The selected strategy and any small-class warnings are written to
`reports/dataset_preparation/dataset_summary.json`, which is ignored by Git.

## Leakage Controls

Evaluation must verify that classifier inputs exclude resolved-answer text, resolution notes, final response text, close notes, post-resolution tags, and fields unavailable before routing.

Train/test contamination checks must be run and reported.

Milestone 2 classifier text is constructed only from:

- `subject`
- `body`

Forbidden classifier inputs include:

- `answer`
- `queue`
- `priority`
- `type`
- `language`
- `version`
- `tag_1` through `tag_8`

The `answer` field remains available only for later retrieval/RAG evidence and
must not be included in classifier preprocessing, feature extraction, model
selection, or threshold selection.

Preparation diagnostics report queue distribution, priority distribution, text
length, missing values, exact duplicate text groups, repeated text patterns, and
high-similarity text signals before any model training.

## Metrics

Primary metrics:

- macro F1
- per-class precision
- per-class recall
- abstention coverage

Secondary metrics:

- accuracy
- confusion matrix
- calibration curves or expected calibration error where appropriate
- human-review precision and recall

Milestone 3 also reports weighted F1, top-k routing accuracy, inference latency,
and selected model artifact size.

## Baseline Requirement

A scikit-learn baseline must be implemented and evaluated before adding TensorFlow text modeling.

The baseline must use only the training and validation splits for model and
threshold selection. The final test split is reserved for final reporting and
must not be used for hyperparameter selection, threshold selection, or
classifier selection.

Milestone 3 fixed baselines:

- Dummy most-frequent classifier
- TF-IDF + logistic regression
- TF-IDF + LinearSVC

Validation model selection uses macro F1. The selected model is
`tfidf_linear_svc` with validation macro F1 0.6680. Because LinearSVC does not
provide calibrated probabilities, the final confidence model uses sigmoid
calibration fit without final-test access.

Final test metrics for the selected calibrated LinearSVC baseline:

- Accuracy: 0.6673
- Macro precision: 0.7765
- Macro recall: 0.6294
- Macro F1: 0.6829
- Weighted F1: 0.6656
- Top-3 routing accuracy: 0.8988

The validation-selected abstention threshold is 0.30. On the final test split it
routes 90.49% automatically, sends 9.51% to review, and reaches 0.7009 accuracy
on automatically routed tickets.

## Reporting

Reports must distinguish verified results from planned or unverified work. Do not claim production readiness without supporting evidence.
