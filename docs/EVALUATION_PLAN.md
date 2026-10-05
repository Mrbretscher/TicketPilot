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

The current split was preserved after a focused near-duplicate cross-split audit
run on 2026-10-05. That audit compared normalized `subject` + `body` text across
train/validation, train/test, and validation/test boundaries with a scikit-learn
TF-IDF cosine nearest-neighbor method. It did not alter train, validation, or
test assignments.

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

The near-duplicate audit writes its machine-readable output to
`reports/near_duplicate_leakage/audit_report.json`, which is ignored by Git. The
current report found 98 cross-split candidate pairs at cosine similarity
`>= 0.90`, 12 at `>= 0.95`, and 4 at `>= 0.98`.

Counts by split pair:

| Split pair | `>= 0.90` | `>= 0.95` | `>= 0.98` |
| --- | ---: | ---: | ---: |
| Train vs validation | 48 | 5 | 0 |
| Train vs final test | 45 | 6 | 4 |
| Validation vs final test | 5 | 1 | 0 |

All 98 candidate pairs at `>= 0.90` have matching queue and priority labels.
The audit found 31 repeated-subject cross-split pairs, including 9 with
different queue labels, but those repeated-subject-only cases were not enough
evidence to justify rebuilding the split because the primary `subject` + `body`
near-duplicate audit did not show material label-conflicting leakage. Existing
metrics are preserved, with a small residual optimism risk noted for the handful
of very high-similarity held-out rows.

## Metrics

Primary metrics:

- macro F1
- per-class precision
- per-class recall
- abstention coverage

Secondary metrics:

- accuracy
- confusion matrix
- calibration diagnostics for confidence-bearing models:
  - multiclass Brier score, computed as the mean summed squared error between
    class probabilities and one-hot labels
  - expected calibration error, computed with 10 equal-width top-label
    confidence bins over `[0, 1]`
  - reliability diagrams for validation and final test reporting
  - confidence distributions and coverage/performance curves across fixed
    confidence thresholds
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

For abstention and review logic, TicketPilot evaluates the selected calibrated
confidence model separately from the uncalibrated validation model-selection
score. The calibrated sklearn confidence model has validation macro F1 0.6455.
Validation calibration diagnostics for the selected calibrated model:

- Brier score: 0.5677
- Expected calibration error: 0.2059
- Median confidence: 0.3971

Final test calibration diagnostics are reporting-only:

- Brier score: 0.5279
- Expected calibration error: 0.1997
- Median confidence: 0.4471

Reliability plots are stored under `reports/queue_baseline/plots/`.

Final test metrics for the selected calibrated LinearSVC baseline:

- Accuracy: 0.6673
- Macro precision: 0.7765
- Macro recall: 0.6294
- Macro F1: 0.6829
- Weighted F1: 0.6656
- Top-3 routing accuracy: 0.8988

The validation-selected abstention threshold is 0.30. On the final test split it
produces above-threshold recommendations for 90.49% of test tickets, flags
9.51% for additional review, and reaches 0.7009 accuracy on the
above-threshold subset. Human review is still required before any response or
operational next step.

The per-queue report identifies low-support queues using support `< 100` and
low-recall queues using recall `< 0.60`. On the final test split, low-support
queues are General Inquiry, Human Resources, Sales and Pre-Sales, and Service
Outages and Maintenance. Low-recall queues are Customer Service, General
Inquiry, IT Support, Returns and Exchanges, and Sales and Pre-Sales. These are
methodology flags only; TicketPilot v1 remains human-reviewed and does not add
autonomous class-specific routing rules.

Deployment-family selection between the sklearn baseline and TensorFlow Conv1D
model is based on validation metrics and documented operational factors, not
final-test metrics. The current regenerated comparison keeps sklearn
`tfidf_linear_svc` because the calibrated sklearn confidence model has
validation macro F1 0.6455 versus TensorFlow validation macro F1 0.3745. Final
test comparison metrics are retained only for reporting after that family
selection.

Methodology limitation: historical project documentation and artifacts have
already inspected final-test metrics. The final test split is therefore still
used as the fixed reporting split for this portfolio milestone, but it should
not be described as a pristine never-inspected holdout for future release
claims.

## Reporting

Reports must distinguish verified results from planned or unverified work. Do not claim production readiness without supporting evidence.
