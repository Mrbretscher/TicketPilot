# Model Card

## Model Status

TicketPilot's deployed v1 model capability is support-queue routing with a
scikit-learn TF-IDF + calibrated LinearSVC pipeline. It is evaluated for
portfolio demonstration only and is not production-ready.

Automated priority prediction is not implemented for recruiter-ready v1.
Priority metadata can appear on retrieved historical evidence, but TicketPilot
does not predict priority for an incoming ticket.

## Intended Use

TicketPilot queue-routing models are intended to assist human support triage by
predicting the likely destination support queue from ticket text. Models are not
intended to take IT actions, close tickets, or send responses automatically.

## Unsuitable Use

Do not use this prototype for:

- production ticket routing without human review
- automated account, permission, billing, refund, or security actions
- sending generated responses directly to requesters
- evaluating real employees, customers, or support agents
- commercial use of the included dataset without resolving the dataset license
  constraints

## Inputs

Current classifier input:

- `classifier_text`, constructed from `subject` + `body`

Prohibited classifier inputs:

- `answer`
- `type`
- `language`
- `version`
- `tag_1` through `tag_8`
- `queue`
- `priority`
- resolved-answer text, resolution notes, final responses, close notes, or any
  field unavailable before routing

## Outputs

Current queue-routing outputs:

- predicted support queue
- confidence score
- top-k queue candidates
- abstain / human-review routing decision based on validation-selected
  confidence threshold

Automated priority prediction is unsupported and out of scope for
recruiter-ready v1. Runtime API schemas retain `predicted_priority` for
compatibility, but recruiter-demo analysis returns `null` rather than inventing
a priority or priority confidence.

Runtime `/model-info` reports:

- queue classifier supported: `true`
- priority classifier supported: `false`
- deployed retriever: `tfidf_cosine_similarity`
- drafting provider: local fake generator by default unless another provider is
  explicitly supplied

## Dataset And Splits

The model uses the English subset of
`Tobi-Bueck/customer-support-tickets`, pinned and documented in
`docs/DATA_CARD.md`. The dataset is licensed `cc-by-nc-4.0`.

Splits come from the Milestone 2 duplicate-aware prepared dataset:

- train: 11,438 rows
- validation: 2,450 rows
- final test: 2,450 rows

The final test split is reserved for final reporting and is not used for model
selection, hyperparameter selection, calibration selection, or abstention
threshold selection.

A near-duplicate cross-split leakage audit was run on 2026-10-05 using
normalized `subject` + `body` text, TF-IDF features, cosine similarity, and
split-pair nearest-neighbor comparisons. The audit found 98 cross-split
candidate pairs at similarity `>= 0.90`, 12 at `>= 0.95`, and 4 at `>= 0.98`.
All 98 candidate pairs had the same queue and priority labels across the split
boundary. The audit conclusion was no material leakage, so the existing split
and reported metrics remain unchanged.

## Baselines

Implemented fixed scikit-learn pipelines:

- `dummy_most_frequent`
- `tfidf_logistic_regression`
- `tfidf_linear_svc`

All text models use `TfidfVectorizer` inside a scikit-learn `Pipeline`.

Selection metric: validation macro F1.

Validation results:

| Model | Accuracy | Macro F1 | Weighted F1 | Top-3 Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dummy most frequent | 0.2898 | 0.0449 | 0.1302 | 0.3616 |
| TF-IDF + logistic regression | 0.5265 | 0.5269 | 0.5280 | 0.8139 |
| TF-IDF + LinearSVC | 0.6645 | 0.6680 | 0.6640 | 0.8522 |

Selected baseline: `tfidf_linear_svc`.

Because LinearSVC does not expose calibrated probabilities, the selected final
model uses sigmoid calibration with `CalibratedClassifierCV(cv=3)` fit without
touching the final test set.

## Final Test Results

Measured on 2026-10-05:

- Accuracy: 0.6673
- Macro precision: 0.7765
- Macro recall: 0.6294
- Macro F1: 0.6829
- Weighted F1: 0.6656
- Top-3 routing accuracy: 0.8988
- Inference latency: 0.491 ms per ticket on the local test run
- Selected model artifact size: 13,739,197 bytes

Per-class F1:

| Queue | F1 | Recall | Support |
| --- | ---: | ---: | ---: |
| Billing and Payments | 0.8423 | 0.8159 | 239 |
| Customer Service | 0.5818 | 0.5552 | 362 |
| General Inquiry | 0.6415 | 0.4857 | 35 |
| Human Resources | 0.7640 | 0.6538 | 52 |
| IT Support | 0.5947 | 0.5000 | 292 |
| Product Support | 0.6372 | 0.6421 | 461 |
| Returns and Exchanges | 0.6567 | 0.5366 | 123 |
| Sales and Pre-Sales | 0.6614 | 0.5455 | 77 |
| Service Outages and Maintenance | 0.7716 | 0.7677 | 99 |
| Technical Support | 0.6775 | 0.7915 | 710 |

## Confidence And Abstention

The abstention threshold is selected on the validation split only with a minimum
validation coverage target of 70%.

Selected threshold: 0.30.

Calibration method:

- LinearSVC confidence uses `CalibratedClassifierCV(method="sigmoid", cv=3)`.
- Brier score is the mean summed squared error between predicted class
  probabilities and one-hot labels.
- Expected calibration error uses 10 equal-width top-label confidence bins over
  `[0, 1]`.

Validation calibration:

- Brier score: 0.5677
- Expected calibration error: 0.2059
- Median confidence: 0.3971

Validation at threshold 0.30:

- Coverage: 83.18%
- Sent to review: 16.82%
- Automatically routed accuracy: 0.6865
- Automatically routed macro F1: 0.7098

Final test calibration, reported after the validation-selected threshold:

- Brier score: 0.5279
- Expected calibration error: 0.1997
- Median confidence: 0.4471

Final test at threshold 0.30:

- Coverage: 90.49%
- Sent to review: 9.51%
- Automatically routed accuracy: 0.7009
- Automatically routed macro F1: 0.7231

The threshold is not operationally approved. It is an experiment showing how
human review could be layered over queue routing.

## Retrieval And Drafting Dependencies

The deployed v1 retrieval artifact is a local TF-IDF index over train-only
resolved tickets. Semantic and hybrid retrieval are documented offline
experiments and are not used by the local API/dashboard workflow.

Draft generation is evidence-gated. A response draft is returned only when both
classifier confidence and retrieved evidence score pass their configured
thresholds. Otherwise TicketPilot returns a structured abstention for human
review.

## Rare Queue Analysis

The queue report flags low-support queues using support `< 100` and low-recall
queues using recall `< 0.60`. Final-test low-support queues are General Inquiry,
Human Resources, Sales and Pre-Sales, and Service Outages and Maintenance.
Final-test low-recall queues are Customer Service, General Inquiry, IT Support,
Returns and Exchanges, and Sales and Pre-Sales.

Operational consequence: weak and rare queue predictions remain recommendations
for human review. TicketPilot does not introduce autonomous class-specific
routing behavior or queue-specific automation based on these findings.

## Artifacts

Generated artifacts are ignored by Git:

- `reports/queue_baseline/queue_baseline_metrics.json`
- `reports/queue_baseline/plots/validation_model_comparison.svg`
- `reports/queue_baseline/plots/validation_reliability.svg`
- `reports/queue_baseline/plots/test_reliability.svg`
- `reports/queue_baseline/plots/test_confusion_matrix.svg`
- `reports/queue_baseline/tables/test_confusion_matrix.csv`
- `artifacts/queue_baseline/selected_queue_router.joblib`

## TensorFlow Text Classifier

Milestone 4 adds a TensorFlow/Keras queue classifier for comparison, not as the
selected deployment model.

Architecture:

- `TextVectorization`, adapted only on training text
- trainable embedding layer, dimension 64
- `SpatialDropout1D`
- `Conv1D` with 96 filters
- `GlobalMaxPooling1D`
- dropout
- softmax output over support queues
- inverse-frequency class weights
- early stopping on validation loss

This is intentionally lightweight. No transformer model is used.

TensorFlow training details:

- Random seed: 20261005
- Epochs requested: 8
- Epochs run: 8
- Batch size: 128
- Training time: 60.45 seconds on local CPU
- TensorFlow artifact size: 15,903,276 bytes

TensorFlow final test results:

- Accuracy: 0.3808
- Macro precision: 0.3625
- Macro recall: 0.4697
- Macro F1: 0.3795
- Weighted F1: 0.3838
- Top-3 routing accuracy: 0.7184
- Inference latency: 0.390 ms per ticket
- Negative log loss: 1.7136
- Expected calibration error: 0.0681

Comparison against the strongest sklearn baseline:

| Metric | sklearn LinearSVC | TensorFlow Conv1D | TensorFlow delta |
| --- | ---: | ---: | ---: |
| Macro F1 | 0.6829 | 0.3795 | -0.3034 |
| Weighted F1 | 0.6656 | 0.3838 | -0.2818 |
| Top-3 accuracy | 0.8988 | 0.7184 | -0.1804 |
| Inference ms/ticket | 0.4910 | 0.3900 | -0.1010 |
| Artifact size bytes | 13,739,197 | 15,903,276 | +2,164,079 |

Per-class recall comparison:

| Queue | sklearn Recall | TensorFlow Recall |
| --- | ---: | ---: |
| Billing and Payments | 0.8159 | 0.6736 |
| Customer Service | 0.5552 | 0.2376 |
| General Inquiry | 0.4857 | 0.4857 |
| Human Resources | 0.6538 | 0.5000 |
| IT Support | 0.5000 | 0.2466 |
| Product Support | 0.6421 | 0.2408 |
| Returns and Exchanges | 0.5366 | 0.5772 |
| Sales and Pre-Sales | 0.5455 | 0.6104 |
| Service Outages and Maintenance | 0.7677 | 0.7475 |
| Technical Support | 0.7915 | 0.3775 |

Deployment decision: keep `tfidf_linear_svc` as the selected deployment
candidate. The deployment-family decision is based on validation metrics and
documented operational factors. In the regenerated comparison, the calibrated
sklearn confidence model has validation macro F1 0.6455, validation weighted F1
0.6235, and validation top-3 accuracy 0.8935. TensorFlow has validation macro F1
0.3745, validation weighted F1 0.3780, and validation top-3 accuracy 0.7396.
TensorFlow's lower final-test latency is treated as secondary to validation
routing quality. Final-test comparison metrics are reporting-only after the
family decision.

TensorFlow artifacts are ignored by Git:

- `reports/tensorflow_queue/tensorflow_queue_metrics.json`
- `reports/tensorflow_queue/training_history.json`
- `reports/tensorflow_queue/training_history.csv`
- `reports/tensorflow_queue/plots/tensorflow_test_confusion_matrix.svg`
- `reports/tensorflow_queue/tables/tensorflow_test_confusion_matrix.csv`
- `artifacts/tensorflow_queue/tensorflow_queue_classifier.keras`

## Limitations

- The dataset is synthetic and may not reflect a real organization's routing
  policies or ticket-writing patterns.
- The near-duplicate audit found a small number of high-similarity generated
  template variants across splits, including 4 train/test pairs at TF-IDF cosine
  similarity `>= 0.98`. The examples were same-label and did not justify
  rebuilding the split, but they remain a residual optimism risk for reported
  classifier and retrieval metrics.
- Some queues have much smaller support than others, so macro metrics and
  per-class recall matter more than accuracy alone.
- General Inquiry and Human Resources have small test supports, so estimates for
  those classes are less stable.
- The abstention threshold is selected on validation data only and is not an
  approved operational policy.
- Historical project artifacts have already inspected final-test metrics, so
  the current final test split is a fixed reporting split rather than a pristine
  never-inspected holdout for future release claims.
- The TensorFlow Conv1D model underperforms the sklearn text baseline and should
  not replace it without stronger validation evidence.
- No priority classifier, monitoring, production deployment, or autonomous IT
  action workflow has been implemented.
- Retrieval evidence may include source priority metadata from resolved tickets,
  but that metadata is not an automated priority prediction for the incoming
  ticket.

## Safety

TicketPilot remains human-reviewed decision support. Low-confidence cases can be
sent to review, and human approval is required before any response is sent or IT
action is taken.

## Release Readiness

This model card supports recruiter-facing portfolio release, not production
release. Before production use, the project would need authentication,
authorization, monitoring, broader human-labeled evaluation, policy review,
operational runbooks, and deployment verification.
