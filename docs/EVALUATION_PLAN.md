# Evaluation Plan

## Evaluation Status

No evaluation has been run yet.

## Classification Tasks

TicketPilot will evaluate:

- support queue classification
- priority classification
- abstention and human-review routing

## Validation Strategy

The validation strategy depends on the final dataset:

- Use stratified splits for suitable balanced classification datasets.
- Use chronological validation when timestamps imply time-dependent deployment.
- Use grouped splits when related tickets, users, organizations, or incidents must remain together.
- Use duplicate-aware splitting for text records.

The selected strategy must be documented before model results are reported.

## Leakage Controls

Evaluation must verify that classifier inputs exclude resolved-answer text, resolution notes, final response text, close notes, post-resolution tags, and fields unavailable before routing.

Train/test contamination checks must be run and reported.

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

## Baseline Requirement

A scikit-learn baseline must be implemented and evaluated before adding TensorFlow text modeling.

## Reporting

Reports must distinguish verified results from planned or unverified work. Do not claim production readiness without supporting evidence.
