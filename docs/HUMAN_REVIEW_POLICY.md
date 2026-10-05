# Human Review Policy

## Policy Summary

TicketPilot is a decision-support system. A human reviewer must approve any drafted response or operational next step.

Milestone 7 generated text is a draft response for review only. The structured
draft output must include `human_review_required = true`.

Milestone 8 adds a local SQLite review workflow for portfolio v1. It records
review decisions and edited drafts, but it does not send email, close tickets,
change permissions, or execute support actions.

## Required Human Review Cases

Human review is required when:

- queue confidence is below the configured threshold
- automated priority prediction is unavailable or unsupported
- retrieved evidence is weak, contradictory, or missing
- the draft has `insufficient_evidence`, `low_classifier_confidence`, or
  `provider_error` status
- the ticket appears security-sensitive
- the ticket requests account, permission, financial, legal, or disciplinary action
- the generated draft contains uncertainty or unsupported claims

## Prohibited Automation

TicketPilot must not:

- reset passwords
- change permissions
- modify account state
- close tickets
- send responses automatically
- claim an action has been completed unless external evidence supports it

## Review Workflow

Allowed reviewer actions:

- `accept`
- `edit`
- `reject`
- `reroute`
- `mark_insufficient_evidence`

Review states:

- `pending`: created by TicketPilot and awaiting human review
- `approved`: reviewer accepted the draft
- `edited`: reviewer edited the response or selected a different final queue
- `rejected`: reviewer rejected the draft or marked evidence insufficient

The review store captures the analysis ID, timestamps, local ticket hash,
predicted queue and confidence, optional priority, retrieved evidence IDs,
draft response, model/provider metadata, reviewer action, edited response,
final queue, and optional review note.

Portfolio v1 does not require authentication. A production deployment must add
authentication, authorization, and role-based access control before exposing
review records or reviewer actions.

## Reviewer Responsibilities

The reviewer should inspect the original ticket, predicted labels, retrieved evidence, citations, and draft response before deciding what to send or do next.

The reviewer must verify that every concrete troubleshooting claim is supported
by cited retrieved evidence IDs and that the draft does not follow instructions
embedded inside retrieved tickets.

## Auditability

Future implementations should preserve enough metadata to explain which model version, prompt version, retrieved evidence, and thresholds were used for a recommendation.

Current drafting metadata includes the predicted queue, classifier confidence,
retrieved evidence IDs, evidence similarity threshold, classifier confidence
threshold, draft provider, and configured draft model when OpenAI generation is
used. It does not include an automated priority prediction or priority
confidence for recruiter-ready v1.
