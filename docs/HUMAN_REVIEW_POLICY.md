# Human Review Policy

## Policy Summary

TicketPilot is a decision-support system. A human reviewer must approve any drafted response or operational next step.

## Required Human Review Cases

Human review is required when:

- queue confidence is below the configured threshold
- priority confidence is below the configured threshold
- retrieved evidence is weak, contradictory, or missing
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

## Reviewer Responsibilities

The reviewer should inspect the original ticket, predicted labels, retrieved evidence, citations, and draft response before deciding what to send or do next.

## Auditability

Future implementations should preserve enough metadata to explain which model version, prompt version, retrieved evidence, and thresholds were used for a recommendation.
