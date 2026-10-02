# RAG Evaluation

## RAG Status

No retrieval or generation system has been implemented yet.

## Retrieval Evaluation

Planned retrieval metrics:

- recall at k
- precision at k
- mean reciprocal rank where relevance labels are available
- evidence coverage for common ticket categories

Retrieved evidence must remain structurally separate from generated draft text.

## Generation Evaluation

Planned generation checks:

- draft cites retrieved evidence
- citations point to real retrieved sources
- draft does not claim unsupported actions were performed
- draft preserves uncertainty where evidence is weak
- draft routes low-confidence or weak-evidence cases to human review

## Prompt Policy

Production prompts must live in version-controlled source code when generation is introduced.

## Test Policy

Tests for RAG behavior must not require paid API calls. Use deterministic fakes, fixtures, or local stubs for automated tests.

## Human Review

Generated drafts are suggestions for human review. TicketPilot must not automatically send responses.
