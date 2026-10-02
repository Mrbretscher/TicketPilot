# Data Card

## Dataset Status

No dataset has been selected yet.

## Approved Data Sources

TicketPilot may use public datasets with compatible licenses or synthetic tickets created specifically for this portfolio project.

Private employer, university, customer, help-desk, or support-ticket data must never be used.

## Planned Schema

Candidate fields:

- ticket_id
- created_at
- title
- body
- requester_context
- product_area
- queue_label
- priority_label
- resolved_answer
- resolution_notes

The final schema is not approved yet.

## Classifier Input Policy

Allowed classifier inputs may include only fields available before routing, such as title, body, and approved request metadata.

Classifier inputs must never include:

- resolved_answer
- resolution_notes
- final response text
- close notes
- post-resolution tags
- any proxy field unavailable at prediction time

Resolved-ticket text may later be used as retrieval evidence, but it must remain separate from classifier features.

## Leakage And Contamination Checks

Before reporting model performance, the project must check for:

- Duplicate or near-duplicate tickets split across train and test.
- Resolution text included in training features.
- Labels or label proxies embedded in feature columns.
- Time-dependent leakage if chronological data is used.
- Retrieval corpus contamination in generation evaluation.

## License And Attribution

Dataset license, source URL, acquisition date, and attribution requirements must be documented before data is committed or used in reports.

## Storage Policy

Downloaded datasets, intermediate data, generated artifacts, model files, vector indexes, and experiment outputs must not be committed.
