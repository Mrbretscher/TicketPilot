# Gold Retrieval Labeling

TicketPilot uses human-labeled gold retrieval sets to evaluate whether the
deployed retriever returns useful resolved-ticket evidence. Do not infer labels
from queue matches alone. Judge whether the candidate evidence would actually
help a human resolve the query ticket.

## Create Labeling CSVs

Run:

```powershell
.\.venv\Scripts\python.exe scripts\build_gold_retrieval_labels.py
```

This uses the deployed local retriever artifact:

- `artifacts/retrieval/tfidf_ticket_retriever.joblib`

It writes two ignored CSV files:

- `reports/gold_retrieval/gold_validation_labeling.csv`
- `reports/gold_retrieval/gold_test_labeling.csv`

Each file contains 20 stratified held-out queries and the top 5 retrieved
evidence candidates for each query. The query samples are selected from their
existing partitions:

- GOLD VALIDATION: validation split only
- GOLD TEST: final test split only

The sampling code tries to include a practical mix of common queues,
lower-support queues, short tickets, longer tickets, easy retrieval cases, and
ambiguous retrieval cases. The final test set must remain untouched for model
selection.

## CSV Columns

The labeling files contain:

- `evaluation_split`
- `query_id`
- `query_queue`
- `query_subject`
- `query_body`
- `candidate_rank`
- `candidate_evidence_id`
- `candidate_queue`
- `candidate_subject`
- `candidate_issue_excerpt`
- `candidate_resolution_excerpt`
- `retrieval_score`
- `relevance_label`
- `reviewer_notes`

Leave `relevance_label` and `reviewer_notes` blank until a human reviewer fills
them in. The CSV generator intentionally does not prefill human relevance
labels, does not create queue-match proxy labels, and does not infer relevance
from the retrieved candidate queue.

## Relevance Scheme

Use exactly these labels:

- `0` = not useful evidence for resolving the query
- `1` = plausibly useful evidence
- `2` = strongly relevant/useful evidence

Label each query/candidate row independently. A candidate can be useful even if
its queue differs from the query queue, and a same-queue candidate can still be
irrelevant. Prefer the actual issue and resolution content over metadata.

Use `reviewer_notes` for uncertainty, edge cases, or short explanations. Notes
are optional, but `relevance_label` is required for every candidate before
scoring.

## Score Completed Labels

After both CSV files have been manually labeled, run:

```powershell
.\.venv\Scripts\python.exe scripts\score_gold_retrieval.py
```

The scorer refuses to run if any required human labels are missing or invalid.
Valid labels are only `0`, `1`, and `2`.

The scorer reports GOLD VALIDATION and GOLD TEST separately:

- Recall@1
- Recall@3
- Recall@5
- MRR
- nDCG@5 using the 0/1/2 graded relevance labels

The default metrics report is:

- `reports/gold_retrieval/gold_retrieval_metrics.json`

## Validation And Test Policy

GOLD VALIDATION labels may later be used for retrieval or evidence-threshold
selection.

GOLD TEST labels must be reporting-only. Do not use the final test labels to
choose thresholds, retriever parameters, hybrid weights, prompt behavior, or any
other model-selection setting.
