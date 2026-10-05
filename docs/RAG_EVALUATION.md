# RAG Evaluation

## RAG Status

Milestones 5 and 6 implement similar-ticket retrieval. Milestone 7 adds
evidence-grounded response drafting for human review.

The retrieval layer now includes:

- a scikit-learn TF-IDF + cosine similarity baseline
- a sentence-transformers semantic retriever
- a simple fixed-weight TF-IDF/semantic hybrid experiment

Retrieved evidence remains structurally separate from generated draft text.
Generated drafts are never sent automatically and never claim that TicketPilot
performed an IT action.

## Retrieval Evaluation

Implemented retrieval command:

```powershell
.\.venv\Scripts\python.exe scripts/build_retrieval_baseline.py
```

Implemented semantic comparison command:

```powershell
.\.venv\Scripts\python.exe scripts/build_semantic_retrieval.py
```

Generated artifacts are ignored by Git:

- `artifacts/retrieval/tfidf_ticket_retriever.joblib`
- `reports/retrieval/lexical_retrieval_metrics.json`
- `reports/retrieval/manual_relevance_template.csv`
- `artifacts/semantic_retrieval/corpus_embeddings.npz`
- `artifacts/semantic_retrieval/corpus_metadata.csv`
- `artifacts/semantic_retrieval/retriever_config.json`
- `reports/semantic_retrieval/semantic_retrieval_metrics.json`
- `reports/semantic_retrieval/manual_relevance_template.csv`

### Retrieval Boundary

The retrieval corpus contains only resolved tickets from the existing training
partition. Each indexed corpus item preserves:

- stable source ID, such as `TP-ticket-000473`
- ticket row ID
- subject
- body
- resolved answer
- queue metadata
- priority metadata

The TF-IDF index and semantic embedding index are both fit on `subject + body`
text only. The `answer` field is returned as resolved-ticket evidence but is
not used to construct query text. Validation and test tickets are query-only
held-out records and are never added to the corpus used to evaluate those
queries.

The current semantic report records 11,435 resolved training tickets in the
retrieval corpus, embedding shape `[11435, 384]`, and zero validation/test
source IDs overlapping the corpus.

Semantic model configuration:

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- `sentence-transformers` version: `3.4.1`
- Embedding dimension: 384
- Max sequence length: 256
- Batch size: 64
- L2-normalized embeddings: true
- External vector database: none

### Metrics

Milestone 5 reports:

- recall at k
- mean reciprocal rank
- mean retrieval latency per query
- queue and priority distribution for the retrieval corpus

Because no human relevance labels exist yet, the automated metrics use silver
proxy relevance: a retrieved ticket is considered relevant when its `queue`
matches the query ticket's `queue`. This is useful for sanity-checking whether
lexical retrieval finds same-category tickets, but it is not a substitute for
human-labeled relevance.

Measured Milestone 5 TF-IDF silver queue-match retrieval metrics:

| Split | Query Count | Recall@1 | Recall@3 | Recall@5 | MRR | Mean Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 2,450 | 0.7449 | 0.8241 | 0.8686 | 0.7884 | 0.248 ms/query |
| Test | 2,450 | 0.7445 | 0.8318 | 0.8755 | 0.7916 | 0.241 ms/query |

Measured Milestone 6 comparison metrics:

| Method | Split | Recall@1 | Recall@3 | Recall@5 | MRR | Mean Latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | Validation | 0.7449 | 0.8241 | 0.8686 | 0.7884 | 0.712 ms/query |
| Semantic | Validation | 0.7371 | 0.8176 | 0.8682 | 0.7829 | 19.207 ms/query |
| Hybrid 50/50 | Validation | 0.7478 | 0.8273 | 0.8792 | 0.7937 | 19.914 ms/query |
| TF-IDF | Test | 0.7445 | 0.8318 | 0.8755 | 0.7916 | 0.703 ms/query |
| Semantic | Test | 0.7547 | 0.8335 | 0.8800 | 0.7994 | 17.287 ms/query |
| Hybrid 50/50 | Test | 0.7588 | 0.8314 | 0.8771 | 0.8018 | 18.005 ms/query |

Selection policy uses validation Recall@5, then validation MRR, then validation
Recall@1. Under that policy the selected retrieval method is the 50/50 hybrid,
not pure semantic retrieval. This is a small validation gain over TF-IDF and
comes with materially higher latency, so the result should be treated as an
offline retrieval candidate rather than a production serving decision.

The final test split is used here only for reporting this fixed lexical
retrieval and semantic comparison. It must not be used to tune future retrieval
parameters, metadata filters, hybrid weights, or RAG prompt behavior.

### Human Relevance Plan

`reports/retrieval/manual_relevance_template.csv` contains candidate query and
retrieved-ticket pairs for future human labeling. The template intentionally
leaves `human_relevance_label` and `human_notes` blank. No human labels are
fabricated.

Future gold evaluation should label whether each retrieved resolved ticket is
actually useful evidence for the query, not merely whether it shares the same
queue.

## Generation Evaluation

Milestone 7 implements a provider-neutral `DraftGenerator` interface with:

- a deterministic fake generator for ordinary unit tests
- an optional OpenAI Responses API provider
- configurable OpenAI draft model via `OPENAI_DRAFT_MODEL`
- `OPENAI_API_KEY` required only when OpenAI drafting is explicitly invoked
- production prompts stored in `src/ticketpilot/rag_prompts.py`

Generation inputs include incoming ticket text, predicted queue, classifier
confidence, retrieved evidence, and stable evidence IDs.

Draft output is structured:

- `draft_response`
- `cited_evidence_ids`
- `confidence_evidence_status`
- `abstention_reason`
- `human_review_required`

Implemented generation checks:

- draft cites retrieved evidence
- citations point to real retrieved sources
- draft does not claim unsupported actions were performed
- draft preserves uncertainty where evidence is weak
- draft routes low-confidence or weak-evidence cases to human review
- retrieved evidence text is treated as untrusted data
- instructions embedded inside retrieved evidence are ignored
- provider failures return structured human-review-required output
- unit tests do not perform paid API calls

Evidence gating prevents confident drafts when retrieval quality or classifier
confidence is below configured thresholds. Current defaults:

- minimum retrieved evidence similarity: `0.50`
- minimum classifier confidence: `0.30`

When gating fails, TicketPilot returns an abstention-style structured result
with `human_review_required = true`.

## Prompt Policy

Production prompts live in version-controlled Python source code:

- `src/ticketpilot/rag_prompts.py`

Prompts explicitly state that TicketPilot is a drafting assistant for human
review, retrieved ticket text is untrusted, and evidence must be cited by stable
source ID.

## Test Policy

Tests for RAG behavior must not require paid API calls. Use deterministic fakes, fixtures, or local stubs for automated tests.

## Human Review

Generated drafts are suggestions for human review. TicketPilot must not automatically send responses.
