# Data Card

## Dataset Status

Milestone 1 selects a public Hugging Face dataset for local experimentation:

- Source repository: `Tobi-Bueck/customer-support-tickets`
- Dataset page: <https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets>
- Dataset file used for recruiter-ready v1: `aa_dataset-tickets-multi-lang-5-2-50-version.csv`
- Pinned revision: `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`
- Source file SHA256: `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`
- DOI listed by Hugging Face: `10.57967/hf/6184`
- License listed by Hugging Face: `cc-by-nc-4.0`
- Retrieval date for this project documentation: 2026-10-02

The full downloaded source CSV and English subset are stored locally under
`data/raw/`, which is ignored by Git. Do not commit the downloaded dataset.

## Acquisition

Fetch and validate the pinned dataset locally:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_customer_support_tickets.py
```

or:

```powershell
.\scripts\fetch_data.ps1
```

The acquisition command writes:

- `data/raw/customer_support_tickets_source.csv`
- `data/raw/customer_support_tickets_en.csv`

If the source file already exists, the command reuses it unless `--overwrite`
is supplied. The source file hash is checked before validation.

## Observed Source Schema

The current pinned CSV was inspected before implementing validation. It contains
28,587 rows and these columns in this order:

1. `subject`
2. `body`
3. `answer`
4. `type`
5. `queue`
6. `priority`
7. `language`
8. `version`
9. `tag_1`
10. `tag_2`
11. `tag_3`
12. `tag_4`
13. `tag_5`
14. `tag_6`
15. `tag_7`
16. `tag_8`

Observed language counts:

- `en`: 16,338
- `de`: 12,249

TicketPilot recruiter-ready v1 uses only `language == "en"` records while
preserving the original fields listed above for analysis.

## English Subset Quality Snapshot

Observed on 2026-10-02 after filtering to English records:

- Rows: 16,338
- Queues:
  - Technical Support: 4,737
  - Product Support: 3,073
  - Customer Service: 2,410
  - IT Support: 1,942
  - Billing and Payments: 1,595
  - Returns and Exchanges: 820
  - Service Outages and Maintenance: 664
  - Sales and Pre-Sales: 513
  - Human Resources: 348
  - General Inquiry: 236
- Priorities:
  - medium: 6,618
  - high: 6,346
  - low: 3,374
- Null values observed:
  - `subject`: 2,607
  - `answer`: 3
  - tag columns contain expected sparse values
- Empty `body` records: 0
- Empty `subject` and `body` records: 0
- Exact duplicate rows: 0
- Exact duplicate `subject` + `body` combinations: 0

Missing `subject` values are retained because `body` remains available and
non-empty. Downstream preprocessing must handle missing subject text explicitly.

## Near-Duplicate Split Audit

A focused near-duplicate audit was run on 2026-10-05 after the Phase 11
evaluation audit flagged repeated and highly similar ticket text patterns. The
audit used normalized `subject` + `body` text as the primary ticket
representation and compared split pairs independently with a documented
scikit-learn TF-IDF cosine nearest-neighbor method. The audit report is written
to the ignored machine-readable artifact
`reports/near_duplicate_leakage/audit_report.json`.

Cross-split candidate pairs:

| Similarity threshold | Candidate pairs | Unique affected tickets |
| --- | ---: | ---: |
| `>= 0.90` | 98 | 187 |
| `>= 0.95` | 12 | 24 |
| `>= 0.98` | 4 | 8 |

Counts by split pair at `>= 0.90`:

- Train vs validation: 48
- Train vs final test: 45
- Validation vs final test: 5

All 98 `>= 0.90` candidate pairs have the same queue and priority labels on
both sides of the split boundary. The highest-scoring examples are mostly
generated-template variants, including one train/test pair that differs only by
punctuation. The audit also found 31 repeated-subject cross-split pairs; 22 have
the same queue and 9 have different queues. These repeated-subject-only matches
are tracked as a risk signal but did not by themselves justify rebuilding the
split because the full `subject` + `body` audit did not show material
label-conflicting near-duplicate leakage.

Audit conclusion: no material cross-split leakage was found. Existing splits
remain defensible, and no train/validation/test assignments were changed.

## Classifier Boundary

Allowed classifier inputs:

- `subject`
- `body`

Allowed classifier labels:

- `queue`
- `priority`

The `answer` field must not be used as a classifier input. It is preserved only
for later resolved-ticket retrieval and RAG evidence. Retrieved evidence must
remain structurally separate from model-generated text.

The following fields are also prohibited as classifier inputs in Milestone 1:

- `answer`
- `type`
- `queue`
- `priority`
- `language`
- `version`
- `tag_1` through `tag_8`

## Validation Checks

Implemented validation covers:

- Required columns and exact schema order.
- Unexpected schema changes.
- Null values in required non-null fields.
- Empty ticket text across `subject` and `body`.
- Language values.
- Queue values.
- Priority values.
- Type values.
- Dataset size thresholds for the source and English subset.
- Label distributions.
- Exact duplicate records.
- Exact duplicate `subject` + `body` combinations.
- TF-IDF cosine near-duplicate candidate pairs crossing split boundaries.
- Repeated-subject patterns crossing split boundaries.
- Suspicious label leakage diagnostics, such as label words appearing in
  classifier text.
- Classifier feature-column policy, including explicit rejection of `answer`.

The regenerated dataset summary also surfaces literal label-mention diagnostics
from `validation.py`. The method flags a ticket when its own queue or priority
label appears verbatim in its public `subject` or `body`; it does not inspect
the resolved answer field and does not remove records. In the current prepared
English dataset, 862 of 16,338 tickets contain at least one such mention
(5.28%). Queue-label mentions affect 97 tickets (0.59%), primarily Technical
Support (48), Returns and Exchanges (16), Product Support (9), Customer Service
(8), Billing and Payments (7), Human Resources (3), IT Support (3), and Service
Outages and Maintenance (3). Priority-label mentions affect 777 tickets
(4.76%), mostly `high` (347) and `low` (427), with 3 `medium` mentions.

These counts are diagnostics rather than automatic failures because
user-authored ticket text can naturally mention urgency or routing terms.
Priority words such as `high` and `low` are especially likely to be ordinary
support language. Exact queue names may be stronger target-proxy signals when
they appear as routing labels, but the safe public examples in
`reports/dataset_preparation/dataset_summary.json` show a mix of natural
phrasing and ambiguous support-template language. Records are retained, and the
risk must be interpreted alongside other leakage checks before reporting model
claims.

## Attribution

Attribute the dataset to the Hugging Face repository
`Tobi-Bueck/customer-support-tickets` and preserve the stated
`cc-by-nc-4.0` license when using this project in public materials. The dataset
card links the creator organization as Softoft.

## Limitations

- The dataset card describes the data as generated by a synthetic IT ticket
  generator, not as real private support data.
- The license is non-commercial (`cc-by-nc-4.0`), so commercial use is out of
  scope without separate permission.
- The data is multilingual, but TicketPilot v1 uses only English records.
- Queue and priority labels are synthetic and may not represent a real
  organization's triage policy.
- Some English records have missing subjects; `body` is therefore the more
  reliable core text field.
- The near-duplicate audit found a small number of high-similarity cross-split
  examples in synthetic generated text. The split remains defensible, but
  future releases should continue to track these candidates before changing
  evaluation claims.
- The `answer` field can support later retrieval/RAG experiments but must never
  be mixed into classifier inputs.
- No model performance, fairness, calibration, or production-readiness claims
  have been established.
