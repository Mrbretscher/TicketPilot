# Data Directory

Raw data files generated locally by TicketPilot belong under `data/raw/`.
That directory is ignored by Git and must not be committed.

Milestone 1 uses the public Hugging Face dataset:

- Repository: `Tobi-Bueck/customer-support-tickets`
- URL: <https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets>
- File: `aa_dataset-tickets-multi-lang-5-2-50-version.csv`
- Pinned revision: `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb`
- License listed by Hugging Face: `cc-by-nc-4.0`
- Retrieval date documented for this project: 2026-10-02

Fetch and validate it locally with:

```powershell
.\scripts\fetch_data.ps1
```

or:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_customer_support_tickets.py
```

This writes:

- `data/raw/customer_support_tickets_source.csv`
- `data/raw/customer_support_tickets_en.csv`

TicketPilot recruiter-ready v1 uses the English subset and preserves the source
columns:

`subject`, `body`, `answer`, `type`, `queue`, `priority`, `language`, `version`,
and `tag_1` through `tag_8`.

Classifier inputs may use only `subject` and `body`. Classifier labels may use
`queue` and `priority`. The `answer` field is preserved for later resolved-ticket
retrieval and RAG evidence, but it must not be used as a classifier input.

See `docs/DATA_CARD.md` for source, attribution, schema, validation checks,
observed label distributions, and limitations.
