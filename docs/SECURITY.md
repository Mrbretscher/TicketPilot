# Security

TicketPilot is a local portfolio prototype. The security posture below is
intended to document prototype controls and gaps; it is not a claim of
production hardening.

## Data Privacy

Do not use private employer, university, customer, or real support-ticket data. Use public datasets with compatible licenses or synthetic examples created for this project.

## Secrets

Never commit passwords, API keys, access tokens, private certificates, connection strings, or real `.env` files. Keep `.env.example` free of real secrets.

The optional OpenAI draft provider reads `OPENAI_API_KEY` only when generation is
explicitly invoked. Ordinary unit tests use deterministic fakes and must not
perform paid API calls. Do not log API keys, provider request headers, or full
ticket bodies.

Docker and CI configuration must not require secrets at build time. The default
runtime path does not require `OPENAI_API_KEY`.

## Untrusted Input

Treat ticket text, retrieved documents, uploaded files, and user-submitted text as untrusted input.

RAG drafting separates system instructions, incoming ticket text, and retrieved
evidence. Retrieved evidence may contain malicious instructions; prompts and
tests require treating those strings as data and ignoring embedded instructions.

Prompt-injection controls currently implemented:

- production prompt tells the model retrieved ticket text is untrusted data
- evidence is wrapped in a distinct retrieved-evidence section
- concrete troubleshooting claims must cite stable evidence IDs
- tests include malicious retrieved text that asks the model to claim a
  password reset and close the ticket
- provider failures return structured human-review-required responses

These controls reduce obvious prompt-injection risks in the prototype but do
not replace production red-team testing, monitoring, or policy review.

## Logging

Do not log secrets or unnecessary personally identifiable information. Logs
should support debugging without exposing private ticket content, full retrieved
answers, or provider payloads.

The current codebase does not implement a production logging pipeline.

## Dependencies

Add new production dependencies only after explaining why they are needed and considering simpler alternatives.

The default Docker/runtime install excludes TensorFlow, sentence-transformers,
Jupyter, training commands, raw datasets, generated reports, and model
artifacts. Runtime containers mount the prepared local classifier and TF-IDF
retrieval artifacts instead of rebuilding or downloading them during startup.
This is intentional for reproducibility: the image is application code, while
large/generated artifacts are separately prepared and ignored by Git.

## Container And CI Hygiene

Docker packaging must not copy `.env` files, API keys, raw datasets, local
SQLite review databases, generated model artifacts, vector indexes, caches, or
reports into the image. Docker Compose uses environment variables for
configuration and a local named volume for the review database.

CI must use synthetic fixtures or ignored local artifacts created during the
test run. It must not call OpenAI, download the full public dataset, run full
training, publish images, or require provider credentials.

## Operational Boundaries

TicketPilot must not autonomously perform IT actions or send generated responses. Human approval is required.

Generated RAG output is a draft response only. It must not claim that a password
was reset, permissions changed, a refund issued, a ticket closed, or any other
action performed unless a human reviewer has independently verified external
evidence.

## Known Security Gaps Before Production

- No authentication or role-based access control.
- No hosted deployment hardening has been verified.
- No rate limiting or abuse monitoring.
- No centralized audit logging or alerting.
- No secrets manager integration.
- No production review of prompt-injection defenses.
- Docker build and Compose runtime verification are pending in the current
  local environment.
