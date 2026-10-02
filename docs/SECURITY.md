# Security

## Data Privacy

Do not use private employer, university, customer, or real support-ticket data. Use public datasets with compatible licenses or synthetic examples created for this project.

## Secrets

Never commit passwords, API keys, access tokens, private certificates, connection strings, or real `.env` files. Keep `.env.example` free of real secrets.

## Untrusted Input

Treat ticket text, retrieved documents, uploaded files, and user-submitted text as untrusted input.

Future RAG implementations must include prompt-injection defenses that separate system instructions, retrieved evidence, and user-controlled ticket content.

## Logging

Do not log secrets or unnecessary personally identifiable information. Future logs should support debugging without exposing private ticket content.

## Dependencies

Add new production dependencies only after explaining why they are needed and considering simpler alternatives.

## Operational Boundaries

TicketPilot must not autonomously perform IT actions or send generated responses. Human approval is required.
