# Security

## Data Privacy

Do not use private employer, university, customer, or real support-ticket data. Use public datasets with compatible licenses or synthetic examples created for this project.

## Secrets

Never commit passwords, API keys, access tokens, private certificates, connection strings, or real `.env` files. Keep `.env.example` free of real secrets.

The optional OpenAI draft provider reads `OPENAI_API_KEY` only when generation is
explicitly invoked. Ordinary unit tests use deterministic fakes and must not
perform paid API calls. Do not log API keys, provider request headers, or full
ticket bodies.

## Untrusted Input

Treat ticket text, retrieved documents, uploaded files, and user-submitted text as untrusted input.

RAG drafting separates system instructions, incoming ticket text, and retrieved
evidence. Retrieved evidence may contain malicious instructions; prompts and
tests require treating those strings as data and ignoring embedded instructions.

## Logging

Do not log secrets or unnecessary personally identifiable information. Logs
should support debugging without exposing private ticket content, full retrieved
answers, or provider payloads.

## Dependencies

Add new production dependencies only after explaining why they are needed and considering simpler alternatives.

## Operational Boundaries

TicketPilot must not autonomously perform IT actions or send generated responses. Human approval is required.

Generated RAG output is a draft response only. It must not claim that a password
was reset, permissions changed, a refund issued, a ticket closed, or any other
action performed unless a human reviewer has independently verified external
evidence.
