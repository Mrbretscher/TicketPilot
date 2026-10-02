# TicketPilot Repository Instructions

Follow the global Codex working agreements. This file only records TicketPilot-specific rules.

## Project Scope

- TicketPilot is a human-reviewed IT support copilot.
- The system must not autonomously perform IT actions.
- The system must not automatically send generated responses.
- Low-confidence or weak-evidence cases must support abstention and human review.

## Data And Modeling Rules

- Classifier inputs must never include resolved-answer text, resolution notes, final response text, or any field unavailable before routing.
- Train/test leakage and train/test contamination must be explicitly checked before reporting model results.
- Private employer, university, customer, or support-ticket data must never be used.
- Downloaded datasets, generated model artifacts, vector indexes, caches, and experiment outputs must not be committed.
- Establish a simple scikit-learn baseline before adding more complex text models.

## Retrieval And Generation Rules

- Retrieved evidence must remain structurally separate from model-generated text.
- Generated drafts must preserve source citations back to retrieved evidence.
- Production prompts must live in version-controlled source code.
- Tests must not require paid API calls.

## Verification

- Add or update tests for changed behavior.
- Run the relevant configured checks before reporting work complete.
- Clearly report any checks that could not be run.

## Code Reuse

- when coding inspect other projects in the ai-engineering-portfolio folder for code that can be reused.
- you are not permited to edit files in any folder other than the ticketpilot folder.
