# Roadmap

## Milestone 0: Initial Scaffold

- Create repository documentation.
- Configure Python packaging and checks.
- Add package-import smoke test.

## Milestone 1: Data Contract And Synthetic Seed Data

- Define ticket schema.
- Add validation utilities.
- Add small synthetic smoke-test data.
- Document dataset source and license requirements.

## Milestone 2: scikit-learn Baseline

- Implement TF-IDF preprocessing.
- Train baseline queue and priority classifiers.
- Add duplicate-aware leakage checks.
- Report baseline metrics.

## Milestone 3: Confidence And Abstention

- Add confidence thresholds.
- Add calibration analysis where appropriate.
- Route uncertain predictions to human review.

## Milestone 4: Retrieval

- Implement similar-ticket retrieval.
- Evaluate retrieval quality.
- Keep retrieved evidence separate from generated text.

## Milestone 5: RAG Drafting

- Add version-controlled prompts.
- Generate cited draft responses.
- Evaluate citation preservation and groundedness.
- Keep all generation tests free of paid API requirements.

## Milestone 6: TensorFlow Text Model

- Add TensorFlow model after the baseline is complete.
- Compare against the baseline with the same splits and metrics.

## Milestone 7: Application Layer

- Add FastAPI inference surface.
- Add Streamlit human-review interface.
- Validate API inputs before model use.

## Milestone 8: Reproducibility And Deployment

- Add Docker.
- Add CI.
- Add reproducible training and evaluation commands.
- Document limitations and operational risks.
