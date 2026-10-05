"""scikit-learn model definitions for TicketPilot queue routing."""

from typing import Literal

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ticketpilot.config import SPLIT_RANDOM_SEED

QueueModelName = Literal[
    "dummy_most_frequent",
    "tfidf_logistic_regression",
    "tfidf_linear_svc",
]


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Build the shared text vectorizer with conservative normalization."""
    return TfidfVectorizer(
        lowercase=True,
        min_df=2,
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[\w][\w.+#/-]*\b",
    )


def build_dummy_queue_pipeline(
    *,
    random_state: int = SPLIT_RANDOM_SEED,
) -> Pipeline:
    """Build the non-informative queue-routing reference model."""
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_logistic_regression_queue_pipeline(
    *,
    random_state: int = SPLIT_RANDOM_SEED,
) -> Pipeline:
    """Build the TF-IDF + logistic-regression queue baseline."""
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_linear_svc_queue_pipeline(
    *,
    random_state: int = SPLIT_RANDOM_SEED,
) -> Pipeline:
    """Build the TF-IDF + LinearSVC queue baseline."""
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    dual="auto",
                    max_iter=5_000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_queue_baseline_pipelines(
    *,
    random_state: int = SPLIT_RANDOM_SEED,
) -> dict[QueueModelName, Pipeline]:
    """Return all fixed Milestone 3 queue-routing baselines."""
    return {
        "dummy_most_frequent": build_dummy_queue_pipeline(random_state=random_state),
        "tfidf_logistic_regression": build_logistic_regression_queue_pipeline(
            random_state=random_state
        ),
        "tfidf_linear_svc": build_linear_svc_queue_pipeline(random_state=random_state),
    }


def build_queue_baseline_pipeline(
    model_name: QueueModelName,
    *,
    random_state: int = SPLIT_RANDOM_SEED,
) -> Pipeline:
    """Build one fixed queue-routing baseline by name."""
    return build_queue_baseline_pipelines(random_state=random_state)[model_name]
