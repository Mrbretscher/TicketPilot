"""Project constants for TicketPilot data acquisition and validation."""

from pathlib import Path

HF_DATASET_REPOSITORY = "Tobi-Bueck/customer-support-tickets"
HF_DATASET_REVISION = "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"
HF_DATASET_FILENAME = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
HF_DATASET_URL = (
    "https://huggingface.co/datasets/"
    f"{HF_DATASET_REPOSITORY}/resolve/{HF_DATASET_REVISION}/{HF_DATASET_FILENAME}"
)
HF_DATASET_SHA256 = "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"

RAW_SOURCE_DATA_PATH = Path("data/raw/customer_support_tickets_source.csv")
RAW_ENGLISH_DATA_PATH = Path("data/raw/customer_support_tickets_en.csv")
PREPARED_DATASET_PATH = Path("reports/dataset_preparation/prepared_dataset.csv")
SPLIT_MANIFEST_PATH = Path("reports/dataset_preparation/split_manifest.csv")
DATASET_SUMMARY_PATH = Path("reports/dataset_preparation/dataset_summary.json")
QUEUE_BASELINE_REPORT_DIR = Path("reports/queue_baseline")
QUEUE_BASELINE_ARTIFACT_DIR = Path("artifacts/queue_baseline")
QUEUE_BASELINE_REPORT_PATH = QUEUE_BASELINE_REPORT_DIR / "queue_baseline_metrics.json"
TENSORFLOW_REPORT_DIR = Path("reports/tensorflow_queue")
TENSORFLOW_ARTIFACT_DIR = Path("artifacts/tensorflow_queue")
TENSORFLOW_REPORT_PATH = TENSORFLOW_REPORT_DIR / "tensorflow_queue_metrics.json"
RETRIEVAL_REPORT_DIR = Path("reports/retrieval")
RETRIEVAL_ARTIFACT_DIR = Path("artifacts/retrieval")
RETRIEVAL_REPORT_PATH = RETRIEVAL_REPORT_DIR / "lexical_retrieval_metrics.json"
RETRIEVAL_INDEX_PATH = RETRIEVAL_ARTIFACT_DIR / "tfidf_ticket_retriever.joblib"
RETRIEVAL_LABEL_TEMPLATE_PATH = RETRIEVAL_REPORT_DIR / "manual_relevance_template.csv"
SEMANTIC_RETRIEVAL_REPORT_DIR = Path("reports/semantic_retrieval")
SEMANTIC_RETRIEVAL_ARTIFACT_DIR = Path("artifacts/semantic_retrieval")
SEMANTIC_RETRIEVAL_REPORT_PATH = (
    SEMANTIC_RETRIEVAL_REPORT_DIR / "semantic_retrieval_metrics.json"
)
SEMANTIC_RETRIEVAL_EMBEDDINGS_PATH = (
    SEMANTIC_RETRIEVAL_ARTIFACT_DIR / "corpus_embeddings.npz"
)
SEMANTIC_RETRIEVAL_METADATA_PATH = (
    SEMANTIC_RETRIEVAL_ARTIFACT_DIR / "corpus_metadata.csv"
)
SEMANTIC_RETRIEVAL_CONFIG_PATH = (
    SEMANTIC_RETRIEVAL_ARTIFACT_DIR / "retriever_config.json"
)
SEMANTIC_RETRIEVAL_LABEL_TEMPLATE_PATH = (
    SEMANTIC_RETRIEVAL_REPORT_DIR / "manual_relevance_template.csv"
)

EXPECTED_COLUMNS = (
    "subject",
    "body",
    "answer",
    "type",
    "queue",
    "priority",
    "language",
    "version",
    "tag_1",
    "tag_2",
    "tag_3",
    "tag_4",
    "tag_5",
    "tag_6",
    "tag_7",
    "tag_8",
)
TAG_COLUMNS = tuple(column for column in EXPECTED_COLUMNS if column.startswith("tag_"))

CLASSIFIER_INPUT_COLUMNS = ("subject", "body")
CLASSIFIER_LABEL_COLUMNS = ("queue", "priority")
PROHIBITED_CLASSIFIER_INPUT_COLUMNS = tuple(
    column for column in EXPECTED_COLUMNS if column not in CLASSIFIER_INPUT_COLUMNS
)

LANGUAGE_VALUES = ("de", "en")
RECRUITER_READY_LANGUAGE = "en"
QUEUE_VALUES = (
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
)
PRIORITY_VALUES = ("high", "low", "medium")
TYPE_VALUES = ("Change", "Incident", "Problem", "Request")

MIN_SOURCE_ROWS = 20_000
MIN_ENGLISH_ROWS = 10_000

SPLIT_RANDOM_SEED = 20261005
TRAIN_SPLIT_FRACTION = 0.70
VALIDATION_SPLIT_FRACTION = 0.15
TEST_SPLIT_FRACTION = 0.15

ABSTENTION_MIN_VALIDATION_COVERAGE = 0.70

SEMANTIC_RETRIEVAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_RETRIEVAL_BATCH_SIZE = 64
SEMANTIC_RETRIEVAL_NORMALIZE_EMBEDDINGS = True
HYBRID_SEMANTIC_WEIGHT = 0.50

OPENAI_DRAFT_MODEL_ENV_VAR = "OPENAI_DRAFT_MODEL"
OPENAI_DRAFT_MODEL_DEFAULT = "gpt-5-mini"
DRAFT_MIN_EVIDENCE_SCORE = 0.50
DRAFT_MIN_CLASSIFIER_CONFIDENCE = 0.30
DRAFT_MAX_TICKET_CHARS = 4_000
DRAFT_MAX_EVIDENCE_FIELD_CHARS = 1_500
REVIEW_DATABASE_PATH = Path("artifacts/review/reviews.sqlite")

TENSORFLOW_RANDOM_SEED = 20261005
TENSORFLOW_MAX_TOKENS = 20_000
TENSORFLOW_SEQUENCE_LENGTH = 160
TENSORFLOW_EMBEDDING_DIM = 64
TENSORFLOW_CONV_FILTERS = 96
TENSORFLOW_BATCH_SIZE = 128
TENSORFLOW_MAX_EPOCHS = 8
TENSORFLOW_EARLY_STOPPING_PATIENCE = 2
