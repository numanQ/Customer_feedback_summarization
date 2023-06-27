"""script for loading configuration."""
import os

PORT = os.environ["PORT"] if os.environ.get("PORT") is not None else 7860

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/cloud-platform.read-only",
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/devstorage.read_only",
    "https://www.googleapis.com/auth/devstorage.read_write"
]

API_BASE_URL = os.getenv("API_BASE_URL", "api/v1")

SERVICE_NAME = os.getenv("SERVICE_NAME", "feedback-summarization")

IS_DEVELOPMENT = bool(os.getenv("IS_DEVELOPMENT", "True").lower() \
    in ("True", "true"))

SUMMARY_MODEL_NAME = "t5-base"

SCORING_MODEL_NAME = "distilbert-base-uncased"

HF_HUB_SUMMARY_MODEL_NAME = "numanBot/customer_feedback_summarization"

HF_HUB_SCORING_MODEL_NAME = "numanBot/summary_annotation_score"

PREFIX = "summarize: "

GENERATION_PARAMS = {"max_length":128, "num_beams":4, "no_repeat_ngram_size":0}
