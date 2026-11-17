"""Environment-driven configuration with zero hard-coded model vendors."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Core OpenAI-compatible model endpoint settings (the ONLY model config required)
    llm_base_url: str = Field(default="http://localhost:8001/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="mock-key-not-secret", alias="LLM_API_KEY")
    llm_model: str = Field(default="mock-vision-strict-v1", alias="LLM_MODEL")

    # Operational LLM controls & cost accounting (USD per 1M tokens)
    llm_cost_input_per_1m: float = Field(default=0.15, alias="LLM_COST_INPUT_PER_1M")
    llm_cost_output_per_1m: float = Field(default=0.60, alias="LLM_COST_OUTPUT_PER_1M")
    llm_max_concurrency: int = Field(default=4, ge=1, le=64, alias="LLM_MAX_CONCURRENCY")
    llm_timeout_seconds: float = Field(default=90.0, gt=0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_repair_attempts: int = Field(default=2, ge=0, le=5, alias="LLM_MAX_REPAIR_ATTEMPTS")
    llm_max_rate_limit_retries: int = Field(default=6, ge=0, le=12, alias="LLM_MAX_RATE_LIMIT_RETRIES")
    llm_rate_limit_base_delay: float = Field(default=0.5, gt=0, alias="LLM_RATE_LIMIT_BASE_DELAY")

    # Extraction & confidence policy
    extraction_mode: Literal["auto", "vision", "ocr_text"] = Field(
        default="auto", alias="EXTRACTION_MODE"
    )
    enable_two_pass: bool = Field(default=True, alias="ENABLE_TWO_PASS")
    review_confidence_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, alias="REVIEW_CONFIDENCE_THRESHOLD"
    )
    max_invoice_age_days: int = Field(default=365, ge=30, alias="MAX_INVOICE_AGE_DAYS")

    # Storage: PostgreSQL + MinIO (with local filesystem fallback for lightweight tests)
    database_url: str = Field(
        default="postgresql+psycopg://ap_user:ap_dev_password_change_me@localhost:5432/doc_intelligence",
        alias="DATABASE_URL",
    )
    minio_endpoint: str = Field(default="localhost:9000", alias="MINIO_ENDPOINT")
    minio_access_key: str = Field(default="minioadmin", alias="MINIO_ACCESS_KEY")
    minio_secret_key: str = Field(default="minioadmin_dev_only", alias="MINIO_SECRET_KEY")
    minio_bucket: str = Field(default="ap-documents", alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")
    local_blob_dir: str | None = Field(default=None, alias="LOCAL_BLOB_DIR")

    # Ingestion, evaluation feedback & downstream exports
    watch_dir: str = Field(default="./data/inbox", alias="WATCH_DIR")
    synthetic_dir: str = Field(default="./data/synthetic", alias="SYNTHETIC_DIR")
    eval_feedback_dir: str = Field(default="./data/eval_feedback", alias="EVAL_FEEDBACK_DIR")
    export_csv_path: str = Field(
        default="./data/exports/approved_payables.csv", alias="EXPORT_CSV_PATH"
    )
    finance_webhook_url: str = Field(
        default="http://localhost:8000/mock-erp/webhooks/payables", alias="FINANCE_WEBHOOK_URL"
    )
    finance_webhook_secret: str = Field(
        default="dev-webhook-hmac-secret-change-me", alias="FINANCE_WEBHOOK_SECRET"
    )
    worker_poll_interval_seconds: float = Field(
        default=0.75, gt=0.05, alias="WORKER_POLL_INTERVAL_SECONDS"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (call get_settings.cache_clear() in tests)."""
    return Settings()
