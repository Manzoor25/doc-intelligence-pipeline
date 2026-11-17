"""SQLAlchemy 2.0 ORM models for documents, queue jobs, caching, metrics, audit, and ERP exports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SupplierMaster(Base):
    """Master vendor directory used by deterministic AP supplier matching."""

    __tablename__ = "suppliers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    tax_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    address: Mapped[str] = mapped_column(String(300), nullable=False)
    default_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class DocumentRecord(Base):
    """Primary lifecycle record for every ingested AP document."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_supplier_docnum", "matched_supplier_id", "document_number_norm"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_channel: Mapped[str] = mapped_column(String(40), nullable=False, default="upload")

    raw_object_key: Mapped[str] = mapped_column(String(300), nullable=False)
    page_object_keys: Mapped[list[str]] = mapped_column(JSON, default=list)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued", index=True)
    document_type: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    extraction_path: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(120), nullable=True)

    ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    pass1_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    pass2_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    validation_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    confidence_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    safety_report: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    overall_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_supplier_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    document_number_norm: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    total_amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    issue_date_iso: Mapped[str | None] = mapped_column(String(16), nullable=True)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    repair_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    processing_latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    reviewed_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    exported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class ProcessingJob(Base):
    """Durable PostgreSQL job queue row claimed with SELECT ... FOR UPDATE SKIP LOCKED."""

    __tablename__ = "processing_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=8, nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )
    locked_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class LLMCacheEntry(Base):
    """Content-addressed LLM completion cache keyed by request digest + model."""

    __tablename__ = "llm_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class LLMCallMetric(Base):
    """Per-request LLM operational telemetry (tokens, USD cost, latency, 429 retries)."""

    __tablename__ = "llm_call_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    call_purpose: Mapped[str] = mapped_column(String(40), nullable=False)
    model: Mapped[str] = mapped_column(String(120), nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    rate_limit_retries: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    succeeded: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    error_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class AuditLogEntry(Base):
    """Immutable append-only audit trail for security, safety, review diffs, and exports."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(120), nullable=False, default="system")
    summary: Mapped[str] = mapped_column(String(500), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ApprovedPayable(Base):
    """Normalized PostgreSQL ledger table exported for approved AP documents."""

    __tablename__ = "approved_payables"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    document_type: Mapped[str] = mapped_column(String(32), nullable=False)
    document_number: Mapped[str] = mapped_column(String(80), nullable=False)
    supplier_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    supplier_name: Mapped[str] = mapped_column(String(200), nullable=False)
    supplier_tax_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issue_date: Mapped[str] = mapped_column(String(16), nullable=False)
    due_date: Mapped[str | None] = mapped_column(String(16), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    subtotal: Mapped[float] = mapped_column(Float, nullable=False)
    tax_amount: Mapped[float] = mapped_column(Float, nullable=False)
    total_amount: Mapped[float] = mapped_column(Float, nullable=False)
    line_items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    approval_source: Mapped[str] = mapped_column(String(32), nullable=False)
    approved_by: Mapped[str] = mapped_column(String(120), nullable=False)
    webhook_delivered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    exported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class EvaluationFeedbackSample(Base):
    """Human-corrected extractions saved as gold-standard evaluation samples."""

    __tablename__ = "evaluation_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    document_type: Mapped[str] = mapped_column(String(32), nullable=False)
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    ground_truth: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    reviewer: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class FinanceWebhookReceipt(Base):
    """Mock downstream Finance ERP ledger storing verified HMAC webhook deliveries."""

    __tablename__ = "finance_webhook_receipts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    document_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    signature_valid: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

