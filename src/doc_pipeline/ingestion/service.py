"""Unified ingestion, execution, and human-in-the-loop correction service."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_pipeline.config import Settings
from doc_pipeline.exports.exporter import export_approved_document
from doc_pipeline.extraction.extractor import run_extraction_pipeline
from doc_pipeline.llm.provider import OpenAICompatibleProvider
from doc_pipeline.ocr.renderer import (
    detect_mime_type,
    extract_embedded_pdf_text,
    render_document_to_png_pages,
)
from doc_pipeline.schemas import ReviewSubmissionRequest
from doc_pipeline.storage.blob import BlobStore
from doc_pipeline.storage.db import append_audit_log
from doc_pipeline.storage.models import (
    DocumentRecord,
    EvaluationFeedbackSample,
    ProcessingJob,
)
from doc_pipeline.validation.rules import normalize_doc_number, validate_document


def ingest_document_bytes(
    *,
    file_name: str,
    raw_bytes: bytes,
    source_channel: str,
    session: Session,
    blob_store: BlobStore,
    force_reprocess: bool = False,
) -> tuple[DocumentRecord, bool]:
    """Ingest a document idempotently keyed by SHA-256 hash, render page PNGs, and enqueue job.

    Returns `(doc_record, newly_created)`.
    """
    if not raw_bytes:
        raise ValueError("Uploaded file is empty")

    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    existing = session.scalar(
        select(DocumentRecord).where(DocumentRecord.file_sha256 == sha256_hex)
    )
    if existing is not None and not force_reprocess:
        append_audit_log(
            session,
            document_id=existing.id,
            event_type="DUPLICATE_HASH_SKIPPED",
            actor=source_channel,
            summary=f"Idempotent hit on SHA-256 {sha256_hex[:12]}...; returning existing record ({existing.status}).",
            payload={"file_name": file_name, "source_channel": source_channel},
        )
        session.flush()
        return existing, False

    mime_type = detect_mime_type(file_name, raw_bytes)
    pages_png, _ = render_document_to_png_pages(raw_bytes, mime_type=mime_type, dpi=200)
    if not pages_png:
        raise ValueError(f"Could not render any pages from {file_name}")

    doc_id = existing.id if existing is not None else str(uuid.uuid4())
    safe_name = Path(file_name).name or f"{doc_id}.bin"
    raw_key = f"raw/{sha256_hex}/{safe_name}"
    blob_store.put_bytes(raw_key, raw_bytes, content_type=mime_type)

    page_keys: list[str] = []
    for idx, png_bytes in enumerate(pages_png, start=1):
        p_key = f"pages/{doc_id}/page_{idx:03d}.png"
        blob_store.put_bytes(p_key, png_bytes, content_type="image/png")
        page_keys.append(p_key)

    if existing is None:
        doc = DocumentRecord(
            id=doc_id,
            file_name=safe_name,
            file_sha256=sha256_hex,
            mime_type=mime_type,
            source_channel=source_channel,
            raw_object_key=raw_key,
            page_object_keys=page_keys,
            page_count=len(page_keys),
            status="queued",
        )
        session.add(doc)
        session.flush()
        job = ProcessingJob(
            document_id=doc.id,
            status="pending",
            attempts=0,
            next_attempt_at=datetime.now(UTC),
        )
        session.add(job)
    else:
        doc = existing
        doc.status = "queued"
        doc.page_object_keys = page_keys
        doc.page_count = len(page_keys)
        doc.error_message = None
        job = session.scalar(
            select(ProcessingJob).where(ProcessingJob.document_id == doc.id)
        )
        if job is None:
            session.add(
                ProcessingJob(
                    document_id=doc.id,
                    status="pending",
                    attempts=0,
                    next_attempt_at=datetime.now(UTC),
                )
            )
        else:
            job.status = "pending"
            job.attempts = 0
            job.next_attempt_at = datetime.now(UTC)
            job.locked_by = None
            job.locked_until = None

    append_audit_log(
        session,
        document_id=doc.id,
        event_type="DOCUMENT_INGESTED",
        actor=source_channel,
        summary=f"Ingested '{safe_name}' ({len(page_keys)} page(s), sha256={sha256_hex[:12]}...) via {source_channel}.",
        payload={
            "file_name": safe_name,
            "mime_type": mime_type,
            "page_count": len(page_keys),
            "source_channel": source_channel,
        },
    )
    session.flush()
    return doc, True


async def process_document_record(
    *,
    document_id: str,
    session: Session,
    blob_store: BlobStore,
    provider: OpenAICompatibleProvider,
    settings: Settings,
    reference_date: date | None = None,
    use_cache: bool = True,
    webhook_client: httpx.AsyncClient | None = None,
) -> DocumentRecord:
    """Execute full extraction, validation, confidence scoring, safety audit, and auto-export."""
    doc = session.get(DocumentRecord, document_id)
    if doc is None:
        raise KeyError(f"Document {document_id} not found")

    t0 = time.perf_counter()
    doc.status = "processing"
    # Commit before async LLM network calls so no write transaction is held across I/O
    session.commit()

    raw_bytes = blob_store.get_bytes(doc.raw_object_key)
    pages_png = [blob_store.get_bytes(k) for k in doc.page_object_keys]
    embedded_pdf_text = extract_embedded_pdf_text(raw_bytes, doc.mime_type)

    result = await run_extraction_pipeline(
        document_id=doc.id,
        pages_png=pages_png,
        embedded_pdf_text=embedded_pdf_text,
        provider=provider,
        session=session,
        settings=settings,
        reference_date=reference_date,
        use_cache=use_cache,
    )

    wall_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    doc.status = result.status
    doc.document_type = result.document_type
    doc.language = result.language
    doc.extraction_path = result.extraction_path
    doc.model_used = settings.llm_model
    doc.ocr_text = result.ocr_text
    doc.extracted_data = result.extracted.model_dump(mode="json") if result.extracted else None
    doc.pass1_data = result.pass1.model_dump(mode="json") if result.pass1 else None
    doc.pass2_data = result.pass2.model_dump(mode="json") if result.pass2 else None
    doc.validation_report = (
        result.validation.model_dump(mode="json") if result.validation else None
    )
    doc.confidence_report = (
        result.confidence.model_dump(mode="json") if result.confidence else None
    )
    doc.safety_report = result.safety.model_dump(mode="json")
    doc.overall_confidence = (
        result.confidence.overall_confidence if result.confidence else 0.0
    )
    doc.matched_supplier_id = (
        result.validation.matched_supplier_id if result.validation else None
    )
    doc.document_number_norm = (
        normalize_doc_number(result.extracted.document_number) if result.extracted else None
    )
    doc.total_amount = result.extracted.total_amount if result.extracted else None
    doc.currency = result.extracted.currency if result.extracted else None
    doc.issue_date_iso = (
        result.extracted.issue_date.isoformat() if result.extracted else None
    )

    doc.prompt_tokens = result.telemetry.prompt_tokens
    doc.completion_tokens = result.telemetry.completion_tokens
    doc.cache_hits = result.telemetry.cache_hits
    doc.repair_attempts = result.telemetry.repair_attempts
    doc.estimated_cost_usd = result.telemetry.cost_usd
    doc.processing_latency_ms = wall_ms
    doc.error_message = result.error_message
    session.flush()

    if result.safety.injection_detected:
        append_audit_log(
            session,
            document_id=doc.id,
            event_type="PROMPT_INJECTION_DETECTED",
            actor="safety_scanner",
            summary=(
                f"Quarantined document due to {len(result.safety.signals)} embedded "
                "prompt-injection signal(s)."
            ),
            payload={
                "signals": [s.model_dump(mode="json") for s in result.safety.signals],
            },
        )

    if doc.status == "approved":
        append_audit_log(
            session,
            document_id=doc.id,
            event_type="AUTO_APPROVED",
            actor="confidence_engine",
            summary=(
                f"Auto-approved ({doc.document_type}, confidence={doc.overall_confidence:.2f}, "
                f"path={doc.extraction_path}, cost=${doc.estimated_cost_usd:.5f})."
            ),
            payload={
                "overall_confidence": doc.overall_confidence,
                "extraction_path": doc.extraction_path,
                "latency_ms": wall_ms,
            },
        )
        await export_approved_document(
            doc=doc,
            session=session,
            settings=settings,
            approval_source="auto",
            approved_by="system:auto-confidence",
            http_client=webhook_client,
        )
    else:
        reasons = (
            result.confidence.review_reasons
            if result.confidence
            else [result.error_message or "Extraction failed"]
        )
        append_audit_log(
            session,
            document_id=doc.id,
            event_type="ROUTED_TO_REVIEW",
            actor="confidence_engine",
            summary=f"Routed to human review ({len(reasons)} reason(s)).",
            payload={
                "overall_confidence": doc.overall_confidence,
                "reasons": reasons,
                "repair_attempts": doc.repair_attempts,
            },
        )

    job = session.scalar(select(ProcessingJob).where(ProcessingJob.document_id == doc.id))
    if job is not None:
        job.status = "completed"
        job.locked_by = None
        job.locked_until = None
    session.flush()
    return doc


def _compute_field_diffs(before: dict[str, Any] | None, after: dict[str, Any]) -> dict[str, Any]:
    if not before:
        return {"_initial_manual_entry": {"before": None, "after": "populated"}}
    diffs: dict[str, Any] = {}
    for key, after_val in after.items():
        before_val = before.get(key)
        if before_val != after_val:
            diffs[key] = {"before": before_val, "after": after_val}
    return diffs


async def apply_human_review(
    *,
    document_id: str,
    submission: ReviewSubmissionRequest,
    session: Session,
    settings: Settings,
    blob_store: BlobStore | None = None,
    reference_date: date | None = None,
    webhook_client: httpx.AsyncClient | None = None,
) -> DocumentRecord:
    """Persist human corrections, promote verified gold sample to eval data, and export if approved."""
    doc = session.get(DocumentRecord, document_id)
    if doc is None:
        raise KeyError(f"Document {document_id} not found")

    before_data = doc.extracted_data
    corrected_dict = submission.extracted_data.model_dump(mode="json")
    diffs = _compute_field_diffs(before_data, corrected_dict)

    revalidated = validate_document(
        submission.extracted_data,
        session,
        current_document_id=doc.id,
        reference_date=reference_date,
        max_age_days=settings.max_invoice_age_days,
    )

    now = datetime.now(UTC)
    doc.extracted_data = corrected_dict
    doc.document_type = submission.extracted_data.document_type
    doc.language = submission.extracted_data.language
    doc.document_number_norm = normalize_doc_number(submission.extracted_data.document_number)
    doc.total_amount = submission.extracted_data.total_amount
    doc.currency = submission.extracted_data.currency
    doc.issue_date_iso = submission.extracted_data.issue_date.isoformat()
    doc.matched_supplier_id = revalidated.matched_supplier_id
    doc.validation_report = revalidated.model_dump(mode="json")
    doc.reviewed_by = submission.reviewer
    doc.reviewed_at = now
    doc.reviewer_notes = submission.reviewer_notes
    doc.status = "approved" if submission.action == "approve" else "rejected"
    session.flush()

    append_audit_log(
        session,
        document_id=doc.id,
        event_type="HUMAN_REVIEW_SAVED",
        actor=submission.reviewer,
        summary=(
            f"Reviewer {submission.action}d document with {len(diffs)} corrected field(s)."
        ),
        payload={
            "action": submission.action,
            "changed_fields": list(diffs.keys()),
            "diffs": diffs,
            "reviewer_notes": submission.reviewer_notes,
        },
    )

    # Promote human-verified document to evaluation dataset (DB + JSON file on disk)
    if submission.action == "approve":
        existing_sample = session.scalar(
            select(EvaluationFeedbackSample).where(
                EvaluationFeedbackSample.document_id == doc.id
            )
        )
        if existing_sample is None:
            session.add(
                EvaluationFeedbackSample(
                    document_id=doc.id,
                    file_sha256=doc.file_sha256,
                    document_type=submission.extracted_data.document_type,
                    language=submission.extracted_data.language,
                    ground_truth=corrected_dict,
                    reviewer=submission.reviewer,
                )
            )
        else:
            existing_sample.ground_truth = corrected_dict
            existing_sample.reviewer = submission.reviewer

        feedback_dir = Path(settings.eval_feedback_dir)
        files_dir = feedback_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        saved_file_path: str | None = None
        if blob_store is not None:
            raw_copy_path = files_dir / f"{doc.id}_{doc.file_name}"
            raw_copy_path.write_bytes(blob_store.get_bytes(doc.raw_object_key))
            saved_file_path = str(raw_copy_path)

        feedback_payload = {
            "document_id": doc.id,
            "file_name": doc.file_name,
            "file_path": saved_file_path,
            "sha256": doc.file_sha256,
            "raw_object_key": doc.raw_object_key,
            "reviewer": submission.reviewer,
            "reviewed_at": now.isoformat(),
            "changed_fields": list(diffs.keys()),
            "ground_truth": corrected_dict,
        }
        (feedback_dir / f"{doc.id}.json").write_text(
            json.dumps(feedback_payload, indent=2), encoding="utf-8"
        )

        await export_approved_document(
            doc=doc,
            session=session,
            settings=settings,
            approval_source="human_review",
            approved_by=submission.reviewer,
            http_client=webhook_client,
        )

    session.flush()
    return doc
