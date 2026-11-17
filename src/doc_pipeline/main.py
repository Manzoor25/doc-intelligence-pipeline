"""FastAPI application: ingestion API, split-pane human Review UI, exports, and mock ERP receiver."""

from __future__ import annotations

import base64
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from doc_pipeline.config import Settings, get_settings
from doc_pipeline.exports.exporter import render_payables_csv, sign_webhook_payload
from doc_pipeline.ingestion.service import (
    apply_human_review,
    ingest_document_bytes,
    process_document_record,
)
from doc_pipeline.llm.provider import OpenAICompatibleProvider
from doc_pipeline.schemas import ReviewSubmissionRequest
from doc_pipeline.storage.blob import get_blob_store
from doc_pipeline.storage.db import init_db, make_session_factory
from doc_pipeline.storage.models import (
    ApprovedPayable,
    AuditLogEntry,
    DocumentRecord,
    EvaluationFeedbackSample,
    FinanceWebhookReceipt,
    ProcessingJob,
)
from doc_pipeline.worker.runner import scan_watched_folder_once

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
TEMPLATES = Jinja2Templates(directory=str(WEB_DIR / "templates"))


class ApiIngestRequest(BaseModel):
    file_name: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=4)
    sync: bool = False
    force_reprocess: bool = False


def _serialize_doc_summary(doc: DocumentRecord) -> dict[str, Any]:
    conf = doc.confidence_report or {}
    safety = doc.safety_report or {}
    extracted = doc.extracted_data or {}
    return {
        "id": doc.id,
        "file_name": doc.file_name,
        "file_sha256": doc.file_sha256,
        "mime_type": doc.mime_type,
        "source_channel": doc.source_channel,
        "page_count": doc.page_count,
        "status": doc.status,
        "document_type": doc.document_type,
        "language": doc.language,
        "extraction_path": doc.extraction_path,
        "model_used": doc.model_used,
        "document_number": extracted.get("document_number"),
        "supplier_name": extracted.get("supplier_name"),
        "matched_supplier_id": doc.matched_supplier_id,
        "total_amount": doc.total_amount,
        "currency": doc.currency,
        "issue_date": doc.issue_date_iso,
        "overall_confidence": doc.overall_confidence,
        "review_reasons": conf.get("review_reasons", []),
        "injection_detected": bool(safety.get("injection_detected", False)),
        "repair_attempts": doc.repair_attempts,
        "estimated_cost_usd": doc.estimated_cost_usd,
        "processing_latency_ms": doc.processing_latency_ms,
        "reviewed_by": doc.reviewed_by,
        "reviewed_at": doc.reviewed_at.isoformat() if doc.reviewed_at else None,
        "exported_at": doc.exported_at.isoformat() if doc.exported_at else None,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


def _serialize_doc_detail(doc: DocumentRecord, audits: list[AuditLogEntry]) -> dict[str, Any]:
    base = _serialize_doc_summary(doc)
    base.update(
        {
            "extracted_data": doc.extracted_data,
            "pass1_data": doc.pass1_data,
            "pass2_data": doc.pass2_data,
            "validation_report": doc.validation_report,
            "confidence_report": doc.confidence_report,
            "safety_report": doc.safety_report,
            "ocr_text": doc.ocr_text,
            "prompt_tokens": doc.prompt_tokens,
            "completion_tokens": doc.completion_tokens,
            "cache_hits": doc.cache_hits,
            "reviewer_notes": doc.reviewer_notes,
            "error_message": doc.error_message,
            "audit_logs": [
                {
                    "id": a.id,
                    "event_type": a.event_type,
                    "actor": a.actor,
                    "summary": a.summary,
                    "payload": a.payload,
                    "created_at": a.created_at.isoformat(),
                }
                for a in audits
            ],
        }
    )
    return base


def create_app(
    settings: Settings | None = None,
    provider: OpenAICompatibleProvider | None = None,
    webhook_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        init_db(cfg.database_url)
        session_factory = make_session_factory(cfg.database_url)
        blob_store = get_blob_store(cfg)
        llm_provider = provider or OpenAICompatibleProvider(
            settings=cfg, session_factory=session_factory
        )

        app_instance.state.settings = cfg
        app_instance.state.session_factory = session_factory
        app_instance.state.blob_store = blob_store
        app_instance.state.provider = llm_provider
        app_instance.state.webhook_client = webhook_client

        # Startup probe: inspect vision + json_schema support and print result
        await llm_provider.probe_capabilities()
        yield
        if provider is None:
            await llm_provider.aclose()

    fastapi_app = FastAPI(
        title="AP Document Intelligence Pipeline",
        description=(
            "Vendor-neutral document intelligence pipeline with startup capability probing, "
            "two-pass extraction, deterministic AP validation, prompt-injection quarantine, "
            "and human-in-the-loop review."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    fastapi_app.mount(
        "/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static"
    )

    @fastapi_app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def review_console(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(request, "index.html", {})

    @fastapi_app.get("/api/v1/health")
    @fastapi_app.get("/healthz")
    async def healthz(request: Request) -> dict[str, Any]:
        sf = request.app.state.session_factory
        prov: OpenAICompatibleProvider = request.app.state.provider
        with sf() as session:
            pending_jobs = (
                session.scalar(
                    select(func.count(ProcessingJob.id)).where(
                        ProcessingJob.status.in_(["pending", "running"])
                    )
                )
                or 0
            )
        caps = prov.capabilities
        return {
            "status": "ok",
            "pending_jobs": pending_jobs,
            "capabilities": caps.model_dump(mode="json") if caps else None,
        }

    @fastapi_app.get("/api/v1/capabilities")
    async def get_capabilities(request: Request) -> dict[str, Any]:
        prov: OpenAICompatibleProvider = request.app.state.provider
        caps = await prov.probe_capabilities()
        return caps.model_dump(mode="json")

    @fastapi_app.post("/api/v1/documents/upload")
    async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        sync: bool = Form(default=False),
        force_reprocess: bool = Form(default=False),
    ) -> dict[str, Any]:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        sf = request.app.state.session_factory
        blob_store = request.app.state.blob_store
        prov: OpenAICompatibleProvider = request.app.state.provider
        app_cfg: Settings = request.app.state.settings
        wh_client = request.app.state.webhook_client

        with sf() as session:
            try:
                doc, created = ingest_document_bytes(
                    file_name=file.filename or "upload.pdf",
                    raw_bytes=raw_bytes,
                    source_channel="upload",
                    session=session,
                    blob_store=blob_store,
                    force_reprocess=force_reprocess,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Could not ingest document: {exc}"
                ) from exc

            if sync and (created or doc.status in ("queued", "failed")):
                doc = await process_document_record(
                    document_id=doc.id,
                    session=session,
                    blob_store=blob_store,
                    provider=prov,
                    settings=app_cfg,
                    webhook_client=wh_client,
                )
            session.commit()
            return {
                "created": created,
                "idempotent_hit": not created,
                "document": _serialize_doc_summary(doc),
            }

    @fastapi_app.post("/api/v1/documents/ingest")
    async def ingest_via_api(
        request: Request,
        payload: ApiIngestRequest,
    ) -> dict[str, Any]:
        try:
            raw_bytes = base64.b64decode(payload.content_base64, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc

        sf = request.app.state.session_factory
        blob_store = request.app.state.blob_store
        prov: OpenAICompatibleProvider = request.app.state.provider
        app_cfg: Settings = request.app.state.settings
        wh_client = request.app.state.webhook_client

        with sf() as session:
            try:
                doc, created = ingest_document_bytes(
                    file_name=payload.file_name,
                    raw_bytes=raw_bytes,
                    source_channel="api",
                    session=session,
                    blob_store=blob_store,
                    force_reprocess=payload.force_reprocess,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400, detail=f"Could not ingest document: {exc}"
                ) from exc

            if payload.sync and (created or doc.status in ("queued", "failed")):
                doc = await process_document_record(
                    document_id=doc.id,
                    session=session,
                    blob_store=blob_store,
                    provider=prov,
                    settings=app_cfg,
                    webhook_client=wh_client,
                )
            session.commit()
            return {
                "created": created,
                "idempotent_hit": not created,
                "document": _serialize_doc_summary(doc),
            }

    @fastapi_app.post("/api/v1/documents/scan-inbox")
    async def trigger_inbox_scan(request: Request) -> dict[str, Any]:
        sf = request.app.state.session_factory
        blob_store = request.app.state.blob_store
        app_cfg: Settings = request.app.state.settings
        count = scan_watched_folder_once(app_cfg.watch_dir, sf, blob_store)
        return {"ingested_count": count, "watch_dir": app_cfg.watch_dir}

    @fastapi_app.get("/api/v1/documents")
    async def list_documents(
        request: Request,
        status: str | None = Query(default=None),
        document_type: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        sf = request.app.state.session_factory
        with sf() as session:
            stmt = select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
            if status:
                stmt = stmt.where(DocumentRecord.status == status)
            if document_type:
                stmt = stmt.where(DocumentRecord.document_type == document_type)
            docs = session.scalars(stmt.limit(limit).offset(offset)).all()

            all_docs = session.scalars(select(DocumentRecord)).all()
            webhooks_count = (
                session.scalar(
                    select(func.count(ApprovedPayable.id)).where(
                        ApprovedPayable.webhook_delivered.is_(True)
                    )
                )
                or 0
            )
            feedback_count = (
                session.scalar(select(func.count(EvaluationFeedbackSample.id))) or 0
            )

            status_counts = {
                "queued": 0,
                "processing": 0,
                "approved": 0,
                "needs_review": 0,
                "rejected": 0,
            }
            total_cost = 0.0
            latencies: list[float] = []
            human_reviewed = 0
            quarantined = 0

            for d in all_docs:
                status_counts[d.status] = status_counts.get(d.status, 0) + 1
                total_cost += d.estimated_cost_usd or 0.0
                if d.processing_latency_ms and d.processing_latency_ms > 0:
                    latencies.append(d.processing_latency_ms)
                if d.reviewed_by:
                    human_reviewed += 1
                if (d.safety_report or {}).get("injection_detected"):
                    quarantined += 1

            n_total = len(all_docs)
            summary = {
                "total_documents": n_total,
                **status_counts,
                "human_reviewed": human_reviewed,
                "quarantined_injections": quarantined,
                "total_cost_usd": round(total_cost, 6),
                "avg_cost_usd": round(total_cost / n_total, 6) if n_total else 0.0,
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
                "webhooks_delivered": webhooks_count,
                "eval_feedback_samples": feedback_count,
            }
            return {
                "summary": summary,
                "documents": [_serialize_doc_summary(d) for d in docs],
            }

    @fastapi_app.get("/api/v1/documents/{document_id}")
    async def get_document(request: Request, document_id: str) -> dict[str, Any]:
        sf = request.app.state.session_factory
        with sf() as session:
            doc = session.get(DocumentRecord, document_id)
            if doc is None:
                raise HTTPException(status_code=404, detail="Document not found")
            audits = session.scalars(
                select(AuditLogEntry)
                .where(AuditLogEntry.document_id == document_id)
                .order_by(AuditLogEntry.created_at.asc(), AuditLogEntry.id.asc())
            ).all()
            return _serialize_doc_detail(doc, list(audits))

    @fastapi_app.get("/api/v1/documents/{document_id}/pages/{page_num}")
    async def get_document_page_image(
        request: Request, document_id: str, page_num: int
    ) -> Response:
        sf = request.app.state.session_factory
        blob_store = request.app.state.blob_store
        with sf() as session:
            doc = session.get(DocumentRecord, document_id)
            if doc is None:
                raise HTTPException(status_code=404, detail="Document not found")
            keys = doc.page_object_keys or []
            if page_num < 1 or page_num > len(keys):
                raise HTTPException(status_code=404, detail="Page out of range")
            png_bytes = blob_store.get_bytes(keys[page_num - 1])
        return Response(content=png_bytes, media_type="image/png")

    @fastapi_app.get("/api/v1/documents/{document_id}/raw")
    async def get_document_raw_file(request: Request, document_id: str) -> Response:
        sf = request.app.state.session_factory
        blob_store = request.app.state.blob_store
        with sf() as session:
            doc = session.get(DocumentRecord, document_id)
            if doc is None:
                raise HTTPException(status_code=404, detail="Document not found")
            raw_bytes = blob_store.get_bytes(doc.raw_object_key)
            mime = doc.mime_type
            fname = doc.file_name
        return Response(
            content=raw_bytes,
            media_type=mime,
            headers={"Content-Disposition": f'inline; filename="{fname}"'},
        )

    @fastapi_app.post("/api/v1/documents/{document_id}/review")
    async def submit_human_review(
        request: Request,
        document_id: str,
        submission: ReviewSubmissionRequest,
    ) -> dict[str, Any]:
        sf = request.app.state.session_factory
        app_cfg: Settings = request.app.state.settings
        wh_client = request.app.state.webhook_client
        with sf() as session:
            if session.get(DocumentRecord, document_id) is None:
                raise HTTPException(status_code=404, detail="Document not found")
            doc = await apply_human_review(
                document_id=document_id,
                submission=submission,
                session=session,
                settings=app_cfg,
                webhook_client=wh_client,
            )
            session.commit()
            audits = session.scalars(
                select(AuditLogEntry)
                .where(AuditLogEntry.document_id == document_id)
                .order_by(AuditLogEntry.created_at.asc(), AuditLogEntry.id.asc())
            ).all()
            return {
                "status": doc.status,
                "promoted_to_eval_dataset": submission.action == "approve",
                "document": _serialize_doc_detail(doc, list(audits)),
            }

    @fastapi_app.get("/api/v1/exports/payables")
    async def list_exported_payables(request: Request) -> dict[str, Any]:
        sf = request.app.state.session_factory
        with sf() as session:
            rows = session.scalars(
                select(ApprovedPayable).order_by(ApprovedPayable.exported_at.desc())
            ).all()
            return {
                "count": len(rows),
                "payables": [
                    {
                        "document_id": r.document_id,
                        "document_type": r.document_type,
                        "document_number": r.document_number,
                        "supplier_id": r.supplier_id,
                        "supplier_name": r.supplier_name,
                        "supplier_tax_id": r.supplier_tax_id,
                        "issue_date": r.issue_date,
                        "due_date": r.due_date,
                        "currency": r.currency,
                        "subtotal": r.subtotal,
                        "tax_amount": r.tax_amount,
                        "total_amount": r.total_amount,
                        "line_items": r.line_items,
                        "approval_source": r.approval_source,
                        "approved_by": r.approved_by,
                        "webhook_delivered": r.webhook_delivered,
                        "exported_at": r.exported_at.isoformat(),
                    }
                    for r in rows
                ],
            }

    @fastapi_app.get("/api/v1/exports/csv")
    async def download_payables_csv(request: Request) -> Response:
        sf = request.app.state.session_factory
        with sf() as session:
            csv_body = render_payables_csv(session)
        return Response(
            content=csv_body,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="approved_payables.csv"'
            },
        )

    @fastapi_app.get("/api/v1/audit-logs")
    async def list_audit_logs(
        request: Request,
        document_id: str | None = Query(default=None),
        event_type: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        sf = request.app.state.session_factory
        with sf() as session:
            stmt = select(AuditLogEntry).order_by(
                AuditLogEntry.created_at.desc(), AuditLogEntry.id.desc()
            )
            if document_id:
                stmt = stmt.where(AuditLogEntry.document_id == document_id)
            if event_type:
                stmt = stmt.where(AuditLogEntry.event_type == event_type)
            entries = session.scalars(stmt.limit(limit)).all()
            return {
                "count": len(entries),
                "audit_logs": [
                    {
                        "id": e.id,
                        "document_id": e.document_id,
                        "event_type": e.event_type,
                        "actor": e.actor,
                        "summary": e.summary,
                        "payload": e.payload,
                        "created_at": e.created_at.isoformat(),
                    }
                    for e in entries
                ],
            }

    @fastapi_app.get("/api/v1/eval-feedback")
    async def list_eval_feedback(request: Request) -> dict[str, Any]:
        sf = request.app.state.session_factory
        with sf() as session:
            rows = session.scalars(
                select(EvaluationFeedbackSample).order_by(
                    EvaluationFeedbackSample.created_at.desc()
                )
            ).all()
            return {
                "count": len(rows),
                "samples": [
                    {
                        "document_id": s.document_id,
                        "file_sha256": s.file_sha256,
                        "document_type": s.document_type,
                        "language": s.language,
                        "reviewer": s.reviewer,
                        "ground_truth": s.ground_truth,
                        "created_at": s.created_at.isoformat(),
                    }
                    for s in rows
                ],
            }

    @fastapi_app.post("/mock-erp/webhooks/payables")
    async def receive_finance_erp_webhook(request: Request) -> dict[str, Any]:
        """Mock downstream Finance ERP webhook endpoint with HMAC-SHA256 verification."""
        app_cfg: Settings = request.app.state.settings
        sf = request.app.state.session_factory
        raw_body = await request.body()
        sig_header = request.headers.get("X-Signature-256", "")
        idem_key = request.headers.get("Idempotency-Key", "")

        expected_sig = sign_webhook_payload(raw_body, app_cfg.finance_webhook_secret)
        if not sig_header or not hmac.compare_digest(sig_header, expected_sig):
            raise HTTPException(status_code=401, detail="Invalid X-Signature-256 HMAC")

        payload = json.loads(raw_body.decode("utf-8"))
        doc_id = str(payload.get("document_id", idem_key or "unknown"))
        effective_key = idem_key or doc_id

        with sf() as session:
            existing = session.scalar(
                select(FinanceWebhookReceipt).where(
                    FinanceWebhookReceipt.idempotency_key == effective_key
                )
            )
            if existing is None:
                receipt = FinanceWebhookReceipt(
                    idempotency_key=effective_key,
                    document_id=doc_id,
                    signature_valid=True,
                    payload=payload,
                )
                session.add(receipt)
            else:
                existing.payload = payload
            session.commit()

        return {
            "accepted": True,
            "erp_voucher_id": f"ERP-VCH-{doc_id[:8].upper()}",
            "idempotency_key": effective_key,
        }

    @fastapi_app.get("/mock-erp/webhooks/payables")
    async def list_finance_erp_receipts(request: Request) -> dict[str, Any]:
        sf = request.app.state.session_factory
        with sf() as session:
            rows = session.scalars(
                select(FinanceWebhookReceipt).order_by(
                    FinanceWebhookReceipt.received_at.desc()
                )
            ).all()
            return {
                "count": len(rows),
                "receipts": [
                    {
                        "idempotency_key": r.idempotency_key,
                        "document_id": r.document_id,
                        "signature_valid": r.signature_valid,
                        "payload": r.payload,
                        "received_at": r.received_at.isoformat(),
                    }
                    for r in rows
                ],
            }

    return fastapi_app


app = create_app()
