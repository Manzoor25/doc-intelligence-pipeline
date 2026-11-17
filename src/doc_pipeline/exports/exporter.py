"""Approved payables exporter: PostgreSQL ledger upsert, CSV generation, and HMAC webhook."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_pipeline.config import Settings
from doc_pipeline.storage.db import append_audit_log
from doc_pipeline.storage.models import ApprovedPayable, DocumentRecord

logger = logging.getLogger(__name__)

CSV_COLUMNS: list[str] = [
    "document_id",
    "document_type",
    "document_number",
    "supplier_id",
    "supplier_name",
    "supplier_tax_id",
    "issue_date",
    "due_date",
    "currency",
    "subtotal",
    "tax_amount",
    "total_amount",
    "line_item_count",
    "approval_source",
    "approved_by",
    "webhook_delivered",
    "exported_at",
]


def sign_webhook_payload(body_bytes: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def render_payables_csv(session: Session) -> str:
    """Render all approved payables in PostgreSQL as an accounting-ready CSV string."""
    rows = session.scalars(
        select(ApprovedPayable).order_by(ApprovedPayable.exported_at.asc(), ApprovedPayable.id.asc())
    ).all()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for r in rows:
        writer.writerow(
            {
                "document_id": r.document_id,
                "document_type": r.document_type,
                "document_number": r.document_number,
                "supplier_id": r.supplier_id or "",
                "supplier_name": r.supplier_name,
                "supplier_tax_id": r.supplier_tax_id or "",
                "issue_date": r.issue_date,
                "due_date": r.due_date or "",
                "currency": r.currency,
                "subtotal": f"{r.subtotal:.2f}",
                "tax_amount": f"{r.tax_amount:.2f}",
                "total_amount": f"{r.total_amount:.2f}",
                "line_item_count": len(r.line_items or []),
                "approval_source": r.approval_source,
                "approved_by": r.approved_by,
                "webhook_delivered": str(r.webhook_delivered).lower(),
                "exported_at": r.exported_at.isoformat(),
            }
        )
    return buf.getvalue()


def write_csv_snapshot(session: Session, csv_path: str | Path) -> Path:
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    csv_text = render_payables_csv(session)
    target.write_text(csv_text, encoding="utf-8")
    return target


async def export_approved_document(
    *,
    doc: DocumentRecord,
    session: Session,
    settings: Settings,
    approval_source: str,
    approved_by: str,
    http_client: httpx.AsyncClient | None = None,
) -> ApprovedPayable:
    """Export an approved document to PostgreSQL, CSV, and signed finance webhook."""
    if not doc.extracted_data:
        raise ValueError(f"Document {doc.id} cannot be exported without extracted_data")

    data: dict[str, Any] = doc.extracted_data
    now = datetime.now(UTC)

    payable = session.scalar(
        select(ApprovedPayable).where(ApprovedPayable.document_id == doc.id)
    )
    if payable is None:
        payable = ApprovedPayable(
            document_id=doc.id,
            document_type=str(data["document_type"]),
            document_number=str(data["document_number"]),
            supplier_id=doc.matched_supplier_id,
            supplier_name=str(data["supplier_name"]),
            supplier_tax_id=data.get("supplier_tax_id"),
            issue_date=str(data["issue_date"]),
            due_date=data.get("due_date"),
            currency=str(data["currency"]),
            subtotal=float(data["subtotal"]),
            tax_amount=float(data["tax_amount"]),
            total_amount=float(data["total_amount"]),
            line_items=list(data.get("line_items", [])),
            approval_source=approval_source,
            approved_by=approved_by,
            webhook_delivered=False,
            exported_at=now,
        )
        session.add(payable)
    else:
        payable.document_type = str(data["document_type"])
        payable.document_number = str(data["document_number"])
        payable.supplier_id = doc.matched_supplier_id
        payable.supplier_name = str(data["supplier_name"])
        payable.supplier_tax_id = data.get("supplier_tax_id")
        payable.issue_date = str(data["issue_date"])
        payable.due_date = data.get("due_date")
        payable.currency = str(data["currency"])
        payable.subtotal = float(data["subtotal"])
        payable.tax_amount = float(data["tax_amount"])
        payable.total_amount = float(data["total_amount"])
        payable.line_items = list(data.get("line_items", []))
        payable.approval_source = approval_source
        payable.approved_by = approved_by
        payable.exported_at = now

    doc.exported_at = now
    session.flush()

    # Deliver signed webhook to mock/external finance ERP if configured
    webhook_ok = False
    if settings.finance_webhook_url:
        webhook_payload = {
            "event": "payable.approved",
            "document_id": doc.id,
            "file_sha256": doc.file_sha256,
            "approval_source": approval_source,
            "approved_by": approved_by,
            "exported_at": now.isoformat(),
            "payable": {
                "document_type": payable.document_type,
                "document_number": payable.document_number,
                "supplier_id": payable.supplier_id,
                "supplier_name": payable.supplier_name,
                "supplier_tax_id": payable.supplier_tax_id,
                "issue_date": payable.issue_date,
                "due_date": payable.due_date,
                "currency": payable.currency,
                "subtotal": payable.subtotal,
                "tax_amount": payable.tax_amount,
                "total_amount": payable.total_amount,
                "line_items": payable.line_items,
            },
        }
        body_bytes = json.dumps(webhook_payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": doc.id,
            "X-Signature-256": sign_webhook_payload(body_bytes, settings.finance_webhook_secret),
        }
        try:
            if http_client is not None:
                resp = await http_client.post(
                    settings.finance_webhook_url, content=body_bytes, headers=headers
                )
                webhook_ok = 200 <= resp.status_code < 300
            else:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        settings.finance_webhook_url, content=body_bytes, headers=headers
                    )
                    webhook_ok = 200 <= resp.status_code < 300
        except Exception as exc:
            logger.warning("Finance ERP webhook delivery failed for doc=%s: %s", doc.id, exc)
            webhook_ok = False

    payable.webhook_delivered = webhook_ok
    session.flush()

    # Refresh CSV snapshot on disk
    write_csv_snapshot(session, settings.export_csv_path)

    append_audit_log(
        session,
        document_id=doc.id,
        event_type="EXPORTED_TO_FINANCE",
        actor=approved_by,
        summary=(
            f"Exported {payable.document_number} ({payable.currency} {payable.total_amount:.2f}) "
            f"to PostgreSQL, CSV, and webhook (delivered={webhook_ok})."
        ),
        payload={
            "approval_source": approval_source,
            "webhook_delivered": webhook_ok,
            "csv_path": str(settings.export_csv_path),
        },
    )
    return payable
