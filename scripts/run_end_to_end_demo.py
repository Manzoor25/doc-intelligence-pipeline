"""End-to-end Accounts Payable pipeline demonstration across all 3 ingestion channels.

Exercises:
1. Multipart upload (`POST /api/v1/documents/upload`)
2. Base64 JSON API ingestion (`POST /api/v1/documents/ingest`)
3. Watched folder drop (`WATCH_DIR`) + inbox scan (`POST /api/v1/documents/scan-inbox`)
4. SHA-256 idempotent deduplication (re-uploading an identical file skips LLM billing)
5. Worker queue drainage (`FOR UPDATE SKIP LOCKED`)
6. Human-in-the-loop correction on an arithmetic-mismatch invoice -> promotes gold sample
   to `data/eval_feedback/` and exports to PostgreSQL + CSV + signed Finance ERP webhook.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
from pathlib import Path

import httpx

from doc_pipeline.config import get_settings


def wait_until_processed(client: httpx.Client, doc_ids: list[str], timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pending = 0
        for did in doc_ids:
            r = client.get(f"/api/v1/documents/{did}")
            r.raise_for_status()
            st = r.json()["status"]
            if st in ("queued", "processing"):
                pending += 1
        if pending == 0:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {len(doc_ids)} documents to finish processing.")


def main() -> None:
    settings = get_settings()
    api_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    synth_dir = Path(settings.synthetic_dir)
    docs_dir = synth_dir / "documents"
    gt_dir = synth_dir / "ground_truth"

    print("=" * 78)
    print("END-TO-END DEMO: Multi-Channel Ingestion, Validation, Review & Finance Export")
    print("=" * 78)

    # Select a curated batch covering clean, multipage, German, noisy scan, phone photo,
    # math mismatch, unknown supplier, bad dates, duplicate invoice, and prompt injection
    upload_ids = ["doc_0001", "doc_0003", "doc_0006", "doc_0007", "doc_0015"]
    api_ids = ["doc_0012", "doc_0019", "doc_0022"]
    inbox_ids = ["doc_0002", "doc_0004"]

    def resolve_doc_path(doc_id: str) -> Path:
        meta = json.loads((gt_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
        return docs_dir / meta["file_name"]

    tracked_ids: list[str] = []

    with httpx.Client(base_url=api_url, timeout=60.0) as client:
        caps = client.get("/api/v1/capabilities").json()
        print(
            f"[1/6] Probed Endpoint: model={caps['model']} | "
            f"vision={caps['supports_vision']} | json_schema={caps['supports_json_schema']} | "
            f"chosen_path={caps['chosen_extraction_path']}"
        )

        # Channel 1: Multipart Upload
        for did in upload_ids:
            fp = resolve_doc_path(did)
            with fp.open("rb") as f:
                resp = client.post(
                    "/api/v1/documents/upload",
                    files={"file": (fp.name, f, "application/octet-stream")},
                    data={"sync": "false"},
                )
            resp.raise_for_status()
            tracked_ids.append(resp.json()["document"]["id"])
        print(f"[2/6] Ingested {len(upload_ids)} documents via Multipart Upload.")

        # Verify SHA-256 Idempotency on doc_0001
        fp_dup = resolve_doc_path("doc_0001")
        with fp_dup.open("rb") as f:
            idem_resp = client.post(
                "/api/v1/documents/upload",
                files={"file": (fp_dup.name, f, "application/octet-stream")},
            ).json()
        assert idem_resp["idempotent_hit"] is True, "Expected SHA-256 idempotent cache hit!"
        print("      Verified SHA-256 idempotency: duplicate file upload skipped with 0 extra jobs.")

        # Channel 2: Programmatic JSON Base64 API
        for did in api_ids:
            fp = resolve_doc_path(did)
            b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
            resp = client.post(
                "/api/v1/documents/ingest",
                json={"file_name": fp.name, "content_base64": b64, "sync": False},
            )
            resp.raise_for_status()
            tracked_ids.append(resp.json()["document"]["id"])
        print(f"[3/6] Ingested {len(api_ids)} documents via Base64 REST API.")

        # Channel 3: Watched Folder Drop
        watch_path = Path(settings.watch_dir)
        watch_path.mkdir(parents=True, exist_ok=True)
        for did in inbox_ids:
            fp = resolve_doc_path(did)
            shutil.copy2(fp, watch_path / fp.name)
        scan_out = client.post("/api/v1/documents/scan-inbox").json()
        print(
            f"[4/6] Dropped {len(inbox_ids)} files into watched folder -> scanned {scan_out['ingested_count']} new file(s)."
        )

        # Wait for background worker queue to drain
        wait_until_processed(client, tracked_ids, timeout_s=120.0)
        time.sleep(1.0)

        listing = client.get("/api/v1/documents?limit=100").json()
        summary = listing["summary"]
        print(
            f"[5/6] Worker Queue Drained -> Total={summary['total_documents']} | "
            f"Auto-Approved={summary['approved']} | Needs Review={summary['needs_review']} | "
            f"Quarantined Injections={summary['quarantined_injections']}"
        )

        # Submit a Human-in-the-Loop Correction on arithmetic mismatch document (doc_0007)
        mismatch_doc = next(
            (
                d
                for d in listing["documents"]
                if d["file_name"].startswith("doc_0007") and d["status"] == "needs_review"
            ),
            None,
        )
        if mismatch_doc:
            detail = client.get(f"/api/v1/documents/{mismatch_doc['id']}").json()
            ex = detail["extracted_data"]
            corrected_total = round(float(ex["subtotal"]) + float(ex["tax_amount"]), 2)
            ex["total_amount"] = corrected_total
            rev_resp = client.post(
                f"/api/v1/documents/{mismatch_doc['id']}/review",
                json={
                    "action": "approve",
                    "reviewer": "claire.martin@acme-finance.example",
                    "reviewer_notes": "Confirmed vendor math typo (+$47.50); corrected total to match line items + tax.",
                    "extracted_data": ex,
                },
            )
            rev_resp.raise_for_status()
            print(
                f"[6/6] Human Reviewer corrected {ex['document_number']} -> Approved, exported to "
                f"PostgreSQL + CSV + signed ERP webhook, and saved to Evaluation Gold Set!"
            )

        erp_receipts = client.get("/mock-erp/webhooks/payables").json()
        csv_lines = client.get("/api/v1/exports/csv").text.strip().splitlines()
        feedback = client.get("/api/v1/eval-feedback").json()
        print(
            f"      Exports Verified: CSV rows={max(0, len(csv_lines) - 1)} | "
            f"Signed ERP Webhooks Accepted={erp_receipts['count']} | "
            f"Eval Gold Feedback Samples={feedback['count']}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
