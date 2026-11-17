"""Failure Demo 2: Malformed JSON / Schema Violation -> Pydantic Repair Loop or Review Routing.

EXPECTED OUTPUT:
  - Case A (Recoverable schema error): Model initially emits `"total_amount": "INVALID_AMOUNT_STRING"`.
    Pydantic catches `float_parsing` validation error, feeds exact diagnostic back to the model,
    repair loop succeeds (`repair_attempts >= 1`), and valid structured fields are saved.
  - Case B (Persistent broken JSON): Model emits truncated JSON on every retry until
    `LLM_MAX_REPAIR_ATTEMPTS` is exhausted. Pipeline catches failure without crashing and
    routes the document safely to `status='needs_review'`.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from doc_pipeline.config import get_settings


def _mock_admin_base(llm_base_url: str) -> str:
    return llm_base_url[:-3] if llm_base_url.endswith("/v1") else llm_base_url.rstrip("/")


def main() -> None:
    settings = get_settings()
    api_url = os.environ.get("API_BASE_URL", "http://localhost:8000")
    mock_admin = os.environ.get("MOCK_ADMIN_URL", _mock_admin_base(settings.llm_base_url))
    docs_dir = Path(settings.synthetic_dir) / "documents"

    doc_recover = next(docs_dir.glob("doc_0040.*"))
    doc_unrecoverable = next(docs_dir.glob("doc_0041.*"))

    print("=" * 78)
    print("FAILURE DEMO 2: Malformed JSON -> Pydantic Self-Repair vs Safe Review Routing")
    print("=" * 78)
    print(
        "[EXPECTED] Case A: Invalid JSON schema output is repaired via Pydantic error feedback.\n"
        "           Case B: Unrecoverable truncated JSON exhausts retries and routes to review."
    )

    with (
        httpx.Client(base_url=api_url, timeout=60.0) as api_client,
        httpx.Client(base_url=mock_admin, timeout=15.0) as chaos_client,
    ):
        try:
            # Case A: Recover on retry
            chaos_client.post(
                "/admin/chaos",
                json={"malformed_mode": "recover_on_retry"},
            ).raise_for_status()

            with doc_recover.open("rb") as f:
                resp_a = api_client.post(
                    "/api/v1/documents/upload",
                    files={"file": (doc_recover.name, f, "application/octet-stream")},
                    data={"sync": "true", "force_reprocess": "true"},
                )
            resp_a.raise_for_status()
            detail_a = api_client.get(
                f"/api/v1/documents/{resp_a.json()['document']['id']}"
            ).json()
            print(
                f"[OBSERVED Case A] status={detail_a['status']} | "
                f"repair_attempts={detail_a['repair_attempts']} | "
                f"extracted_total={detail_a['total_amount']}"
            )
            assert detail_a["repair_attempts"] >= 1, "Expected >=1 Pydantic repair attempt"
            assert detail_a["extracted_data"] is not None, "Expected recovered structured JSON"

            # Case B: Persistent malformed JSON -> routed to review
            chaos_client.post(
                "/admin/chaos",
                json={"malformed_mode": "always_malformed"},
            ).raise_for_status()

            with doc_unrecoverable.open("rb") as f:
                resp_b = api_client.post(
                    "/api/v1/documents/upload",
                    files={"file": (doc_unrecoverable.name, f, "application/octet-stream")},
                    data={"sync": "true", "force_reprocess": "true"},
                )
            resp_b.raise_for_status()
            detail_b = api_client.get(
                f"/api/v1/documents/{resp_b.json()['document']['id']}"
            ).json()
            print(
                f"[OBSERVED Case B] status={detail_b['status']} | "
                f"repair_attempts={detail_b['repair_attempts']} | "
                f"error='{str(detail_b['error_message'])[:72]}...'"
            )
            assert detail_b["status"] == "needs_review", "Unrecoverable JSON must route to review"
        finally:
            chaos_client.post("/admin/chaos", json={"malformed_mode": "off"})

    print("[RESULT]   PASS — Transient schema errors repaired; persistent errors routed to review.")
    print("=" * 78)


if __name__ == "__main__":
    main()
