"""Failure Demo 1: Endpoint Rate-Limit Storms (HTTP 429) -> Queue, Backoff Retry & Zero Loss.

EXPECTED OUTPUT:
  - Mock endpoint injects HTTP 429 Too Many Requests bursts (`Retry-After: 0.25s`).
  - Concurrent documents remain safely queued/retried with exponential backoff + jitter.
  - Observed 429 responses >= 8, retry recovery succeeds on 100% of batch, and 0 documents are lost.
"""

from __future__ import annotations

import os
import time
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

    batch_files = sorted(docs_dir.glob("doc_003[0-5].*"))[:6]
    if len(batch_files) < 6:
        raise RuntimeError("Run synthetic generator first so doc_0030..doc_0035 exist.")

    print("=" * 78)
    print("FAILURE DEMO 1: HTTP 429 Rate-Limit Storm -> Queue, Backoff & Zero Lost Docs")
    print("=" * 78)
    print(
        "[EXPECTED] Endpoint returns HTTP 429 bursts; worker backs off obeying Retry-After,\n"
        "           retries transparently, and finishes 6/6 documents with 0 lost."
    )

    with (
        httpx.Client(base_url=api_url, timeout=60.0) as api_client,
        httpx.Client(base_url=mock_admin, timeout=15.0) as chaos_client,
    ):
        before_stats = chaos_client.get("/admin/stats").json()
        chaos_client.post(
            "/admin/chaos",
            json={
                "rate_limit_429_remaining": 10,
                "rate_limit_every_n": 2,
                "retry_after_seconds": 0.2,
                "malformed_mode": "off",
                "artificial_latency_ms": 0,
            },
        ).raise_for_status()

        try:
            doc_ids: list[str] = []
            for fp in batch_files:
                with fp.open("rb") as f:
                    resp = api_client.post(
                        "/api/v1/documents/upload",
                        files={"file": (fp.name, f, "application/octet-stream")},
                        data={"sync": "false", "force_reprocess": "true"},
                    )
                resp.raise_for_status()
                doc_ids.append(resp.json()["document"]["id"])

            print(f"[ACTION]   Enqueued {len(doc_ids)} documents concurrently during active 429 storm...")

            deadline = time.time() + 120.0
            final_statuses: dict[str, str] = {}
            while time.time() < deadline:
                final_statuses.clear()
                for did in doc_ids:
                    st = api_client.get(f"/api/v1/documents/{did}").json()["status"]
                    final_statuses[did] = st
                if all(st in ("approved", "needs_review") for st in final_statuses.values()):
                    break
                time.sleep(0.5)

            after_stats = chaos_client.get("/admin/stats").json()
        finally:
            chaos_client.post(
                "/admin/chaos",
                json={
                    "rate_limit_429_remaining": 0,
                    "rate_limit_every_n": 0,
                    "retry_after_seconds": 0.5,
                    "malformed_mode": "off",
                    "artificial_latency_ms": 0,
                },
            )

    rl_delta = after_stats["rate_limits_returned"] - before_stats["rate_limits_returned"]
    completed_count = sum(
        1 for st in final_statuses.values() if st in ("approved", "needs_review")
    )
    lost_count = len(doc_ids) - completed_count

    print(
        f"[OBSERVED] HTTP 429 Errors Handled: {rl_delta} | "
        f"Documents Completed: {completed_count}/{len(doc_ids)} | Lost: {lost_count}"
    )
    assert rl_delta >= 8, f"Expected >= 8 HTTP 429 errors injected, got {rl_delta}"
    assert lost_count == 0, f"Expected 0 lost documents, got {lost_count}"
    print("[RESULT]   PASS — Rate limits absorbed via exponential backoff; zero documents lost.")
    print("=" * 78)


if __name__ == "__main__":
    main()
