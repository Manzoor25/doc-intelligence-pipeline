"""Integration tests running the full pipeline & API against the mock OpenAI endpoint."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from doc_pipeline.config import Settings
from doc_pipeline.llm.provider import OpenAICompatibleProvider
from doc_pipeline.main import create_app
from doc_pipeline.mock_services.openai_mock import ChaosConfigRequest, configure_chaos, reset_state
from doc_pipeline.mock_services.openai_mock import app as mock_llm_app
from doc_pipeline.storage.db import dispose_engine, init_db, make_session_factory
from doc_pipeline.synthetic.generator import generate_dataset


@pytest.fixture(scope="module")
def mini_corpus(tmp_path_factory) -> Path:
    corpus_dir = tmp_path_factory.mktemp("synth_corpus")
    generate_dataset(corpus_dir, count=8, seed=42)
    return corpus_dir


@pytest.mark.asyncio
async def test_startup_probe_selects_vision_vs_ocr_paths(tmp_path) -> None:
    await reset_state()
    db_url = f"sqlite:///{(tmp_path / 'probe.db').as_posix()}"
    sf = make_session_factory(db_url)
    init_db(db_url)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_llm_app), base_url="http://mock.local"
        ) as mock_http:
            # 1. Vision + strict JSON schema model
            vis_cfg = Settings(
                LLM_BASE_URL="http://mock.local/v1",
                LLM_API_KEY="test",
                LLM_MODEL="mock-vision-strict-v1",
                DATABASE_URL=db_url,
            )
            prov_vis = OpenAICompatibleProvider(
                settings=vis_cfg, session_factory=sf, http_client=mock_http
            )
            cap_vis = await prov_vis.probe_capabilities(force=True)
            assert cap_vis.supports_vision is True
            assert cap_vis.supports_json_schema is True
            assert cap_vis.chosen_extraction_path == "vision_strict_schema"

            # 2. Text-only model without json_schema -> OCR + prompt JSON repair fallback
            txt_cfg = Settings(
                LLM_BASE_URL="http://mock.local/v1",
                LLM_API_KEY="test",
                LLM_MODEL="mock-text-promptjson-v1",
                DATABASE_URL=db_url,
            )
            prov_txt = OpenAICompatibleProvider(
                settings=txt_cfg, session_factory=sf, http_client=mock_http
            )
            cap_txt = await prov_txt.probe_capabilities(force=True)
            assert cap_txt.supports_vision is False
            assert cap_txt.supports_json_schema is False
            assert cap_txt.chosen_extraction_path == "ocr_json_repair"
    finally:
        dispose_engine(db_url)


@pytest.mark.asyncio
async def test_end_to_end_upload_idempotency_review_export_and_chaos(
    tmp_path, mini_corpus: Path, monkeypatch
) -> None:
    await reset_state()
    monkeypatch.setenv("SYNTHETIC_DATA_DIR", str(mini_corpus))

    db_url = f"sqlite:///{(tmp_path / 'integration.db').as_posix()}"
    cfg = Settings(
        LLM_BASE_URL="http://mock.local/v1",
        LLM_API_KEY="test-key",
        LLM_MODEL="mock-vision-strict-v1",
        DATABASE_URL=db_url,
        LOCAL_BLOB_DIR=str(tmp_path / "blobs"),
        WATCH_DIR=str(tmp_path / "inbox"),
        SYNTHETIC_DIR=str(mini_corpus),
        EVAL_FEEDBACK_DIR=str(tmp_path / "eval_feedback"),
        EXPORT_CSV_PATH=str(tmp_path / "exports" / "payables.csv"),
        FINANCE_WEBHOOK_URL="http://app.local/mock-erp/webhooks/payables",
        FINANCE_WEBHOOK_SECRET="integration-hmac-secret",
    )

    init_db(db_url)
    sf = make_session_factory(db_url)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_llm_app), base_url="http://mock.local"
    ) as mock_llm_http:
        provider = OpenAICompatibleProvider(
            settings=cfg, session_factory=sf, http_client=mock_llm_http
        )
        fastapi_app = create_app(settings=cfg, provider=provider)

        async with (
            fastapi_app.router.lifespan_context(fastapi_app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=fastapi_app), base_url="http://app.local"
            ) as client,
        ):
            # Route outgoing webhooks in-process to the FastAPI app
            fastapi_app.state.webhook_client = client

            docs_dir = mini_corpus / "documents"
            clean_file = next(docs_dir.glob("doc_0001.*"))
            mismatch_file = next(docs_dir.glob("doc_0002.*"))  # anomaly: math_mismatch
            injection_file = next(docs_dir.glob("doc_0005.*"))  # anomaly: prompt_injection

            # 1. Clean document -> auto-approved + exported to Postgres, CSV & signed ERP webhook
            r1 = await client.post(
                "/api/v1/documents/upload",
                files={"file": (clean_file.name, clean_file.read_bytes(), "application/pdf")},
                data={"sync": "true"},
            )
            assert r1.status_code == 200
            d1 = r1.json()["document"]
            assert d1["status"] == "approved"

            # 2. Re-uploading identical bytes triggers SHA-256 idempotency
            r1_dup = await client.post(
                "/api/v1/documents/upload",
                files={"file": (clean_file.name, clean_file.read_bytes(), "application/pdf")},
                data={"sync": "true"},
            )
            assert r1_dup.json()["idempotent_hit"] is True
            assert r1_dup.json()["document"]["id"] == d1["id"]

            # 3. Forged webhook signature is rejected by mock Finance ERP
            bad_wh = await client.post(
                "/mock-erp/webhooks/payables",
                content=b'{"document_id":"fake"}',
                headers={"X-Signature-256": "sha256=deadbeef"},
            )
            assert bad_wh.status_code == 401

            # 4. Arithmetic-mismatch invoice -> routed to review -> human fixes & approves
            r2 = await client.post(
                "/api/v1/documents/upload",
                files={"file": (mismatch_file.name, mismatch_file.read_bytes(), "application/octet-stream")},
                data={"sync": "true"},
            )
            d2_id = r2.json()["document"]["id"]
            assert r2.json()["document"]["status"] == "needs_review"

            detail2 = (await client.get(f"/api/v1/documents/{d2_id}")).json()
            corrected = detail2["extracted_data"]
            corrected["total_amount"] = round(
                float(corrected["subtotal"]) + float(corrected["tax_amount"]), 2
            )
            rev_res = await client.post(
                f"/api/v1/documents/{d2_id}/review",
                json={
                    "action": "approve",
                    "reviewer": "qa.reviewer@acme.example",
                    "reviewer_notes": "Corrected vendor arithmetic error.",
                    "extracted_data": corrected,
                },
            )
            assert rev_res.status_code == 200
            assert rev_res.json()["status"] == "approved"
            assert rev_res.json()["promoted_to_eval_dataset"] is True

            # Confirm evaluation gold sample + CSV + 2 signed ERP webhooks
            fb = (await client.get("/api/v1/eval-feedback")).json()
            assert fb["count"] == 1
            erp = (await client.get("/mock-erp/webhooks/payables")).json()
            assert erp["count"] == 2
            csv_text = (await client.get("/api/v1/exports/csv")).text
            assert corrected["document_number"] in csv_text

            # 5. Prompt-injection invoice -> instructions ignored & quarantined in needs_review
            r3 = await client.post(
                "/api/v1/documents/upload",
                files={"file": (injection_file.name, injection_file.read_bytes(), "application/octet-stream")},
                data={"sync": "true"},
            )
            d3 = r3.json()["document"]
            assert d3["status"] == "needs_review"
            assert d3["injection_detected"] is True

            # 6. Chaos: 429 rate-limit burst + recoverable malformed JSON repair loop
            await configure_chaos(
                ChaosConfigRequest(
                    rate_limit_429_remaining=3,
                    retry_after_seconds=0.05,
                    malformed_mode="recover_on_retry",
                )
            )
            other_file = next(docs_dir.glob("doc_0003.*"))
            r4 = await client.post(
                "/api/v1/documents/upload",
                files={"file": (other_file.name, other_file.read_bytes(), "application/octet-stream")},
                data={"sync": "true"},
            )
            assert r4.status_code == 200
            assert r4.json()["document"]["repair_attempts"] >= 1
            await reset_state()

        await provider.aclose()
    dispose_engine(db_url)
