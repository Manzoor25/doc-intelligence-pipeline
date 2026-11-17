"""Deterministic OpenAI-compatible mock server for zero-key quickstart, CI, and chaos demos.

Supports:
- Vision + strict JSON schema model (`mock-vision-v1`)
- Text-only + JSON schema model (`mock-text-v1`) -> triggers Tesseract OCR fallback path
- Text-only prompt-JSON model (`mock-text-promptjson-v1`) -> triggers prompt-repair JSON path
- Admin chaos controls (`/admin/chaos`, `/admin/stats`, `/admin/reset`) for rate-limit (HTTP 429)
  and malformed-JSON repair/routing demonstrations.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from doc_pipeline.ocr.renderer import render_document_to_png_pages, tesseract_available

app = FastAPI(title="Mock OpenAI-Compatible Endpoint", version="1.0.0")


class ChaosConfigRequest(BaseModel):
    rate_limit_429_remaining: int = 0
    rate_limit_every_n: int = 0
    retry_after_seconds: float = 0.5
    malformed_mode: Literal["off", "recover_on_retry", "always_malformed"] = "off"
    artificial_latency_ms: int = 0


@dataclass
class MockState:
    chaos: ChaosConfigRequest = field(default_factory=ChaosConfigRequest)
    total_requests: int = 0
    successful_completions: int = 0
    rate_limits_returned: int = 0
    malformed_outputs_returned: int = 0
    repair_recoveries_served: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    by_doc_number: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_ahash: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    corpus_loaded_from: str | None = None


STATE = MockState()


def _compute_ahash(png_bytes: bytes) -> int:
    """Compute a 64-bit average perceptual hash invariant to minor resize differences."""
    img = Image.open(io.BytesIO(png_bytes)).convert("L").resize((8, 8), Image.Resampling.BILINEAR)
    pixels = list(img.getdata())
    avg = sum(pixels) / 64.0
    bits = 0
    for idx, px in enumerate(pixels):
        if px >= avg:
            bits |= 1 << idx
    return bits


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _ensure_corpus_indexed() -> None:
    synth_dir = Path(os.environ.get("SYNTHETIC_DATA_DIR", "data/synthetic"))
    gt_dir = synth_dir / "ground_truth"
    docs_dir = synth_dir / "documents"
    resolved = str(synth_dir.resolve())
    if STATE.corpus_loaded_from == resolved and STATE.by_doc_number:
        return
    if not gt_dir.exists():
        return

    by_num: dict[str, dict[str, Any]] = {}
    by_hash: list[tuple[int, dict[str, Any]]] = []
    for gt_file in sorted(gt_dir.glob("doc_*.json")):
        try:
            payload = json.loads(gt_file.read_text(encoding="utf-8"))
            gt = payload["ground_truth"]
            doc_num = str(gt["document_number"]).upper()
            by_num[doc_num] = gt

            doc_file = docs_dir / payload["file_name"]
            if doc_file.exists():
                raw = doc_file.read_bytes()
                pages, _ = render_document_to_png_pages(raw, mime_type="", dpi=120)
                if pages:
                    by_hash.append((_compute_ahash(pages[0]), gt))
        except Exception:
            continue

    STATE.by_doc_number = by_num
    STATE.by_ahash = by_hash
    STATE.corpus_loaded_from = resolved


def _extract_messages_content(messages: list[dict[str, Any]]) -> tuple[str, list[bytes]]:
    texts: list[str] = []
    images: list[bytes] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    if url.startswith("data:image/") and ";base64," in url:
                        b64 = url.split(";base64,", 1)[1]
                        try:
                            images.append(base64.b64decode(b64))
                        except Exception:
                            pass
    return "\n".join(texts), images


def _normalize_ocr_doc_num(raw_token: str) -> str:
    prefix, year, seq = raw_token.upper().split("-")
    seq_fixed = (
        seq.replace("O", "0")
        .replace("Q", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("B", "8")
    )
    year_fixed = year.replace("O", "0").replace("I", "1")
    return f"{prefix}-{year_fixed}-{seq_fixed}"


def _parse_money_from_text(label_pattern: str, text: str) -> float | None:
    pattern = re.compile(
        rf"(?:{label_pattern})[^\n\d$-]{{0,25}}(?:USD|EUR|GBP|\$)?\s*([0-9]{{1,3}}(?:,[0-9]{{3}})*(?:\.[0-9]{{2}})|[0-9]+\.[0-9]{{2}})",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _extract_from_ocr_or_free_text(full_text: str) -> dict[str, Any]:
    """Extract structured AP fields from OCR text or ad-hoc test prompts.

    Never executes instructions embedded inside untrusted document text.
    """
    _ensure_corpus_indexed()

    # Isolate the document section if wrapped in <untrusted_document_text>
    doc_section = full_text
    tag_match = re.search(
        r"<untrusted_document_text>(.*?)</untrusted_document_text>",
        full_text,
        flags=re.DOTALL,
    )
    if tag_match:
        doc_section = tag_match.group(1)

    # Check if OCR recognized a synthetic corpus document number
    num_match = re.search(r"\b(INV|RCP|PO)-[0-9OI]{4}-[0-9OIQlsbSB]{4}\b", doc_section)
    if num_match:
        norm_num = _normalize_ocr_doc_num(num_match.group(0))
        if norm_num in STATE.by_doc_number:
            # Start from corpus baseline, then overlay actual OCR-parsed totals/dates
            # so genuine Tesseract OCR errors on noisy scans & phone photos are reflected!
            candidate = json.loads(json.dumps(STATE.by_doc_number[norm_num]))
            ocr_sub = _parse_money_from_text(r"Subtotal|Zwischensumme", doc_section)
            ocr_tax = _parse_money_from_text(r"Tax|MwSt\.?", doc_section)
            ocr_tot = _parse_money_from_text(r"Total Due|Gesamtbetrag", doc_section)
            if ocr_sub is not None:
                candidate["subtotal"] = ocr_sub
            if ocr_tax is not None:
                candidate["tax_amount"] = ocr_tax
            if ocr_tot is not None:
                candidate["total_amount"] = ocr_tot

            iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", doc_section)
            if iso_dates:
                candidate["issue_date"] = iso_dates[0]
                if len(iso_dates) >= 2 and candidate.get("due_date") is not None:
                    candidate["due_date"] = iso_dates[1]
            return candidate

    # Generic fallback parser for arbitrary non-corpus documents / unit tests
    lower = doc_section.lower()
    if "purchase order" in lower or "bestellung" in lower or "po-" in lower:
        doc_type = "purchase_order"
    elif "receipt" in lower or "quittung" in lower or "kassenbeleg" in lower or "rcp-" in lower:
        doc_type = "receipt"
    else:
        doc_type = "invoice"

    lang = "de" if any(w in lower for w in ("rechnung", "betrag", "mwst", "gesamtbetrag")) else "en"
    generic_num = re.search(
        r"(?:Number|No\.?|Belegnummer)\s*[:#]?\s*([A-Z0-9][A-Z0-9-]{3,24})",
        doc_section,
        re.IGNORECASE,
    )
    doc_number = generic_num.group(1) if generic_num else "INV-GENERIC-0001"

    sup_match = re.search(r"(?:Supplier|Vendor|From)\s*:\s*([^\n]+)", doc_section, re.IGNORECASE)
    lines_nonempty = [ln.strip() for ln in doc_section.splitlines() if ln.strip() and not ln.strip().startswith("---")]
    supplier_name = (
        sup_match.group(1).strip()
        if sup_match
        else (lines_nonempty[0] if lines_nonempty else "Northwind Industrial Supply Inc.")
    )

    tax_id_match = re.search(
        r"(?:Tax ID|VAT|USt-IdNr\.?)\s*:\s*([A-Z0-9-]+)", doc_section, re.IGNORECASE
    )
    currency = "EUR" if "eur" in lower else ("GBP" if "gbp" in lower else "USD")
    iso_dates = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", doc_section)
    issue_date = iso_dates[0] if iso_dates else "2026-03-15"
    due_date = iso_dates[1] if len(iso_dates) >= 2 else ("2026-04-14" if doc_type != "receipt" else None)

    subtotal = _parse_money_from_text(r"Subtotal|Zwischensumme", doc_section) or 100.00
    tax_amount = _parse_money_from_text(r"Tax|MwSt\.?", doc_section)
    if tax_amount is None:
        tax_amount = 19.00 if lang == "de" else 0.00
    total_amount = _parse_money_from_text(r"Total Due|Total|Gesamtbetrag", doc_section) or round(
        subtotal + tax_amount, 2
    )

    # Preserve any suspicious or handwritten notes strictly as inert string data
    note_match = re.search(r"(?:Note|Notes|Override)\s*[:\-]?\s*([^\n]+)", doc_section, re.IGNORECASE)
    notes_val = note_match.group(0).strip() if note_match else None

    return {
        "document_type": doc_type,
        "language": lang,
        "document_number": doc_number,
        "supplier_name": supplier_name,
        "supplier_tax_id": tax_id_match.group(1) if tax_id_match else "US-94-3184721",
        "supplier_address": None,
        "issue_date": issue_date,
        "due_date": due_date,
        "currency": currency,
        "subtotal": subtotal,
        "tax_rate": round(tax_amount / subtotal, 4) if subtotal > 0 else 0.0,
        "tax_amount": tax_amount,
        "total_amount": total_amount,
        "line_items": [
            {
                "sku": "GEN-01",
                "description": "Standard goods / services item",
                "quantity": 1.0,
                "unit_price": subtotal,
                "line_total": subtotal,
            }
        ],
        "handwritten_notes": notes_val,
    }


def _extract_from_vision_images(images: list[bytes], full_text: str) -> dict[str, Any]:
    _ensure_corpus_indexed()
    if images and STATE.by_ahash:
        target_hash = _compute_ahash(images[0])
        best_dist = 999
        best_gt: dict[str, Any] | None = None
        for cand_hash, gt in STATE.by_ahash:
            dist = _hamming_distance(target_hash, cand_hash)
            if dist < best_dist:
                best_dist = dist
                best_gt = gt
        if best_gt is not None and best_dist <= 8:
            return json.loads(json.dumps(best_gt))

    # Fallback: if Tesseract is installed, OCR page 1 for arbitrary non-corpus uploads
    if images and tesseract_available():
        import pytesseract

        img = Image.open(io.BytesIO(images[0])).convert("L")
        ocr_txt = pytesseract.image_to_string(img)
        return _extract_from_ocr_or_free_text(ocr_txt + "\n" + full_text)

    return _extract_from_ocr_or_free_text(full_text)


@app.get("/health")
@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "mock-vision-strict-v1", "object": "model", "owned_by": "doc-pipeline-mock"},
            {"id": "mock-vision-v1", "object": "model", "owned_by": "doc-pipeline-mock"},
            {"id": "mock-text-strict-v1", "object": "model", "owned_by": "doc-pipeline-mock"},
            {"id": "mock-text-v1", "object": "model", "owned_by": "doc-pipeline-mock"},
            {
                "id": "mock-text-promptjson-v1",
                "object": "model",
                "owned_by": "doc-pipeline-mock",
            },
        ],
    }


@app.post("/admin/chaos")
async def configure_chaos(cfg: ChaosConfigRequest) -> dict[str, Any]:
    async with STATE.lock:
        STATE.chaos = cfg
    return {"status": "updated", "chaos": cfg.model_dump()}


@app.post("/admin/reset")
async def reset_state() -> dict[str, Any]:
    async with STATE.lock:
        STATE.chaos = ChaosConfigRequest()
        STATE.total_requests = 0
        STATE.successful_completions = 0
        STATE.rate_limits_returned = 0
        STATE.malformed_outputs_returned = 0
        STATE.repair_recoveries_served = 0
    return {"status": "reset"}


@app.get("/admin/stats")
async def get_stats() -> dict[str, Any]:
    return {
        "total_requests": STATE.total_requests,
        "successful_completions": STATE.successful_completions,
        "rate_limits_returned": STATE.rate_limits_returned,
        "malformed_outputs_returned": STATE.malformed_outputs_returned,
        "repair_recoveries_served": STATE.repair_recoveries_served,
        "chaos": STATE.chaos.model_dump(),
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    model = str(body.get("model", "mock-vision-v1"))
    messages = body.get("messages", [])
    response_format = body.get("response_format") or {}
    rf_type = response_format.get("type")

    full_text, images = _extract_messages_content(messages)
    lower_model = model.lower()

    # Enforce model capability boundaries for startup probing & fallback routing
    if images and ("text" in lower_model or "novision" in lower_model):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{model}' is text-only and does not accept image_url inputs.",
                    "type": "invalid_request_error",
                    "code": "vision_not_supported",
                }
            },
        )

    if rf_type == "json_schema" and ("promptjson" in lower_model or "nostruct" in lower_model):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Model '{model}' does not support response_format.type='json_schema'.",
                    "type": "invalid_request_error",
                    "code": "json_schema_not_supported",
                }
            },
        )

    # Check if this is the lightweight startup probe
    is_probe = (
        '{"ok": true}' in full_text
        or "Reply with the single word OK" in full_text
        or (
            rf_type == "json_schema"
            and response_format.get("json_schema", {}).get("name")
            in ("startup_capability_probe", "capability_probe", "ProbeResponse")
        )
    )

    async with STATE.lock:
        STATE.total_requests += 1
        req_num = STATE.total_requests
        artificial_delay = STATE.chaos.artificial_latency_ms
        if not is_probe:
            should_429 = False
            if STATE.chaos.rate_limit_429_remaining > 0:
                STATE.chaos.rate_limit_429_remaining -= 1
                should_429 = True
            elif (
                STATE.chaos.rate_limit_every_n > 0
                and req_num % STATE.chaos.rate_limit_every_n == 0
            ):
                should_429 = True

            if should_429:
                STATE.rate_limits_returned += 1
                retry_after = str(STATE.chaos.retry_after_seconds)
                return JSONResponse(
                    status_code=429,
                    headers={"Retry-After": retry_after},
                    content={
                        "error": {
                            "message": "Rate limit exceeded (simulated 429 burst). Please retry after backoff.",
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded",
                        }
                    },
                )

    if artificial_delay > 0:
        await asyncio.sleep(artificial_delay / 1000.0)

    if is_probe:
        payload_obj: Any = {"ok": True}
    else:
        schema_name = response_format.get("json_schema", {}).get("name", "")
        is_classification = (
            schema_name in ("ClassificationResult", "document_classification")
            or "Classify this" in full_text
        )
        extracted_doc = (
            _extract_from_vision_images(images, full_text)
            if images
            else _extract_from_ocr_or_free_text(full_text)
        )
        if is_classification:
            payload_obj = {
                "document_type": extracted_doc["document_type"],
                "language": extracted_doc["language"],
                "reasoning": f"Header layout and terminology match {extracted_doc['document_type']} ({extracted_doc['language']}).",
            }
        else:
            payload_obj = extracted_doc

    # Chaos injection for malformed JSON / schema violation repair loop
    lower_full = full_text.lower()
    is_repair_turn = (
        "previous output failed" in lower_full
        or "failed strict json/pydantic validation" in lower_full
        or "validation error" in lower_full
    )
    content_str = json.dumps(payload_obj)

    if not is_probe and STATE.chaos.malformed_mode != "off":
        is_classification_turn = "document_type" in payload_obj and "line_items" not in payload_obj
        if not is_classification_turn:
            async with STATE.lock:
                if STATE.chaos.malformed_mode == "always_malformed":
                    STATE.malformed_outputs_returned += 1
                    content_str = '{"document_type": "invoice", "total_amount": "NOT_A_FLOAT", "line_items": ['
                elif STATE.chaos.malformed_mode == "recover_on_retry":
                    if not is_repair_turn:
                        STATE.malformed_outputs_returned += 1
                        # Deliberately violate Pydantic schema (string total_amount + negative qty)
                        broken = dict(payload_obj)
                        broken["total_amount"] = "INVALID_AMOUNT_STRING"
                        content_str = json.dumps(broken)
                    else:
                        STATE.repair_recoveries_served += 1

    async with STATE.lock:
        STATE.successful_completions += 1

    # Realistic token accounting: vision image tiles consume ~850 tokens/page vs ~260 OCR text tokens
    prompt_tokens = max(48, len(full_text) // 4 + len(images) * 820)
    completion_tokens = max(16, len(content_str) // 4)

    return JSONResponse(
        status_code=200,
        content={
            "id": f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_str},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )
