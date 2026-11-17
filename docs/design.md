# System Design — Accounts Payable Document Intelligence Pipeline

## 1. Requirements & Operating Envelope

### Business Context
An Accounts Payable (AP) team receives **5,000 commercial documents per month**—vendor invoices, employee expense receipts, and purchase orders—arriving as native PDFs, multi-page statements, skewed noisy scans, and smartphone photos in **English (`en`) and German (`de`)**. Manual entry takes 3–5 minutes per document (~250–400 hours/month) and introduces transposition errors, missed early-payment discounts, and duplicate supplier payouts.

### Functional Requirements
1. **Vendor-Agnostic Model Access:** Every LLM call routes through one small OpenAI-compatible module reading `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`. Zero hard-coded vendor SDKs.
2. **Startup Capability Probing:** Automatically detect whether the endpoint supports multimodal `image_url` inputs and strict `json_schema` structured outputs, log the result, and choose the execution path dynamically.
3. **Multi-Channel Ingestion:** Accept files via Web UI upload, REST API (`POST /api/documents/upload`), and an automated watched inbox directory (`WATCH_DIR`), deduplicated idempotently by `SHA-256` content hash.
4. **Strict Extraction & Repair:** Extract supplier identity, tax ID, document type/number, ISO dates, currency, subtotal, tax, grand total, handwritten notes, and every line item across pages. Validate against strict Pydantic v2 models and retry with targeted validation errors on failure.
5. **Non-Vision Fallback:** When `accepts_images=False`, rasterize PDFs/images with `pypdfium2` + `Pillow`, preprocess contrast/noise, run `Tesseract OCR (eng+deu)`, and perform text-only structured extraction.
6. **Model-Free Business Validation:** Verify line-item sum (`sum(lines) == subtotal`), tax arithmetic (`subtotal + tax == total`), date plausibility, fuzzy supplier master matching (`RapidFuzz`), and semantic duplicate invoice detection.
7. **Calibrated Per-Field Confidence:** Score every field from two-pass extraction agreement and deterministic rule coupling—never from model self-reported confidence. Route documents scoring `< 0.85` or failing rules/safety to a human review queue.
8. **Human-in-the-Loop Flywheel:** Display rendered document pages beside extracted fields, confidence badges, and rule diagnostics; persist reviewer edits as audit diffs and promote them automatically into the evaluation dataset (`data/eval_feedback/`).
9. **Multi-Sink Finance Export:** Export approved and reviewer-verified documents idempotently to PostgreSQL (`approved_payables`), an append-safe CSV ledger, and an `HMAC-SHA256`-signed ERP webhook.

### Non-Functional Targets (5,000 Docs / Month)
- **Throughput:** 5,000 docs/month averages ~230 docs/business day (~0.5 docs/minute during an 8-hour workday, with month-end spikes of 500 docs/hour). Bounded async concurrency (`LLM_MAX_CONCURRENCY=4`) processes 500 docs well inside an hour.
- **Zero Silent Data Loss:** Upstream `HTTP 429` rate limits, `5xx` errors, or truncated JSON never lose an uploaded file; blobs are persisted before queueing, retried with exponential backoff, and degraded cleanly to `needs_review` if retries exhaust.
- **Zero Unchecked Financial Mutation:** No arithmetic discrepancy or prompt-injection attempt may auto-export to the finance system (`Anomaly Review Recall = 1.00` target on hard accounting/safety rules).

---

## 2. Extraction, Capability Probe & Fallback Design

### 2.1 Startup Capability Probe (`src/doc_pipeline/llm/provider.py`)
Vendor endpoints differ widely in feature parity across OpenAI-compatible `/v1/chat/completions` implementations ([ADR-0001](adr/0001-openai-compatible-provider-and-startup-probe.md)). On startup (and whenever `LLM_MODEL` changes), `probe_capabilities()` executes two lightweight probes:
1. **Structured Output Probe:** Requests `{"ok": true}` using `response_format={"type": "json_schema", "json_schema": {"strict": true, ...}}`. On `HTTP 400`, falls back to probing `{"type": "json_object"}`, and finally prompt-guided JSON.
2. **Vision Probe:** Sends a generated `32×32` RGB PNG (`data:image/png;base64,...`) inside a multimodal content array. If accepted (`HTTP 200`), `supports_vision=True`; if rejected (`HTTP 400/415/422`), `supports_vision=False`.

| `supports_vision` | `supports_json_schema` | Selected Execution Path | Behavior |
| :---: | :---: | :--- | :--- |
| `True` | `True` | `vision_strict_schema` | Page PNGs sent directly (`<= VISION_MAX_IMAGE_EDGE`); grammar-constrained JSON output + Pydantic check. |
| `True` | `False` | `vision_json_repair` | Page PNGs sent directly; JSON schema injected into system prompt + multi-turn Pydantic repair loop. |
| `False` | `True` | `ocr_strict_schema` | `pypdfium2` -> `Pillow` autocontrast/median denoise -> `Tesseract (eng+deu)` -> grammar-constrained JSON. |
| `False` | `False` | `ocr_json_repair` | Local Tesseract OCR text -> prompt-guided JSON extraction + multi-turn Pydantic repair loop. |

### 2.2 Pydantic Schema Validation & Error-Guided Repair Loop
Even when `json_schema` grammar is supported, semantic constraints—such as ISO `YYYY-MM-DD` date parsing (`15/05/2026` fails), 3-letter uppercase ISO currency normalization, or non-empty line-item arrays—require post-parse validation (`src/doc_pipeline/schemas.py`).

When `json.loads()` or `ExtractedDocument.model_validate()` fails:
1. `complete_structured()` formats the exact offending JSON snippet and Pydantic field path errors (e.g., `issue_date: Input should be a valid date or datetime, invalid character in year`).
2. It appends the assistant's failed response and a concise repair prompt to the conversation for up to `LLM_MAX_REPAIR_ATTEMPTS` (default `2`) turns.
3. Crucially, the response cache writes **only after** Pydantic validation succeeds, preventing transient malformed responses from poisoning cached retries. If all repair attempts fail, the document transitions safely to `needs_review` with full telemetry preserved.

### 2.3 Untrusted Data Isolation & Prompt-Injection Defense
Every invoice or receipt is untrusted external input ([ADR-0005](adr/0005-untrusted-document-isolation-and-injection-defense.md)):
- **Privilege Separation:** The LLM has **no tool definitions**, no database handle, and no `status` field in `ExtractedDocument`. It cannot approve payments or invoke webhooks.
- **Boundary Tagging:** OCR text is enclosed inside `<untrusted_document_text>...</untrusted_document_text>` tags with system-level instructions directing the model to ignore imperatives inside vendor text.
- **Multi-Layer Scanner (`src/doc_pipeline/safety/injection.py`):** Scans native embedded PDF text layers (catching invisible white-on-white text), OCR text, and extracted string fields (`handwritten_notes`, line-item descriptions) for instruction-override, role-hijack, and payment-diversion patterns. Any match sets `SafetyReport.quarantined = True`, forces `status = "needs_review"`, blocks ERP export, and records a high-severity `AuditLog` event.

---

## 3. Deterministic Business Validation (`src/doc_pipeline/validation/rules.py`)

Five pure-Python rules run in `< 5 ms` per document with zero model calls:

1. **`LINE_ITEMS_SUM` (`error`):** Checks $|\sum_{i} \text{line\_total}_i - \text{subtotal}| \le 0.05$ AND per-row $|\text{qty}_i \times \text{unit\_price}_i - \text{line\_total}_i| \le 0.05$. Catches dropped page-2 items, misread quantities, and shifted decimal points.
2. **`TAX_ARITHMETIC` (`error` / `warning`):** Checks $|(\text{subtotal} + \text{tax\_amount}) - \text{total\_amount}| \le 0.05$ (`error`) and validates the implied effective tax rate (`0%–27%` commercial VAT/sales-tax band; `warning` if unusual).
3. **`SENSIBLE_DATES` (`error`):** Rejects `issue_date` more than 7 days in the future, older than `MAX_INVOICE_AGE_DAYS` (default `365`), or `due_date < issue_date`.
4. **`SUPPLIER_MASTER_MATCH` (`warning`):** Matches `supplier_tax_id` (exact normalized VAT/EIN) or `supplier_name` (`RapidFuzz` `WRatio >= 82.0`) against the `suppliers` master table (`12` seeded EN/DE vendors), flagging inactive/unapproved vendors or currency mismatches.
5. **`DUPLICATE_DOCUMENT` (`error`):** Queries PostgreSQL for any prior non-rejected document sharing `(canonical_supplier_key, normalized_document_number, round(total_amount, 2))` with a different `document_id`. Catches cross-channel duplicates (e.g., PDF upload followed days later by a smartphone photo of the printout).

---

## 4. Calibrated Per-Field Confidence (`src/doc_pipeline/confidence/scorer.py`)

Rather than asking the model `"How confident are you?"`, each field $f$ receives a calibrated score $C_f \in [0, 1]$ built from three orthogonal signals ([ADR-0003](adr/0003-two-pass-agreement-and-rule-coupled-confidence.md)):

$$C_f = \operatorname{clamp}_{[0,1]}\Big(0.55 \cdot A_f + 0.35 \cdot V_f + 0.10 \cdot S_f - P_{\text{repair}}\Big)$$

Where:
- **$A_f \in [0, 1]$ — Cross-Pass Agreement (`55%` weight):**
  - Numeric fields (`subtotal`, `tax_amount`, `total_amount`): `1.0` if $|v_1 - v_2| \le 0.01$, `0.5` if within `1%`, else `0.0`.
  - Dates & currency (`issue_date`, `due_date`, `currency`, `document_type`): `1.0` on exact canonical match, else `0.0`.
  - Strings (`supplier_name`, `document_number`, `supplier_tax_id`): normalized `RapidFuzz` similarity (`1.0` when $\ge 0.95$, linear down to `0.0`).
  - Line-item table (`line_items`): row-count match + mean row-level `(qty, unit_price, line_total, description)` agreement.
  - *(When `ENABLE_TWO_PASS=false`, $A_f$ defaults conservatively to `0.78` so single-pass runs rely more heavily on deterministic rule coupling.)*
- **$V_f \in [0, 1]$ — Rule-Coupled Validation Score (`35%` weight):**
  - Starts at `1.0`. Each failed rule mapped to field $f$ deducts **`0.75`** (`severity="error"`, plus an explicit `-0.25` coupling floor penalty) or **`0.35`** (`severity="warning"`).
  - Example: If a receipt has a tampered or misread tax line (`subtotal=100.00, tax=19.00, total=129.00`), `TAX_ARITHMETIC` fails (`error`), penalizing `subtotal`, `tax_amount`, and `total_amount` simultaneously down to `~0.48` even if Pass 1 and Pass 2 both read `129.00` identically!
- **$S_f \in [0, 1]$ — Schema Presence (`10%` weight):** `1.0` when non-null and canonically formatted (`0.85` for legitimately optional `None` fields like receipt `due_date`).
- **$P_{\text{repair}}$ — Repair Penalty:** `0.06` deducted per Pydantic repair retry required during extraction.

### Document Routing Decision
$$\text{Overall Confidence} = \min\Big(\operatorname{mean}_{f}(C_f),\; \min_{f \in \text{Critical}}(C_f) + 0.05\Big)$$

A document is auto-approved (`status = "approved"`) **if and only if**:
1. `Overall Confidence >= REVIEW_CONFIDENCE_THRESHOLD` (`0.85`), **and**
2. `validation.passed == True` (zero `error`-severity rule violations), **and**
3. `safety.quarantined == False` (zero prompt-injection matches).

Otherwise, it routes to `status = "needs_review"`.

---

## 5. Evaluation Methodology (`src/doc_pipeline/eval/runner.py`)

1. **Deterministic Stratified Corpus (`seed=42`, 200 Documents):** Generated via ReportLab + Pillow (`src/doc_pipeline/synthetic/generator.py`, [ADR-0004](adr/0004-vision-primary-with-tesseract-ocr-fallback.md)) across 3 document types (`97` invoices, `56` purchase orders, `47` receipts), 2 languages (`136` English, `64` German), and 4 physical conditions (`85` clean PDFs, `23` multi-page PDFs, `42` skewed/speckled scans, `50` perspective-warped/shadowed phone photos), plus controlled anomalies (intentional tax/math errors, stale dates, unknown vendors, duplicate submissions, and embedded prompt injections).
2. **Header Field Precision / Recall / F1:** Evaluated across all 10 commercial header fields per document (`2,000+` field instances on the full corpus). Numeric fields match within `±0.01`; dates and ISO currencies require exact match; supplier names match at `token_sort_ratio >= 88`.
3. **Line-Item Table Accuracy:** Greedy best-first bipartite matching pairs predicted rows against ground-truth rows requiring simultaneous numeric match (`quantity`, `unit_price`, `line_total` within `±0.02`) and description fuzzy match (`>= 75`), reporting micro-averaged Precision, Recall, F1, and **Exact Table Match Rate** (percentage of documents with zero false-positive or false-negative rows).
4. **Anomaly Routing Recall:** Measures the fraction of documents containing ground-truth accounting errors, duplicates, or prompt injections that were prevented from straight-through auto-approval (`needs_review`).
5. **CI Regression Gate (`.github/workflows/ci.yml`):** Runs a 24-document stratified subset (`--subset ci --in-process-mock --fail-on-regression`) on every push, failing the build if Header Field F1 `< 0.92`, Line-Item F1 `< 0.90`, or Anomaly Review Recall `< 1.00`.

---

## 6. Unit Economics & Monthly Cost at 5,000 Documents/Month

Measured token consumption across our benchmark (`3,202` avg input tokens and `569` avg output tokens per document on two-pass Vision; `1,156` avg input tokens and `565` avg output tokens on two-pass Tesseract OCR fallback):

| Operating Mode | Avg Prompt Tokens / Doc | Avg Output Tokens / Doc | Cost / Doc | Monthly Cost (5,000 Docs) | Notes |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Manual AP Data Entry Baseline** | — | — | ~$2.50–$4.00 | **$12,500–$20,000/mo** | 4 min/doc @ $38–$60/hr loaded clerk cost |
| **Pipeline — Compact Cloud VLM** *(e.g. `gpt-4o-mini` / `gemini-2.5-flash` @ `$0.15/$0.60` per 1M)* | 3,202 | 569 | **$0.00082** | **$4.10/mo** | Full 2-pass Vision + classification |
| **Pipeline — OCR Fallback + Text LLM** *(Tesseract + `$0.15/$0.60` per 1M)* | 1,156 | 565 | **$0.00051** | **$2.55/mo** | **63.9% fewer input tokens**, +0.9s CPU OCR |
| **Pipeline — Frontier Cloud VLM** *(e.g. `$2.50/$10.00` per 1M)* | 3,202 | 569 | **$0.01370** | **$68.48/mo** | Useful escalation tier for hard reviews |
| **Dedicated Cloud Document AI** *(e.g. `$0.01–$0.03` per page, avg 1.25 pages/doc)* | — | — | **$0.0125–$0.0375** | **$62.50–$187.50/mo** | Prebuilt invoice OCR APIs |
| **Self-Hosted Local Ollama** *(`qwen2.5vl:3b` / `qwen2.5:3b`)* | 1,680 *(1-pass)* | 310 *(1-pass)* | **$0.00 API** | **Infra only (~$35–$120/mo GPU VM)** | Full air-gapped data residency |

Even factoring in ~12–18% human review handling (~30 seconds per flagged exception vs. 4 minutes of full manual entry), total AP processing effort drops by **>80%** for under **$5–$70/month** in model spend.

---

## 7. Trade-Offs: General Vision LLM vs. Dedicated OCR / Document AI Service

A key architectural question is whether to send page images directly to a multimodal LLM (`vision_strict_schema`) or use a specialized OCR / Document AI service (Azure Document Intelligence, AWS Textract, Google Document AI, or open-source Tesseract / Surya / Docling).

### Why General Vision Model Primary + OCR Fallback Won Here ([ADR-0004](adr/0004-vision-primary-with-tesseract-ocr-fallback.md))
1. **Layout Generalization Across Languages:** Vision models jointly see spatial table alignment, currency glyphs (`€` vs `$`), handwritten margin notes, and multi-column headers without brittle reading-order serialization mistakes. On degraded scans (`noisy_scan`), our OCR fallback benchmark showed real Tesseract character errors dropping `noisy_scan` Line-Item F1 whereas spatial vision preserved structure.
2. **Single Vendor-Neutral Contract:** Supporting Groq, Vertex AI, Azure Foundry, and local Ollama through one OpenAI-compatible interface avoids locking customers into a cloud-specific Document AI SDK while retaining automatic local Tesseract fallback whenever a text-only model is configured.
3. **Lower Cost at Mid-Market Scale:** At 5,000 docs/month, compact VLMs (`$0.00082/doc`) cost **15×–45× less** than commercial per-page invoice extraction APIs (`$0.01–$0.03/page`).

### Where Dedicated OCR / Document AI Wins
1. **Exact Word-Level Bounding Boxes:** Commercial Document AI APIs return exact polygon coordinates `(x1, y1, ..., x4, y4)` for every extracted token, enabling click-to-zoom field highlighting on 20-page logistics invoices.
2. **20–100 Page Invoices:** Vision context windows and latency scale steeply past 10 high-DPI pages; layout OCR + page-chunked text extraction scales linearly and cuts prompt tokens by **~64%**.
3. **Deterministic Character Fidelity on Tiny Fonts:** Dedicated 300-DPI OCR models rarely hallucinate plausible-sounding vendor names on blurry text; instead they output literal garbled characters that deterministic fuzzy matching cleanly flags.

---

## 8. What Would Change for Enterprise Production

Moving from this single-repo Docker Compose deployment to a regulated multi-entity enterprise environment entails six targeted upgrades:

1. **Three-Way ERP Matching (PO <-> Goods Receipt <-> Invoice):** Extend `validation/rules.py` beyond supplier master matching to query live ERP Purchase Orders and Goods Receipt Notes (SAP S/4HANA, NetSuite, Dynamics 365) for line-level quantity and unit-price variance tolerances.
2. **Enterprise Identity, RBAC & Dual-Control Approval:** Put OIDC/Entra ID authentication in front of FastAPI, enforce role separation (`ap_reviewer` vs. `ap_approver` for invoices `> $25,000`), and bind `reviewer` identity in `review_corrections` and `audit_logs` to verified JWT claims.
3. **Database Migrations & Partitioned Queueing:** Add Alembic schema migrations, configure PgBouncer connection pooling, add a periodic stale-lock reaper (`status='processing' AND locked_at < now() - interval '10 minutes'`), and partition `llm_call_metrics` and `audit_logs` by month.
4. **Tiered Model Escalation:** Run Pass 1 on a fast, low-cost compact VLM (`gemini-2.5-flash` / `gpt-4o-mini`) and run Pass 2 (or repair retries) on a distinct model family or higher-tier model—eliminating correlated blind spots between Pass 1 and Pass 2 while spending frontier tokens only where needed.
5. **Word-Level Bounding Box Overlays in Review UI:** Run lightweight local layout detection (e.g., Surya / DocTR word boxes) alongside vision extraction so clicking `tax_amount` in the Review Console highlights the exact rectangle on the PDF canvas.
6. **PII Lifecycle, KMS Encryption & Zero-Data-Retention:** Enable SSE-KMS on S3/MinIO, enforce Zero-Data-Retention (ZDR) agreements on the configured `LLM_BASE_URL`, hash/redact IBANs and employee names in application logs, and run automated TTL jobs to purge `llm_response_cache` and rasterized `pages/*.png` after statutory archive transfer.

---

## 9. Architecture Decision Records Index

- [**ADR-0001:** Single OpenAI-Compatible Provider & Startup Capability Probe](adr/0001-openai-compatible-provider-and-startup-probe.md)
- [**ADR-0002:** PostgreSQL `FOR UPDATE SKIP LOCKED` Worker Queue & MinIO Object Storage](adr/0002-postgres-skip-locked-worker-queue-and-minio.md)
- [**ADR-0003:** Two-Pass Agreement & Rule-Coupled Per-Field Confidence Scoring](adr/0003-two-pass-agreement-and-rule-coupled-confidence.md)
- [**ADR-0004:** Multimodal Vision Primary Path with Local Tesseract OCR Fallback](adr/0004-vision-primary-with-tesseract-ocr-fallback.md)
- [**ADR-0005:** Untrusted Document Isolation & Prompt-Injection Quarantine](adr/0005-untrusted-document-isolation-and-injection-defense.md)
- [**ADR-0006:** Human Review Feedback Loop to Automated Evaluation Corpus](adr/0006-human-review-feedback-loop-to-eval-corpus.md)
