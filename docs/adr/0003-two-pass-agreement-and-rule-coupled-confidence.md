# ADR 0003: Two-Pass Agreement and Rule-Coupled Per-Field Confidence

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

Asking an LLM to output `"confidence": 0.98` alongside extracted values produces poorly calibrated numbers: models routinely emit `0.95+` while hallucinating transposed digits or misreading smudged totals. Accounts payable automation requires high precision on auto-approved invoices because false approvals trigger erroneous payments, whereas false rejections merely add a 20-second human review.

## Decision

Never request or use self-reported model confidence. Compute per-field confidence in `[0.0, 1.0]` deterministically in `src/doc_pipeline/confidence/scorer.py` from three independent signals:

1. **Two-pass extraction agreement (weight 0.55):** Execute two extraction passes with distinct prompt framing and sampling (`temperature=0.0` primary extraction vs `temperature=0.15` independent verification pass). Exact canonical match on numbers/dates/codes yields `1.0`; numeric disagreement yields `0.25` and records both candidates for the human reviewer; string fields (`supplier_name`, `description`) use normalized Levenshtein similarity (`rapidfuzz`).
2. **Deterministic accounting corroboration (weight 0.35):** Cross-field invariants (`sum(line_items) == subtotal`, `subtotal + tax_amount == total_amount`, `qty * unit_price == line_total`, `due_date >= issue_date`) boost corroborated numeric/date fields when satisfied and cap implicated fields below the review threshold (`<= 0.50`) when violated.
3. **Master-data & repair-loop signal (weight 0.10):** Matching an active vendor in `suppliers` (by tax ID or fuzzy name >= 88) adds corroboration; requiring Pydantic repair retries applies a `0.10 * retries` penalty.

Any document with `min(critical_field_confidences) < REVIEW_CONFIDENCE_THRESHOLD` (default `0.85`), any hard business rule failure, or any prompt-injection alert is routed to `needs_review`.

## Consequences

- **Positive:** Calibration correlates strongly with true field accuracy; arithmetic cross-checks catch single-digit OCR/vision slips even when both passes make correlated visual errors on one number.
- **Negative:** Running two passes roughly doubles token consumption per document (mitigated by content-addressed caching and negligible ~$0.001–$0.004/doc cost on modern flash/mini models). Can be toggled via `ENABLE_TWO_PASS=false` for low-risk triage.
