"""Indirect prompt-injection detector and PII log redaction utilities.

Every incoming PDF, scan, and OCR transcript is untrusted third-party input.
"""

from __future__ import annotations

import re

from doc_pipeline.schemas import ExtractedDocument, InjectionSignal, SafetyReport

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b.{0,45}\b(rules?|instructions?|prompts?|guidelines?|validations?|checks?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "high",
    ),
    (
        "system_role_spoof",
        re.compile(
            r"(\bsystem\s+override\b|\bimportant\s+ai\s+instruction\b|\bdeveloper\s+mode\s+enabled\b|<\|im_start\|>|\[INST\]|###\s*System\s*:)",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "payment_approval_hijack",
        re.compile(
            r"\b(approve\s+this\s+payment|mark\s+status\s+approved|pre-approved\s+executive\s+payment|skip\s+human\s+review|wire\s+\$?\d[\d,\.]*\s+to\s+attacker)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        "field_tamper_directive",
        re.compile(
            r"\b(set|change|output)\s+(total_amount|subtotal|supplier_name|status)\s+(to|as)\b",
            re.IGNORECASE,
        ),
        "high",
    ),
]

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def scan_for_prompt_injection(
    *,
    embedded_pdf_text: str = "",
    ocr_text: str = "",
    extracted: ExtractedDocument | None = None,
) -> SafetyReport:
    """Scan raw text layers and extracted strings for instructions aimed at the LLM."""
    signals: list[InjectionSignal] = []
    seen: set[tuple[str, str]] = set()

    sources: list[tuple[str, str]] = []
    if embedded_pdf_text:
        sources.append(("embedded_pdf_text", embedded_pdf_text))
    if ocr_text:
        sources.append(("ocr_text", ocr_text))
    if extracted is not None:
        sources.append(("extracted.supplier_name", extracted.supplier_name))
        if extracted.supplier_address:
            sources.append(("extracted.supplier_address", extracted.supplier_address))
        if extracted.handwritten_notes:
            sources.append(("extracted.handwritten_notes", extracted.handwritten_notes))
        for idx, item in enumerate(extracted.line_items):
            sources.append((f"extracted.line_items[{idx}].description", item.description))

    for location, text in sources:
        for pattern_name, regex, severity in _INJECTION_PATTERNS:
            match = regex.search(text)
            if match:
                key = (pattern_name, location.split(".")[0])
                if key in seen:
                    continue
                seen.add(key)
                snippet = " ".join(match.group(0).split())[:140]
                signals.append(
                    InjectionSignal(
                        pattern_name=pattern_name,
                        source_location=location,
                        matched_excerpt=snippet,
                        severity=severity,  # type: ignore[arg-type]
                    )
                )

    detected = len(signals) > 0
    return SafetyReport(
        injection_detected=detected,
        quarantined_for_review=detected,
        signals=signals,
        pii_redacted_in_logs=True,
    )


def redact_pii_for_log(text: str) -> str:
    """Mask email addresses, IBANs, and phone numbers before writing operational logs."""
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    out = _IBAN_RE.sub("[REDACTED_IBAN]", out)
    out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
    return out
