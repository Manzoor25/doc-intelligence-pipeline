"""Per-field and document-level confidence engine.

Built exclusively from:
  1. Agreement between two independent extraction passes (weight 0.55)
  2. Deterministic business-rule validation outcomes (weight 0.35)
  3. Master-table & schema-stability corroboration (weight 0.10)
Never relies on self-reported LLM confidence numbers.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from doc_pipeline.schemas import (
    ConfidenceReport,
    ExtractedDocument,
    FieldConfidence,
    LineItem,
    SafetyReport,
    ValidationReport,
)
from doc_pipeline.validation.rules import normalize_doc_number, normalize_tax_id

CRITICAL_FIELDS: tuple[str, ...] = (
    "document_type",
    "document_number",
    "supplier_name",
    "issue_date",
    "currency",
    "subtotal",
    "tax_amount",
    "total_amount",
    "line_items",
)

ALL_SCORED_FIELDS: tuple[str, ...] = (
    *CRITICAL_FIELDS,
    "supplier_tax_id",
    "due_date",
)


def _compare_line_items(items1: list[LineItem], items2: list[LineItem]) -> tuple[bool, float]:
    if len(items1) != len(items2):
        overlap = min(len(items1), len(items2)) / max(1, max(len(items1), len(items2)))
        return False, round(0.45 * overlap, 4)

    row_scores: list[float] = []
    exact_all = True
    for a, b in zip(items1, items2, strict=True):
        desc_sim = fuzz.token_sort_ratio(a.description.lower(), b.description.lower()) / 100.0
        nums_match = (
            abs(a.quantity - b.quantity) <= 0.01
            and abs(a.unit_price - b.unit_price) <= 0.02
            and abs(a.line_total - b.line_total) <= 0.02
        )
        if nums_match and desc_sim >= 0.90:
            row_scores.append(1.0)
        elif nums_match:
            exact_all = False
            row_scores.append(0.75 + 0.20 * desc_sim)
        else:
            exact_all = False
            row_scores.append(0.20 * desc_sim)

    return exact_all, round(sum(row_scores) / max(1, len(row_scores)), 4)


def _field_agreement(field: str, v1: Any, v2: Any) -> tuple[bool, float]:
    if v1 is None and v2 is None:
        return True, 1.0
    if v1 is None or v2 is None:
        return False, 0.30

    if field == "line_items":
        return _compare_line_items(v1, v2)

    if field in ("subtotal", "tax_amount", "total_amount"):
        diff = abs(float(v1) - float(v2))
        if diff <= 0.01:
            return True, 1.0
        if diff <= 0.05:
            return False, 0.70
        return False, 0.20

    if field == "document_number":
        agreed = normalize_doc_number(str(v1)) == normalize_doc_number(str(v2))
        if agreed:
            return True, 1.0
        sim = fuzz.ratio(str(v1).upper(), str(v2).upper()) / 100.0
        return False, round(0.45 * sim, 4)

    if field == "supplier_tax_id":
        agreed = normalize_tax_id(str(v1)) == normalize_tax_id(str(v2))
        return (True, 1.0) if agreed else (False, 0.25)

    if field == "supplier_name":
        sim = fuzz.token_sort_ratio(str(v1).lower(), str(v2).lower()) / 100.0
        return (sim >= 0.92), round(sim, 4)

    # Exact discrete/date/enum fields
    agreed = str(v1) == str(v2)
    return (True, 1.0) if agreed else (False, 0.20)


def _serialize_val(field: str, val: Any) -> Any:
    if field == "line_items" and isinstance(val, list):
        return [it.model_dump(mode="json") if isinstance(it, LineItem) else it for it in val]
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return val


def score_confidence(
    *,
    pass1: ExtractedDocument,
    pass2: ExtractedDocument | None,
    validation: ValidationReport,
    safety: SafetyReport,
    repair_attempts: int = 0,
    threshold: float = 0.85,
) -> ConfidenceReport:
    """Calculate calibrated per-field confidence and decide human-review routing."""
    failed_error_fields: dict[str, list[str]] = {}
    failed_warn_fields: dict[str, list[str]] = {}
    for rule in validation.results:
        if not rule.passed:
            target_map = failed_error_fields if rule.severity == "error" else failed_warn_fields
            for f in rule.fields:
                target_map.setdefault(f, []).append(rule.rule_code)

    repair_penalty = min(0.24, 0.08 * repair_attempts)
    field_reports: dict[str, FieldConfidence] = {}

    for fname in ALL_SCORED_FIELDS:
        v1 = getattr(pass1, fname)
        v2 = getattr(pass2, fname) if pass2 is not None else None
        reasons: list[str] = []

        if pass2 is not None:
            agreed, agreement_score = _field_agreement(fname, v1, v2)
            if agreed:
                reasons.append("Pass-1 and Pass-2 agreed.")
            else:
                reasons.append("Disagreement detected between Pass-1 and Pass-2.")
        else:
            agreed = True
            agreement_score = 0.82
            reasons.append("Single-pass mode configured (no second-pass cross-check).")

        err_rules = failed_error_fields.get(fname, [])
        warn_rules = failed_warn_fields.get(fname, [])
        if err_rules:
            validation_score = 0.20
            reasons.append(f"Failed validation rule(s): {', '.join(err_rules)}.")
        elif warn_rules:
            validation_score = 0.65
            reasons.append(f"Validation warning(s): {', '.join(warn_rules)}.")
        else:
            validation_score = 1.0
            reasons.append("Corroborated by deterministic validation rules.")

        # Corroboration bonus/penalty
        corroboration = 0.90
        if fname in ("supplier_name", "supplier_tax_id") and validation.matched_supplier_id:
            corroboration = min(1.0, validation.supplier_match_score / 100.0)
        elif fname in ("subtotal", "tax_amount", "total_amount", "line_items") and not err_rules:
            corroboration = 1.0
        corroboration = max(0.0, round(corroboration - repair_penalty, 4))
        if repair_attempts > 0:
            reasons.append(f"Schema repair penalty applied ({repair_attempts} retry/retries).")

        raw_score = 0.55 * agreement_score + 0.35 * validation_score + 0.10 * corroboration
        if err_rules:
            # Hard business validation error caps implicated field confidence below review threshold
            raw_score = min(raw_score, 0.48)
        elif not agreed and fname in CRITICAL_FIELDS:
            raw_score = min(raw_score, 0.72)

        field_reports[fname] = FieldConfidence(
            field_name=fname,
            score=round(max(0.0, min(1.0, raw_score)), 4),
            two_pass_agreed=agreed,
            agreement_score=round(agreement_score, 4),
            validation_score=round(validation_score, 4),
            corroboration_score=round(corroboration, 4),
            pass1_value=_serialize_val(fname, v1),
            pass2_value=_serialize_val(fname, v2) if pass2 is not None else None,
            reasons=reasons,
        )

    overall = round(min(field_reports[k].score for k in CRITICAL_FIELDS), 4)
    review_reasons: list[str] = []

    if safety.injection_detected:
        review_reasons.append(
            "Safety quarantine: embedded prompt-injection instructions detected in document."
        )
    for rule in validation.results:
        if not rule.passed and rule.severity == "error":
            review_reasons.append(f"[{rule.rule_code}] {rule.message}")
    low_conf_fields = [
        f"{k} ({field_reports[k].score:.2f})"
        for k in CRITICAL_FIELDS
        if field_reports[k].score < threshold
    ]
    if low_conf_fields:
        review_reasons.append(
            f"Field confidence below {threshold:.2f} threshold: {', '.join(low_conf_fields)}"
        )

    requires_review = (
        safety.injection_detected or (not validation.passed) or (overall < threshold)
    )

    return ConfidenceReport(
        overall_confidence=overall,
        requires_review=requires_review,
        review_reasons=review_reasons,
        repair_attempts_used=repair_attempts,
        two_pass_executed=pass2 is not None,
        fields=field_reports,
    )
