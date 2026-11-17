"""Business validation engine requiring zero LLM calls.

Checks line-item multiplication, subtotal summation, tax arithmetic, date sanity,
master-supplier verification, and duplicate invoice detection.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_pipeline.schemas import ExtractedDocument, ValidationReport, ValidationRuleResult
from doc_pipeline.storage.models import DocumentRecord, SupplierMaster

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


def normalize_doc_number(doc_number: str) -> str:
    return _NON_ALNUM_RE.sub("", doc_number.upper().strip())


def normalize_tax_id(tax_id: str | None) -> str:
    if not tax_id:
        return ""
    return _NON_ALNUM_RE.sub("", tax_id.upper().strip())


def validate_document(
    doc: ExtractedDocument,
    session: Session,
    *,
    current_document_id: str | None = None,
    reference_date: date | None = None,
    max_age_days: int = 365,
) -> ValidationReport:
    """Run all deterministic AP business checks against an extracted document."""
    today = reference_date or date.today()
    results: list[ValidationRuleResult] = []

    # 1. Each line item: qty * unit_price == line_total
    bad_lines: list[dict[str, float | int]] = []
    for idx, item in enumerate(doc.line_items):
        expected_line = round(item.quantity * item.unit_price, 2)
        if abs(expected_line - round(item.line_total, 2)) > 0.03:
            bad_lines.append(
                {
                    "index": idx,
                    "expected": expected_line,
                    "actual": round(item.line_total, 2),
                }
            )
    results.append(
        ValidationRuleResult(
            rule_code="LINE_ITEM_MATH",
            passed=len(bad_lines) == 0,
            severity="error",
            fields=["line_items"],
            message=(
                f"All {len(doc.line_items)} line items satisfy qty * unit_price == line_total."
                if not bad_lines
                else f"{len(bad_lines)} line item(s) failed qty * unit_price multiplication."
            ),
            details={"mismatches": bad_lines},
        )
    )

    # 2. Sum of line_items == subtotal
    summed_lines = round(sum(it.line_total for it in doc.line_items), 2)
    subtotal_diff = round(abs(summed_lines - doc.subtotal), 2)
    subtotal_ok = subtotal_diff <= 0.03
    results.append(
        ValidationRuleResult(
            rule_code="LINE_ITEMS_SUM",
            passed=subtotal_ok,
            severity="error",
            fields=["line_items", "subtotal"],
            message=(
                f"Line items sum ({summed_lines:.2f}) matches subtotal ({doc.subtotal:.2f})."
                if subtotal_ok
                else f"Line items sum ({summed_lines:.2f}) differs from subtotal ({doc.subtotal:.2f}) by {subtotal_diff:.2f}."
            ),
            details={"summed_lines": summed_lines, "subtotal": doc.subtotal, "diff": subtotal_diff},
        )
    )

    # 3. Tax & grand-total arithmetic: subtotal + tax_amount == total_amount (and rate check)
    expected_total = round(doc.subtotal + doc.tax_amount, 2)
    total_diff = round(abs(expected_total - doc.total_amount), 2)
    rate_ok = True
    expected_tax: float | None = None
    if doc.tax_rate is not None:
        expected_tax = round(doc.subtotal * doc.tax_rate, 2)
        rate_tol = max(0.05, round(doc.subtotal * 0.0025, 2))
        rate_ok = abs(expected_tax - doc.tax_amount) <= rate_tol

    tax_ok = (total_diff <= 0.03) and rate_ok
    results.append(
        ValidationRuleResult(
            rule_code="TAX_ARITHMETIC",
            passed=tax_ok,
            severity="error",
            fields=["subtotal", "tax_rate", "tax_amount", "total_amount"],
            message=(
                f"Subtotal ({doc.subtotal:.2f}) + tax ({doc.tax_amount:.2f}) == total ({doc.total_amount:.2f})."
                if tax_ok
                else (
                    f"Arithmetic mismatch: subtotal ({doc.subtotal:.2f}) + tax ({doc.tax_amount:.2f}) "
                    f"= {expected_total:.2f}, printed total is {doc.total_amount:.2f}"
                    + (
                        f" (expected tax from rate {doc.tax_rate:.4f}: {expected_tax:.2f})."
                        if expected_tax is not None and not rate_ok
                        else "."
                    )
                )
            ),
            details={
                "expected_total": expected_total,
                "actual_total": doc.total_amount,
                "expected_tax_from_rate": expected_tax,
            },
        )
    )

    # 4. Sensible dates check
    date_issues: list[str] = []
    if doc.issue_date > today + timedelta(days=1):
        date_issues.append(f"issue_date ({doc.issue_date.isoformat()}) is in the future")
    age_days = (today - doc.issue_date).days
    if age_days > max_age_days:
        date_issues.append(
            f"issue_date ({doc.issue_date.isoformat()}) is {age_days} days old (> {max_age_days}d policy)"
        )
    if doc.due_date is not None:
        if doc.due_date < doc.issue_date:
            date_issues.append(
                f"due_date ({doc.due_date.isoformat()}) precedes issue_date ({doc.issue_date.isoformat()})"
            )
        elif (doc.due_date - doc.issue_date).days > 365:
            date_issues.append("payment term exceeds 365 days")

    dates_ok = len(date_issues) == 0
    results.append(
        ValidationRuleResult(
            rule_code="SENSIBLE_DATES",
            passed=dates_ok,
            severity="error",
            fields=["issue_date", "due_date"],
            message=(
                "Document issue_date and due_date are within AP policy bounds."
                if dates_ok
                else "Date validation failed: " + "; ".join(date_issues)
            ),
            details={"issues": date_issues},
        )
    )

    # 5. Supplier matched against master table
    suppliers = session.scalars(select(SupplierMaster)).all()
    doc_tax_norm = normalize_tax_id(doc.supplier_tax_id)
    matched_supplier: SupplierMaster | None = None
    best_score = 0.0

    for sup in suppliers:
        if doc_tax_norm and normalize_tax_id(sup.tax_id) == doc_tax_norm:
            matched_supplier = sup
            best_score = 100.0
            break
        candidates = [sup.name, *(sup.aliases or [])]
        for cand in candidates:
            score = float(fuzz.token_sort_ratio(doc.supplier_name.lower(), cand.lower()))
            if score > best_score:
                best_score = score
                if score >= 88.0:
                    matched_supplier = sup

    supplier_ok = matched_supplier is not None and matched_supplier.active
    matched_id = matched_supplier.id if matched_supplier else None
    matched_name = matched_supplier.name if matched_supplier else None

    results.append(
        ValidationRuleResult(
            rule_code="SUPPLIER_MASTER_MATCH",
            passed=supplier_ok,
            severity="error",
            fields=["supplier_name", "supplier_tax_id"],
            message=(
                f"Matched active master vendor {matched_id} ({matched_name}, score={best_score:.1f})."
                if supplier_ok
                else f"Supplier '{doc.supplier_name}' (tax_id={doc.supplier_tax_id}) not found in active master table (best_score={best_score:.1f})."
            ),
            details={
                "matched_supplier_id": matched_id,
                "matched_supplier_name": matched_name,
                "match_score": round(best_score, 1),
            },
        )
    )

    if matched_supplier and matched_supplier.default_currency != doc.currency:
        results.append(
            ValidationRuleResult(
                rule_code="SUPPLIER_CURRENCY_CHECK",
                passed=False,
                severity="warning",
                fields=["currency", "supplier_name"],
                message=(
                    f"Currency {doc.currency} differs from master vendor default "
                    f"{matched_supplier.default_currency}."
                ),
                details={"expected_currency": matched_supplier.default_currency},
            )
        )

    # 6. Duplicate invoice / document detection across non-rejected ingested records
    norm_num = normalize_doc_number(doc.document_number)
    duplicate_of_id: str | None = None
    stmt = select(DocumentRecord).where(
        DocumentRecord.status.in_(["approved", "needs_review", "processing"])
    )
    if current_document_id:
        stmt = stmt.where(DocumentRecord.id != current_document_id)
    prior_docs = session.scalars(stmt).all()

    for prior in prior_docs:
        same_vendor = False
        if matched_id and prior.matched_supplier_id == matched_id:
            same_vendor = True
        elif prior.extracted_data:
            prior_sup = str(prior.extracted_data.get("supplier_name", ""))
            if fuzz.token_sort_ratio(doc.supplier_name.lower(), prior_sup.lower()) >= 90:
                same_vendor = True

        if not same_vendor:
            continue

        if norm_num and prior.document_number_norm == norm_num:
            duplicate_of_id = prior.id
            break

        # Secondary duplicate check: identical vendor + issue date + exact total on invoices
        if (
            doc.document_type == "invoice"
            and prior.document_type == "invoice"
            and prior.issue_date_iso == doc.issue_date.isoformat()
            and prior.total_amount is not None
            and abs(prior.total_amount - doc.total_amount) <= 0.01
        ):
            duplicate_of_id = prior.id
            break

    is_dup = duplicate_of_id is not None
    results.append(
        ValidationRuleResult(
            rule_code="DUPLICATE_INVOICE",
            passed=not is_dup,
            severity="error",
            fields=["document_number", "supplier_name", "total_amount"],
            message=(
                "No duplicate document found in ledger history."
                if not is_dup
                else f"Duplicate detected: matches existing document {duplicate_of_id} ({doc.document_number})."
            ),
            details={"duplicate_of_document_id": duplicate_of_id},
        )
    )

    hard_errors = [r for r in results if not r.passed and r.severity == "error"]
    return ValidationReport(
        passed=len(hard_errors) == 0,
        matched_supplier_id=matched_id,
        matched_supplier_name=matched_name,
        supplier_match_score=round(best_score, 1),
        is_duplicate=is_dup,
        duplicate_of_document_id=duplicate_of_id,
        results=results,
    )
