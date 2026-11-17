"""Strict Pydantic schemas for extraction, validation, confidence, safety, and API I/O."""

from __future__ import annotations

import copy
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentType = Literal["invoice", "receipt", "purchase_order"]
LanguageCode = Literal["en", "de"]
CurrencyCode = Literal["USD", "EUR", "GBP"]
DocumentStatus = Literal[
    "queued",
    "processing",
    "approved",
    "needs_review",
    "rejected",
    "failed",
]


class ClassificationResult(BaseModel):
    """Document type classification result."""

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    language: LanguageCode
    reasoning_brief: str = Field(min_length=1, max_length=240)


class LineItem(BaseModel):
    """Single commercial line item on an invoice, receipt, or purchase order."""

    model_config = ConfigDict(extra="forbid")

    sku: str | None = None
    description: str = Field(min_length=1, max_length=300)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    line_total: float = Field(ge=0)

    @field_validator("quantity", "unit_price", "line_total")
    @classmethod
    def round_monetary(cls, v: float) -> float:
        return round(float(v), 4)


class ExtractedDocument(BaseModel):
    """Canonical strict extraction schema enforced on every AP document.

    Notice privilege separation: status, approval, payment routing, and bank accounts
    are intentionally excluded so the model has zero authority over payment approval.
    """

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    language: LanguageCode
    document_number: str = Field(min_length=1, max_length=80)
    supplier_name: str = Field(min_length=1, max_length=200)
    supplier_tax_id: str | None = Field(default=None, max_length=64)
    supplier_address: str | None = Field(default=None, max_length=300)
    issue_date: date
    due_date: date | None = None
    currency: CurrencyCode
    subtotal: float = Field(ge=0)
    tax_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    tax_amount: float = Field(ge=0)
    total_amount: float = Field(gt=0)
    line_items: list[LineItem] = Field(min_length=1)
    handwritten_notes: str | None = Field(default=None, max_length=500)

    @field_validator("subtotal", "tax_amount", "total_amount")
    @classmethod
    def round_currency(cls, v: float) -> float:
        return round(float(v), 2)

    @field_validator("tax_rate")
    @classmethod
    def normalize_tax_rate(cls, v: float | None) -> float | None:
        if v is None:
            return None
        return round(float(v), 4)

    @field_validator("document_number", "supplier_name")
    @classmethod
    def strip_non_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Field cannot be blank or whitespace only")
        return cleaned


class ValidationRuleResult(BaseModel):
    """Outcome of a single deterministic business validation check."""

    rule_code: str
    passed: bool
    severity: Literal["error", "warning", "info"]
    fields: list[str]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Aggregated deterministic business validation results."""

    passed: bool
    matched_supplier_id: str | None = None
    matched_supplier_name: str | None = None
    supplier_match_score: float = 0.0
    is_duplicate: bool = False
    duplicate_of_document_id: str | None = None
    results: list[ValidationRuleResult]


class FieldConfidence(BaseModel):
    """Calibrated confidence breakdown for a single extracted field."""

    field_name: str
    score: float = Field(ge=0.0, le=1.0)
    two_pass_agreed: bool
    agreement_score: float = Field(ge=0.0, le=1.0)
    validation_score: float = Field(ge=0.0, le=1.0)
    corroboration_score: float = Field(ge=0.0, le=1.0)
    pass1_value: Any = None
    pass2_value: Any = None
    reasons: list[str] = Field(default_factory=list)


class ConfidenceReport(BaseModel):
    """Document-level and per-field confidence assessment."""

    overall_confidence: float = Field(ge=0.0, le=1.0)
    requires_review: bool
    review_reasons: list[str]
    repair_attempts_used: int = 0
    two_pass_executed: bool = True
    fields: dict[str, FieldConfidence]


class InjectionSignal(BaseModel):
    """Detected prompt-injection or instruction-hijacking indicator."""

    pattern_name: str
    source_location: str
    matched_excerpt: str
    severity: Literal["high", "medium"]


class SafetyReport(BaseModel):
    """Security and untrusted-data inspection report for an ingested document."""

    injection_detected: bool = False
    quarantined_for_review: bool = False
    signals: list[InjectionSignal] = Field(default_factory=list)
    pii_redacted_in_logs: bool = True


class ReviewSubmissionRequest(BaseModel):
    """Payload submitted by a human reviewer in the Review UI."""

    action: Literal["approve", "reject"]
    reviewer: str = Field(default="ap-reviewer@company.local", min_length=1, max_length=120)
    reviewer_notes: str | None = Field(default=None, max_length=1000)
    extracted_data: ExtractedDocument


class CapabilityProfile(BaseModel):
    """Result of startup probing against the configured OpenAI-compatible endpoint."""

    base_url: str
    model: str
    reachable: bool
    supports_vision: bool
    supports_json_schema: bool
    supports_json_object: bool
    chosen_extraction_path: Literal[
        "vision_strict_schema",
        "vision_json_repair",
        "ocr_strict_schema",
        "ocr_json_repair",
    ]
    probe_latency_ms: float
    notes: list[str] = Field(default_factory=list)
    probed_at: datetime


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Inline JSON Schema $defs/$ref and enforce OpenAI/Groq/Azure strict schema rules."""
    if isinstance(node, dict):
        if "$ref" in node:
            ref_path = node["$ref"]
            ref_name = ref_path.rsplit("/", 1)[-1]
            resolved = copy.deepcopy(defs[ref_name])
            return _inline_refs(resolved, defs)

        cleaned: dict[str, Any] = {}
        is_date_fmt = node.get("format") == "date"
        # Strip value-bound facets unsupported by OpenAI/Azure strict json_schema and
        # known to cause O(N^2) GBNF grammar state blowup in Ollama / llama.cpp;
        # Pydantic enforces all range/length/date constraints post-parse in Python.
        stripped_keys = {
            "$defs",
            "title",
            "default",
            "minLength",
            "maxLength",
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "minItems",
            "maxItems",
            "format",
        }
        for k, v in node.items():
            if k in stripped_keys:
                continue
            cleaned[k] = _inline_refs(v, defs)
        if is_date_fmt and "description" not in cleaned:
            cleaned["description"] = "ISO 8601 date formatted YYYY-MM-DD"

        # Convert anyOf: [{"type": T}, {"type": "null"}] into clean strict nullable schema
        if "anyOf" in cleaned and isinstance(cleaned["anyOf"], list):
            cleaned["anyOf"] = [_inline_refs(sub, defs) for sub in cleaned["anyOf"]]

        if cleaned.get("type") == "object" and "properties" in cleaned:
            props = cleaned["properties"]
            cleaned["additionalProperties"] = False
            cleaned["required"] = list(props.keys())

        return cleaned
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def to_strict_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Generate a strict-mode JSON Schema compatible across OpenAI, Groq, Azure, Vertex, and Ollama.

    All objects set `additionalProperties: false` and list all keys in `required`
    (with optional fields represented as nullable `anyOf` unions).
    """
    raw = model_cls.model_json_schema()
    defs = raw.get("$defs", {})
    strict_schema = _inline_refs(raw, defs)
    strict_schema["additionalProperties"] = False
    if "properties" in strict_schema:
        strict_schema["required"] = list(strict_schema["properties"].keys())
    return strict_schema
