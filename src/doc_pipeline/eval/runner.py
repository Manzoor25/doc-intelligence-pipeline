"""Evaluation harness for the Accounts Payable Document Intelligence Pipeline.

Measures:
- Field-level Precision, Recall, and F1 across all commercial header fields
- Line-item extraction Precision, Recall, F1, and exact line-table match rate
- Anomaly / review-queue routing recall & precision
- Breakdown by document type (`invoice`, `receipt`, `purchase_order`) and physical condition
- Per-document cost (USD), token usage, and p50/mean/p95 latency (ms)
- Optional side-by-side comparison (`--compare-ocr` or model A vs model B)
- CI regression gate (`--fail-on-regression`)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from rapidfuzz import fuzz

from doc_pipeline.config import Settings, get_settings
from doc_pipeline.ingestion.service import ingest_document_bytes, process_document_record
from doc_pipeline.llm.provider import OpenAICompatibleProvider
from doc_pipeline.mock_services.openai_mock import app as mock_openai_app
from doc_pipeline.storage.blob import LocalFilesystemBlobStore
from doc_pipeline.storage.db import dispose_engine, init_db, make_session_factory
from doc_pipeline.synthetic.generator import REFERENCE_TODAY
from doc_pipeline.validation.rules import normalize_doc_number

HEADER_FIELDS: tuple[str, ...] = (
    "document_type",
    "document_number",
    "supplier_name",
    "supplier_tax_id",
    "issue_date",
    "due_date",
    "currency",
    "subtotal",
    "tax_amount",
    "total_amount",
)


@dataclass
class CountsPRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return round(self.tp / denom, 4) if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return round(self.tp / denom, 4) if denom > 0 else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return round((2 * p * r) / (p + r), 4) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class BucketAccumulator:
    docs: int = 0
    field_counts: CountsPRF = field(default_factory=CountsPRF)
    line_counts: CountsPRF = field(default_factory=CountsPRF)
    exact_line_tables: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.docs,
            "field_precision": self.field_counts.precision,
            "field_recall": self.field_counts.recall,
            "field_f1": self.field_counts.f1,
            "line_item_precision": self.line_counts.precision,
            "line_item_recall": self.line_counts.recall,
            "line_item_f1": self.line_counts.f1,
            "exact_table_match_rate": round(self.exact_line_tables / self.docs, 4)
            if self.docs
            else 0.0,
        }


def _norm_tax_id(val: Any) -> str | None:
    if val is None:
        return None
    cleaned = "".join(ch for ch in str(val).upper() if ch.isalnum())
    return cleaned or None


def _fields_match(field_name: str, expected: Any, predicted: Any) -> bool:
    if expected is None and predicted is None:
        return True
    if expected is None or predicted is None:
        return False

    if field_name in ("subtotal", "tax_amount", "total_amount"):
        try:
            return abs(float(expected) - float(predicted)) <= 0.02
        except (TypeError, ValueError):
            return False

    if field_name == "document_number":
        return normalize_doc_number(str(expected)) == normalize_doc_number(str(predicted))

    if field_name == "supplier_tax_id":
        return _norm_tax_id(expected) == _norm_tax_id(predicted)

    if field_name == "supplier_name":
        return fuzz.token_set_ratio(str(expected).lower(), str(predicted).lower()) >= 88.0

    return str(expected).strip().lower() == str(predicted).strip().lower()


def _score_field(counts: CountsPRF, field_name: str, expected: Any, predicted: Any) -> None:
    # Normalize empty strings to None
    exp = None if expected == "" else expected
    pred = None if predicted == "" else predicted

    if exp is None and pred is None:
        counts.tn += 1
    elif exp is None and pred is not None:
        counts.fp += 1
    elif exp is not None and pred is None:
        counts.fn += 1
    elif _fields_match(field_name, exp, pred):
        counts.tp += 1
    else:
        # Wrong non-null prediction is both a false positive and a missed ground-truth value
        counts.fp += 1
        counts.fn += 1


def _line_items_match(gt_item: dict[str, Any], pred_item: dict[str, Any]) -> bool:
    desc_score = fuzz.token_set_ratio(
        str(gt_item.get("description", "")).lower(),
        str(pred_item.get("description", "")).lower(),
    )
    if desc_score < 72.0:
        return False
    try:
        qty_ok = abs(float(gt_item["quantity"]) - float(pred_item["quantity"])) <= 0.02
        unit_ok = abs(float(gt_item["unit_price"]) - float(pred_item["unit_price"])) <= 0.05
        tot_ok = abs(float(gt_item["line_total"]) - float(pred_item["line_total"])) <= 0.05
        return qty_ok and unit_ok and tot_ok
    except (KeyError, TypeError, ValueError):
        return False


def _score_line_items(
    gt_lines: list[dict[str, Any]], pred_lines: list[dict[str, Any]]
) -> tuple[int, int, int, bool]:
    matched_pred: set[int] = set()
    tp = 0
    for gt in gt_lines:
        for j, pred in enumerate(pred_lines):
            if j in matched_pred:
                continue
            if _line_items_match(gt, pred):
                matched_pred.add(j)
                tp += 1
                break
    fp = len(pred_lines) - tp
    fn = len(gt_lines) - tp
    exact = tp == len(gt_lines) and fp == 0 and fn == 0
    return tp, fp, fn, exact


def load_eval_samples(
    synthetic_dir: str | Path,
    subset: str = "benchmark",
    limit: int | None = None,
    feedback_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(synthetic_dir)
    gt_dir = root / "ground_truth"
    docs_dir = root / "documents"
    if not gt_dir.exists():
        raise FileNotFoundError(
            f"Synthetic dataset not found at {gt_dir}. Run `python -m doc_pipeline.synthetic.generator` first."
        )

    samples: list[dict[str, Any]] = []
    for gt_path in sorted(gt_dir.glob("doc_*.json")):
        meta = json.loads(gt_path.read_text(encoding="utf-8"))
        if subset != "all" and subset not in meta.get("subsets", []):
            continue
        doc_path = docs_dir / meta["file_name"]
        samples.append(
            {
                "doc_id": meta["doc_id"],
                "file_name": meta["file_name"],
                "file_path": doc_path,
                "condition": meta.get("condition", "clean_pdf"),
                "anomaly_type": meta.get("anomaly_type", "none"),
                "expected_needs_review": bool(meta.get("expected_needs_review", False)),
                "ground_truth": meta["ground_truth"],
            }
        )

    if feedback_dir:
        fb_root = Path(feedback_dir)
        if fb_root.exists():
            for fb_file in sorted(fb_root.glob("*.json")):
                fb = json.loads(fb_file.read_text(encoding="utf-8"))
                # Feedback samples include human-verified ground truth
                if "ground_truth" in fb and fb.get("file_path") and Path(fb["file_path"]).exists():
                    samples.append(
                        {
                            "doc_id": f"feedback_{fb['document_id'][:8]}",
                            "file_name": fb["file_name"],
                            "file_path": Path(fb["file_path"]),
                            "condition": "human_feedback",
                            "anomaly_type": "none",
                            "expected_needs_review": False,
                            "ground_truth": fb["ground_truth"],
                        }
                    )

    if limit is not None and limit > 0:
        samples = samples[:limit]
    return samples


async def evaluate_pipeline(
    *,
    samples: list[dict[str, Any]],
    base_settings: Settings,
    model_override: str | None = None,
    extraction_mode_override: str | None = None,
    enable_two_pass: bool | None = None,
    in_process_mock: bool = False,
    reference_date: date = REFERENCE_TODAY,
) -> dict[str, Any]:
    """Run an isolated end-to-end evaluation pass over `samples` and compute metrics."""
    with tempfile.TemporaryDirectory(prefix="ap_eval_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        eval_db_url = f"sqlite:///{(tmp_path / 'eval.db').as_posix()}"
        blob_store = LocalFilesystemBlobStore(tmp_path / "blobs")

        cfg = base_settings.model_copy(
            update={
                "database_url": eval_db_url,
                "local_blob_dir": str(tmp_path / "blobs"),
                "export_csv_path": str(tmp_path / "exports" / "approved.csv"),
                "finance_webhook_url": "",  # Disable network webhook during isolated scoring
                "llm_model": model_override or base_settings.llm_model,
                "extraction_mode": extraction_mode_override or base_settings.extraction_mode,
                "enable_two_pass": (
                    enable_two_pass
                    if enable_two_pass is not None
                    else base_settings.enable_two_pass
                ),
            }
        )

        init_db(eval_db_url)
        session_factory = make_session_factory(eval_db_url)

        custom_http: httpx.AsyncClient | None = None
        if in_process_mock:
            cfg = cfg.model_copy(update={"llm_base_url": "http://mock.local/v1"})
            custom_http = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=mock_openai_app),
                base_url="http://mock.local",
            )

        provider = OpenAICompatibleProvider(
            settings=cfg,
            session_factory=session_factory,
            http_client=custom_http,
        )

        try:
            caps = await provider.probe_capabilities(force=True)

            per_field: dict[str, CountsPRF] = {f: CountsPRF() for f in HEADER_FIELDS}
            overall_fields = CountsPRF()
            overall_lines = CountsPRF()
            anomaly_routing = CountsPRF()
            by_doc_type: dict[str, BucketAccumulator] = {}
            by_condition: dict[str, BucketAccumulator] = {}

            latencies: list[float] = []
            costs: list[float] = []
            prompt_tokens_total = 0
            completion_tokens_total = 0
            repair_attempts_total = 0
            extraction_failures = 0
            exact_tables_total = 0

            # Process sequentially in order so duplicate-invoice detection behaves deterministically
            for sample in samples:
                raw_bytes = Path(sample["file_path"]).read_bytes()
                gt = sample["ground_truth"]
                doc_type_gt = str(gt["document_type"])
                cond = str(sample["condition"])

                with session_factory() as session:
                    doc, _ = ingest_document_bytes(
                        file_name=sample["file_name"],
                        raw_bytes=raw_bytes,
                        source_channel="eval_harness",
                        session=session,
                        blob_store=blob_store,
                        force_reprocess=True,
                    )
                    doc = await process_document_record(
                        document_id=doc.id,
                        session=session,
                        blob_store=blob_store,
                        provider=provider,
                        settings=cfg,
                        reference_date=reference_date,
                        use_cache=False,
                    )
                    session.commit()
                    pred = doc.extracted_data or {}
                    status = doc.status
                    lat_ms = doc.processing_latency_ms or 0.0
                    cost_usd = doc.estimated_cost_usd or 0.0
                    p_tok = doc.prompt_tokens or 0
                    c_tok = doc.completion_tokens or 0
                    repairs = doc.repair_attempts or 0

                if not pred:
                    extraction_failures += 1

                latencies.append(lat_ms)
                costs.append(cost_usd)
                prompt_tokens_total += p_tok
                completion_tokens_total += c_tok
                repair_attempts_total += repairs

                type_bucket = by_doc_type.setdefault(doc_type_gt, BucketAccumulator())
                cond_bucket = by_condition.setdefault(cond, BucketAccumulator())
                type_bucket.docs += 1
                cond_bucket.docs += 1

                for f_name in HEADER_FIELDS:
                    exp_val = gt.get(f_name)
                    pred_val = pred.get(f_name) if pred else None
                    _score_field(per_field[f_name], f_name, exp_val, pred_val)
                    _score_field(overall_fields, f_name, exp_val, pred_val)
                    _score_field(type_bucket.field_counts, f_name, exp_val, pred_val)
                    _score_field(cond_bucket.field_counts, f_name, exp_val, pred_val)

                li_tp, li_fp, li_fn, exact_tbl = _score_line_items(
                    gt.get("line_items", []),
                    pred.get("line_items", []) if pred else [],
                )
                for target_lc in (overall_lines, type_bucket.line_counts, cond_bucket.line_counts):
                    target_lc.tp += li_tp
                    target_lc.fp += li_fp
                    target_lc.fn += li_fn
                if exact_tbl:
                    exact_tables_total += 1
                    type_bucket.exact_line_tables += 1
                    cond_bucket.exact_line_tables += 1

                expected_review = bool(sample["expected_needs_review"])
                actual_review = status == "needs_review"
                if expected_review and actual_review:
                    anomaly_routing.tp += 1
                elif not expected_review and actual_review:
                    anomaly_routing.fp += 1
                elif expected_review and not actual_review:
                    anomaly_routing.fn += 1
                else:
                    anomaly_routing.tn += 1

            sorted_lat = sorted(latencies)
            p50_idx = int(0.50 * (len(sorted_lat) - 1)) if sorted_lat else 0
            p95_idx = int(0.95 * (len(sorted_lat) - 1)) if sorted_lat else 0

            n_docs = len(samples)
            return {
                "model": cfg.llm_model,
                "base_url": cfg.llm_base_url,
                "chosen_extraction_path": caps.chosen_extraction_path,
                "supports_vision": caps.supports_vision,
                "supports_json_schema": caps.supports_json_schema,
                "two_pass_enabled": cfg.enable_two_pass,
                "documents_evaluated": n_docs,
                "extraction_failures": extraction_failures,
                "overall": {
                    "field_precision": overall_fields.precision,
                    "field_recall": overall_fields.recall,
                    "field_f1": overall_fields.f1,
                    "line_item_precision": overall_lines.precision,
                    "line_item_recall": overall_lines.recall,
                    "line_item_f1": overall_lines.f1,
                    "exact_line_table_match_rate": round(exact_tables_total / n_docs, 4)
                    if n_docs
                    else 0.0,
                    "anomaly_review_precision": anomaly_routing.precision,
                    "anomaly_review_recall": anomaly_routing.recall,
                    "anomaly_review_f1": anomaly_routing.f1,
                },
                "telemetry": {
                    "prompt_tokens_total": prompt_tokens_total,
                    "completion_tokens_total": completion_tokens_total,
                    "avg_prompt_tokens": round(prompt_tokens_total / n_docs, 1) if n_docs else 0.0,
                    "avg_completion_tokens": round(completion_tokens_total / n_docs, 1)
                    if n_docs
                    else 0.0,
                    "total_cost_usd": round(sum(costs), 6),
                    "avg_cost_per_doc_usd": round(sum(costs) / n_docs, 6) if n_docs else 0.0,
                    "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
                    "p50_latency_ms": round(sorted_lat[p50_idx], 1) if sorted_lat else 0.0,
                    "p95_latency_ms": round(sorted_lat[p95_idx], 1) if sorted_lat else 0.0,
                    "repair_attempts_total": repair_attempts_total,
                },
                "per_field": {k: v.to_dict() for k, v in per_field.items()},
                "by_document_type": {k: v.to_dict() for k, v in sorted(by_doc_type.items())},
                "by_condition": {k: v.to_dict() for k, v in sorted(by_condition.items())},
            }
        finally:
            await provider.aclose()
            if custom_http is not None:
                await custom_http.aclose()
            dispose_engine(eval_db_url)


def format_report_markdown(report: dict[str, Any]) -> str:
    ov = report["overall"]
    tel = report["telemetry"]
    lines = [
        f"## Evaluation Summary — `{report['model']}` (`{report['chosen_extraction_path']}`)",
        "",
        f"- **Documents Evaluated:** {report['documents_evaluated']} (failures: {report['extraction_failures']})",
        f"- **Header Field Precision / Recall / F1:** {ov['field_precision']:.4f} / {ov['field_recall']:.4f} / **{ov['field_f1']:.4f}**",
        f"- **Line-Item Precision / Recall / F1:** {ov['line_item_precision']:.4f} / {ov['line_item_recall']:.4f} / **{ov['line_item_f1']:.4f}** "
        f"(Exact Table Match: {ov['exact_line_table_match_rate'] * 100:.1f}%)",
        f"- **Anomaly Routing Recall / Precision:** {ov['anomaly_review_recall']:.4f} / {ov['anomaly_review_precision']:.4f}",
        f"- **Cost & Latency:** `${tel['avg_cost_per_doc_usd']:.5f}`/doc "
        f"({tel['avg_prompt_tokens']:.0f} in / {tel['avg_completion_tokens']:.0f} out tokens) | "
        f"Mean `{tel['mean_latency_ms']:.1f} ms`, P95 `{tel['p95_latency_ms']:.1f} ms` | "
        f"Repairs `{tel['repair_attempts_total']}`",
        "",
        "### Per-Field Precision & Recall",
        "| Field | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for f_name, metrics in report["per_field"].items():
        lines.append(
            f"| `{f_name}` | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### Accuracy by Document Type",
            "| Document Type | Docs | Field P | Field R | Field F1 | Line-Item F1 | Exact Table % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dtype, b in report["by_document_type"].items():
        lines.append(
            f"| `{dtype}` | {b['documents']} | {b['field_precision']:.4f} | {b['field_recall']:.4f} | "
            f"{b['field_f1']:.4f} | {b['line_item_f1']:.4f} | {b['exact_table_match_rate'] * 100:.1f}% |"
        )

    lines.extend(
        [
            "",
            "### Accuracy by Physical Scan Condition",
            "| Condition | Docs | Field F1 | Line-Item F1 | Exact Table % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cond, b in report["by_condition"].items():
        lines.append(
            f"| `{cond}` | {b['documents']} | {b['field_f1']:.4f} | {b['line_item_f1']:.4f} | "
            f"{b['exact_table_match_rate'] * 100:.1f}% |"
        )
    return "\n".join(lines)


def format_comparison_markdown(rep_a: dict[str, Any], rep_b: dict[str, Any]) -> str:
    oa, ob = rep_a["overall"], rep_b["overall"]
    ta, tb = rep_a["telemetry"], rep_b["telemetry"]
    label_a = f"{rep_a['model']} ({rep_a['chosen_extraction_path']})"
    label_b = f"{rep_b['model']} ({rep_b['chosen_extraction_path']})"
    return "\n".join(
        [
            "## Side-by-Side Extraction Path / Model Comparison",
            "",
            f"| Metric | `{label_a}` | `{label_b}` | Delta (B - A) |",
            "|---|---:|---:|---:|",
            f"| Header Field Precision | {oa['field_precision']:.4f} | {ob['field_precision']:.4f} | {ob['field_precision'] - oa['field_precision']:+.4f} |",
            f"| Header Field Recall | {oa['field_recall']:.4f} | {ob['field_recall']:.4f} | {ob['field_recall'] - oa['field_recall']:+.4f} |",
            f"| Header Field F1 | {oa['field_f1']:.4f} | {ob['field_f1']:.4f} | {ob['field_f1'] - oa['field_f1']:+.4f} |",
            f"| Line-Item F1 | {oa['line_item_f1']:.4f} | {ob['line_item_f1']:.4f} | {ob['line_item_f1'] - oa['line_item_f1']:+.4f} |",
            f"| Exact Line-Table Match % | {oa['exact_line_table_match_rate'] * 100:.1f}% | {ob['exact_line_table_match_rate'] * 100:.1f}% | {(ob['exact_line_table_match_rate'] - oa['exact_line_table_match_rate']) * 100:+.1f} pp |",
            f"| Anomaly Review Recall | {oa['anomaly_review_recall']:.4f} | {ob['anomaly_review_recall']:.4f} | {ob['anomaly_review_recall'] - oa['anomaly_review_recall']:+.4f} |",
            f"| Avg Prompt Tokens / Doc | {ta['avg_prompt_tokens']:.0f} | {tb['avg_prompt_tokens']:.0f} | {tb['avg_prompt_tokens'] - ta['avg_prompt_tokens']:+.0f} |",
            f"| Avg Cost / Doc (USD) | ${ta['avg_cost_per_doc_usd']:.5f} | ${tb['avg_cost_per_doc_usd']:.5f} | ${tb['avg_cost_per_doc_usd'] - ta['avg_cost_per_doc_usd']:+.5f} |",
            f"| Mean Latency / Doc (ms) | {ta['mean_latency_ms']:.1f} | {tb['mean_latency_ms']:.1f} | {tb['mean_latency_ms'] - ta['mean_latency_ms']:+.1f} |",
            f"| P95 Latency / Doc (ms) | {ta['p95_latency_ms']:.1f} | {tb['p95_latency_ms']:.1f} | {tb['p95_latency_ms'] - ta['p95_latency_ms']:+.1f} |",
        ]
    )


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description="Run AP Document Intelligence Evaluation Harness")
    parser.add_argument("--subset", choices=["ci", "benchmark", "all"], default="benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on sample count")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL")
    parser.add_argument(
        "--extraction-mode",
        choices=["auto", "vision", "ocr_text"],
        default=None,
        help="Override EXTRACTION_MODE",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Disable Pass-2 verification (useful for fast local model benchmarking)",
    )
    parser.add_argument(
        "--compare-ocr",
        action="store_true",
        help="Run both Vision path and Tesseract OCR fallback path and print comparison",
    )
    parser.add_argument(
        "--compare-model",
        default=None,
        help="Optional second LLM_MODEL to compare against primary run",
    )
    parser.add_argument(
        "--in-process-mock",
        action="store_true",
        help="Evaluate against built-in in-process OpenAI-compatible mock server (CI mode)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if field F1, line-item F1, or anomaly review recall breaches gate",
    )
    parser.add_argument("--min-field-f1", type=float, default=0.92)
    parser.add_argument("--min-line-f1", type=float, default=0.90)
    parser.add_argument("--min-anomaly-recall", type=float, default=1.00)
    parser.add_argument("--output-json", default=None, help="Write JSON results to file")
    args = parser.parse_args()

    settings = get_settings()
    samples = load_eval_samples(
        settings.synthetic_dir,
        subset=args.subset,
        limit=args.limit,
        feedback_dir=settings.eval_feedback_dir,
    )

    primary = await evaluate_pipeline(
        samples=samples,
        base_settings=settings,
        model_override=args.model,
        extraction_mode_override=args.extraction_mode,
        enable_two_pass=False if args.single_pass else None,
        in_process_mock=args.in_process_mock,
    )
    print(format_report_markdown(primary), flush=True)

    output_bundle: dict[str, Any] = {"primary": primary}

    if args.compare_ocr or args.compare_model:
        ocr_model = args.compare_model or (
            "mock-text-strict-v1"
            if ("mock" in (args.model or settings.llm_model))
            else (args.model or settings.llm_model)
        )
        secondary = await evaluate_pipeline(
            samples=samples,
            base_settings=settings,
            model_override=ocr_model,
            extraction_mode_override="auto" if args.compare_model else "ocr_text",
            enable_two_pass=False if args.single_pass else None,
            in_process_mock=args.in_process_mock,
        )
        print("\n" + format_report_markdown(secondary), flush=True)
        print("\n" + format_comparison_markdown(primary, secondary), flush=True)
        output_bundle["ocr_fallback"] = secondary

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_bundle, indent=2), encoding="utf-8")

    if args.fail_on_regression:
        ov = primary["overall"]
        failures: list[str] = []
        if ov["field_f1"] < args.min_field_f1:
            failures.append(f"field_f1 {ov['field_f1']:.4f} < {args.min_field_f1:.4f}")
        if ov["line_item_f1"] < args.min_line_f1:
            failures.append(f"line_item_f1 {ov['line_item_f1']:.4f} < {args.min_line_f1:.4f}")
        if ov["anomaly_review_recall"] < args.min_anomaly_recall:
            failures.append(
                f"anomaly_review_recall {ov['anomaly_review_recall']:.4f} < {args.min_anomaly_recall:.4f}"
            )
        if failures:
            print(f"\n[CI REGRESSION FAILURE] {'; '.join(failures)}", file=sys.stderr)
            return 1
        print(
            f"\n[CI REGRESSION GATE PASSED] field_f1={ov['field_f1']:.4f}, "
            f"line_item_f1={ov['line_item_f1']:.4f}, "
            f"anomaly_review_recall={ov['anomaly_review_recall']:.4f}"
        )
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_async_main()))


if __name__ == "__main__":
    main()
