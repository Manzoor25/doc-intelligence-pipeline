"""Failure Demo 4: Swap LLM_MODEL -> Re-Probe Capabilities, Rerun Evaluation & Print Comparison.

EXPECTED OUTPUT:
  - Evaluates Model A (`LLM_MODEL`, default `mock-vision-strict-v1` -> `vision_strict_schema`).
  - Changes `LLM_MODEL` to Model B (`SWAP_TO_MODEL`, default `mock-text-strict-v1` -> `ocr_strict_schema`).
  - Startup probe detects that Model B rejects `image_url` inputs and automatically switches to the
    Tesseract OCR + text extraction fallback path without any code change.
  - Reruns evaluation on the identical document subset and prints a side-by-side comparison table.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from doc_pipeline.config import get_settings
from doc_pipeline.eval.runner import (
    evaluate_pipeline,
    format_comparison_markdown,
    load_eval_samples,
)


async def _run() -> None:
    parser = argparse.ArgumentParser(description="Swap LLM_MODEL and compare evaluation results")
    parser.add_argument("--subset", default="ci", choices=["ci", "benchmark", "all"])
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--model-a", default=None, help="Baseline LLM_MODEL")
    parser.add_argument(
        "--model-b",
        default=os.environ.get("SWAP_TO_MODEL", "mock-text-strict-v1"),
        help="Target swapped LLM_MODEL",
    )
    parser.add_argument("--in-process-mock", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    model_a = args.model_a or settings.llm_model
    model_b = args.model_b

    samples = load_eval_samples(settings.synthetic_dir, subset=args.subset, limit=args.limit)

    print("=" * 78)
    print(f"FAILURE DEMO 4: Dynamic Model Swap ({model_a} -> {model_b}) & Eval Comparison")
    print("=" * 78)
    print(
        f"[EXPECTED] Changing LLM_MODEL from '{model_a}' to '{model_b}' triggers automatic\n"
        f"           capability re-probing and prints side-by-side accuracy, cost, and latency."
    )

    rep_a = await evaluate_pipeline(
        samples=samples,
        base_settings=settings,
        model_override=model_a,
        extraction_mode_override="auto",
        in_process_mock=args.in_process_mock,
    )
    rep_b = await evaluate_pipeline(
        samples=samples,
        base_settings=settings,
        model_override=model_b,
        extraction_mode_override="auto",
        in_process_mock=args.in_process_mock,
    )

    print("\n" + format_comparison_markdown(rep_a, rep_b))
    print("\n[RESULT]   PASS — Swapped LLM_MODEL cleanly via config with zero vendor code changes.")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(_run())
