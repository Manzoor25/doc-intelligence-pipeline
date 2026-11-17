"""CI evaluation regression test: runs the 24-document stratified CI subset and enforces F1 gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_pipeline.config import get_settings
from doc_pipeline.eval.runner import evaluate_pipeline, load_eval_samples
from doc_pipeline.mock_services.openai_mock import reset_state
from doc_pipeline.synthetic.generator import generate_dataset


@pytest.mark.asyncio
async def test_ci_evaluation_subset_regression_gate(monkeypatch) -> None:
    await reset_state()
    synth_dir = Path("data/synthetic")
    manifest = synth_dir / "manifest.json"
    if not manifest.exists():
        generate_dataset(synth_dir, count=200, seed=42)

    monkeypatch.setenv("SYNTHETIC_DATA_DIR", str(synth_dir.resolve()))
    settings = get_settings()
    samples = load_eval_samples(synth_dir, subset="ci")
    assert len(samples) == 24

    report = await evaluate_pipeline(
        samples=samples,
        base_settings=settings,
        model_override="mock-vision-strict-v1",
        extraction_mode_override="vision",
        in_process_mock=True,
    )

    overall = report["overall"]
    assert report["extraction_failures"] == 0
    assert overall["field_f1"] >= 0.95, f"Header field F1 regressed: {overall['field_f1']}"
    assert overall["line_item_f1"] >= 0.95, f"Line-item F1 regressed: {overall['line_item_f1']}"
    assert (
        overall["anomaly_review_recall"] >= 1.0
    ), f"Anomaly review recall regressed: {overall['anomaly_review_recall']}"
