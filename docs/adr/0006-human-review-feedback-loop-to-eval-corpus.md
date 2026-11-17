# ADR 0006: Human-in-the-Loop Corrections Promoted to Evaluation Data

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

In production document AI systems, initial prompts and schemas degrade silently as new vendor layouts appear. Most teams treat human review UIs purely as operational queues and discard reviewer edits after exporting to the ERP, losing their highest-signal supervision data (specifically the documents where the current pipeline had low confidence or failed validation).

## Decision

1. Build a split-pane web review station (`GET /`) displaying rendered document pages alongside editable header fields, line items, per-field confidence badges, Pass-1 vs Pass-2 disagreements, and validation diagnostics.
2. When a reviewer submits corrections (`POST /api/v1/documents/{id}/review`):
   - Persist field-level `before -> after` diffs and reviewer notes in `audit_logs`.
   - Automatically export a canonical ground-truth record (`data/eval_feedback/{doc_id}.json` and `evaluation_samples` in PostgreSQL) pairing the immutable document hash/pages with the human-verified extraction.
   - Trigger downstream PostgreSQL payable ledger upsert, CSV export refresh, and signed ERP webhook delivery upon approval.
3. Configure the evaluation harness (`doc_pipeline.eval.runner`) to evaluate both the synthetic benchmark corpus and accumulated `eval_feedback` gold samples, with a fast deterministic subset gating every push in GitHub Actions CI.

## Consequences

- **Positive:** Every human review makes the regression suite stronger on real failure modes; model upgrades (`LLM_MODEL` swaps) can be validated quantitatively against historical edge cases before cutover.
- **Negative:** Human reviewers can occasionally save typos; production deployments should add periodic spot-auditing or dual-review before promoting high-value corrections into CI hard gates.
