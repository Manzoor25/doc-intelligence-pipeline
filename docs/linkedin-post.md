Passing a clean PDF invoice to an LLM takes an afternoon. Making Accounts Payable trust the output across 5,000 skewed scans, German multi-page POs, and parking-lot phone receipts a month is an 80% systems engineering problem.

I built and open-sourced a runnable, vendor-agnostic AP Document Intelligence Pipeline (`FastAPI` + `PostgreSQL SKIP LOCKED` + `MinIO` + Review UI + Docker Compose) configured by just `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`.

What sits around the model:
• Startup capability probe: auto-detects vision & strict JSON schema, falling back to local Tesseract OCR (`eng+deu`) + Pydantic error-guided repair—cutting prompt tokens by 63.9% (`3,202 -> 1,155` tokens/doc) on text-only endpoints
• Model-free accounting rules: line-item sums, tax math, RapidFuzz supplier matching & cross-channel duplicate invoice detection
• Calibrated per-field confidence from two-pass agreement + rule coupling (never model self-claims)
• Prompt-injection quarantine + human review corrections that auto-promote into CI regression tests

Full write-up (including a surprising CPU GBNF grammar bottleneck on 152k-vocab local models) and code:
https://github.com/Manzoor25/doc-intelligence-pipeline

#DocumentAI #AIEngineering #LLMSystems #FinTech #OpenSource
