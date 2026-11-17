# ADR 0004: Vision-First Extraction with Preprocessed Tesseract OCR Fallback

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

Invoices and purchase orders carry spatial semantics—multi-column headers, nested tax tables, skewed phone photos, and handwritten margin notes—that plain text extraction flattens or scrambles. However, many cost-effective or local open-weights endpoints (e.g., `llama-3.3-70b-versatile` on Groq, `qwen3:8b` on CPU Ollama) only accept text.

## Decision

1. Render every ingested PDF page to a 200-DPI RGB PNG via `pypdfium2` and normalize uploaded photos/scans via `Pillow`.
2. **Primary path (Vision):** When the startup probe confirms vision support, send page images directly to the multimodal chat endpoint (capped per request with multi-page batching/stitching).
3. **Fallback path (OCR -> Text LLM):** When vision is unavailable (or `EXTRACTION_MODE=ocr_text` is forced for benchmarking), preprocess each page image (grayscale conversion, autocontrast, median denoise, deskew estimation) and run `Tesseract OCR` (`eng+deu`) to produce page-delimited text (`--- PAGE k ---`), then run the identical schema extraction and Pydantic repair pipeline.
4. Benchmark both paths side by side in `doc_pipeline.eval.runner --compare-ocr` on accuracy, token usage, cost, and latency.

## Consequences

- **Positive:** Works against any text or vision chat endpoint while preserving spatial accuracy when vision is available. Gives engineering teams empirical data on the accuracy-vs-cost gap on their own documents.
- **Negative:** Local Tesseract struggles on curved phone photos and handwritten annotations compared to vision models or dedicated cloud OCR (Azure Document Intelligence / AWS Textract).
