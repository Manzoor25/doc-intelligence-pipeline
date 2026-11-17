# ADR 0001: Single OpenAI-Compatible Provider Module with Startup Capability Probing

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

Enterprise accounts payable teams deploy across heterogeneous model hosts: cloud inference APIs (Groq, Google Cloud Vertex AI, Azure AI Foundry / Azure OpenAI) and air-gapped on-premises servers (Ollama, vLLM). Hard-coding vendor SDKs (`groq`, `google-cloud-aiplatform`, `azure-ai-inference`) creates vendor lock-in and divergent code paths. Furthermore, OpenAI-compatible endpoints vary widely in two capabilities critical to document extraction:

1. Multimodal image input (`image_url` base64 data URIs in `/v1/chat/completions`).
2. Constrained JSON Schema structured outputs (`response_format: {"type": "json_schema", "json_schema": {"strict": true, ...}}`).

## Decision

1. Centralize **all** LLM network calls in a single module (`src/doc_pipeline/llm/provider.py`) configured exclusively via `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` using the standard `openai` Python client.
2. Execute a lightweight **startup capability probe** (`probe_capabilities()`) before processing documents:
   - **Vision probe:** Sends a minimal 32x32 PNG data URI and checks for a valid completion versus HTTP 400/404/415/422 unsupported-modality errors.
   - **Structured-output probe:** Sends a tiny strict JSON Schema (`{"ok": boolean}`) using `response_format.type = "json_schema"`. If unsupported, probes `json_object` mode, falling back to prompt-enforced JSON extraction.
3. Log and print the discovered capability profile at startup and route extraction automatically:
   - `supports_vision=True` -> render pages to PNGs and send base64 data URIs directly.
   - `supports_vision=False` -> run Tesseract OCR on rendered pages and send structured page text.
   - `supports_json_schema=True` -> enforce strict JSON Schema (`additionalProperties: false`, all keys required with nullable unions).
   - `supports_json_schema=False` -> inject canonical JSON Schema into the system prompt, use `json_object` if available, and rely on the Pydantic validation repair loop.

## Consequences

- **Positive:** Zero code changes when switching between Groq, Vertex AI, Azure Foundry, Ollama, or the CI mock server. Graceful degradation when an operator swaps from a vision model to a cheaper text-only model.
- **Negative:** Startup incurs two tiny probe requests (~50–150 tokens total), mitigated by caching probe results per `(base_url, model)` in memory for worker lifetime.
