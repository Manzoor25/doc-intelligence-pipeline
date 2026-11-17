"""Single OpenAI-compatible LLM provider with startup capability probing, caching,
rate-limit exponential backoff, cost/latency accounting, and Pydantic repair loop.

Reads ONLY LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from settings. No vendor is hard-coded.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from PIL import Image
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from doc_pipeline.config import Settings, get_settings
from doc_pipeline.schemas import CapabilityProfile, to_strict_json_schema
from doc_pipeline.storage.models import LLMCacheEntry, LLMCallMetric

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


@dataclass
class CallTelemetry:
    """Aggregated token, cost, retry, and latency metrics across one extraction operation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    repair_attempts: int = 0
    rate_limit_retries: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class StructuredExtractionOutcome[TModel]:
    """Result of structured extraction with Pydantic validation and repair retries."""

    parsed: TModel | None
    raw_text: str | None
    succeeded: bool
    telemetry: CallTelemetry
    error_summary: str | None = None


def _make_probe_png_data_uri() -> str:
    img = Image.new("RGB", (32, 32), color=(30, 64, 175))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_json_object_text(raw: str) -> str:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _parse_retry_after(exc: APIStatusError) -> float | None:
    headers = getattr(exc, "response", None)
    if headers is None or not hasattr(headers, "headers"):
        return None
    hdrs = headers.headers
    ms_val = hdrs.get("retry-after-ms")
    if ms_val:
        try:
            return max(0.05, float(ms_val) / 1000.0)
        except ValueError:
            pass
    sec_val = hdrs.get("retry-after")
    if sec_val:
        try:
            return max(0.05, float(sec_val))
        except ValueError:
            pass
    return None


class OpenAICompatibleProvider:
    """Sole gateway for all LLM interactions across the pipeline."""

    def __init__(
        self,
        settings: Settings | None = None,
        session_factory: sessionmaker[Session] | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._session_factory = session_factory
        self._semaphore = asyncio.Semaphore(self.settings.llm_max_concurrency)
        # Disable SDK's internal silent retries so our explicit backoff & telemetry track 429s accurately
        self._client = AsyncOpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=0,
            http_client=http_client,
        )
        self._capabilities: CapabilityProfile | None = None
        self._use_max_completion_tokens: bool = False
        self._omit_temperature: bool = False

    @property
    def capabilities(self) -> CapabilityProfile | None:
        return self._capabilities

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        in_cost = (prompt_tokens / 1_000_000.0) * self.settings.llm_cost_input_per_1m
        out_cost = (completion_tokens / 1_000_000.0) * self.settings.llm_cost_output_per_1m
        return round(in_cost + out_cost, 8)

    def _compute_cache_key(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
        temperature: float,
        cache_namespace: str,
    ) -> str:
        payload = json.dumps(
            {
                "base_url": self.settings.llm_base_url,
                "model": self.settings.llm_model,
                "namespace": cache_namespace,
                "temperature": round(temperature, 3),
                "response_format": response_format,
                "messages": messages,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, cache_key: str) -> LLMCacheEntry | None:
        if self._session_factory is None:
            return None
        with self._session_factory() as session:
            return session.get(LLMCacheEntry, cache_key)

    def _write_cache(
        self, cache_key: str, response_text: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        if self._session_factory is None:
            return
        with self._session_factory() as session:
            if session.get(LLMCacheEntry, cache_key) is None:
                session.add(
                    LLMCacheEntry(
                        cache_key=cache_key,
                        model=self.settings.llm_model,
                        response_text=response_text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                )
                session.commit()

    def _record_metric(
        self,
        *,
        document_id: str | None,
        call_purpose: str,
        cache_hit: bool,
        attempt_number: int,
        rate_limit_retries: int,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
        succeeded: bool,
        error_type: str | None = None,
    ) -> None:
        if self._session_factory is None:
            return
        with self._session_factory() as session:
            session.add(
                LLMCallMetric(
                    document_id=document_id,
                    call_purpose=call_purpose,
                    model=self.settings.llm_model,
                    cache_hit=cache_hit,
                    attempt_number=attempt_number,
                    rate_limit_retries=rate_limit_retries,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=cost_usd,
                    latency_ms=latency_ms,
                    succeeded=succeeded,
                    error_type=error_type,
                )
            )
            session.commit()

    async def _raw_chat_call(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        document_id: str | None = None,
        call_purpose: str = "extract",
        attempt_number: int = 1,
        use_cache: bool = True,
        cache_namespace: str = "default",
    ) -> tuple[str, int, int, float, int, bool]:
        """Execute one chat.completions request with concurrency cap, caching, and 429 backoff."""
        cache_key: str | None = None
        if use_cache:
            cache_key = self._compute_cache_key(
                messages, response_format, temperature, cache_namespace
            )
            cached = self._read_cache(cache_key)
            if cached is not None:
                self._record_metric(
                    document_id=document_id,
                    call_purpose=call_purpose,
                    cache_hit=True,
                    attempt_number=attempt_number,
                    rate_limit_retries=0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.5,
                    succeeded=True,
                )
                return cached.response_text, 0, 0, 0.5, 0, True

        rate_limit_retries = 0
        start_perf = time.perf_counter()

        async with self._semaphore:
            while True:
                kwargs: dict[str, Any] = {
                    "model": self.settings.llm_model,
                    "messages": messages,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if not self._omit_temperature:
                    kwargs["temperature"] = temperature
                if self._use_max_completion_tokens:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

                try:
                    resp = await self._client.chat.completions.create(**kwargs)
                    latency_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
                    content = resp.choices[0].message.content or ""
                    usage = resp.usage
                    p_tok = int(usage.prompt_tokens) if usage and usage.prompt_tokens else 0
                    c_tok = int(usage.completion_tokens) if usage and usage.completion_tokens else 0
                    cost = self._estimate_cost(p_tok, c_tok)

                    if use_cache and cache_key and content:
                        self._write_cache(cache_key, content, p_tok, c_tok)

                    self._record_metric(
                        document_id=document_id,
                        call_purpose=call_purpose,
                        cache_hit=False,
                        attempt_number=attempt_number,
                        rate_limit_retries=rate_limit_retries,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        cost_usd=cost,
                        latency_ms=latency_ms,
                        succeeded=True,
                    )
                    return content, p_tok, c_tok, latency_ms, rate_limit_retries, False

                except BadRequestError as exc:
                    msg_lower = str(exc).lower()
                    # Automatically adapt to reasoning models (e.g. Azure GPT-5 / o-series)
                    if "max_completion_tokens" in msg_lower and not self._use_max_completion_tokens:
                        self._use_max_completion_tokens = True
                        continue
                    if "temperature" in msg_lower and not self._omit_temperature:
                        self._omit_temperature = True
                        continue
                    latency_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
                    self._record_metric(
                        document_id=document_id,
                        call_purpose=call_purpose,
                        cache_hit=False,
                        attempt_number=attempt_number,
                        rate_limit_retries=rate_limit_retries,
                        prompt_tokens=0,
                        completion_tokens=0,
                        cost_usd=0.0,
                        latency_ms=latency_ms,
                        succeeded=False,
                        error_type="BadRequestError",
                    )
                    raise

                except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                    if rate_limit_retries >= self.settings.llm_max_rate_limit_retries:
                        latency_ms = round((time.perf_counter() - start_perf) * 1000.0, 2)
                        self._record_metric(
                            document_id=document_id,
                            call_purpose=call_purpose,
                            cache_hit=False,
                            attempt_number=attempt_number,
                            rate_limit_retries=rate_limit_retries,
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost_usd=0.0,
                            latency_ms=latency_ms,
                            succeeded=False,
                            error_type=type(exc).__name__,
                        )
                        raise

                    retry_after = (
                        _parse_retry_after(exc) if isinstance(exc, APIStatusError) else None
                    )
                    base = self.settings.llm_rate_limit_base_delay * (2**rate_limit_retries)
                    jitter = random.uniform(0.02, max(0.05, base * 0.35))
                    sleep_s = min(30.0, (retry_after or base) + jitter)
                    rate_limit_retries += 1
                    logger.warning(
                        "LLM transient/rate-limit error (%s); retry %d/%d in %.2fs",
                        type(exc).__name__,
                        rate_limit_retries,
                        self.settings.llm_max_rate_limit_retries,
                        sleep_s,
                    )
                    await asyncio.sleep(sleep_s)

    async def probe_capabilities(self, force: bool = False) -> CapabilityProfile:
        """Probe the endpoint on startup for image input and JSON-schema structured output."""
        if self._capabilities is not None and not force:
            return self._capabilities

        t0 = time.perf_counter()
        notes: list[str] = []
        reachable = True
        supports_vision = False
        supports_json_schema = False
        supports_json_object = False

        # 1. Probe JSON-Schema structured output (simultaneously verifies text reachability)
        probe_schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        try:
            text_out, _, _, _, _, _ = await self._raw_chat_call(
                messages=[
                    {"role": "system", "content": "Return JSON matching the schema."},
                    {"role": "user", "content": 'Set ok to true and return {"ok": true}.'},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "startup_capability_probe",
                        "strict": True,
                        "schema": probe_schema,
                    },
                },
                max_tokens=48,
                call_purpose="probe_json_schema",
                use_cache=False,
            )
            parsed_probe = json.loads(_extract_json_object_text(text_out))
            if isinstance(parsed_probe, dict) and isinstance(parsed_probe.get("ok"), bool):
                supports_json_schema = True
                supports_json_object = True
                notes.append("Endpoint supports strict json_schema structured output.")
            else:
                notes.append("Endpoint returned JSON without matching strict probe schema.")
        except APIStatusError as exc:
            notes.append(f"json_schema unsupported (HTTP {exc.status_code}); testing json_object.")
        except Exception as exc:
            if isinstance(exc, (APIConnectionError, APITimeoutError)):
                reachable = False
                notes.append(f"Endpoint unreachable during probe: {exc}")
            else:
                notes.append(f"json_schema probe rejected: {type(exc).__name__}")

        if reachable and not supports_json_schema:
            try:
                text_out, _, _, _, _, _ = await self._raw_chat_call(
                    messages=[
                        {"role": "user", "content": 'Respond in JSON: {"ok": true}'},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=48,
                    call_purpose="probe_json_object",
                    use_cache=False,
                )
                json.loads(_extract_json_object_text(text_out))
                supports_json_object = True
                notes.append("Endpoint supports json_object mode; Pydantic repair loop active.")
            except Exception:
                notes.append("Falling back to prompt-guided JSON extraction with Pydantic repair.")

        # 2. Probe multimodal vision support (base64 PNG data URI)
        if reachable:
            data_uri = _make_probe_png_data_uri()
            try:
                await self._raw_chat_call(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Reply with the single word OK."},
                                {"type": "image_url", "image_url": {"url": data_uri}},
                            ],
                        }
                    ],
                    max_tokens=16,
                    call_purpose="probe_vision",
                    use_cache=False,
                )
                supports_vision = True
                notes.append("Endpoint accepts base64 image_url vision inputs.")
            except APIStatusError as exc:
                supports_vision = False
                notes.append(
                    f"Endpoint rejected image_url input (HTTP {exc.status_code}) -> using OCR text fallback."
                )
            except Exception as exc:
                supports_vision = False
                notes.append(f"Vision probe failed ({type(exc).__name__}) -> using OCR text fallback.")

        # Choose code path respecting explicit EXTRACTION_MODE override if set
        use_vision = supports_vision
        if self.settings.extraction_mode == "ocr_text":
            use_vision = False
            notes.append("EXTRACTION_MODE=ocr_text forced by configuration.")
        elif self.settings.extraction_mode == "vision":
            use_vision = True
            notes.append("EXTRACTION_MODE=vision forced by configuration.")

        if use_vision and supports_json_schema:
            chosen = "vision_strict_schema"
        elif use_vision:
            chosen = "vision_json_repair"
        elif supports_json_schema:
            chosen = "ocr_strict_schema"
        else:
            chosen = "ocr_json_repair"

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        profile = CapabilityProfile(
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
            reachable=reachable,
            supports_vision=supports_vision,
            supports_json_schema=supports_json_schema,
            supports_json_object=supports_json_object,
            chosen_extraction_path=chosen,  # type: ignore[arg-type]
            probe_latency_ms=elapsed_ms,
            notes=notes,
            probed_at=datetime.now(UTC),
        )
        self._capabilities = profile

        banner = (
            f"[LLM STARTUP PROBE] base_url={profile.base_url} | model={profile.model} | "
            f"reachable={profile.reachable} | accepts_images={profile.supports_vision} | "
            f"json_schema={profile.supports_json_schema} | chosen_path={profile.chosen_extraction_path} "
            f"({profile.probe_latency_ms:.1f} ms)"
        )
        print(banner, flush=True)
        logger.info(banner)
        return profile

    async def complete_structured(
        self,
        *,
        schema_cls: type[TModel],
        schema_name: str,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 2400,
        document_id: str | None = None,
        call_purpose: str = "extract_pass1",
        use_cache: bool = True,
        cache_namespace: str = "pass1",
    ) -> StructuredExtractionOutcome[TModel]:
        """Extract against `schema_cls` and repair automatically with Pydantic error feedback."""
        caps = await self.probe_capabilities()
        strict_schema = to_strict_json_schema(schema_cls)

        response_format: dict[str, Any] | None = None
        effective_system = system_prompt
        if caps.supports_json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": strict_schema,
                },
            }
        else:
            schema_json = json.dumps(strict_schema, indent=2)
            effective_system = (
                f"{system_prompt}\n\n"
                "Respond ONLY with a single valid JSON object conforming strictly to this JSON Schema "
                "(no markdown fences, no commentary):\n"
                f"{schema_json}"
            )
            if caps.supports_json_object:
                response_format = {"type": "json_object"}

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": user_content},
        ]

        telemetry = CallTelemetry()
        last_raw: str | None = None
        max_attempts = 1 + self.settings.llm_max_repair_attempts

        for attempt in range(1, max_attempts + 1):
            is_repair = attempt > 1
            if is_repair:
                telemetry.repair_attempts += 1

            try:
                raw_text, p_tok, c_tok, lat_ms, rl_retries, hit = await self._raw_chat_call(
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    document_id=document_id,
                    call_purpose="repair" if is_repair else call_purpose,
                    attempt_number=attempt,
                    use_cache=use_cache and not is_repair,
                    cache_namespace=cache_namespace,
                )
                last_raw = raw_text
                telemetry.prompt_tokens += p_tok
                telemetry.completion_tokens += c_tok
                telemetry.latency_ms = round(telemetry.latency_ms + lat_ms, 2)
                telemetry.rate_limit_retries += rl_retries
                if hit:
                    telemetry.cache_hits += 1
                else:
                    telemetry.cost_usd = round(
                        telemetry.cost_usd + self._estimate_cost(p_tok, c_tok), 8
                    )
            except Exception as exc:
                err_msg = f"LLM transport failure on attempt {attempt}: {type(exc).__name__}: {exc}"
                telemetry.errors.append(err_msg)
                return StructuredExtractionOutcome(
                    parsed=None,
                    raw_text=last_raw,
                    succeeded=False,
                    telemetry=telemetry,
                    error_summary=err_msg,
                )

            # Parse & validate against strict Pydantic model
            try:
                candidate_json = _extract_json_object_text(raw_text)
                payload = json.loads(candidate_json)
                validated = schema_cls.model_validate(payload)
                return StructuredExtractionOutcome(
                    parsed=validated,
                    raw_text=raw_text,
                    succeeded=True,
                    telemetry=telemetry,
                )
            except (json.JSONDecodeError, ValidationError, TypeError) as val_err:
                err_detail = str(val_err)
                telemetry.errors.append(f"Attempt {attempt} validation failed: {err_detail[:400]}")
                logger.warning(
                    "Structured output validation failed on attempt %d/%d for doc=%s: %s",
                    attempt,
                    max_attempts,
                    document_id,
                    err_detail[:220],
                )
                if attempt < max_attempts:
                    messages.append({"role": "assistant", "content": raw_text or "{}"})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response failed strict JSON/Pydantic validation with "
                                f"the following error:\n{err_detail}\n\n"
                                "Fix ONLY the schema/formatting errors based strictly on the document "
                                "data and return the corrected JSON object."
                            ),
                        }
                    )

        return StructuredExtractionOutcome(
            parsed=None,
            raw_text=last_raw,
            succeeded=False,
            telemetry=telemetry,
            error_summary=telemetry.errors[-1] if telemetry.errors else "Schema validation exhausted",
        )

    async def aclose(self) -> None:
        await self._client.close()
