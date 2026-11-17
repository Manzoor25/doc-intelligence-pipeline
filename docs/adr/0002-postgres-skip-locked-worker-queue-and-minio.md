# ADR 0002: PostgreSQL SKIP LOCKED Durable Queue and MinIO Object Storage

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

The target workload is 5,000 AP documents/month (~230 documents/business day, with end-of-month bursts of 50–100 documents/hour). Documents arrive via web upload, REST API, and watched SFTP/network folders, and must never be dropped when model endpoints return HTTP 429 rate limits or transient 5xx errors.

## Decision

1. Store raw uploaded files and rendered 200-DPI page PNGs in **MinIO** (S3-compatible object storage), keyed under deterministic prefixes (`raw/{sha256}/{filename}` and `pages/{doc_id}/page_{n}.png`).
2. Implement the asynchronous worker queue directly in **PostgreSQL** (`processing_jobs` table) using transactional `SELECT ... FOR UPDATE SKIP LOCKED`, `next_attempt_at` scheduled retries, and visibility-timeout heartbeats (`locked_until`), rather than introducing Redis/RabbitMQ as an additional broker.
3. Enforce idempotency at ingestion via a unique index on `documents.file_sha256`.

## Consequences

- **Positive:** Atomic enqueue alongside document metadata eliminates dual-write split-brain between database and message broker. Native `next_attempt_at` timestamps cleanly handle `Retry-After` rate-limit backoff without blocking worker coroutines. Easy to inspect, replay, and audit with SQL.
- **Negative:** Polling Postgres (`SKIP LOCKED`) tops out around several thousand jobs/second—well above 5,000 docs/month (and even 5M docs/month), but high-frequency sub-10ms streaming workloads at >50M docs/month would warrant migrating the queue transport to Kafka or Redis Streams.
