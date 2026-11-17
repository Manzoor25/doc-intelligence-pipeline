"""Background worker queue runner + watched-folder ingestion loop.

Claims pending jobs from PostgreSQL using `SELECT ... FOR UPDATE SKIP LOCKED`,
processes documents concurrently, reschedules transient failures with exponential
backoff (zero lost documents), and watches `WATCH_DIR` for dropped files.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import or_, select

from doc_pipeline.config import Settings, get_settings
from doc_pipeline.ingestion.service import ingest_document_bytes, process_document_record
from doc_pipeline.llm.provider import OpenAICompatibleProvider
from doc_pipeline.storage.blob import BlobStore, get_blob_store
from doc_pipeline.storage.db import init_db, make_session_factory
from doc_pipeline.storage.models import DocumentRecord, ProcessingJob

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def scan_watched_folder_once(
    watch_dir: str | Path,
    session_factory,
    blob_store: BlobStore,
) -> int:
    """Scan `watch_dir` for dropped files, ingest into MinIO+Postgres, and archive originals."""
    inbox = Path(watch_dir)
    processed_dir = inbox / "processed"
    failed_dir = inbox / "failed"
    inbox.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    ingested_count = 0
    for entry in sorted(inbox.iterdir()):
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            raw_bytes = entry.read_bytes()
            with session_factory() as session:
                ingest_document_bytes(
                    file_name=entry.name,
                    raw_bytes=raw_bytes,
                    source_channel="watched_folder",
                    session=session,
                    blob_store=blob_store,
                )
                session.commit()
            dest = processed_dir / entry.name
            shutil.move(str(entry), str(dest))
            ingested_count += 1
            logger.info("Watched folder ingested '%s' -> archived to processed/", entry.name)
        except Exception as exc:
            logger.error("Watched folder failed on '%s': %s", entry.name, exc)
            dest = failed_dir / entry.name
            try:
                shutil.move(str(entry), str(dest))
            except OSError:
                pass
    return ingested_count


def claim_next_job(session_factory, worker_id: str) -> str | None:
    """Atomically claim one ready job using PostgreSQL `FOR UPDATE SKIP LOCKED`."""
    now = datetime.now(UTC)
    lock_expiry = now + timedelta(minutes=5)
    with session_factory() as session:
        stmt = (
            select(ProcessingJob)
            .where(
                or_(
                    ProcessingJob.status == "pending",
                    (ProcessingJob.status == "running") & (ProcessingJob.locked_until < now),
                ),
                ProcessingJob.next_attempt_at <= now,
            )
            .order_by(ProcessingJob.next_attempt_at.asc(), ProcessingJob.id.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = session.scalar(stmt)
        if job is None:
            return None

        job.status = "running"
        job.attempts += 1
        job.locked_by = worker_id
        job.locked_until = lock_expiry
        doc_id = job.document_id
        session.commit()
        return doc_id


async def execute_claimed_job(
    document_id: str,
    session_factory,
    blob_store: BlobStore,
    provider: OpenAICompatibleProvider,
    settings: Settings,
) -> None:
    try:
        with session_factory() as session:
            await process_document_record(
                document_id=document_id,
                session=session,
                blob_store=blob_store,
                provider=provider,
                settings=settings,
            )
            session.commit()
    except Exception as exc:
        logger.exception("Job execution error on document %s: %s", document_id, exc)
        with session_factory() as session:
            job = session.scalar(
                select(ProcessingJob).where(ProcessingJob.document_id == document_id)
            )
            doc = session.get(DocumentRecord, document_id)
            if job is not None:
                job.last_error = f"{type(exc).__name__}: {exc}"
                job.locked_by = None
                job.locked_until = None
                if job.attempts >= job.max_attempts:
                    job.status = "dead"
                    if doc is not None:
                        doc.status = "needs_review"
                        doc.error_message = (
                            f"Max queue retries reached ({job.attempts}); routed to human review: {exc}"
                        )
                else:
                    backoff = min(60.0, 1.5 * (2 ** min(job.attempts, 6)))
                    job.status = "pending"
                    job.next_attempt_at = datetime.now(UTC) + timedelta(seconds=backoff)
                    if doc is not None:
                        doc.status = "queued"
            session.commit()


async def run_worker_loop(settings: Settings | None = None) -> None:
    cfg = settings or get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    init_db(cfg.database_url)
    session_factory = make_session_factory(cfg.database_url)
    blob_store = get_blob_store(cfg)
    provider = OpenAICompatibleProvider(settings=cfg, session_factory=session_factory)
    worker_id = f"{socket.gethostname()}-pid{os.getpid()}"

    # Probe model capabilities on worker startup and print chosen path
    await provider.probe_capabilities()
    logger.info(
        "Worker %s started (concurrency=%d, watch_dir=%s)",
        worker_id,
        cfg.llm_max_concurrency,
        cfg.watch_dir,
    )

    active_tasks: set[asyncio.Task[None]] = set()
    try:
        while True:
            scan_watched_folder_once(cfg.watch_dir, session_factory, blob_store)

            while len(active_tasks) < cfg.llm_max_concurrency:
                doc_id = claim_next_job(session_factory, worker_id=worker_id)
                if doc_id is None:
                    break
                task = asyncio.create_task(
                    execute_claimed_job(
                        doc_id, session_factory, blob_store, provider, cfg
                    )
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)

            await asyncio.sleep(cfg.worker_poll_interval_seconds)
    finally:
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(run_worker_loop())
