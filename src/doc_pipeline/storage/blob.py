"""Object storage abstraction backed by MinIO (S3) with local-filesystem test fallback."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol

from minio import Minio

from doc_pipeline.config import Settings, get_settings


class BlobStore(Protocol):
    """Minimal object storage interface for raw documents and rendered page PNGs."""

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...

    def get_bytes(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class MinioBlobStore:
    """S3-compatible object storage backed by MinIO."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.minio_bucket
        self._client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self._bucket_ready = False

    def _ensure_bucket(self) -> None:
        if not self._bucket_ready:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
            self._bucket_ready = True

    def put_bytes(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        self._ensure_bucket()
        self._client.put_object(
            bucket_name=self._bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def get_bytes(self, key: str) -> bytes:
        self._ensure_bucket()
        resp = self._client.get_object(self._bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def exists(self, key: str) -> bool:
        self._ensure_bucket()
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except Exception:
            return False


class LocalFilesystemBlobStore:
    """Filesystem-backed object store used in standalone CI / unit test runs."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        safe_parts = [p for p in Path(key).parts if p not in ("..", "/", "\\")]
        return self._root.joinpath(*safe_parts)

    def put_bytes(
        self, key: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return key

    def get_bytes(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()


def get_blob_store(settings: Settings | None = None) -> BlobStore:
    cfg = settings or get_settings()
    if cfg.local_blob_dir:
        return LocalFilesystemBlobStore(cfg.local_blob_dir)
    return MinioBlobStore(cfg)
