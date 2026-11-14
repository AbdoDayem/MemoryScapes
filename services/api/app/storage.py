"""Read-only access to the pipeline's blob storage (same layout as services/workers).

Keys look like ``{memory_id}/{stage}/...``. Configured with the same variables as the
workers: ``STORAGE_BACKEND=local|s3``; ``STORAGE_ROOT`` for local (defaults to
services/workers/data/storage, where the worker CLI writes when run from its folder);
``S3_BUCKET``, ``S3_ENDPOINT``, ``S3_ACCESS_KEY``, ``S3_SECRET_KEY``, ``S3_REGION`` for S3.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol

DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parents[2] / "workers" / "data" / "storage"


class BlobStore(Protocol):
    def exists(self, key: str) -> bool: ...
    def read_bytes(self, key: str) -> bytes: ...
    def list(self, prefix: str) -> list[str]: ...


class LocalStore:
    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).resolve()

    def path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Key escapes storage root: {key!r}")
        return path

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def read_bytes(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def list(self, prefix: str) -> list[str]:
        base = self.path(prefix)
        if not base.is_dir():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())


class S3Store:
    def __init__(self) -> None:
        import boto3

        self.bucket = os.environ.get("S3_BUCKET", "memories")
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT") or None,
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY") or None,
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY") or None,
            region_name=os.environ.get("S3_REGION") or None,
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def read_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)

    def presigned_url(self, key: str, expires_s: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s)


@lru_cache(maxsize=1)
def get_store() -> LocalStore | S3Store:
    backend = os.environ.get("STORAGE_BACKEND", "local")
    if backend == "local":
        return LocalStore(os.environ.get("STORAGE_ROOT") or DEFAULT_LOCAL_ROOT)
    if backend == "s3":
        return S3Store()
    raise ValueError(f"Unknown storage backend: {backend}")
