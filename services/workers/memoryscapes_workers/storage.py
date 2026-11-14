"""Object storage used by pipeline stages.

Keys are always relative to the bucket, e.g. ``mem_123/poses/sparse/0/cameras.bin``.
Two backends:

- ``LocalStorage``: a directory on disk (``STORAGE_ROOT``). For local development,
  tests, and Modal volumes.
- ``S3Storage``: S3, R2 or MinIO (``S3_ENDPOINT``, ``S3_BUCKET``, ...).

``get_storage()`` picks one from the environment: ``STORAGE_BACKEND=local|s3``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def list(self, prefix: str) -> list[str]: ...

    def exists(self, key: str) -> bool: ...

    def read_bytes(self, key: str) -> bytes: ...

    def write_bytes(self, key: str, data: bytes) -> None: ...

    def download(self, key: str, dest: Path) -> None: ...

    def upload(self, src: Path, key: str) -> None: ...

    def delete_prefix(self, prefix: str) -> None: ...


def _norm_prefix(prefix: str) -> str:
    return prefix if not prefix or prefix.endswith("/") else prefix + "/"


def download_prefix(storage: Storage, prefix: str, dest: Path) -> list[Path]:
    """Download every object under ``prefix`` into ``dest``, keeping relative paths."""
    prefix = _norm_prefix(prefix)
    paths = []
    for key in storage.list(prefix):
        target = dest / key[len(prefix) :]
        storage.download(key, target)
        paths.append(target)
    return paths


def upload_dir(storage: Storage, src: Path, prefix: str) -> list[str]:
    """Upload every file under ``src`` to ``prefix``, keeping relative paths."""
    prefix = _norm_prefix(prefix)
    keys = []
    for path in sorted(p for p in src.rglob("*") if p.is_file()):
        key = prefix + path.relative_to(src).as_posix()
        storage.upload(path, key)
        keys.append(key)
    return keys


class LocalStorage:
    """A directory that mimics a bucket. Keys map to ``root/key``."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError(f"Key escapes storage root: {key!r}")
        return path

    def _walk(self, prefix: str) -> Iterator[Path]:
        # Prefixes are string prefixes, like S3: "a/b" matches "a/b/c" and "a/bc".
        if not prefix:
            base = self.root
        elif prefix.endswith("/"):
            base = self._path(prefix)
        else:
            base = self._path(prefix).parent
        if not base.exists():
            return
        for path in base.rglob("*"):
            if path.is_file() and path.relative_to(self.root).as_posix().startswith(prefix):
                yield path

    def list(self, prefix: str) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self._walk(prefix))

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), dest)

    def upload(self, src: Path, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(path)

    def delete_prefix(self, prefix: str) -> None:
        prefix = _norm_prefix(prefix)
        base = self._path(prefix)
        if base.is_dir():
            shutil.rmtree(base)


class S3Storage:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ) -> None:
        import boto3

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            region_name=region or None,
        )

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return sorted(keys)

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

    def write_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))

    def upload(self, src: Path, key: str) -> None:
        self.client.upload_file(str(src), self.bucket, key)

    def delete_prefix(self, prefix: str) -> None:
        keys = self.list(_norm_prefix(prefix))
        for i in range(0, len(keys), 1000):
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in keys[i : i + 1000]], "Quiet": True},
            )


def get_storage(backend: str | None = None) -> Storage:
    backend = backend or os.environ.get("STORAGE_BACKEND", "local")
    if backend == "local":
        return LocalStorage(os.environ.get("STORAGE_ROOT", "data/storage"))
    if backend == "s3":
        return S3Storage(
            bucket=os.environ.get("S3_BUCKET", "memories"),
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            access_key=os.environ.get("S3_ACCESS_KEY"),
            secret_key=os.environ.get("S3_SECRET_KEY"),
            region=os.environ.get("S3_REGION"),
        )
    raise ValueError(f"Unknown storage backend: {backend}")
