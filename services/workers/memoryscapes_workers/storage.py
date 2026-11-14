"""Blob storage for stage artifacts.

Stages only talk to `Storage`, so the same code runs against a local folder (dev, tests)
or S3/R2/MinIO (deployed). Keys are relative, e.g. ``mem_123/separate/stems.json`` or
``mem_123/poses/sparse/0/cameras.bin``.

- ``LocalStorage``: a directory on disk (``STORAGE_ROOT``). For local development and tests.
- ``S3Storage``: S3, R2 or MinIO (``S3_ENDPOINT``, ``S3_BUCKET``, ...).

``get_storage()`` picks one from the environment: ``STORAGE_BACKEND=local|s3``.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Storage(Protocol):
    def list(self, prefix: str) -> list[str]: ...

    def exists(self, key: str) -> bool: ...

    def read_bytes(self, key: str) -> bytes: ...

    def write_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None: ...

    def download(self, key: str, dest: Path) -> None: ...

    def upload(self, src: Path, key: str) -> None: ...

    def delete_prefix(self, prefix: str) -> None: ...

    def url_for(self, key: str, expires_s: int = 3600) -> str:
        """A URL another service (e.g. SATIE) can fetch the object from."""
        ...


def _retry_locked(fn, attempts: int = 8, delay_s: float = 0.05):
    """Run ``fn``, retrying while Windows reports the file as in use.

    On Windows a file another process has open (the API streaming it to a viewer, an
    indexer, antivirus) can't be replaced or deleted; it's usually free again within
    moments. POSIX doesn't have the problem, so there the first attempt just succeeds.
    """
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay_s * (2**i))
    return None


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

    def write_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        _retry_locked(lambda: tmp.replace(path))

    def download(self, key: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(key), dest)

    def upload(self, src: Path, key: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        shutil.copyfile(src, tmp)
        _retry_locked(lambda: tmp.replace(path))

    def delete_prefix(self, prefix: str) -> None:
        prefix = _norm_prefix(prefix)
        base = self._path(prefix)
        if base.is_dir():
            _retry_locked(lambda: shutil.rmtree(base))

    def url_for(self, key: str, expires_s: int = 3600) -> str:
        return self._path(key).as_uri()


class S3Storage:
    def __init__(
        self,
        bucket: str,
        client: Any | None = None,
        *,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
    ) -> None:
        self.bucket = bucket
        if client is None:
            import boto3

            client = boto3.client(
                "s3",
                endpoint_url=endpoint_url or None,
                aws_access_key_id=access_key or None,
                aws_secret_access_key=secret_key or None,
                region_name=region or None,
            )
        self.client = client

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

    def write_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

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

    def url_for(self, key: str, expires_s: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s
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


# ---------- typed helpers ----------


def read_json(storage: Storage, key: str) -> Any:
    return json.loads(storage.read_bytes(key))


def write_json(storage: Storage, key: str, value: Any) -> None:
    storage.write_bytes(key, json.dumps(value, indent=2).encode(), "application/json")


def read_wav(storage: Storage, key: str) -> tuple[np.ndarray, int]:
    """Returns (audio, sample_rate) with audio shaped (channels, samples), float32.

    Any format libsndfile reads (WAV, FLAC, ...) works, whatever the key's extension."""
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(storage.read_bytes(key)), dtype="float32", always_2d=True)
    return data.T.copy(), int(sr)


def write_wav(storage: Storage, key: str, audio: np.ndarray, sample_rate: int) -> None:
    """audio shaped (channels, samples) or (samples,). Written as 16-bit PCM, which is
    what SATIE accepts for uploaded audio."""
    import soundfile as sf

    audio = np.atleast_2d(audio)
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio.T, -1.0, 1.0), sample_rate, subtype="PCM_16", format="WAV")
    storage.write_bytes(key, buf.getvalue(), "audio/wav")


def read_npy(storage: Storage, key: str) -> np.ndarray:
    return np.load(io.BytesIO(storage.read_bytes(key)), allow_pickle=False)


def write_npy(storage: Storage, key: str, array: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    storage.write_bytes(key, buf.getvalue(), "application/octet-stream")
