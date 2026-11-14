"""Blob storage for stage artifacts.

Stages only talk to `Storage`, so the same code runs against a local folder (dev, tests)
or S3/R2/MinIO (deployed). Keys are relative, e.g. "mem_123/separate/stems.json".
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Storage(Protocol):
    def read_bytes(self, key: str) -> bytes: ...
    def write_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None: ...
    def exists(self, key: str) -> bool: ...
    def url_for(self, key: str, expires_s: int = 3600) -> str:
        """A URL another service (e.g. SATIE) can fetch the object from."""
        ...


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def write_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def url_for(self, key: str, expires_s: int = 3600) -> str:
        return self._path(key).resolve().as_uri()


class S3Storage:
    def __init__(self, bucket: str, client: Any | None = None) -> None:
        import boto3

        self.bucket = bucket
        self.client = client or boto3.client("s3")

    def read_bytes(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def write_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        extra = {"ContentType": content_type} if content_type else {}
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except self.client.exceptions.ClientError:
            return False

    def url_for(self, key: str, expires_s: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s
        )


# ---------- typed helpers ----------


def read_json(storage: Storage, key: str) -> Any:
    return json.loads(storage.read_bytes(key))


def write_json(storage: Storage, key: str, value: Any) -> None:
    storage.write_bytes(key, json.dumps(value, indent=2).encode(), "application/json")


def read_wav(storage: Storage, key: str) -> tuple[np.ndarray, int]:
    """Returns (audio, sample_rate) with audio shaped (channels, samples), float32."""
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
