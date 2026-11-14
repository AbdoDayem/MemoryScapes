"""Shared machinery that makes stages idempotent and resumable.

Every stage runs through :func:`run_stage`:

1. Fingerprint the stage's config, code version and upstream fingerprints.
2. If ``{memory_id}/{stage}/_stage.json`` exists with the same fingerprint, skip.
3. Otherwise clear the stage prefix, download inputs into a scratch dir, run the
   stage body, upload its output dir, and write the marker last.

A crash before step 3 finishes leaves no marker, so the next run starts clean.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.storage import download_prefix, upload_dir

STAGE_MARKER = "_stage.json"
UPLOAD_PREFIX = "upload"

log = logging.getLogger("memoryscapes.stage")


class StageError(RuntimeError):
    """A stage failed for a reason the user can act on (shown in the UI)."""

    def __init__(self, stage: str, message: str, hint: str | None = None) -> None:
        super().__init__(f"[{stage}] {message}" + (f" Hint: {hint}" if hint else ""))
        self.stage = stage
        self.message = message
        self.hint = hint


@dataclass
class StageIO:
    ctx: StageContext
    stage: str
    work_dir: Path
    out_dir: Path
    _inputs: dict[str, Path] = field(default_factory=dict)

    def input_dir(self, upstream: str) -> Path:
        """Download ``{memory_id}/{upstream}/`` (once) and return the local copy."""
        if upstream not in self._inputs:
            dest = self.work_dir / "inputs" / upstream
            download_prefix(self.ctx.storage, self.ctx.prefix(upstream), dest)
            (dest / STAGE_MARKER).unlink(missing_ok=True)
            self._inputs[upstream] = dest
        return self._inputs[upstream]

    def read_upstream_json(self, upstream: str, name: str) -> Any:
        return json.loads(self.ctx.storage.read_bytes(self.ctx.prefix(upstream) + name))

    def publish_early(self, rel_path: str) -> str:
        """Upload one output file before the stage finishes (e.g. a preview splat)."""
        key = self.ctx.prefix(self.stage) + rel_path
        self.ctx.storage.upload(self.out_dir / rel_path, key)
        return key

    def progress(self, fraction: float, message: str = "") -> None:
        log.info("[%s] %3.0f%% %s", self.stage, fraction * 100, message)
        if self.ctx.on_progress is not None:
            self.ctx.on_progress(self.stage, fraction, message)


def _fingerprint(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def read_marker(ctx: StageContext, stage: str) -> dict[str, Any] | None:
    key = ctx.prefix(stage) + STAGE_MARKER
    if not ctx.storage.exists(key):
        return None
    return json.loads(ctx.storage.read_bytes(key))


def run_stage(
    ctx: StageContext,
    stage: str,
    *,
    version: str,
    config: BaseModel,
    upstream: list[str],
    body: Callable[[StageIO], dict[str, Any]],
    extra_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``body`` as an idempotent stage and return its summary dict.

    ``extra_inputs`` joins the fingerprint, for inputs without a stage marker (e.g. a
    digest of an output written by a stage that doesn't use run_stage).
    """
    upstream_fps: dict[str, str] = {}
    for name in upstream:
        if name == UPLOAD_PREFIX:
            keys = ctx.storage.list(ctx.prefix(UPLOAD_PREFIX))
            if not keys:
                raise StageError(stage, "No uploaded files found.", "Upload a video first.")
            upstream_fps[name] = _fingerprint({"keys": keys})
            continue
        marker = read_marker(ctx, name)
        if marker is None:
            raise StageError(stage, f"Upstream stage {name!r} has not completed.")
        upstream_fps[name] = marker["fingerprint"]

    payload = {"stage": stage, "version": version, "config": config.model_dump(mode="json"),
               "upstream": upstream_fps}
    if extra_inputs:  # only when given, so existing fingerprints stay valid
        payload["extra"] = extra_inputs
    fingerprint = _fingerprint(payload)
    existing = read_marker(ctx, stage)
    if existing and existing.get("fingerprint") == fingerprint and not ctx.force:
        log.info("[%s] up to date (%s), skipping", stage, fingerprint)
        return existing.get("summary", {})

    ctx.storage.delete_prefix(ctx.prefix(stage))
    started = time.time()
    if ctx.work_root is not None:
        ctx.work_root.mkdir(parents=True, exist_ok=True)
    # ignore_cleanup_errors: on Windows a file another process still holds can't be deleted.
    with tempfile.TemporaryDirectory(prefix=f"ms-{stage}-", dir=ctx.work_root,
                                     ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        io = StageIO(ctx=ctx, stage=stage, work_dir=work, out_dir=work / "out")
        io.out_dir.mkdir()
        io.progress(0.0, "starting")
        summary = body(io)
        upload_dir(ctx.storage, io.out_dir, ctx.prefix(stage))

    marker = {
        "stage": stage,
        "version": version,
        "fingerprint": fingerprint,
        "upstream": upstream_fps,
        "config": config.model_dump(mode="json"),
        "seconds": round(time.time() - started, 2),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
    }
    ctx.storage.write_bytes(ctx.prefix(stage) + STAGE_MARKER, json.dumps(marker, indent=2).encode())
    log.info("[%s] done in %.1fs", stage, marker["seconds"])
    return summary


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
