from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memoryscapes_workers.storage import Storage


@dataclass(frozen=True)
class StageContext:
    """Shared inputs for every pipeline stage.

    Each stage reads from and writes to {memory_id}/{stage}/ in `storage` and must be
    idempotent, so a failed workflow can resume from the last completed stage.

    The raw upload (written by the API through presigned URLs) lives under
    ``{memory_id}/upload/``. Per-stage options go in ``config[stage]``.
    """

    memory_id: str
    storage: Storage = field(compare=False, repr=False)
    input_kind: str = "video"  # "video" | "panoramas"
    config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    # Scratch space for downloads and intermediate files. Defaults to a temp dir.
    work_root: Path | None = None
    # Re-run stages even when their completion marker matches.
    force: bool = False
    # Optional progress hook (stage, fraction 0..1, message), e.g. to feed SSE events.
    on_progress: Callable[[str, float, str], None] | None = field(
        default=None, compare=False, repr=False
    )

    def prefix(self, stage: str) -> str:
        return f"{self.memory_id}/{stage}/"

    def key(self, stage: str, name: str) -> str:
        return f"{self.memory_id}/{stage}/{name}"

    def stage_config(self, stage: str) -> Mapping[str, Any]:
        return self.config.get(stage, {})
