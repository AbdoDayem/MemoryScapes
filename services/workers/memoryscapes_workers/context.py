from __future__ import annotations

from dataclasses import dataclass

from memoryscapes_workers.storage import Storage


@dataclass(frozen=True)
class StageContext:
    """Shared inputs for every pipeline stage.

    Each stage reads from and writes to {memory_id}/{stage}/ in `storage` and must be
    idempotent, so a failed workflow can resume from the last completed stage.
    """

    memory_id: str
    storage: Storage
    input_kind: str = "video"  # "video" | "panoramas"

    def prefix(self, stage: str) -> str:
        return f"{self.memory_id}/{stage}/"

    def key(self, stage: str, name: str) -> str:
        return f"{self.memory_id}/{stage}/{name}"
