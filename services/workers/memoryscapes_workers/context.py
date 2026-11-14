from dataclasses import dataclass


@dataclass(frozen=True)
class StageContext:
    """Shared inputs for every pipeline stage.

    Each stage reads from and writes to s3://{bucket}/{memory_id}/{stage}/ and must be
    idempotent, so a failed workflow can resume from the last completed stage.
    """

    memory_id: str
    bucket: str
    input_kind: str = "video"  # "video" | "panoramas"

    def prefix(self, stage: str) -> str:
        return f"{self.memory_id}/{stage}/"
