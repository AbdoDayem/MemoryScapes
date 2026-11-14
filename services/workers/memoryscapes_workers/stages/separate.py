"""Audio source separation and labeling.

- General sound events: query-based separation (AudioSep / SAM-Audio-style prompts).
- Speech: enhancement + diarization. Music: HTDemucs.
- Tag each stem with CLAP or BEATs. The residual becomes the ambient bed.
"""

from memoryscapes_workers.context import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError
