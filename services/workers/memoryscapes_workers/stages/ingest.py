"""Ingest: demux audio and frames, read metadata, pick keyframes.

- ffmpeg demux; keep the original audio channel layout and sample rate.
- Choose 150-400 keyframes by sharpness (Laplacian variance) and parallax.
- Panoramas: split equirectangular frames into cubemap/perspective crops.
- Read GPS and capture time from EXIF or container metadata for the globe pin.
"""

from memoryscapes_workers.context import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError
