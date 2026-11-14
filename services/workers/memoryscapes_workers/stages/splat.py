"""Gaussian splat training and packaging.

- Train with gsplat (Nerfstudio Splatfacto).
- Remove floaters; crop to the convex hull of the camera trajectory.
- Compress to .spz (or SOG) and produce 2-3 LOD levels.
- Optionally publish an early low-iteration preview splat.
"""

from memoryscapes_workers.context import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError
