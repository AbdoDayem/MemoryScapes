"""Camera poses.

- Quality path: GLOMAP / COLMAP.
- Fast path: feed-forward VGGT or MASt3R-SfM (poses + dense points in minutes).
- Estimate metric scale (monocular metric depth, or IMU/ARKit later) so audio
  distance attenuation works in meters.
"""

from memoryscapes_workers.context import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError
