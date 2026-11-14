"""Render an example memory to an .mp4 (video + its stereo recording), to test the upload path.

    .venv/Scripts/python examples/make_video.py park_walk out.mp4 [--fps 8]

The video is the scene's point cloud rendered along its camera path. It's crude, but it's
textured enough for COLMAP and its sounds sit at known positions, so localization on
the uploaded file can be checked against debug ground truth.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scenes import SCENARIOS, camera_path, render  # noqa: E402
from splat_fake import png_bytes, point_colors, render_view  # noqa: E402

import shutil  # noqa: E402


def render_video(scenario, out: str | Path, fps: float = 8.0):
    """Render the scenario to an .mp4 at `out`; returns the Rendered scene (ground truth)."""
    r = render(scenario)
    cams = camera_path(scenario.conditions.camera, scenario.duration_s, n=int(scenario.duration_s * fps) + 1)
    colors = point_colors(r)
    # denser points so the frames have enough texture for feature matching
    rng = np.random.default_rng(0)
    pts = np.concatenate([r.points + rng.normal(0, 0.04, r.points.shape) for _ in range(3)])
    cols = np.concatenate([np.clip(colors.astype(int) + rng.integers(-25, 26, colors.shape), 0, 255).astype(np.uint8) for _ in range(3)])

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        for i, cam in enumerate(cams):
            (t / f"{i:05d}.png").write_bytes(png_bytes(render_view(cam, pts, cols, width=640, height=480, radius_m=0.12)))
        sf.write(t / "audio.wav", r.recording.T, r.sample_rate, subtype="PCM_16")
        subprocess.run(
            [shutil.which("ffmpeg") or "ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(t / "%05d.png"), "-i", str(t / "audio.wav"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)],
            check=True,
        )
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=list(SCENARIOS))
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=8.0)
    args = ap.parse_args()
    r = render_video(SCENARIOS[args.scenario], args.out, args.fps)
    channels = "stereo" if r.recording.shape[0] == 2 else "mono"
    print(f"wrote {args.out}: {r.scenario.duration_s} s, {channels} audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
