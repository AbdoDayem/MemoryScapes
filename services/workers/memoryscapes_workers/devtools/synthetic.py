"""Synthetic test footage with ground truth.

Ray-casts a small textured courtyard (ground, walls, boxes) along a handheld-style
walking path at a known eye height, and mixes a stereo audio track with two placed
sound sources. Writes an .mp4 (H.264 + AAC, with creation time and GPS metadata) and
``ground_truth.json`` (per-frame camera poses, intrinsics, blurred frames, sources).

World frame: meters, +y up, ground at y=0 (the same frame the pipeline produces).
Camera poses use the OpenCV convention (x right, y down, z forward), camera-to-world.

    python -m memoryscapes_workers.devtools.synthetic data/synthetic --seconds 10

Needs numpy and ffmpeg only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EYE_HEIGHT_M = 1.5

# (min corner, max corner, palette A, palette B) in meters / RGB 0..1
BOXES: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple, tuple]] = [
    ((-20, -0.2, -20), (20, 0.0, 20), (0.42, 0.36, 0.28), (0.20, 0.30, 0.16)),  # ground
    ((-7.2, 0, -12), (-7.0, 4, 10), (0.70, 0.62, 0.52), (0.45, 0.30, 0.25)),  # west wall
    ((7.0, 0, -12), (7.2, 4, 10), (0.55, 0.60, 0.68), (0.25, 0.28, 0.40)),  # east wall
    ((-7.2, 0, -12.2), (7.2, 4, -12), (0.75, 0.70, 0.55), (0.35, 0.40, 0.30)),  # north wall
    ((-7.2, 0, 10), (7.2, 4, 10.2), (0.60, 0.50, 0.60), (0.30, 0.25, 0.35)),  # south wall
    ((-3.1, 0, 0.4), (-1.9, 1.2, 1.6), (0.80, 0.45, 0.20), (0.40, 0.18, 0.10)),
    ((3.0, 0, 1.5), (4.0, 1.5, 2.5), (0.25, 0.55, 0.75), (0.10, 0.20, 0.35)),
    ((-2.5, 0, -3.5), (-0.5, 1.0, -2.5), (0.85, 0.80, 0.30), (0.45, 0.35, 0.10)),
    ((2.6, 0, -2.9), (3.4, 2.0, -2.1), (0.60, 0.25, 0.55), (0.25, 0.10, 0.25)),
    ((-0.25, 0, -7.75), (1.25, 0.8, -6.25), (0.30, 0.70, 0.40), (0.10, 0.30, 0.15)),
]

SUN = np.array([0.4, 0.8, 0.3]) / np.linalg.norm([0.4, 0.8, 0.3])


@dataclass
class SyntheticSpec:
    seconds: float = 10.0
    fps: int = 30
    width: int = 640
    height: int = 360
    hfov_deg: float = 65.0
    # Fraction of the clip (start, end) where the camera stands still.
    pause: tuple[float, float] = (0.45, 0.55)
    # Frame indices rendered with heavy motion blur (keyframe selection should skip them).
    blur_every: int = 13
    sample_rate: int = 48000
    seed: int = 7
    lat_lon_alt: tuple[float, float, float] = (37.4275, -122.1697, 30.0)
    captured_at: str = "2026-06-01T18:30:00Z"
    sources: list[dict] = field(
        default_factory=lambda: [
            {"id": "bird", "label": "birdsong", "position": [3.5, 2.0, 2.0]},
            {"id": "fountain", "label": "water", "position": [-1.5, 0.5, -3.0]},
        ]
    )

    @property
    def num_frames(self) -> int:
        return int(round(self.seconds * self.fps))

    @property
    def focal(self) -> float:
        return 0.5 * self.width / np.tan(np.radians(self.hfov_deg) / 2)


# --------------------------------------------------------------------------- camera path


def _progress(t: float, spec: SyntheticSpec) -> float:
    """Map time to path progress s in [0, 1], holding still during the pause."""
    u = t / spec.seconds
    p0, p1 = spec.pause
    if u < p0:
        return 0.5 * (u / p0)
    if u < p1:
        return 0.5
    return 0.5 + 0.5 * (u - p1) / (1 - p1)


def camera_pose(t: float, spec: SyntheticSpec) -> np.ndarray:
    """Camera-to-world 4x4 (OpenCV axes) at time t."""
    s = _progress(t, spec)
    pos = np.array(
        [
            1.6 * np.sin(np.pi * s) + 0.3 * np.sin(6 * np.pi * s),
            EYE_HEIGHT_M + 0.02 * np.sin(2 * np.pi * 8 * s),
            5.0 - 9.0 * s,
        ]
    )
    yaw = np.radians(-30.0 * np.sin(2 * np.pi * s))
    pitch = np.radians(-10.0)
    fwd = np.array([-np.sin(yaw) * np.cos(pitch), np.sin(pitch), -np.cos(yaw) * np.cos(pitch)])
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, fwd], axis=1)
    c2w[:3, 3] = pos
    return c2w


# --------------------------------------------------------------------------- rendering


def _hash01(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (seed * 83492791)
    h = (h & 0xFFFFFFFF).astype(np.uint32)
    h ^= h >> np.uint32(13)
    h *= np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) / float(0xFFFFFF)


def _value_noise(u: np.ndarray, v: np.ndarray, freqs, footprint: np.ndarray, seed: int):
    """Non-repeating multi-octave value noise in [0, 1], band-limited per pixel."""
    total = np.zeros_like(u, dtype=np.float32)
    weight_sum = 0.0
    for k, f in enumerate(freqs):
        x, y = u * f, v * f
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        s = seed * 31 + k
        a, b = _hash01(x0, y0, s), _hash01(x0 + 1, y0, s)
        c, d = _hash01(x0, y0 + 1, s), _hash01(x0 + 1, y0 + 1, s)
        n = (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy
        # Fade octaves above Nyquist to their mean so distant surfaces don't alias.
        fade = np.clip((0.5 - f * footprint) / 0.25, 0.0, 1.0)
        amp = 1.0 / (k + 1)
        total += amp * (0.5 + (n - 0.5) * fade)
        weight_sum += amp
    return total / weight_sum


def render(c2w: np.ndarray, spec: SyntheticSpec) -> np.ndarray:
    """Render one RGB frame (H, W, 3) uint8."""
    w, h, f = spec.width, spec.height, spec.focal
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dirs_cam = np.stack([(us - w / 2) / f, (vs - h / 2) / f, np.ones_like(us)], axis=-1)
    dirs = dirs_cam.reshape(-1, 3) @ c2w[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origin = c2w[:3, 3]
    safe = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)

    n_rays = dirs.shape[0]
    t_best = np.full(n_rays, np.inf)
    box_id = np.full(n_rays, -1)
    axis_id = np.zeros(n_rays, dtype=int)
    for i, (lo, hi, _, _) in enumerate(BOXES):
        t0 = (np.asarray(lo) - origin) / safe
        t1 = (np.asarray(hi) - origin) / safe
        tnear = np.minimum(t0, t1)
        tmin = tnear.max(axis=1)
        tmax = np.maximum(t0, t1).min(axis=1)
        hit = (tmax >= tmin) & (tmin > 1e-4) & (tmin < t_best)
        t_best[hit] = tmin[hit]
        box_id[hit] = i
        axis_id[hit] = tnear[hit].argmax(axis=1)

    # Sky gradient for misses.
    sky_t = np.clip(dirs[:, 1] * 2 + 0.3, 0, 1)[:, None]
    color = (1 - sky_t) * np.array([0.85, 0.88, 0.92]) + sky_t * np.array([0.45, 0.62, 0.85])

    hit = box_id >= 0
    p = origin + dirs[hit] * t_best[hit, None]
    ax = axis_id[hit]
    normal = np.zeros_like(p)
    normal[np.arange(len(ax)), ax] = -np.sign(safe[hit][np.arange(len(ax)), ax])
    tangent_axes = np.array([[1, 2], [0, 2], [0, 1]])[ax]
    u = p[np.arange(len(ax)), tangent_axes[:, 0]]
    v = p[np.arange(len(ax)), tangent_axes[:, 1]]
    cos_inc = np.clip(np.abs((normal * dirs[hit]).sum(axis=1)), 0.2, 1.0)
    footprint = t_best[hit] / (f * cos_inc)

    shaded = np.zeros_like(p)
    for i, (_, _, ca, cb) in enumerate(BOXES):
        m = box_id[hit] == i
        if not m.any():
            continue
        mix = _value_noise(u[m], v[m], [0.35, 1.1], footprint[m], seed=spec.seed + 2 * i)
        detail = _value_noise(u[m], v[m], [3.0, 8.0, 20.0, 45.0], footprint[m], spec.seed + 2 * i + 1)
        mix = np.clip((mix - 0.5) * 2.5 + 0.5, 0, 1)[:, None]
        detail = np.clip((detail - 0.5) * 3.0 + 0.5, 0, 1)
        base = (1 - mix) * np.asarray(ca) + mix * np.asarray(cb)
        shaded[m] = base * (0.45 + 1.1 * detail[:, None])
    lambert = 0.35 + 0.65 * np.clip(normal @ SUN, 0, 1)
    color[hit] = shaded * lambert[:, None]
    img = np.clip(color.reshape(h, w, 3), 0, 1)
    return (img * 255 + 0.5).astype(np.uint8)


def motion_blur(img: np.ndarray, length: int = 15) -> np.ndarray:
    acc = np.zeros(img.shape, dtype=np.float32)
    for dx in range(-(length // 2), length // 2 + 1):
        acc += np.roll(img, dx, axis=1)
    return (acc / length).astype(np.uint8)


# --------------------------------------------------------------------------- audio


def synth_audio(spec: SyntheticSpec, poses: list[np.ndarray]) -> np.ndarray:
    """Stereo float32 [n, 2]: diffuse bed + sources panned/attenuated by camera pose."""
    rng = np.random.default_rng(spec.seed)
    sr = spec.sample_rate
    n = int(spec.seconds * sr)
    t = np.arange(n) / sr
    bed = rng.standard_normal((n, 2)).astype(np.float32)
    bed = np.cumsum(bed, axis=0)
    bed -= np.convolve(bed[:, 0], np.ones(2048) / 2048, mode="same")[:, None]
    bed = 0.02 * bed / (np.abs(bed).max() + 1e-9)

    chirp_env = (np.sin(2 * np.pi * 3.0 * t) > 0.6).astype(np.float32)
    bird = 0.3 * chirp_env * np.sin(2 * np.pi * (3000 + 800 * np.sin(2 * np.pi * 9 * t)) * t)
    water = rng.standard_normal(n).astype(np.float32)
    water = 0.15 * np.convolve(water, np.ones(8) / 8, mode="same")
    signals = {"bird": bird, "fountain": water}

    out = bed.copy()
    frame_idx = np.minimum((t * spec.fps).astype(int), len(poses) - 1)
    for src in spec.sources:
        pos = np.asarray(src["position"], dtype=float)
        sig = signals.get(src["id"], np.zeros(n, dtype=np.float32))
        gain_l = np.zeros(len(poses))
        gain_r = np.zeros(len(poses))
        for k, c2w in enumerate(poses):
            rel = c2w[:3, :3].T @ (pos - c2w[:3, 3])  # camera frame, x right, z forward
            dist = max(np.linalg.norm(rel), 1.0)
            pan = np.clip(rel[0] / max(np.linalg.norm(rel[[0, 2]]), 1e-6), -1, 1)
            theta = (pan + 1) * np.pi / 4
            gain_l[k], gain_r[k] = np.cos(theta) / dist, np.sin(theta) / dist
        out[:, 0] += sig * gain_l[frame_idx]
        out[:, 1] += sig * gain_r[frame_idx]
    return np.clip(out, -1, 1).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (audio * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(audio.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------- main


def iso6709(lat: float, lon: float, alt: float) -> str:
    return f"{lat:+08.4f}{lon:+09.4f}{alt:+08.3f}/"


def generate(out_dir: Path, spec: SyntheticSpec | None = None) -> Path:
    """Render the clip into ``out_dir`` and return the .mp4 path."""
    spec = spec or SyntheticSpec()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "video.mp4"
    wav_path = out_dir / "audio.wav"

    times = [i / spec.fps for i in range(spec.num_frames)]
    poses = [camera_pose(t, spec) for t in times]
    blurred = [i for i in range(spec.num_frames) if spec.blur_every and i % spec.blur_every == 5]
    write_wav(wav_path, synth_audio(spec, poses), spec.sample_rate)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{spec.width}x{spec.height}",
        "-r", str(spec.fps), "-i", "-",
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-g", str(spec.fps),
        "-c:a", "aac", "-b:a", "192k",
        "-metadata", f"creation_time={spec.captured_at}",
        "-metadata", f"location={iso6709(*spec.lat_lon_alt)}",
        "-shortest", str(video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for i, c2w in enumerate(poses):
        frame = render(c2w, spec)
        if i in blurred:
            frame = motion_blur(frame)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the synthetic video")
    wav_path.unlink()

    gt = {
        "description": "MemoryScapes synthetic courtyard. World: meters, +y up, ground y=0.",
        "eye_height_m": EYE_HEIGHT_M,
        "fps": spec.fps,
        "width": spec.width,
        "height": spec.height,
        "intrinsics": {"fx": spec.focal, "fy": spec.focal, "cx": spec.width / 2, "cy": spec.height / 2},
        "pause_seconds": [spec.pause[0] * spec.seconds, spec.pause[1] * spec.seconds],
        "blurred_frames": blurred,
        "frames": [
            {"index": i, "t": t, "c2w_opencv": np.round(c2w, 6).tolist()}
            for i, (t, c2w) in enumerate(zip(times, poses, strict=True))
        ],
        "sources": spec.sources,
        "location": list(spec.lat_lon_alt),
        "captured_at": spec.captured_at,
    }
    (out_dir / "ground_truth.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")
    return video_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()
    path = generate(
        args.out_dir,
        SyntheticSpec(seconds=args.seconds, fps=args.fps, width=args.width, height=args.height),
    )
    print(path)


if __name__ == "__main__":
    main()
