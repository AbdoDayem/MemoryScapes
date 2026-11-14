"""Ingest (ffmpeg) and the pose-normalization math. The ffmpeg tests skip if ffmpeg isn't
installed; COLMAP itself is exercised by running the pipeline, not here."""

import json
import subprocess

import numpy as np
import pytest

from memoryscapes_workers import run as runner
from memoryscapes_workers.media import MediaError, find_tool, parse_iso6709
from memoryscapes_workers.stages import ingest
from memoryscapes_workers.stages.poses import normalize_frame, rotation_between
from memoryscapes_workers.storage import read_json, read_wav

try:
    FFMPEG = find_tool("ffmpeg")
except MediaError:
    FFMPEG = None
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


def make_video(path, seconds=4, stereo=True, audio=True):
    args = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=10:duration={seconds}"]
    if audio:
        layout = "stereo" if stereo else "mono"
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=16000:duration={seconds}", "-af", f"aformat=channel_layouts={layout}", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-metadata", "location=+37.4275-122.1697+030.000/", "-shortest", str(path)]
    subprocess.run(args, check=True)


def upload(ctx, video_name="video.mp4"):
    root = ctx.storage.root
    (root / ctx.memory_id / "upload").mkdir(parents=True, exist_ok=True)
    (root / ctx.memory_id / "upload" / "meta.json").write_text(json.dumps({"video": video_name, "audio": None}))
    return root / ctx.memory_id / "upload" / video_name


@needs_ffmpeg
def test_ingest_extracts_audio_frames_and_metadata(ctx):
    make_video(upload(ctx), seconds=4)
    meta = ingest.run(ctx, ingest.IngestConfig(target_frames=12, drop_blurry=0))
    assert meta["mic"] == {"channels": 2, "layout": "stereo"}
    assert meta["location"]["lat"] == pytest.approx(37.4275)
    audio, sr = read_wav(ctx.storage, ctx.key("ingest", "audio.wav"))
    assert sr == 16000 and audio.shape[0] == 2 and abs(audio.shape[1] / sr - 4) < 0.1
    frames = read_json(ctx.storage, ctx.key("ingest", "frames.json"))
    assert 10 <= len(frames) <= 14
    assert frames[0]["t"] == 0 and all(b["t"] > a["t"] for a, b in zip(frames, frames[1:]))
    assert all(ctx.storage.exists(f["key"]) for f in frames)


@needs_ffmpeg
def test_ingest_caps_duration(ctx):
    make_video(upload(ctx), seconds=6, stereo=False)
    meta = ingest.run(ctx, ingest.IngestConfig(max_duration_s=2, target_frames=8, drop_blurry=0))
    assert meta["mic"]["layout"] == "mono" and meta["source"]["used_s"] == 2
    audio, sr = read_wav(ctx.storage, ctx.key("ingest", "audio.wav"))
    assert abs(audio.shape[1] / sr - 2) < 0.1


@needs_ffmpeg
def test_ingest_without_audio_explains_what_to_do(ctx):
    make_video(upload(ctx), audio=False)
    with pytest.raises(MediaError, match="no audio track"):
        ingest.run(ctx)


def test_parse_iso6709():
    assert parse_iso6709("+37.4275-122.1697+030.000/") == {"lat": 37.4275, "lon": -122.1697, "alt": 30.0}
    assert parse_iso6709("-33.8688+151.2093/") == {"lat": -33.8688, "lon": 151.2093}
    assert parse_iso6709("nonsense") is None


def test_rotation_between_maps_vectors_including_opposites():
    for a, b in [((1, 0, 0), (0, 1, 0)), ((0, 0, 1), (0, 0, -1)), ((0.3, -0.9, 0.2), (0, -1, 0))]:
        R = rotation_between(np.array(a, float), np.array(b, float))
        np.testing.assert_allclose(R @ (np.array(a) / np.linalg.norm(a)), np.array(b) / np.linalg.norm(b), atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_normalize_frame_recovers_upright_metric_frame():
    """Build a known scene (cameras 1.5 m above a ground plane, walking +x, looking +z),
    scramble it with a random rotation, scale and offset as COLMAP would, and check the
    normalization undoes it."""
    rng = np.random.default_rng(0)
    c2w_true = []
    for i in range(10):
        c2w = np.eye(4)
        c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = [-1, 0, 0], [0, -1, 0], [0, 0, 1]  # right, down, forward
        c2w[:3, 3] = [0.5 * i, 1.5, 0]
        c2w_true.append(c2w)
    c2w_true = np.stack(c2w_true)
    ground = np.column_stack([rng.uniform(-3, 8, 500), np.zeros(500), rng.uniform(0, 10, 500)])

    R, s, t = _random_rotation(rng), 0.37, np.array([4.0, -2.0, 7.0])
    sim = np.eye(4)
    sim[:3, :3], sim[:3, 3] = R * s, t
    c2w_colmap = np.einsum("ij,kjl->kil", sim, c2w_true)
    c2w_colmap[:, :3, :3] /= s  # rotations stay orthonormal
    pts_colmap = (R @ ground.T).T * s + t

    c2w, pts, info = normalize_frame(c2w_colmap, pts_colmap, camera_height_m=1.5)
    np.testing.assert_allclose(c2w[:, :3, 1], np.tile([0, -1, 0], (10, 1)), atol=1e-6)  # upright
    np.testing.assert_allclose(c2w[0, :3, 2], [0, 0, 1], atol=1e-6)  # first camera looks +z
    np.testing.assert_allclose(c2w[:, 1, 3], 1.5, atol=0.05)  # 1.5 m above the ground
    np.testing.assert_allclose(np.linalg.norm(c2w[-1, :3, 3] - c2w[0, :3, 3]), 4.5, rtol=0.03)  # metric
    assert abs(np.percentile(pts[:, 1], 50)) < 0.05  # ground at y = 0
    assert info["scale_method"].startswith("median camera height")


def test_runner_skips_unimplemented_and_blocks_dependents(tmp_path, monkeypatch):
    """With poses failing, localize and soundscape are blocked; splat/publish are skipped;
    separate still runs."""
    calls = []

    def ok(name):
        return lambda ctx: calls.append(name)

    def boom(ctx):
        raise RuntimeError("no texture")

    def todo(ctx):
        raise NotImplementedError

    dag = {
        "ingest": (ok("ingest"), []),
        "poses": (boom, ["ingest"]),
        "splat": (todo, ["poses"]),
        "separate": (ok("separate"), ["ingest"]),
        "localize": (ok("localize"), ["poses", "separate"]),
        "soundscape": (ok("soundscape"), ["localize"]),
        "publish": (todo, ["splat", "soundscape"]),
    }
    monkeypatch.setattr(runner, "DAG", dag)
    monkeypatch.setattr(runner, "topological_order", lambda: list(dag))
    from memoryscapes_workers.stages import ingest as ing, poses as pos, separate as sep, localize as loc

    monkeypatch.setattr(ing, "run", lambda ctx, cfg=None: calls.append("ingest"))
    monkeypatch.setattr(pos, "run", boom)
    monkeypatch.setattr(sep, "run", lambda ctx: calls.append("separate"))
    monkeypatch.setattr(loc, "run", lambda ctx: calls.append("localize"))

    code = runner.main(["--root", str(tmp_path), "--memory", "m"])
    status = json.loads((tmp_path / "m" / "job" / "status.json").read_text())
    stages = {k: v["status"] for k, v in status["stages"].items()}
    assert code == 1 and status["state"] == "failed"
    assert stages == {"ingest": "done", "poses": "failed", "splat": "blocked", "separate": "done", "localize": "blocked", "soundscape": "blocked", "publish": "blocked"}
    assert "no texture" in status["stages"]["poses"]["error"]
    assert calls == ["ingest", "separate"]
    assert (tmp_path / "m" / "job" / "log.txt").exists()
