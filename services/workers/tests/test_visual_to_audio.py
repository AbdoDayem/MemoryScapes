"""The audio stages run on what the real ingest and poses stages write.

The audio tests use synthetic inputs written straight into storage; this one produces
them from a video instead (ingest -> poses), then runs separate -> localize ->
soundscape with stand-in models. The detector projects a known scene point through the
posed cameras, so localization must recover that point: that only works if
cameras.json (K, c2w, image_key) and points3d.npy mean what the audio stages assume.

Needs ffmpeg, OpenCV and pycolmap, so it is skipped in CI.
"""

import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy import signal

from markers import has_module, needs_cv2, needs_ffmpeg, needs_pycolmap
from memoryscapes_workers.audio.tagging import TagResult
from memoryscapes_workers.models import Camera, MicInfo, StemsIndex
from memoryscapes_workers.providers.soundscape import StubProvider
from memoryscapes_workers.stages import localize, publish, separate, soundscape
from memoryscapes_workers.storage import read_json, read_npy
from memoryscapes_workers.vision.grounding import Detection

pytestmark = [needs_ffmpeg, needs_cv2, needs_pycolmap]

SECONDS = 6.0


@pytest.fixture(scope="module")
def walk_video(tmp_path_factory):
    """The smallest synthetic walk COLMAP reliably reconstructs (~15 s to render)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from memoryscapes_workers.devtools.synthetic import SyntheticSpec, generate

    return generate(tmp_path_factory.mktemp("walk"),
                    SyntheticSpec(seconds=SECONDS, fps=15, width=480, height=270))


class BirdTagger:
    """Hears birdsong throughout (the synthetic track has a 3-3.8 kHz chirping bird)."""

    def tag(self, mono, sr, classes):
        times = np.arange(0.5, len(mono) / sr, 1.0)
        return TagResult(times, {c.label: np.full(len(times), 0.9 if c.label == "birdsong" else 0.01)
                                 for c in classes})


class HighpassSeparator:
    def separate(self, mono, sr, classes):
        sos = signal.butter(8, 2000, "highpass", fs=sr, output="sos")
        return {c.label: signal.sosfiltfilt(sos, mono).astype(np.float32) for c in classes}


class ProjectingDetector:
    """Boxes a fixed world point in every frame where it is visible."""

    def __init__(self, target, cameras, storage, half_px=12):
        self.target, self.half = target, half_px
        self.by_image = {hashlib.sha1(storage.read_bytes(c.image_key)).hexdigest(): c for c in cameras}
        self.calls = 0

    def detect(self, image_bytes, prompt):
        self.calls += 1
        cam = self.by_image[hashlib.sha1(image_bytes).hexdigest()]
        u, v, z = project(cam, self.target)
        if z <= 0.1 or not (self.half <= u < cam.width - self.half and self.half <= v < cam.height - self.half):
            return []
        return [Detection((u - self.half, v - self.half, u + self.half, v + self.half), 0.9, "bird")]


def project(cam: Camera, p):
    pc = np.linalg.inv(cam.c2w) @ np.append(p, 1.0)
    uvw = cam.K @ pc[:3]
    return uvw[0] / uvw[2], uvw[1] / uvw[2], pc[2]


def most_visible_point(points, cameras, margin=20):
    counts = np.zeros(len(points), int)
    for cam in cameras:
        pc = (np.c_[points, np.ones(len(points))] @ np.linalg.inv(cam.c2w).T)[:, :3]
        uv = (pc @ cam.K.T)[:, :2] / pc[:, 2:3]
        counts += (pc[:, 2] > 0.5) & (uv[:, 0] > margin) & (uv[:, 0] < cam.width - margin) \
            & (uv[:, 1] > margin) & (uv[:, 1] < cam.height - margin)
    return points[int(np.argmax(counts))], int(counts.max())


def test_audio_stages_run_on_real_ingest_and_poses(ctx, walk_video):
    # Through the CLI, which runs each stage in its own process: pycolmap (poses) can't
    # share a process with torch on macOS, and other tests in this session load torch.
    env = {**os.environ, "STORAGE_BACKEND": "local", "STORAGE_ROOT": str(ctx.storage.root)}
    cli = [sys.executable, "-m", "memoryscapes_workers"]
    trainer = ["--set", "splat.trainer=torch", "--set", "splat.train.steps=20",
               "--set", "splat.train.render_max_side=120"] if has_module("torch") else []
    for args in (["import", ctx.memory_id, str(walk_video)],
                 ["run", ctx.memory_id, "--until", "splat", *trainer]):
        res = subprocess.run([*cli, *args], env=env, capture_output=True, text=True, check=False)
        assert res.returncode == 0, res.stderr[-3000:]
    s = ctx.storage

    # ---- the documented contract (services/workers/README.md)
    mic = MicInfo.from_json(read_json(s, ctx.key("ingest", "meta.json"))["mic"])
    assert (mic.channels, mic.layout) == (2, "stereo")
    cameras = [Camera.from_json(c) for c in read_json(s, ctx.key("poses", "cameras.json"))]
    points = read_npy(s, ctx.key("poses", "points3d.npy")).astype(float)
    assert len(cameras) >= 10 and points.ndim == 2 and points.shape[1] == 3 and len(points) > 100
    assert all(s.exists(c.image_key) for c in cameras)
    times = [c.t for c in cameras]
    assert times == sorted(times) and 0 <= times[0] and times[-1] <= SECONDS
    for c in cameras:  # proper rigid c2w, metric frame with the ground near y = 0
        np.testing.assert_allclose(c.c2w[:3, :3] @ c.c2w[:3, :3].T, np.eye(3), atol=1e-4)
        assert 0.8 < c.position[1] < 2.5  # eye height above the ground

    # ---- audio stages with stand-in models
    index = separate.run(ctx, tagger=BirdTagger(), separator=HighpassSeparator())
    assert isinstance(index, StemsIndex) and [st.label for st in index.stems] == ["birdsong"]
    assert index.channels == 2 and index.sample_rate == 48000

    target, n_views = most_visible_point(points, cameras)
    assert n_views >= 3
    detector = ProjectingDetector(target, cameras, s)
    localize.run(ctx, detector=detector)
    result = read_json(s, ctx.key("localize", "sources.json"))
    (bird,) = result["sources"]
    assert detector.calls > 0
    assert "av" in bird["method"], bird
    # The detection box covers a patch of surface around the target, so allow ~a box width.
    assert np.linalg.norm(np.array(bird["position"]) - target) < 0.75, (bird["position"], target)

    soundscape.run(ctx, provider=StubProvider())
    scene = read_json(s, ctx.key("soundscape", "scene.json"))
    assert scene

    # ---- splat outputs (docs/SPLAT_OUTPUTS.md), checked the way tools/debugger does
    check_splat_outputs(ctx, cameras, points, rendered=has_module("torch"))

    # ---- manifest: validated against the schema; every URL resolves to a stored file
    publish.run(ctx)
    m = read_json(s, ctx.key("publish", "manifest.json"))
    assert m["audio"]["sources"][0]["label"] == "birdsong" and m["audio"]["sources"][0]["position"]
    urls = [lod["url"] for lod in m["splat"]["lods"]] + [x["url"] for x in m["audio"]["sources"]]
    urls += [m["audio"]["ambient_bed"]["url"], m["navigation"]["original_path"]]
    assert all(s.exists(f"{ctx.memory_id}/{u}") for u in urls), urls


def check_splat_outputs(ctx, cameras, points, rendered):
    from scipy.spatial import cKDTree

    from memoryscapes_workers.visual.splat_io import read_ply

    s = ctx.storage
    meta = read_json(s, ctx.key("splat", "meta.json"))
    assert (meta["units"], meta["up"]) == ("meters", "+y")
    assert {"trainer", "trainer_version", "iterations", "sh_degree", "eval_frames"} <= set(meta)
    ids = {c.frame_id for c in cameras}
    assert set(meta["eval_frames"]) <= ids
    metrics = read_json(s, ctx.key("splat", "metrics.json"))
    assert len({len(v) for v in metrics.values()}) == 1
    assert s.exists(ctx.key("splat", "lods/lod0.spz"))
    colors = read_npy(s, ctx.key("poses", "points3d_rgb.npy"))
    assert colors.dtype == np.uint8 and colors.shape == points.shape

    # Alignment: the splat must sit on the poses' point cloud (same metric frame).
    s.download(ctx.key("splat", "point_cloud.ply"), ctx.work_root / "check" / "pc.ply")
    means = read_ply(ctx.work_root / "check" / "pc.ply").means
    gap = np.median(cKDTree(points).query(means)[0])
    extent = np.linalg.norm(np.percentile(points, 98, 0) - np.percentile(points, 2, 0))
    assert gap < max(0.3, 0.03 * extent), gap
    scale = np.std(means, axis=0).mean() / np.std(points, axis=0).mean()
    assert abs(scale - 1) < 0.1, scale
    if rendered:
        assert meta["eval_frames"] and len(metrics["iteration"]) >= 2
        assert {f"{ctx.memory_id}/splat/renders/{i}.png" for i in ids} <= set(s.list(ctx.prefix("splat")))
