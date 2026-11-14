"""Inputs at the edges: tiny or silent recordings, one camera, no point cloud,
missing metadata, and the decision rules in isolation."""

import numpy as np
import pytest

from memoryscapes_workers.audio.doa import DoaEstimate
from memoryscapes_workers.audio.fusion import Cluster, fuse
from memoryscapes_workers.audio.tagging import TagResult
from memoryscapes_workers.models import Camera
from memoryscapes_workers.providers.mcp_transport import McpError
from memoryscapes_workers.providers.soundscape import SatieMcpProvider, StubProvider
from memoryscapes_workers.stages import localize, separate, soundscape
from memoryscapes_workers.stages.separate import SeparationConfig, find_redundant, is_present
from memoryscapes_workers.storage import read_json, write_json, write_npy, write_wav
from memoryscapes_workers.vision.grounding import lift_box, touches_border
from synth import SR, look_camera, noise, project, wall_points


class Tagger:
    def __init__(self, labels):
        self.labels = labels

    def tag(self, mono, sr, classes):
        return TagResult(np.array([0.5, 1.0]), {c.label: np.full(2, 0.6 if c.label in self.labels else 0.0) for c in classes})


class Separator:
    def separate(self, mono, sr, classes):
        return {c.label: mono * 0.5 for c in classes}


def _ingest(ctx, audio, layout="stereo", spacing=None):
    write_wav(ctx.storage, ctx.key("ingest", "audio.wav"), audio, SR)
    mic = {"channels": int(np.atleast_2d(audio).shape[0]), "layout": layout}
    if spacing:
        mic["spacing_m"] = spacing
    write_json(ctx.storage, ctx.key("ingest", "meta.json"), {"mic": mic})


def _poses(ctx, cams, points):
    for c in cams:
        c.image_key = ctx.key("ingest", f"frames/{c.frame_id}.jpg")
        ctx.storage.write_bytes(c.image_key, c.frame_id.encode())
    write_json(ctx.storage, ctx.key("poses", "cameras.json"), [c.to_json() for c in cams])
    write_npy(ctx.storage, ctx.key("poses", "points3d.npy"), points)


def _all_stages(ctx, tagger, detector=None):
    separate.run(ctx, tagger=tagger, separator=Separator())
    result = localize.run(ctx, detector=detector)
    scene = soundscape.run(ctx, provider=StubProvider())
    return result, scene


def test_recording_shorter_than_every_analysis_window(ctx):
    _ingest(ctx, np.stack([noise(0.3), noise(0.3, seed=1)]))
    _poses(ctx, [look_camera(np.array([0, 1.6, 0.0]), 0, 0.0, "f0")], wall_points())
    result, scene = _all_stages(ctx, Tagger({"birdsong"}))
    assert len(result["sources"]) == 1  # analysed without crashing; placement may be diffuse
    assert scene["sources"][0]["label"] == "birdsong"


def test_silent_recording(ctx):
    _ingest(ctx, np.zeros((2, SR * 3), np.float32))
    _poses(ctx, [look_camera(np.array([0, 1.6, 0.0]), 0, 0.0, "f0")], wall_points())
    result, scene = _all_stages(ctx, Tagger(set()))
    assert result["sources"] == [] and scene["sources"] == []


def test_single_camera_and_no_point_cloud(ctx):
    _ingest(ctx, np.stack([noise(3.0), noise(3.0, seed=1)]))
    _poses(ctx, [look_camera(np.array([0, 1.6, 0.0]), 0, 0.0, "f0")], np.zeros((0, 3), np.float32))
    result, _ = _all_stages(ctx, Tagger({"water"}))
    assert result["sources"][0]["label"] == "water"


def test_stereo_without_spacing_uses_default(ctx):
    _ingest(ctx, np.stack([noise(3.0), noise(3.0, seed=1)]), spacing=None)
    _poses(ctx, [look_camera(np.array([x, 1.6, 0.0]), 0, t, f"f{i}") for i, (x, t) in enumerate([(-1, 0.0), (1, 3.0)])], wall_points())
    result, _ = _all_stages(ctx, Tagger({"water"}))
    assert len(result["sources"]) == 1


def test_mono_file_labelled_stereo_skips_doa(ctx):
    _ingest(ctx, noise(3.0), layout="stereo")
    _poses(ctx, [look_camera(np.array([0, 1.6, 0.0]), 0, 0.0, "f0")], wall_points())
    result, _ = _all_stages(ctx, Tagger({"water"}))
    assert result["sources"][0]["method"] == "diffuse"


# ---------- decision rules


@pytest.mark.parametrize(
    "scores, present",
    [([0.21, 0.05, 0.03], False), ([0.21, 0.2, 0.0], True), ([0.4, 0.0, 0.0], True), ([0.25], True), ([], False)],
)
def test_tag_presence_rule(scores, present):
    assert is_present(np.array(scores), SeparationConfig()) is present


def test_find_redundant_keeps_the_better_supported_look_alike():
    rng = np.random.default_rng(0)
    a, b = rng.random((40, 30)), rng.random((40, 30))
    mags = np.stack([a, b, 0.4 * (a + b)])  # third is leakage of the first two
    assert set(find_redundant(mags, [0.6, 0.5, 0.3], 0.9)) == {2}
    assert find_redundant(mags[:2], [0.6, 0.5], 0.9) == {}


def test_lift_box_cut_off_by_frame_edge_still_lands_on_the_object():
    cam = look_camera(np.array([0.0, 1.6, 0.0]), 0.0, 0.0, "f0")
    obj = np.array([2.2, 1.2, 3.0])
    rng = np.random.default_rng(0)
    d = rng.standard_normal((400, 3))
    blob = obj + 0.3 * d / np.linalg.norm(d, axis=1, keepdims=True)
    points = np.concatenate([wall_points(z=8.0), blob])
    u, v = project(cam, obj)
    half = cam.K[0, 0] * 0.3 / 3.0
    box = (u - half, v - half, cam.width, v + half)  # right side cut off by the frame
    assert touches_border(box, cam)
    p = lift_box(box, cam, points)
    assert p is not None and np.linalg.norm(p - obj) < 0.5


def test_fusion_distrusts_doa_that_contradicts_the_camera():
    seen = [Cluster(np.array([0.0, 1.0, 10.0]), 1.0, 0.5, 2)]
    doa = DoaEstimate(position=np.array([5.0, 8.0, -1.0]), score=0.7, spread_m=0.5)
    placement = fuse(seen, doa_scorer=lambda p: np.zeros(len(p)), doa=doa)
    assert placement.method != "doa"


def test_satie_network_error_falls_back(ctx):
    class Broken:
        def call_tool(self, name, args):
            raise McpError("connection reset")

    write_json(ctx.storage, ctx.key("localize", "sources.json"), {"bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}, "sources": []})
    write_json(
        ctx.storage,
        ctx.key("separate", "stems.json"),
        {"sample_rate": SR, "channels": 1, "layout": "mono", "duration_s": 1.0, "window_times": [], "stems": [], "ambient_key": "a.wav"},
    )
    scene = soundscape.run(ctx, provider=SatieMcpProvider(Broken()))
    assert scene["provider"]["name"] == "stub"
    assert "connection reset" in scene["provider"]["fallback_reason"]
    assert read_json(ctx.storage, ctx.key("soundscape", "scene.json")) == scene
