import numpy as np

from memoryscapes_workers.audio import doa
from memoryscapes_workers.audio.fusion import cluster_candidates
from memoryscapes_workers.models import SceneBounds, StemInfo, StemsIndex
from memoryscapes_workers.stages import localize
from memoryscapes_workers.storage import read_json, write_json, write_npy, write_wav
from memoryscapes_workers.vision.grounding import Detection, lift_box
from synth import SR, look_camera, noise, project, stereo_from_source, walking_path, wall_points

SPACING = 0.15
SOURCE = np.array([2.5, 1.6, 5.0])  # on the wall, at mic height
DURATION = 8.0


def test_stereo_doa_triangulates_source_along_moving_path():
    cams = walking_path(DURATION)
    stereo = stereo_from_source(noise(DURATION, seed=5), SOURCE, cams, SPACING)
    bearings = doa.stereo_bearings(stereo, SR, SPACING, activity=lambda t: 1.0)
    assert len(bearings) > 10

    path = doa.CameraPath(cams)
    pts = wall_points()
    bounds = SceneBounds.from_points(np.concatenate([pts, np.stack([c.position for c in cams])]))
    cands = doa.candidate_points(pts, np.array(bounds.min), np.array(bounds.max))
    est = doa.triangulate(cands, doa.bearing_likelihood(cands, bearings, path))
    assert est is not None
    assert np.linalg.norm(est.position - SOURCE) < 1.0, est


def test_lift_box_puts_detection_on_the_wall():
    cam = look_camera(np.array([0.0, 1.6, 0.0]), 0.0, 0.0, "f0")
    target = np.array([1.0, 1.0, 5.0])
    u, v = project(cam, target)
    p = lift_box((u - 30, v - 30, u + 30, v + 30), cam, wall_points())
    assert p is not None
    assert np.linalg.norm(p - target) < 0.3


def test_cluster_candidates_prefers_consistent_views_over_one_strong_outlier():
    pts = np.array([[1, 1, 5], [1.2, 1, 5], [0.9, 1.1, 5], [-4, 1, 5]], float)
    w = np.array([0.5, 0.5, 0.5, 0.9])
    clusters = cluster_candidates(pts, w)
    assert clusters[0].count == 3
    assert np.linalg.norm(clusters[0].centre - [1.03, 1.03, 5]) < 0.1


class FakeDetector:
    """Finds the true source in every frame where it's in view, plus a distractor
    'bird' in a single frame."""

    def __init__(self, cams, truth):
        self.cams = {c.frame_id: c for c in cams}
        self.truth = truth
        self.calls = 0

    def detect(self, image_bytes, prompt):
        self.calls += 1
        cam = self.cams[image_bytes.decode()]
        out = []
        pc = np.linalg.inv(cam.c2w)[:3] @ np.append(self.truth, 1)
        u, v = project(cam, self.truth)
        if pc[2] > 0 and 0 <= u < cam.width and 0 <= v < cam.height:
            out.append(Detection((u - 40, v - 40, u + 40, v + 40), 0.8, "bird"))
        if cam.frame_id == "f004":
            d = project(cam, np.array([-3.0, 2.0, 5.0]))
            out.append(Detection((d[0] - 40, d[1] - 40, d[0] + 40, d[1] + 40), 0.9, "bird"))
        return out


def _write_inputs(ctx, layout="stereo"):
    s = ctx.storage
    cams = walking_path(DURATION)
    for c in cams:
        c.image_key = ctx.key("ingest", f"frames/{c.frame_id}.jpg")
        s.write_bytes(c.image_key, c.frame_id.encode())
    write_json(s, ctx.key("poses", "cameras.json"), [c.to_json() for c in cams])
    write_npy(s, ctx.key("poses", "points3d.npy"), wall_points())
    write_json(s, ctx.key("ingest", "meta.json"), {"mic": {"channels": 2, "layout": layout, "spacing_m": SPACING}})

    bird = stereo_from_source(noise(DURATION, seed=7), SOURCE, cams, SPACING)
    wind = np.stack([noise(DURATION, seed=8), noise(DURATION, seed=9)])
    times = list(np.arange(0.25, DURATION, 0.25))
    stems = []
    for sid, label, audio in (("s1_birdsong", "birdsong", bird), ("s2_wind", "wind", wind)):
        key = ctx.key("separate", f"stems/{sid}.wav")
        write_wav(s, key, audio, SR)
        stems.append(StemInfo(sid, label, "event", key, -26.0, [1.0] * len(times), 0.5))
    index = StemsIndex(SR, 2, layout, DURATION, times, stems, ctx.key("separate", "ambient.wav"))
    write_json(s, ctx.key("separate", "stems.json"), index.to_json())
    return cams


def test_localize_stage_fuses_av_and_doa(ctx):
    cams = _write_inputs(ctx)
    detector = FakeDetector(cams, SOURCE)
    result = localize.run(ctx, detector=detector)

    by_id = {s["id"]: s for s in result["sources"]}
    bird = by_id["s1_birdsong"]
    assert bird["method"] == "av+doa"
    assert np.linalg.norm(np.array(bird["position"]) - SOURCE) < 0.5
    assert bird["level_db_at_1m"] > bird["level_dbfs"]  # source is metres away
    assert by_id["s2_wind"]["method"] == "diffuse" and by_id["s2_wind"]["position"] is None

    saved = read_json(ctx.storage, ctx.key("localize", "sources.json"))
    assert saved == result


def test_localize_stage_uses_doa_alone_without_a_detector(ctx):
    _write_inputs(ctx)
    result = localize.run(ctx, detector=None)
    bird = next(s for s in result["sources"] if s["id"] == "s1_birdsong")
    assert bird["method"] == "doa"
    assert np.linalg.norm(np.array(bird["position"]) - SOURCE) < 1.0


def test_localize_stage_mono_recording_uses_av_only(ctx):
    cams = _write_inputs(ctx, layout="mono")
    result = localize.run(ctx, detector=FakeDetector(cams, SOURCE))
    bird = next(s for s in result["sources"] if s["id"] == "s1_birdsong")
    assert bird["method"] == "av"
    assert np.linalg.norm(np.array(bird["position"]) - SOURCE) < 0.5


def test_lift_box_on_sparse_far_points_is_finite():
    """Few points spread over a large depth range (a real outdoor walk): the depth band
    must still hold a point, not average an empty set into NaN."""
    cam = look_camera([0, 1.5, 0], 0, 0.0, "f0")
    # Seven points straight ahead at very different depths.
    # Two near, three far: the interpolated 35th-percentile depth (~41 m) has no point
    # within 20% of it. Yaw 0 looks along +Z.
    pts = np.array([[0.0, 1.5, d] for d in (2.0, 2.1, 100.0, 101.0, 102.0)])
    u, v = 320.0, 240.0
    p = lift_box((u - 40, v - 40, u + 40, v + 40), cam, pts, min_points=5)
    assert p is not None and np.isfinite(p).all()


def test_clustering_ignores_nan_points_and_zero_weights():
    pts = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]])
    clusters = cluster_candidates(pts, np.array([1.0, 1.0, 1.0, 0.0]))
    assert len(clusters) == 1 and np.isfinite(clusters[0].centre).all() and clusters[0].count == 2
