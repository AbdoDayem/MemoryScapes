"""Spatial (stereo) video: probing, both eyes at ingest, and metric scale from disparity."""

import json
from dataclasses import dataclass

import numpy as np
import pytest

from memoryscapes_workers.visual import alignment as al
from memoryscapes_workers.visual import stereo
from memoryscapes_workers.visual.video import parse_probe, parse_stereo

from markers import needs_cv2, needs_ffmpeg

# ffprobe's video stream for an iPhone 16 Plus spatial video (MV-HEVC), trimmed.
SPATIAL_STREAM = {
    "codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080,
    "avg_frame_rate": "30/1", "view_ids_available": "0,1", "view_pos_available": "2,1",
    "side_data_list": [{"side_data_type": "Stereo 3D", "type": "unspecified", "view": "packed",
                        "baseline": 17738, "horizontal_disparity_adjustment": "200/10000",
                        "horizontal_field_of_view": "63400/1000"}],
}


def test_spatial_video_probe_names_the_eyes_and_the_lens_distance():
    info = parse_probe({"format": {"duration": "34.8"}, "streams": [SPATIAL_STREAM]}, "IMG_9661.MOV")
    assert info.stereo.left_view == 1 and info.stereo.right_view == 0  # view position 1 = left
    assert info.stereo.baseline_m == pytest.approx(0.017738)
    assert info.stereo.hfov_deg == pytest.approx(63.4)
    plain = {k: v for k, v in SPATIAL_STREAM.items() if "view" not in k}
    assert parse_stereo(plain) is None
    assert parse_stereo({**SPATIAL_STREAM, "view_pos_available": "0,0"}) is None


def _texture(h=360, w=640, seed=0):
    import cv2

    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 1.5)
    return cv2.normalize(t, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _right_eye(left, disp_of_x, dy=1.0):
    """The right eye's view: a point at left x appears at x - d(x), dy pixels higher."""
    import cv2

    h, w = left.shape
    xr = np.arange(w, dtype=np.float32)
    src_x = xr + disp_of_x(xr)  # d varies slowly, so d(x_r) ~ d(x)
    map_x = np.tile(src_x, (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None] + dy, (1, w))
    return cv2.remap(left, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


@needs_cv2
def test_disparity_is_measured_to_a_fraction_of_a_pixel():
    left = _texture()
    right = _right_eye(left, lambda x: np.full_like(x, 6.3))
    pts = np.array([[x, y] for x in range(120, 600, 40) for y in range(40, 320, 40)], float)
    disp, score, vert = stereo.measure_disparity(left, right, pts)
    ok = np.isfinite(disp)
    assert ok.mean() > 0.9 and np.all(score[ok] > 0.85)
    assert np.median(np.abs(disp[ok] - 6.3)) < 0.15
    assert np.median(vert[ok]) == pytest.approx(-1.0, abs=0.15)  # the right eye sits 1 px higher
    flat = np.full_like(left, 128)
    assert np.isnan(stereo.measure_disparity(flat, flat, pts)[0]).all()  # nothing to match


def test_the_scale_fit_survives_outliers_and_a_constant_view_shift():
    rng = np.random.default_rng(1)
    s_true, d0, fx, B = 2.4, 0.7, 1300.0, 0.0177
    z_metric = rng.uniform(1.5, 14.0, 1500)
    z_colmap = z_metric / s_true
    disp = fx * B / z_metric + d0 + rng.normal(0, 0.15, z_metric.size)
    bad = rng.random(z_metric.size) < 0.2
    disp[bad] = rng.uniform(1, 40, bad.sum())  # mismatches
    fit = stereo.fit_scale(disp, z_colmap, np.full(z_metric.size, fx), B)
    assert fit.scale == pytest.approx(s_true, rel=0.02)
    assert fit.disparity_offset_px == pytest.approx(d0, abs=0.2)
    assert fit.confidence > 0.6
    assert stereo.fit_scale(disp[:10], z_colmap[:10], np.full(10, fx), B) is None  # too few points


@dataclass
class _P2D:
    xy: np.ndarray
    point3D_id: int


class _Pose:
    def __mul__(self, p):  # identity camera at the origin
        return np.asarray(p, float)


class _Image:
    has_pose = True

    def __init__(self, name, obs):
        self.name, self.camera_id, self._obs = name, 1, obs

    def get_observation_points2D(self):
        return self._obs

    def cam_from_world(self):
        return _Pose()


class _Camera:
    def __init__(self, fx, w, h):
        self.K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])

    def calibration_matrix(self):
        return self.K


@needs_cv2
def test_scale_from_a_reconstruction_and_the_right_eyes_frames(tmp_path):
    import cv2

    fx, B, s_true = 600.0, 0.0177, 3.0
    h, w = 360, 640
    left = _texture(h, w, seed=2)
    z_of_x = lambda x: 1.5 + 6.0 * x / w  # a wall slanting away: 1.5 m to 7.5 m  # noqa: E731
    right = _right_eye(left, lambda x: fx * B / z_of_x(x))
    (tmp_path / "L").mkdir(), (tmp_path / "R").mkdir()
    cv2.imwrite(str(tmp_path / "L" / "frame_00000.jpg"), left, [cv2.IMWRITE_JPEG_QUALITY, 98])
    cv2.imwrite(str(tmp_path / "R" / "frame_00000.jpg"), right, [cv2.IMWRITE_JPEG_QUALITY, 98])
    rng = np.random.default_rng(3)
    xs, ys = rng.uniform(80, w - 30, 400), rng.uniform(30, h - 30, 400)
    points3D, obs = {}, []
    for k, (x, y) in enumerate(zip(xs, ys)):
        z_colmap = z_of_x(x) / s_true
        points3D[k] = type("P3", (), {"xyz": np.array([(x - w / 2) / fx * z_colmap, (y - h / 2) / fx * z_colmap, z_colmap])})()
        obs.append(_P2D(np.array([x, y]), k))
    rec = type("Rec", (), {"images": {1: _Image("frame_00000.jpg", obs)}, "points3D": points3D,
                           "cameras": {1: _Camera(fx, w, h)}})()
    fit = stereo.scale_from_reconstruction(rec, tmp_path / "L", tmp_path / "R", B)
    assert fit is not None and fit.frames == 1 and fit.points > 150
    assert fit.scale == pytest.approx(s_true, rel=0.05)


def test_a_measured_scale_replaces_the_eye_height_guess():
    rng = np.random.default_rng(0)
    centers = np.stack([np.linspace(0, 10, 40), np.zeros(40), np.zeros(40)], axis=1)
    rots = np.tile(np.diag([1.0, -1.0, -1.0]), (40, 1, 1))  # looking along +z, y down
    # cam_up = c2w @ (0, -1, 0) = +y here, so the floor is below the cameras at y = -3
    ground = np.stack([rng.uniform(-5, 15, 500), np.full(500, -3.0), rng.uniform(1, 10, 500)], axis=1)
    guess = al.estimate_alignment(centers, rots, ground)
    measured = al.estimate_alignment(centers, rots, ground, measured_scale=(0.42, 0.9, "stereo"))
    assert guess.scale_method == "eye_height"
    assert measured.scale == pytest.approx(0.42) and measured.scale_method == "stereo"
    assert measured.scale_confidence == pytest.approx(0.9)
    fixed = al.estimate_alignment(centers, rots, ground, fixed_scale=1.0, measured_scale=(0.42, 0.9, "stereo"))
    assert fixed.scale == 1.0 and fixed.scale_method == "fixed"  # set by hand still wins


@needs_ffmpeg
@needs_cv2
def test_ingest_keeps_both_eyes_of_a_spatial_video(ctx, small_video, monkeypatch):
    """No MV-HEVC encoder in CI: probe says stereo, and both 'views' decode the same clip."""
    from memoryscapes_workers.stages import ingest
    from memoryscapes_workers.visual import video as vid
    from memoryscapes_workers.cli import import_upload

    clip, _ = small_video
    import_upload(ctx.storage, ctx.memory_id, clip, None)
    real_probe, real_iter = vid.probe, vid.iter_frames
    views_used = set()

    def probe(path):
        info = real_probe(path)
        info.stereo = vid.StereoInfo(left_view=1, right_view=0, baseline_m=0.017738, hfov_deg=63.4)
        return info

    def iter_frames(src, size, fps, gray=False, view=None):
        views_used.add(view)
        return real_iter(src, size, fps, gray=gray)

    monkeypatch.setattr(vid, "probe", probe)
    monkeypatch.setattr(vid, "iter_frames", iter_frames)
    monkeypatch.setattr(vid, "supports_views", lambda: True)
    summary = ingest.run(ctx)
    assert summary["stereo"] is True and views_used == {0, 1}
    meta = json.loads(ctx.storage.read_bytes(ctx.key("ingest", "meta.json")))
    assert meta["stereo"]["baseline_m"] == pytest.approx(0.017738)
    left = ctx.storage.list(ctx.key("ingest", "frames/"))
    right = ctx.storage.list(ctx.key("ingest", "frames_right/"))
    assert len(left) == len(right) == meta["keyframes"]["count"]
    w = meta["keyframes"]["width"]
    assert meta["keyframes"]["focal_px"] == pytest.approx(w / 2 / np.tan(np.radians(63.4 / 2)), abs=0.1)
