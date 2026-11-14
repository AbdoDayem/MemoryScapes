"""Metric scale: eye height above the ground plane, checked against walking pace."""

import numpy as np

from memoryscapes_workers.visual.alignment import estimate_alignment


def scene(path_units, floor_below, seconds=20.0, n=40, seed=0):
    """Cameras walking along -z (OpenCV axes, level) and a floor plane below them."""
    rng = np.random.default_rng(seed)
    z = -np.linspace(0, path_units, n)
    centers = np.stack([np.zeros(n), np.zeros(n), z], -1)
    rots = np.repeat(np.diag([1.0, -1.0, -1.0])[None], n, axis=0)  # c2w: cam y down = world -y
    floor = np.stack([rng.uniform(-3, 3, 800), np.full(800, -floor_below), rng.uniform(z.min() - 2, 2, 800)], -1)
    clutter = rng.uniform(-5, 5, (200, 3)) + [0, 2, 0]
    return centers, rots, np.vstack([floor, clutter]), np.linspace(0, seconds, n)


def speed(a, centers, times):
    return a.scale * np.linalg.norm(np.diff(centers, axis=0), axis=1).sum() / (times[-1] - times[0])


def test_plausible_eye_height_scale_is_kept():
    c, r, p, t = scene(path_units=8.0, floor_below=0.6)  # 1.5/0.6 = 2.5 m/unit -> 20 m in 20 s
    a = estimate_alignment(c, r, p, times=t)
    assert a.scale_method == "eye_height" and a.scale_note is None
    assert abs(a.scale - 2.5) < 0.05 and 0.9 < speed(a, c, t) < 1.1


def test_implausible_eye_height_scale_falls_back_to_walking_pace():
    # A "floor" 40 units below a 0.4-unit path: eye height says 0.04 m/s, not a walk.
    c, r, p, t = scene(path_units=0.4, floor_below=40.0)
    a = estimate_alignment(c, r, p, times=t)
    assert a.scale_method == "walking_speed" and "m/s" in a.scale_note
    assert abs(speed(a, c, t) - 1.2) < 1e-6 and a.scale_confidence < 0.4
    assert a.up_method == "camera_up"


def test_without_times_nothing_changes():
    c, r, p, _ = scene(path_units=0.4, floor_below=40.0)
    assert estimate_alignment(c, r, p).scale_method == "eye_height"
