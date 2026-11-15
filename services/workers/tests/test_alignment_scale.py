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


def test_jumps_between_shots_are_not_walking():
    """An edited video: two shots, the second starting 50 units away. Only the walking
    inside each shot counts toward the speed."""
    c, r, p, t = scene(path_units=0.4, floor_below=40.0)
    c2 = c.copy()
    c2[20:, 2] -= 50.0  # the cut: the camera "teleports"
    seg = (np.arange(len(c)) >= 20).astype(int)
    a = estimate_alignment(c2, r, p, times=t, segments=seg)
    within = sum(np.linalg.norm(np.diff(c2[seg == k], axis=0), axis=1).sum() for k in (0, 1))
    span = sum(t[seg == k][-1] - t[seg == k][0] for k in (0, 1))
    assert a.scale_method == "walking_speed" and abs(a.scale * within / span - 1.2) < 1e-6


def _brisk(unsure):
    """About 2.2 m/s by eye height: fast, but inside the 2.5 m/s limit. Unsure: a sparse
    floor among points that aren't on it, and a camera height that sways, as in a Tampere
    underpass whose eye-height scale came out at 0.18 confidence and ~2.4 m/s."""
    c, r, p, t = scene(path_units=17.0, floor_below=0.6)  # 2.5 m/unit -> 42.5 m in 20 s
    if unsure:
        rng = np.random.default_rng(2)
        c = c.copy()
        c[:, 1] += 0.4 * np.sin(np.linspace(0, 2 * np.pi, len(c)))
        below = np.stack([rng.uniform(-3, 3, 1500), rng.uniform(-3, -0.1, 1500), rng.uniform(-20, 2, 1500)], -1)
        p = np.vstack([p[:100], p[800:], below])
    return estimate_alignment(c, r, p, times=t), c, t


def test_a_confident_eye_height_may_imply_a_brisk_walk():
    a, c, t = _brisk(unsure=False)
    assert a.scale_method == "eye_height" and a.scale_confidence > 0.5
    assert 2.0 < speed(a, c, t) < 2.5


def test_an_unsure_eye_height_must_imply_an_ordinary_walk():
    a, c, t = _brisk(unsure=True)
    assert a.scale_method == "walking_speed" and "assumed a 1.2 m/s walk" in a.scale_note
    assert abs(speed(a, c, t) - 1.2) < 1e-6


def test_without_times_nothing_changes():
    c, r, p, _ = scene(path_units=0.4, floor_below=40.0)
    assert estimate_alignment(c, r, p).scale_method == "eye_height"
