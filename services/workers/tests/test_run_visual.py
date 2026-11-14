"""The frame alignment used by examples/run_visual.py (true cameras -> recovered ones)."""

import numpy as np

from run_visual import fit_similarity
from scenes import camera_path


def _rotation(rng):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q * np.sign(np.linalg.det(q))


def _transform(c2ws, s, R, t):
    out = c2ws.copy()
    out[:, :3, :3] = np.einsum("ij,kjl->kil", R, c2ws[:, :3, :3])
    out[:, :3, 3] = s * c2ws[:, :3, 3] @ R.T + t
    return out


def test_recovers_scale_rotation_and_offset_on_a_near_straight_walk():
    """walk_pan is almost a straight line, so positions alone can't fix the rotation
    about the walking direction; the orientations must."""
    rng = np.random.default_rng(1)
    true = np.stack([c.c2w for c in camera_path("walk_pan", 12.0, n=30)])
    s, R, t = 0.42, _rotation(rng), np.array([3.0, -1.0, 8.0])
    sim = fit_similarity(true, _transform(true, s, R, t))
    assert abs(sim.scale - s) < 1e-6
    np.testing.assert_allclose(sim.R, R, atol=1e-6)
    np.testing.assert_allclose(sim.t, t, atol=1e-6)
    assert sim.rms_m < 1e-6
    p = np.array([2.5, 1.6, 5.0])
    np.testing.assert_allclose(sim.apply(p), s * R @ p + t, atol=1e-6)


def test_noisy_poses_give_a_close_fit_and_report_the_residual():
    rng = np.random.default_rng(2)
    true = np.stack([c.c2w for c in camera_path("walk_pan", 12.0, n=40)])
    s, R, t = 1.3, _rotation(rng), np.zeros(3)
    pipe = _transform(true, s, R, t)
    pipe[:, :3, 3] += rng.normal(0, 0.03, (len(pipe), 3))
    sim = fit_similarity(true, pipe)
    assert abs(sim.scale - s) / s < 0.03
    assert 0.02 < sim.rms_m < 0.08
