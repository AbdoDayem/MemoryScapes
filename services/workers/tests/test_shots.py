"""Shot boundaries: cross-dissolves and cuts are found, camera motion isn't mistaken
for one, and poses links shots / reports what an edited video kept."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from memoryscapes_workers.stages.poses import cross_shot_pairs, shot_coverage  # noqa: E402
from memoryscapes_workers.visual.shots import find_shots  # noqa: E402

FPS = 12.0


def texture(seed, h=27, w=260):
    """A smooth scene (like real footage, frames change gradually as the camera pans)."""
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 255, (3, 9)).astype(np.float32)
    return cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)


def pan(tex, n, speed=0.6, start=0.0):
    """Frames of a camera panning across a wide texture (48 px wide window)."""
    return [tex[:, int(start + i * speed) : int(start + i * speed) + 48] for i in range(n)]


def video(transition="dissolve"):
    a = pan(texture(1), 60)
    b = pan(texture(2), 60, start=40)
    if transition == "dissolve":
        mix = [((1 - w) * a[-1] + w * b[0]) for w in np.linspace(0, 1, 12)[1:-1]]
        frames = a + mix + b
    else:
        frames = a + b
    return np.clip(np.array(frames), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("transition", ["dissolve", "cut"])
def test_finds_the_transition_between_two_shots(transition):
    s = find_shots(video(transition), FPS)
    assert len(s.segments) == 2
    (a0, a1), (b0, b1) = s.segments
    assert 50 <= a1 <= 62 and 58 <= b0 <= 74, s.segments
    if transition == "dissolve":
        assert s.transition[60:70].all()  # the blended frames are flagged


def test_camera_motion_is_not_a_transition():
    frames = np.clip(np.array(pan(texture(3), 120, speed=1.2)), 0, 255).astype(np.uint8)
    s = find_shots(frames, FPS)
    assert s.segments == [(0, 120)] and not s.transition.any()


def test_cross_shot_pairs_sample_each_shot():
    shot_of = {f"frame_{i:05d}.jpg": (0 if i < 30 else 1 if i < 50 else 2) for i in range(80)}
    pairs = cross_shot_pairs(shot_of, per_shot=4)
    assert len(pairs) == 3 * 4 * 4  # 3 shot pairs x 4 x 4
    assert all(shot_of[a] != shot_of[b] for a, b in pairs)
    assert ("frame_00000.jpg", "frame_00030.jpg") in pairs  # first frames included


def test_shot_coverage_reports_what_was_placed():
    meta = {"shots": [{"start_s": 0, "end_s": 10}, {"start_s": 11, "end_s": 20}],
            "keyframes": {"frames": [{"file": f"frames/f{i}.jpg", "shot": int(i >= 4)} for i in range(8)]}}
    cov = shot_coverage(meta, {"f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg", "f4.jpg"})
    assert [c["placed"] for c in cov] == [1.0, 0.25] and cov[1]["start_s"] == 11
