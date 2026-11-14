import numpy as np
import pytest
from scipy import signal

from memoryscapes_workers.visual.audio_align import Alignment, apply_alignment, apply_offset, find_offset


def scene(seconds, sr, seed=0):
    """Something both microphones hear: noise bursts plus a few tones."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    rate, tone = rng.uniform(0.4, 1.2), rng.uniform(300, 900)
    env = (np.sin(2 * np.pi * rate * t + rng.uniform(0, 6)) > 0.3) * 0.5 + 0.1
    x = rng.standard_normal(len(t)) * env + 0.3 * np.sin(2 * np.pi * tone * t) * (t % 3 < 1)
    return x.astype(np.float32)


def recorder(world, world_sr, sr, start_s, seconds, seed=1, drift_ppm=0.0):
    """Another device: own sample rate, started at start_s of world time, different
    frequency response and noise, optionally a slightly fast clock."""
    i0 = int(start_s * world_sr)
    n = int(seconds * world_sr)
    clip = world[max(0, i0) : i0 + n] if i0 >= 0 else np.concatenate([np.zeros(-i0, np.float32), world[: i0 + n]])
    y = signal.resample_poly(clip, sr // np.gcd(sr, world_sr), world_sr // np.gcd(sr, world_sr))
    if drift_ppm:  # a fast clock counts more samples for the same stretch of world time
        y = signal.resample(y, int(round(len(y) * (1 + drift_ppm * 1e-6))))
    y = signal.sosfilt(signal.butter(2, [150, 3000], "bandpass", fs=sr, output="sos"), y)
    y = y + 0.05 * np.random.default_rng(seed).standard_normal(len(y))
    return np.stack([y, 0.8 * y]).astype(np.float32)  # stereo


WORLD_SR = 48000


@pytest.mark.parametrize("offset", [3.217, -1.5, 0.0])
def test_finds_offset_between_devices(offset):
    world = scene(40, WORLD_SR)
    video = recorder(world, WORLD_SR, 48000, start_s=10.0, seconds=20, seed=2)
    # The separate recorder started `offset` seconds before the video.
    sep = recorder(world, WORLD_SR, 44100, start_s=10.0 - offset, seconds=25, seed=3)
    a = find_offset(video, 48000, sep, 44100)
    assert a.offset_s == pytest.approx(offset, abs=0.002)
    assert a.reliable


def test_unrelated_audio_has_low_confidence():
    video = recorder(scene(30, WORLD_SR, seed=5), WORLD_SR, 48000, 0, 20)
    sep = recorder(scene(30, WORLD_SR, seed=6), WORLD_SR, 44100, 0, 20)
    assert not find_offset(video, 48000, sep, 44100).reliable


def test_reports_clock_drift():
    world = scene(120, WORLD_SR)
    video = recorder(world, WORLD_SR, 48000, start_s=5, seconds=90, seed=2)
    sep = recorder(world, WORLD_SR, 48000, start_s=3, seconds=100, seed=3, drift_ppm=100)
    a = find_offset(video, 48000, sep, 48000)
    # 100 ppm (the high end for consumer devices) = 9 ms over the 90 s video.
    assert a.offset_s == pytest.approx(2.0, abs=0.005)  # well under a video frame
    assert a.drift_ms == pytest.approx(9.0, abs=2.0)
    # Resampled along the fitted line, the end of the clip lines up too.
    aligned = apply_alignment(sep, 48000, a, 90.0)
    b = find_offset(video[:, -20 * 48000:], 48000, aligned[:, -20 * 48000:], 48000)
    assert abs(b.offset_s) < 0.001


def test_small_drift_is_a_plain_shift():
    audio = np.arange(30, dtype=np.float32)[None]
    a = Alignment(offset_s=1.0, confidence=500, rate=1.00001, drift_ms=0.5)
    np.testing.assert_array_equal(apply_alignment(audio, 10, a, 1.0), apply_offset(audio, 10, 1.0, 1.0))


def test_apply_offset_trims_and_pads():
    sr = 10
    audio = np.arange(30, dtype=np.float32)[None]  # 3 s
    np.testing.assert_array_equal(apply_offset(audio, sr, 1.0, 1.0)[0], np.arange(10, 20))
    late = apply_offset(audio, sr, -0.5, 1.0)[0]  # recorder started 0.5 s after the video
    np.testing.assert_array_equal(late, np.r_[np.zeros(5), np.arange(5)])
    assert apply_offset(audio, sr, 2.5, 1.0)[0].tolist() == [25, 26, 27, 28, 29, 0, 0, 0, 0, 0]
