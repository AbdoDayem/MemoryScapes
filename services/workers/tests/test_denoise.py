"""Cleaning before separation (audio/denoise.py): high-pass and two-mic wind suppression."""

import numpy as np
import pytest
from scipy import signal

from memoryscapes_workers.audio.denoise import CleaningConfig, clean_recording, highpass, suppress_wind

SR = 16_000
C, SPACING = 343.0, 0.12


def _place(mono, az_deg):
    """A sound arriving from az_deg at two mics 12 cm apart (fractional delay)."""
    tau = SPACING * np.sin(np.radians(az_deg)) / C
    spec = np.fft.rfft(mono)
    f = np.fft.rfftfreq(len(mono), 1 / SR)
    return np.stack([np.fft.irfft(spec * np.exp(1j * np.pi * f * tau), len(mono)),
                     np.fft.irfft(spec * np.exp(-1j * np.pi * f * tau), len(mono))]).astype(np.float32)


def _scene(seconds=8.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    band = signal.butter(4, [150, 5000], "band", fs=SR, output="sos")
    voice = signal.sosfiltfilt(band, rng.standard_normal(n)) * (1 + np.sin(2 * np.pi * 3 * np.arange(n) / SR))
    return _place(0.2 * voice / np.std(voice), 30)


def _mic_wind(n, seed=1):
    """Gusts hit both mics at once; the turbulence itself differs at each capsule."""
    rng = np.random.default_rng(seed)
    slow = signal.sosfiltfilt(signal.butter(2, 0.8, "low", fs=SR, output="sos"), rng.standard_normal(n))
    gust = np.maximum(slow / np.std(slow) + 0.3, 0) ** 2
    lp = signal.butter(4, 500, "low", fs=SR, output="sos")
    return np.stack([signal.sosfiltfilt(lp, rng.standard_normal(n)) * gust for _ in range(2)]).astype(np.float32)


def _si_sdr(est, ref):
    vals = []
    for e, r in zip(est, ref):
        a = np.dot(e, r) / np.dot(r, r)
        vals.append(10 * np.log10(np.sum((a * r) ** 2) / np.sum((a * r - e) ** 2)))
    return float(np.mean(vals))


def _lag(x):
    """Inter-channel delay in samples (cross-correlation peak)."""
    corr = signal.correlate(x[0], x[1], mode="full", method="fft")
    return int(np.argmax(corr)) - (x.shape[1] - 1)


def test_highpass_removes_rumble_and_keeps_the_rest():
    t = np.arange(4 * SR) / SR
    x = np.stack([np.sin(2 * np.pi * 25 * t), np.sin(2 * np.pi * 1000 * t)]).astype(np.float32)
    y = highpass(x, SR, 60)
    mid = slice(SR, 3 * SR)  # away from the edges
    assert 20 * np.log10(np.std(y[0, mid]) / np.std(x[0, mid])) < -20
    assert abs(20 * np.log10(np.std(y[1, mid]) / np.std(x[1, mid]))) < 0.1


def test_wind_is_suppressed_and_the_sound_and_its_direction_kept():
    clean = _scene()
    wind = _mic_wind(clean.shape[1])
    wind *= np.sqrt(np.sum(clean ** 2) / np.sum(wind ** 2))  # as much wind as sound
    noisy = clean + wind
    out, report = clean_recording(noisy, SR)
    assert report["wind"] == "suppressed" and report["wind_like_frames"] > 0.05  # the gusts (calm gaps between)
    reference = highpass(clean, SR, 60)
    # dB better. This wind is flat up to 500 Hz, so it overlaps the voice more than real wind
    # (which falls off steeply); the 1/f-shaped wind of the evaluation gained +10 dB.
    assert _si_sdr(out, reference) - _si_sdr(noisy, reference) > 3
    assert report["removed_db"]["20-100 Hz"] > 3 and abs(report["removed_db"]["500-2000 Hz"]) < 1.5
    assert _lag(out) == _lag(clean)  # same gain on both channels: direction cues survive


def test_a_recording_without_wind_passes_almost_unchanged():
    clean = _scene(seed=3)
    out, report = clean_recording(clean, SR)
    assert report["wind"] == "suppressed"
    assert _si_sdr(out, highpass(clean, SR, 60)) > 20
    assert report["wind_like_frames"] < 0.02


def _phone_stereo(seconds=8.0, seed=4):
    """Still air, but channels that hardly agree above ~500 Hz, as phones' processed stereo
    does: each channel keeps the scene below 500 Hz and gets its own version above it."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    low = signal.butter(4, 500, "low", fs=SR, output="sos")
    high = signal.butter(4, [500, 5000], "band", fs=SR, output="sos")
    voice = signal.sosfiltfilt(signal.butter(4, [150, 5000], "band", fs=SR, output="sos"), rng.standard_normal(n))
    voice *= 0.2 * (1 + np.sin(2 * np.pi * 3 * np.arange(n) / SR)) / np.std(voice)
    common = _place(signal.sosfiltfilt(low, voice), 30)
    own = [signal.sosfiltfilt(high, rng.standard_normal(n)) * (1 + np.sin(2 * np.pi * 3 * np.arange(n) / SR))
           for _ in range(2)]
    own = np.stack(own) * (np.std(signal.sosfiltfilt(high, voice)) / np.std(own))
    return (common + own).astype(np.float32)


def test_phone_stereo_without_wind_keeps_its_mids():
    """iPhone clips lost 3-5 dB at 0.5-2 kHz with no wind at all: above 500 Hz their
    channels disagree, which an ungated gain reads as wind. The 50-200 Hz band shows no
    wind, so above 200 Hz the gain stays off."""
    x = _phone_stereo()
    _, report = clean_recording(x, SR)
    assert report["wind"] == "suppressed" and report["wind_like_frames"] < 0.02
    assert abs(report["removed_db"]["500-2000 Hz"]) < 0.3
    ungated = CleaningConfig(wind_full_hz=1000.0, wind_fade_hz=1000.0, wind_max_hz=1001.0)  # the gain everywhere below 1 kHz
    _, before = clean_recording(x, SR, ungated)
    assert before["removed_db"]["500-2000 Hz"] > 1.0


def test_phone_stereo_with_wind_still_has_its_wind_removed():
    x = _phone_stereo(seed=5)
    wind = _mic_wind(x.shape[1], seed=6)
    wind *= np.sqrt(np.sum(x ** 2) / np.sum(wind ** 2))
    _, report = clean_recording(x + wind, SR)
    assert report["wind_like_frames"] > 0.05 and report["removed_db"]["20-100 Hz"] > 3


@pytest.mark.parametrize("channels, expected", [
    (1, "skipped: 1 channel"), (4, "skipped: 4 channels")])
def test_wind_needs_exactly_two_channels(channels, expected):
    x = np.tile(_scene()[0], (channels, 1))
    _, report = clean_recording(x, SR)
    assert report["wind"].startswith(expected)


def test_near_identical_channels_are_left_alone():
    mono = _scene()[0]
    out, report = clean_recording(np.stack([mono, mono * 0.98]), SR)
    assert report["wind"].startswith("skipped: the channels are near-identical")
    np.testing.assert_allclose(out, highpass(np.stack([mono, mono * 0.98]), SR, 60), atol=1e-6)


def test_blocks_join_without_seams():
    noisy = _scene(seconds=14.0) + 0.5 * _mic_wind(14 * SR)
    blocks = CleaningConfig(block_s=5.0, overlap_s=2.0, crossfade_s=0.5)
    whole = CleaningConfig(block_s=100.0)
    a, _ = suppress_wind(noisy, SR, blocks)
    b, _ = suppress_wind(noisy, SR, whole)
    assert np.max(np.abs(a - b)) < 0.02 * np.max(np.abs(b))


def test_cleaning_can_be_configured_off():
    x = _scene() + _mic_wind(8 * SR)
    out, report = clean_recording(x, SR, CleaningConfig(highpass_hz=0, wind=False))
    np.testing.assert_array_equal(out, x)
    assert report["wind"] == "off" and report["highpass_hz"] is None
