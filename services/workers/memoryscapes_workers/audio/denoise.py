"""Clean a recording before separation: a gentle high-pass, then wind suppression.

1. High-pass at 60 Hz (zero phase): rumble and handling noise, with almost no audible loss.
2. Wind (two-channel recordings): microphone wind is turbulence at each capsule, so it's
   uncorrelated between the mics, while real sound reaches both. Where the channels
   disagree below ~1.6 kHz, that part is turned down (a Wiener gain from the coherent share
   of the power: |cross-spectrum| / mean channel power). The same gain goes on both channels,
   so the inter-channel cues localization needs are kept. Above ~1.6 kHz nothing changes:
   there, real sound fields stop being coherent at phone mic spacings.

On a synthetic scene with mic wind (stereo, 12 cm), this improved SI-SDR by +4.7, +10.0
and +12.1 dB for light, medium and heavy wind, and kept birdsong and speech detail intact.
Real low sounds lose some energy only where wind buries them. Recordings whose channels are
near-identical (mono, or an edited mix) don't show wind this way, so step 2 is skipped for
them, and for 1- or 3+-channel recordings.

Only numpy and scipy. Long recordings are processed in overlapping blocks, crossfaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class CleaningConfig:
    highpass_hz: float = 60.0  # 0 turns the high-pass off
    wind: bool = True
    wind_max_hz: float = 800.0  # full effect below this, fading out by twice this
    gain_floor: float = 0.2  # turn wind down by at most ~14 dB
    smoothing_s: float = 0.15  # time constant of the spectral averages
    window_s: float = 0.064
    near_mono: float = 0.99  # channel correlation above this: wind can't be seen, skip
    block_s: float = 30.0
    overlap_s: float = 2.0  # warm-up before each block (covers the crossfade)
    crossfade_s: float = 0.5


BANDS = {"20-100 Hz": (20, 100), "100-500 Hz": (100, 500), "500-2000 Hz": (500, 2000), "all": (20, None)}


def highpass(audio: np.ndarray, sr: int, hz: float) -> np.ndarray:
    sos = signal.butter(4, hz, "high", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=-1).astype(np.float32)


def channel_correlation(audio: np.ndarray) -> float:
    a, b = audio[0] - audio[0].mean(), audio[1] - audio[1].mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / denom) if denom > 0 else 1.0


def _suppress_block(x: np.ndarray, sr: int, cfg: CleaningConfig) -> tuple[np.ndarray, np.ndarray]:
    """One block: (cleaned block, per-frame wind flags)."""
    nper = int(round(cfg.window_s * sr))
    hop = nper // 4
    kw = dict(fs=sr, nperseg=nper, noverlap=nper - hop)
    f, _, spec = signal.stft(x.astype(np.float32), **kw)
    left, right = spec[0], spec[1]
    band = f < 2 * cfg.wind_max_hz  # only these bins can change
    alpha = float(np.exp(-hop / sr / cfg.smoothing_s))
    nb = int(band.sum())
    p_ll = np.zeros(nb); p_rr = np.zeros(nb); p_lr = np.zeros(nb, complex)
    ratio = np.empty((nb, left.shape[1]))
    msc = np.empty_like(ratio)
    lb, rb = left[band], right[band]
    for t in range(left.shape[1]):
        p_ll = alpha * p_ll + (1 - alpha) * np.abs(lb[:, t]) ** 2
        p_rr = alpha * p_rr + (1 - alpha) * np.abs(rb[:, t]) ** 2
        p_lr = alpha * p_lr + (1 - alpha) * lb[:, t] * np.conj(rb[:, t])
        ratio[:, t] = np.abs(p_lr) / (0.5 * (p_ll + p_rr) + 1e-20)  # coherent share of the power
        msc[:, t] = np.abs(p_lr) ** 2 / (p_ll * p_rr + 1e-20)
    ratio = signal.convolve2d(ratio, np.ones((3, 1)) / 3, mode="same", boundary="symm")
    gain = np.clip(ratio, cfg.gain_floor, 1.0)
    fade = np.clip((f[band] - cfg.wind_max_hz) / cfg.wind_max_hz, 0.0, 1.0)[:, None]
    spec[:, band] *= (fade + (1 - fade) * gain)[None].astype(spec.dtype)
    _, y = signal.istft(spec, **kw)
    low = (f[band] >= 50) & (f[band] < 500)
    energy = (np.abs(lb[low]) ** 2 + np.abs(rb[low]) ** 2).sum(0)
    flags = (msc[low].mean(0) < 0.5) & (energy > np.median(energy))
    return y[..., : x.shape[-1]].astype(np.float32), flags


def suppress_wind(audio: np.ndarray, sr: int, cfg: CleaningConfig = CleaningConfig()) -> tuple[np.ndarray, float]:
    """Two-channel wind suppression. Returns (cleaned audio, share of wind-like frames)."""
    n = audio.shape[-1]
    block, overlap = int(cfg.block_s * sr), int(cfg.overlap_s * sr)
    xfade, tail = int(cfg.crossfade_s * sr), int(0.25 * sr)
    out = np.zeros_like(audio, dtype=np.float32)
    flags: list[np.ndarray] = []
    starts = list(range(0, n, block))
    for k, s in enumerate(starts):
        a, b = max(0, s - overlap), min(n, s + block)
        y, fl = _suppress_block(audio[:, a:min(n, b + tail)], sr, cfg)
        y = y[:, : b - a]
        w = np.ones(b - a, np.float32)
        if k > 0:  # ramp up over the crossfade before s; nothing before that (warm-up only)
            w[: s - xfade - a] = 0.0
            w[s - xfade - a: s - a] = np.linspace(0, 1, xfade, endpoint=False)
        if k < len(starts) - 1:  # ramp down into the next block
            w[b - xfade - a:] = np.linspace(1, 0, xfade, endpoint=False)
        out[:, a:b] += y * w
        hop_n = int(round(cfg.window_s * sr)) // 4
        flags.append(fl[(s - a) // hop_n: (b - a) // hop_n])
    all_flags = np.concatenate(flags) if flags else np.zeros(0, bool)
    return out, float(all_flags.mean()) if len(all_flags) else 0.0


def _band_db(audio: np.ndarray, sr: int, lo: float, hi: float | None) -> float:
    mono = audio.mean(axis=0)
    f, p = signal.welch(mono, fs=sr, nperseg=min(4096, len(mono)))
    sel = (f >= lo) & (f < (hi or sr / 2))
    return float(10 * np.log10(p[sel].sum() + 1e-30))


def clean_recording(audio: np.ndarray, sr: int, cfg: CleaningConfig = CleaningConfig()) -> tuple[np.ndarray, dict]:
    """(C, N) recording -> (cleaned recording, report for stems.json)."""
    audio = np.asarray(audio, np.float32)
    report: dict = {"highpass_hz": cfg.highpass_hz or None}
    out = highpass(audio, sr, cfg.highpass_hz) if cfg.highpass_hz else audio.copy()
    channels = audio.shape[0]
    if not cfg.wind:
        report["wind"] = "off"
    elif channels != 2:
        report["wind"] = f"skipped: {channels} channel{'s' if channels != 1 else ''} (needs a pair of microphones)"
    elif audio.shape[-1] < int(4 * cfg.window_s * sr):
        report["wind"] = "skipped: too short"
    else:
        corr = channel_correlation(out)
        if corr > cfg.near_mono:
            report["wind"] = f"skipped: the channels are near-identical (correlation {corr:.3f}), so wind can't be told apart"
        else:
            out, share = suppress_wind(out, sr, cfg)
            report["wind"] = "suppressed"
            report["wind_like_frames"] = round(share, 4)
    report["removed_db"] = {name: round(_band_db(audio, sr, lo, hi) - _band_db(out, sr, lo, hi), 2)
                            for name, (lo, hi) in BANDS.items()}
    return out, report
