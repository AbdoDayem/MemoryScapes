"""Signal helpers shared by separation and localization (numpy/scipy only)."""

from __future__ import annotations

from collections.abc import Callable
from math import gcd

import numpy as np
from scipy import signal


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return x.mean(axis=0)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g, axis=-1).astype(np.float32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad the last axis to n samples."""
    if x.shape[-1] >= n:
        return x[..., :n]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
    return np.pad(x, pad)


def stft(x: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    _, _, z = signal.stft(x, nperseg=nfft, noverlap=nfft - hop, axis=-1)
    return z


def istft(z: np.ndarray, nfft: int, hop: int, length: int) -> np.ndarray:
    _, x = signal.istft(z, nperseg=nfft, noverlap=nfft - hop)
    return fit_length(x, length).astype(np.float32)


def consistent_masks(
    mix_mag: np.ndarray, stem_mags: np.ndarray, power: float = 2.0, eps: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Turn independent stem estimates into masks that partition the mixture.

    Query-based separators run once per label, so their estimates can overlap or exceed
    the mixture. Normalizing Wiener-style gives masks with sum(stem masks) + residual = 1
    in every time-frequency bin, so stems + ambient reconstructs the recording exactly.

    mix_mag: (F, T); stem_mags: (K, F, T). Returns (stem_masks (K, F, T), residual (F, T)).
    """
    p = stem_mags**power
    denom = np.maximum(mix_mag**power, p.sum(axis=0)) + eps
    masks = p / denom
    residual = np.clip(1.0 - masks.sum(axis=0), 0.0, 1.0)
    return masks, residual


def apply_masks(
    mix: np.ndarray, masks: np.ndarray, residual: np.ndarray, nfft: int, hop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply single-channel masks to every channel of the mixture.

    Using the same real-valued mask on each channel keeps inter-channel time and level
    differences intact, which is what multichannel DOA needs later.

    mix: (C, N). Returns (stems (K, C, N), ambient (C, N)).
    """
    n = mix.shape[-1]
    z = stft(mix, nfft, hop)  # (C, F, T)
    stems = np.stack([istft(z * m[None], nfft, hop, n) for m in masks]) if len(masks) else np.zeros((0, *mix.shape), np.float32)
    ambient = istft(z * residual[None], nfft, hop, n)
    return stems, ambient


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-10))


def activity_envelope(
    mono: np.ndarray, sr: int, hop_s: float = 0.25, win_s: float = 0.5, floor_db: float = -40.0
) -> tuple[np.ndarray, np.ndarray]:
    """RMS envelope normalized to 0..1 (1 = loudest window, 0 = floor_db below it)."""
    hop = max(1, int(hop_s * sr))
    win = max(hop, int(win_s * sr))
    starts = np.arange(0, max(1, len(mono) - win + 1), hop)
    rms = np.array([np.sqrt(np.mean(mono[s : s + win] ** 2) + 1e-20) for s in starts])
    times = (starts + win / 2) / sr
    db = 20 * np.log10(rms / (rms.max() + 1e-20) + 1e-20)
    return times, np.clip(1.0 - db / floor_db, 0.0, 1.0)


def process_in_chunks(
    x: np.ndarray, chunk: int, overlap: int, fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Run fn on overlapping chunks of a mono signal and crossfade them back together.
    Keeps model memory bounded on long recordings."""
    n = len(x)
    if n <= chunk:
        return fit_length(fn(x), n)
    out = np.zeros(n, np.float32)
    weight = np.zeros(n, np.float32)
    step = chunk - overlap
    fade = np.ones(chunk, np.float32)
    if overlap:
        # Strictly positive ramps, so weighted overlap-add normalizes cleanly at the edges.
        ramp = np.linspace(0, 1, overlap + 2, dtype=np.float32)[1:-1]
        fade[:overlap], fade[-overlap:] = ramp, ramp[::-1]
    start = 0
    while True:
        seg = x[start : start + chunk]
        w = fade[: len(seg)]
        out[start : start + len(seg)] += fit_length(fn(seg), len(seg)) * w
        weight[start : start + len(seg)] += w
        if start + chunk >= n:
            break
        start += step
    return out / weight
