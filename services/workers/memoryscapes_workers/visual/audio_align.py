"""Align a separately recorded audio file to the video's clock.

Cameras get timestamps from the video (``t`` in poses/cameras.json), and the audio stages
assume ``ingest/audio.wav`` is on that same clock. A separate recorder starts at some
other moment, so we find the offset from the video's own soundtrack: both microphones
heard the same scene.

Method: both tracks to mono at 8 kHz, then GCC-PHAT (cross-correlation with the
spectrum whitened, which tolerates the two microphones' different frequency
responses). The peak's lag is the offset; its height relative to the rest of the
correlation is the confidence. Device clocks also run at slightly different rates (tens
of ppm, i.e. tens of ms over 10 minutes), so the first and last thirds are aligned
separately and a line ``s = offset + rate * t`` (separate-recording time s for video
time t) is fitted through them; ``apply_alignment`` resamples along that line.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ALIGN_SR = 8000
# Confidence: the correlation summed over +-1 ms of lags (so clock drift smearing the
# peak over a few lags doesn't hide it), at its peak, over its robust noise level.
# Matching recordings score 50-100+ even with 100 ppm drift; unrelated audio about 6.
SMEAR_LAGS = 17  # 2 ms at ALIGN_SR
MIN_CONFIDENCE = 20.0
# The drift windows only search +-2 s around the known offset: far fewer lags compete.
DRIFT_MIN_CONFIDENCE = 12.0
DRIFT_WINDOW_S = 15.0
COARSE_WINDOW_S = 20.0


@dataclass
class Alignment:
    # Seconds into the separate recording at which the video starts. Negative: the
    # recorder started after the video.
    offset_s: float
    confidence: float
    rate: float = 1.0  # separate-recording seconds per video second
    drift_ms: float | None = None  # offset change over the video's length, if measured

    @property
    def reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    def to_dict(self) -> dict:
        d = asdict(self)
        digits = {"offset_s": 4, "confidence": 1, "rate": 8, "drift_ms": 2}
        return {k: (round(v, digits[k]) if isinstance(v, float) else v) for k, v in d.items()}


def _prep(x: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import resample_poly

    mono = x.mean(axis=0) if x.ndim == 2 else x
    g = np.gcd(int(sr), ALIGN_SR)
    y = resample_poly(mono.astype(np.float64), ALIGN_SR // g, int(sr) // g)
    return y - y.mean()


def _gcc_phat_lag(ref: np.ndarray, sig: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) such that sig[n + lag] ~ ref[n], and the peak's confidence.

    The peak is found on the correlation summed over SMEAR_LAGS neighbouring lags (robust
    to drift and to an inverted microphone's polarity), then refined on the raw one.
    """
    n = len(ref) + len(sig)
    nfft = 1 << (n - 1).bit_length()
    cross = np.fft.rfft(sig, nfft) * np.conj(np.fft.rfft(ref, nfft))
    cross /= np.abs(cross) + 1e-12
    cc = np.fft.irfft(cross, nfft)
    max_lag = min(max_lag, nfft // 2 - 1)
    vals = np.concatenate([cc[nfft - max_lag:], cc[: max_lag + 1]])  # lags -max_lag..max_lag
    summed = np.abs(np.convolve(vals, np.ones(SMEAR_LAGS), "same"))
    i = int(np.argmax(summed))
    noise = np.median(summed) * 1.4826 + 1e-12  # robust sigma
    half = SMEAR_LAGS
    lo = max(0, i - half)
    j = lo + int(np.argmax(np.abs(vals[lo : i + half + 1])))
    return j - max_lag, float(summed[i] / noise)


def find_offset(video_audio: np.ndarray, video_sr: int, separate: np.ndarray, separate_sr: int,
                max_offset_s: float = 300.0) -> Alignment:
    """Offset (and clock rate) of the separate recording relative to the video's track.

    The coarse search uses only the first COARSE_WINDOW_S of the video's audio against
    the whole recording: over a long clip, clock drift would smear the correlation peak.
    Short windows at both ends then measure the drift.
    """
    ref = _prep(video_audio, video_sr)
    sig = _prep(separate, separate_sr)
    head = ref[: int(COARSE_WINDOW_S * ALIGN_SR)]
    max_lag = int(min(max_offset_s * ALIGN_SR, len(head) + len(sig)))
    lag, conf = _gcc_phat_lag(head, sig, max_lag)
    result = Alignment(offset_s=lag / ALIGN_SR, confidence=conf)
    w = int(DRIFT_WINDOW_S * ALIGN_SR)
    if conf < MIN_CONFIDENCE or len(ref) < 3 * w:  # too short to tell drift from noise
        return result
    search = int(2 * ALIGN_SR)  # +-2 s around the coarse offset
    points = []  # (video time, offset) at the centre of a window at each end
    for start in (0, len(ref) - w):
        lo = max(0, start + lag - search)
        part = sig[lo : start + lag + w + search]
        if len(part) < w:
            return result
        sub_lag, sub_conf = _gcc_phat_lag(ref[start : start + w], part, len(part))
        if sub_conf < DRIFT_MIN_CONFIDENCE:
            return result
        points.append(((start + w / 2) / ALIGN_SR, (lo + sub_lag - start) / ALIGN_SR))
    (t1, o1), (t2, o2) = points
    slope = (o2 - o1) / (t2 - t1)
    result.offset_s = o1 - slope * t1
    result.rate = 1.0 + slope
    result.drift_ms = slope * len(ref) / ALIGN_SR * 1000.0
    return result


def apply_alignment(audio: np.ndarray, sr: int, a: Alignment, duration_s: float) -> np.ndarray:
    """(channels, samples) on the video clock: t=0 is the first frame, length duration_s.
    Shifts by whole samples, and resamples along the fitted line when the clocks drift
    by more than a couple of milliseconds over the clip."""
    if a.drift_ms is None or abs(a.drift_ms) < 2.0:
        return apply_offset(audio, sr, a.offset_s, duration_s)
    n = int(round(duration_s * sr))
    pos = (a.offset_s + a.rate * np.arange(n) / sr) * sr  # sample positions in `audio`
    inside = (pos >= 0) & (pos <= audio.shape[1] - 1)
    out = np.zeros((audio.shape[0], n), audio.dtype)
    idx = np.arange(audio.shape[1])
    for c in range(audio.shape[0]):
        out[c, inside] = np.interp(pos[inside], idx, audio[c])
    return out


def apply_offset(audio: np.ndarray, sr: int, offset_s: float, duration_s: float) -> np.ndarray:
    """(channels, samples) shifted so t=0 is the video start, cut or padded to duration_s."""
    shift = int(round(offset_s * sr))
    n = int(round(duration_s * sr))
    out = np.zeros((audio.shape[0], n), audio.dtype)
    src_start, dst_start = (shift, 0) if shift >= 0 else (0, -shift)
    count = max(0, min(audio.shape[1] - src_start, n - dst_start))
    out[:, dst_start : dst_start + count] = audio[:, src_start : src_start + count]
    return out
