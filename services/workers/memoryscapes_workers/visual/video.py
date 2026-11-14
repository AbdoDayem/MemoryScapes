"""ffprobe / ffmpeg wrappers: probing, audio demux and frame decoding."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp"}
AUDIO_EXTS = {".wav", ".flac", ".m4a", ".mp3", ".aac", ".ogg", ".opus", ".caf", ".aiff", ".aif"}

_ISO6709 = re.compile(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?")


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH")


@dataclass
class AudioStreamInfo:
    codec: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None


@dataclass
class MediaInfo:
    path: str
    container: str | None
    duration_s: float
    size_bytes: int
    # Video (None for audio-only files)
    codec: str | None = None
    width: int | None = None  # coded size
    height: int | None = None
    rotation: int = 0  # degrees, display rotation
    fps: float | None = None
    num_frames: int | None = None
    audio: AudioStreamInfo | None = None
    geo: tuple[float, float, float | None] | None = None  # lat, lon, alt
    captured_at: str | None = None  # ISO 8601 UTC

    @property
    def display_size(self) -> tuple[int, int]:
        assert self.width and self.height
        if abs(self.rotation) % 180 == 90:
            return self.height, self.width
        return self.width, self.height

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.width:
            d["display_width"], d["display_height"] = self.display_size
        return d


def _ratio(s: str | None) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    if "/" in s:
        num, den = s.split("/", 1)
        return float(num) / float(den) if float(den) else None
    return float(s)


def parse_iso6709(value: str) -> tuple[float, float, float | None] | None:
    m = _ISO6709.match(value.strip())
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon, float(m.group(3)) if m.group(3) else None


def _parse_time(value: str) -> str | None:
    value = value.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.fromisoformat(value) if fmt is None else datetime.strptime(value, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def _rotation(stream: dict) -> int:
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(round(float(sd["rotation"])))
    tag = (stream.get("tags") or {}).get("rotate")
    return int(tag) if tag else 0


def parse_probe(data: dict, path: str = "") -> MediaInfo:
    """Build MediaInfo from ``ffprobe -show_format -show_streams -of json`` output."""
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next(
        (s for s in streams if s.get("codec_type") == "video"
         and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    tags: dict[str, str] = {}
    for src in [fmt] + ([video] if video else []) + ([audio] if audio else []):
        for k, v in (src.get("tags") or {}).items():
            tags.setdefault(k.lower(), v)

    geo = None
    for key in ("com.apple.quicktime.location.iso6709", "location", "location-eng"):
        if key in tags and (geo := parse_iso6709(tags[key])):
            break
    captured = None
    # Apple's creationdate is the real capture time (with offset); creation_time can be
    # the export time.
    for key in ("com.apple.quicktime.creationdate", "creation_time", "date"):
        if key in tags and (captured := _parse_time(tags[key])):
            break

    duration = float(fmt.get("duration") or (video or audio or {}).get("duration") or 0.0)
    info = MediaInfo(
        path=path,
        container=fmt.get("format_name"),
        duration_s=duration,
        size_bytes=int(fmt.get("size") or 0),
        geo=geo,
        captured_at=captured,
    )
    if video:
        info.codec = video.get("codec_name")
        info.width, info.height = int(video["width"]), int(video["height"])
        info.rotation = _rotation(video)
        info.fps = _ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))
        nb = video.get("nb_frames")
        info.num_frames = int(nb) if nb and nb.isdigit() else None
    if audio:
        info.audio = AudioStreamInfo(
            codec=audio.get("codec_name"),
            sample_rate=int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            channels=int(audio["channels"]) if audio.get("channels") else None,
            channel_layout=audio.get("channel_layout"),
        )
    return info


def probe(path: Path) -> MediaInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise ValueError(f"ffprobe could not read {path.name}: {out.stderr.strip()[:300]}")
    return parse_probe(json.loads(out.stdout), path=path.name)


def demux_audio(src: Path, dest_stem: Path, channels: int | None) -> Path:
    """Extract the first audio stream losslessly, keeping sample rate and channel layout.

    FLAC up to 8 channels, otherwise 32-bit float WAV. Returns the written path.
    """
    if channels is not None and channels > 8:
        dest, codec = dest_stem.with_suffix(".wav"), ["-c:a", "pcm_f32le"]
    else:
        dest, codec = dest_stem.with_suffix(".flac"), ["-c:a", "flac"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-map", "0:a:0", "-vn", *codec, str(dest)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"audio demux failed: {res.stderr.strip()[:300]}")
    return dest


def fit_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Scale to fit ``max_side`` (never upscale); dimensions even for yuv codecs."""
    scale = min(1.0, max_side / max(width, height))
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return w, h


def iter_frames(
    src: Path, size: tuple[int, int], sample_fps: float, gray: bool = False
) -> Iterator[np.ndarray]:
    """Decode frames at a constant ``sample_fps``, rotated upright and resized to ``size``.

    Frame i is at t = i / sample_fps from the start of the stream. ffmpeg applies the
    display rotation (autorotate) before the scale filter.
    """
    w, h = size
    pix_fmt, channels = ("gray", 1) if gray else ("rgb24", 3)
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(src), "-map", "0:v:0", "-an",
        "-vf", f"fps={sample_fps:.6f},scale={w}:{h}:flags=area",
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    frame_bytes = w * h * channels
    finished = False
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                finished = True
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, channels)
            yield frame[..., 0] if gray else frame
    finally:
        if not finished:  # caller stopped early (or raised): don't report the broken pipe
            proc.kill()
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg decode failed ({code}): {stderr.strip()[:300]}")
