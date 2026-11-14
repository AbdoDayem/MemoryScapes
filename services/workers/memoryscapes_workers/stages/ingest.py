"""Ingest: demux audio and frames, read metadata, pick keyframes.

Inputs (written by the upload):
    upload/meta.json      {"video": "<file name>", "audio": "<file name>" | null}
    upload/<files>        the uploaded video, and optionally a separate audio recording

Outputs:
    ingest/audio.wav      original channels and sample rate (from the separate audio file if given)
    ingest/frames/<id>.jpg  keyframes, longest side <= max_side
    ingest/frames.json    [{"frame_id", "t", "key", "sharpness"}], t on the audio clock
    ingest/meta.json      {"mic": {...}, "source": {...}, "location"?, "captured_at"?}

Frames are sampled evenly at a rate that gives about `target_frames`, then the blurriest
`drop_blurry` fraction is dropped (motion blur hurts SfM and detection alike).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.media import MediaError, find_tool, location_from_tags, probe, run as ff
from memoryscapes_workers.storage import read_json, write_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestConfig:
    max_duration_s: float = 60.0
    target_frames: int = 60
    min_fps: float = 0.5
    max_fps: float = 4.0
    max_side: int = 960
    drop_blurry: float = 0.15


def _sharpness(path: Path) -> float:
    """Variance of a Laplacian on a downscaled grey image."""
    from PIL import Image

    img = Image.open(path).convert("L")
    img.thumbnail((320, 320))
    a = np.asarray(img, dtype=np.float32)
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())


def _local(ctx: StageContext, key: str, tmp: Path) -> Path:
    """A local path for a stored object (storage may be S3)."""
    root = getattr(ctx.storage, "root", None)
    if root is not None and (Path(root) / key).exists():
        return Path(root) / key
    path = tmp / Path(key).name
    path.write_bytes(ctx.storage.read_bytes(key))
    return path


def run(ctx: StageContext, cfg: IngestConfig = IngestConfig()) -> dict:
    storage = ctx.storage
    upload = read_json(storage, ctx.key("upload", "meta.json"))
    if not upload.get("video"):
        raise MediaError("upload/meta.json names no video")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video = _local(ctx, ctx.key("upload", upload["video"]), tmp)
        info = probe(video)
        streams = info.get("streams", [])
        vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
        if vstream is None:
            raise MediaError(f"{upload['video']} has no video stream")
        duration = float(info.get("format", {}).get("duration") or vstream.get("duration") or 0)
        use_s = min(duration, cfg.max_duration_s) if duration else cfg.max_duration_s
        tags = {**info.get("format", {}).get("tags", {}), **vstream.get("tags", {})}

        # ---------- audio
        audio_src = _local(ctx, ctx.key("upload", upload["audio"]), tmp) if upload.get("audio") else video
        ainfo = probe(audio_src) if audio_src != video else info
        astream = next((s for s in ainfo.get("streams", []) if s.get("codec_type") == "audio"), None)
        if astream is None:
            raise MediaError("no audio track: upload a video with sound, or a separate audio file")
        channels = int(astream.get("channels", 1))
        wav = tmp / "audio.wav"
        ff([find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(audio_src), "-t", f"{use_s:.3f}", "-vn", "-c:a", "pcm_s16le", str(wav)])
        storage.write_bytes(ctx.key("ingest", "audio.wav"), wav.read_bytes(), "audio/wav")
        log.info("audio: %d ch, %s Hz, %.1f s", channels, astream.get("sample_rate"), use_s)

        # ---------- frames
        fps = float(np.clip(cfg.target_frames / max(use_s, 0.1), cfg.min_fps, cfg.max_fps))
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        scale = f"scale='if(gt(iw,ih),min({cfg.max_side},iw),-2)':'if(gt(iw,ih),-2,min({cfg.max_side},ih))'"
        ff([find_tool("ffmpeg"), "-y", "-v", "error", "-i", str(video), "-t", f"{use_s:.3f}", "-vf", f"fps={fps:.4f},{scale}", "-q:v", "3", str(frames_dir / "%05d.jpg")])
        paths = sorted(frames_dir.glob("*.jpg"))
        if len(paths) < 3:
            raise MediaError(f"only {len(paths)} frames extracted; the video may be too short")
        frames = [{"frame_id": f"f{i:04d}", "t": round(i / fps, 4), "path": p, "sharpness": _sharpness(p)} for i, p in enumerate(paths)]
        if cfg.drop_blurry and len(frames) > 12:
            cutoff = np.quantile([f["sharpness"] for f in frames], cfg.drop_blurry)
            kept = [f for f in frames if f["sharpness"] >= cutoff]
            log.info("frames: kept %d of %d after dropping the blurriest", len(kept), len(frames))
            frames = kept
        index = []
        for f in frames:
            key = ctx.key("ingest", f"frames/{f['frame_id']}.jpg")
            storage.write_bytes(key, f["path"].read_bytes(), "image/jpeg")
            index.append({"frame_id": f["frame_id"], "t": f["t"], "key": key, "sharpness": round(f["sharpness"], 2)})
        write_json(storage, ctx.key("ingest", "frames.json"), index)

    layout = {1: "mono", 2: "stereo"}.get(channels, "multichannel")
    meta = {
        "mic": {"channels": channels, "layout": layout},  # spacing unknown: localize uses its default
        "source": {
            "video": upload["video"],
            "audio": upload.get("audio"),
            "duration_s": duration,
            "used_s": use_s,
            "width": vstream.get("width"),
            "height": vstream.get("height"),
            "video_codec": vstream.get("codec_name"),
            "audio_codec": astream.get("codec_name"),
            "sample_rate": int(astream.get("sample_rate", 0)),
            "frame_rate_sampled": round(fps, 3),
        },
    }
    loc = location_from_tags(tags)
    if loc:
        meta["location"] = loc
    if tags.get("creation_time"):
        meta["captured_at"] = tags["creation_time"]
    if upload.get("audio"):
        meta["source"]["note"] = "separate audio file: assumed to start with the video"
    write_json(storage, ctx.key("ingest", "meta.json"), meta)
    return meta
