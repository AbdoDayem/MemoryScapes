"""Ingest: demux audio and frames, read metadata, pick keyframes.

- ffmpeg demux; keep the original audio channel layout and sample rate.
- Choose 150-400 keyframes by sharpness (Laplacian variance) and parallax.
- Panoramas: split equirectangular frames into cubemap/perspective crops (not yet).
- Read GPS and capture time from EXIF or container metadata for the globe pin.

Reads  {memory_id}/upload/        the uploaded video (+ optional separate audio file)
Writes {memory_id}/ingest/
    frames/frame_00000.jpg ...    upright keyframes, long side <= output_max_side
    audio.wav                     primary audio (separate recording if uploaded,
                                  otherwise the video's track), original rate/channels,
                                  on the video's clock: t=0 is the first video frame
    video_audio.wav               the video's own track, when a separate file was given

A separate recording is aligned to the video by cross-correlating it with the video's
own soundtrack (visual/audio_align.py), then trimmed or padded to the video's length.
    meta.json                     media info, geo, capture time, keyframe timeline, and
                                  the "mic" block the audio stages read

The audio stages' input contract is in services/workers/README.md.
"""

from __future__ import annotations

import math
from pathlib import Path

from pydantic import BaseModel, Field

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.stagekit import UPLOAD_PREFIX, StageError, StageIO, run_stage, write_json

STAGE = "ingest"
VERSION = "3"  # 2: audio-stage contract outputs (audio.wav, meta.json); 3: separate audio aligned


class IngestConfig(BaseModel):
    max_keyframes: int = 400
    min_keyframes: int = 150  # short clips can yield fewer (one per candidate at most)
    # Desired camera motion between keyframes: median optical flow / image diagonal.
    target_step: float = Field(0.035, gt=0)
    # Candidates decoded per keyframe slot, before selection.
    oversample: float = 3.0
    analysis_max_side: int = 640
    output_max_side: int = 1600
    # Reject a keyframe whose sharpness is below this x the local (~1 s) median.
    blur_reject: float = 0.7
    # Total motion (in image diagonals) below which we fail: the camera barely moved.
    min_total_motion: float = 0.25
    jpeg_quality: int = 95
    max_duration_s: float = 600.0


def run(ctx: StageContext) -> dict:
    cfg = IngestConfig(**ctx.stage_config(STAGE))
    return run_stage(ctx, STAGE, version=VERSION, config=cfg, upstream=[UPLOAD_PREFIX],
                     body=lambda io: _ingest(io, cfg))


def mic_info(audio_meta: dict | None) -> dict:
    """The ``mic`` block for the audio stages. Only "stereo" enables DOA localization; the
    mic spacing is unknown from a phone video, so localize falls back to its default."""
    channels = int(audio_meta.get("channels") or 0) if audio_meta else 0
    layout = (audio_meta or {}).get("channel_layout") or ""
    if channels <= 1:
        kind = "mono"
    elif channels == 2:
        kind = "stereo"
    elif channels == 4 and "ambisonic" in layout.lower():
        kind = "foa"
    else:
        kind = "multichannel"
    return {"channels": channels, "layout": kind, "spacing_m": None}


def _align_separate(raw: Path, video_track: Path | None, dest: Path, duration_s: float,
                    warnings: list[str]) -> dict:
    """Write ``dest`` = the separate recording on the video's clock; return what was done."""
    import shutil

    import soundfile as sf

    from memoryscapes_workers.visual.audio_align import apply_alignment, find_offset

    if video_track is None:
        shutil.copyfile(raw, dest)
        warnings.append("The video has no sound to align the separate recording with; "
                        "assuming both started together.")
        return {"applied": False, "reason": "video has no audio track"}
    sep, sep_sr = sf.read(str(raw), dtype="float32", always_2d=True)
    vid, vid_sr = sf.read(str(video_track), dtype="float32", always_2d=True)
    a = find_offset(vid.T, vid_sr, sep.T, sep_sr)
    sep_len = len(sep) / sep_sr
    overlap = min(sep_len, a.offset_s + duration_s) - max(0.0, a.offset_s)
    if not a.reliable or overlap < min(5.0, 0.5 * duration_s):
        shutil.copyfile(raw, dest)
        warnings.append(f"Couldn't match the separate recording to the video's sound "
                        f"(confidence {a.confidence:.0f}); assuming both started together.")
        return {"applied": False, "reason": "no reliable match", **a.to_dict()}
    aligned = apply_alignment(sep.T, sep_sr, a, duration_s)
    sf.write(str(dest), aligned.T, sep_sr, subtype="PCM_24")
    if overlap < duration_s - 0.5:
        warnings.append(f"The separate recording covers only {overlap:.0f} s of the "
                        f"{duration_s:.0f} s video; the rest is silent.")
    return {"applied": True, "method": "gcc-phat", **a.to_dict(), "overlap_s": round(overlap, 2)}


def _pick_inputs(upload: Path) -> tuple[Path, Path | None]:
    from memoryscapes_workers.visual.video import AUDIO_EXTS, VIDEO_EXTS

    files = sorted(p for p in upload.rglob("*") if p.is_file())
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    audios = [p for p in files if p.suffix.lower() in AUDIO_EXTS]
    if not videos:
        names = ", ".join(p.name for p in files) or "nothing"
        raise StageError(STAGE, f"No video found in upload (got {names}).",
                         "Supported: " + ", ".join(sorted(VIDEO_EXTS)))
    if len(videos) > 1:
        # Largest file wins; multi-clip memories are not supported yet.
        videos.sort(key=lambda p: p.stat().st_size, reverse=True)
    return videos[0], (audios[0] if audios else None)


def _ingest(io: StageIO, cfg: IngestConfig) -> dict:
    import cv2

    from memoryscapes_workers.visual import keyframes, video

    if io.ctx.input_kind != "video":
        raise StageError(STAGE, f"input_kind={io.ctx.input_kind!r} is not supported yet.",
                         "Panorama input is planned for phase 5; upload a video.")
    video.require_ffmpeg()
    src, separate_audio = _pick_inputs(io.input_dir(UPLOAD_PREFIX))
    info = video.probe(src)
    if not info.width or not info.fps:
        raise StageError(STAGE, f"{src.name} has no decodable video stream.")
    if info.duration_s <= 0.5:
        raise StageError(STAGE, "Video is too short.", "Record at least ~10 seconds.")
    if info.duration_s > cfg.max_duration_s:
        raise StageError(STAGE, f"Video is {info.duration_s:.0f}s; the limit is "
                         f"{cfg.max_duration_s:.0f}s.", "Trim the clip and upload again.")
    warnings: list[str] = []

    # ---- audio
    out = io.out_dir
    audio_meta = None
    if separate_audio is not None:
        a_info = video.probe(separate_audio)
        raw = video.demux_audio(separate_audio, io.work_dir / "separate_raw",
                                a_info.audio and a_info.audio.channels)
        audio_meta = {"file": "audio.wav", "source": "separate_file", "original": separate_audio.name,
                      **(vars(a_info.audio) if a_info.audio else {})}
        video_track = video.demux_audio(src, out / "video_audio", info.audio.channels) if info.audio else None
        audio_meta["alignment"] = _align_separate(raw, video_track, out / "audio.wav",
                                                  info.duration_s, warnings)
    elif info.audio:
        path = video.demux_audio(src, out / "audio", info.audio.channels)
        audio_meta = {"file": path.name, "source": "video_track", **vars(info.audio)}
    else:
        warnings.append("No audio found; the memory will be silent.")
    io.progress(0.1, "audio extracted")

    # ---- candidate analysis (low-res grayscale pass)
    disp_w, disp_h = info.display_size
    sample_fps = min(info.fps, max(2.0, cfg.max_keyframes * cfg.oversample / info.duration_s))
    a_size = video.fit_size(disp_w, disp_h, cfg.analysis_max_side)
    stats = keyframes.analyze(video.iter_frames(src, a_size, sample_fps, gray=True))
    n_candidates = len(stats.sharpness)
    if n_candidates < 3:
        raise StageError(STAGE, "Could not decode enough frames from the video.")
    io.progress(0.5, f"analyzed {n_candidates} candidate frames")

    sel = keyframes.select(
        stats,
        target_step=cfg.target_step,
        min_keyframes=cfg.min_keyframes,
        max_keyframes=cfg.max_keyframes,
        blur_reject=cfg.blur_reject,
        sharpness_window=max(3, int(round(sample_fps))),
    )
    warnings += sel.warnings
    if sel.total_motion < cfg.min_total_motion:
        raise StageError(
            STAGE,
            f"The camera barely moved (total motion {sel.total_motion:.2f} image diagonals).",
            "Walk through the space or orbit the subject slowly; rotating in place is not enough.",
        )
    if len(sel.indices) < 20:
        raise StageError(STAGE, f"Only {len(sel.indices)} usable keyframes.",
                         "Record a longer, steadier clip with more camera movement.")

    # ---- write selected frames at output resolution (second decode pass)
    o_size = video.fit_size(disp_w, disp_h, cfg.output_max_side)
    wanted = {idx: k for k, idx in enumerate(sel.indices)}
    frames_dir = out / "frames"
    frames_dir.mkdir()
    frames_meta: list[dict] = []
    params = [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]
    last = max(wanted)
    for i, rgb in enumerate(video.iter_frames(src, o_size, sample_fps)):
        if i in wanted:
            name = f"frame_{wanted[i]:05d}.jpg"
            video.imwrite_rgb(frames_dir / name, rgb, params)
            frames_meta.append({
                "file": f"frames/{name}",
                "t": round(i / sample_fps, 4),
                "candidate": i,
                "sharpness": round(float(sel.rel_sharpness[i]), 3),
            })
        if i >= last:
            break
    if len(frames_meta) != len(sel.indices):
        raise StageError(STAGE, "Frame extraction pass returned fewer frames than analysis.")
    io.progress(0.95, f"wrote {len(frames_meta)} keyframes")

    meta = {
        "source": {"file": src.name, "container": info.container,
                   "duration_s": info.duration_s, "size_bytes": info.size_bytes},
        "video": {k: v for k, v in info.to_dict().items()
                  if k in ("codec", "width", "height", "display_width", "display_height",
                           "rotation", "fps", "num_frames")},
        "audio": audio_meta,
        "geo": dict(zip(("lat", "lon", "alt"), info.geo, strict=True)) if info.geo else None,
        "captured_at": info.captured_at,
        "keyframes": {
            "count": len(frames_meta),
            "width": o_size[0],
            "height": o_size[1],
            "sample_fps": round(sample_fps, 4),
            "candidates": n_candidates,
            "total_motion": round(sel.total_motion, 4),
            "frames": frames_meta,
        },
        "warnings": warnings,
    }
    meta["mic"] = mic_info(audio_meta)
    write_json(out / "meta.json", meta)
    return {
        "keyframes": len(frames_meta),
        "candidates": n_candidates,
        "has_audio": audio_meta is not None,
        "geo": meta["geo"],
        "captured_at": info.captured_at,
        "warnings": warnings,
        "duration_s": round(info.duration_s, 2),
        "frame_size": list(o_size),
        "_motion_per_keyframe": round(sel.total_motion / max(1, len(frames_meta)), 4),
    }
