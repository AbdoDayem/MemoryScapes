import json

import numpy as np
import pytest

from memoryscapes_workers.stagekit import StageError
from memoryscapes_workers.visual import keyframes
from memoryscapes_workers.visual.video import fit_size, parse_iso6709, parse_probe

from markers import needs_cv2, needs_ffmpeg

PROBE_IPHONE = {
    "format": {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "12.5", "size": "1000",
        "tags": {
            "creation_time": "2026-06-02T01:31:00.000000Z",
            "com.apple.quicktime.location.ISO6709": "+37.4275-122.1697+030.000/",
            "com.apple.quicktime.creationdate": "2026-06-01T18:30:00-0700",
            "com.apple.quicktime.camera.focal_length.35mm_equivalent": "26",
        },
    },
    "streams": [
        {"codec_type": "video", "codec_name": "hevc", "width": 1920, "height": 1080,
         "avg_frame_rate": "30000/1001", "nb_frames": "375",
         "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 4,
         "channel_layout": "4.0"},
    ],
}


def test_parse_probe_iphone_portrait():
    info = parse_probe(PROBE_IPHONE, "clip.mov")
    assert info.display_size == (1080, 1920)  # portrait after rotation
    assert info.fps == pytest.approx(29.97, abs=0.01)
    assert info.geo == (37.4275, -122.1697, 30.0)
    # Apple creationdate (true capture time, with offset) wins over creation_time.
    assert info.captured_at == "2026-06-02T01:30:00Z"
    assert info.audio.channels == 4 and info.audio.sample_rate == 48000
    assert info.focal_35mm == 26.0
    assert parse_probe({"format": {"duration": "1"}, "streams": []}, "x.mp4").focal_35mm is None


def test_parse_iso6709_variants():
    assert parse_iso6709("+37.4275-122.1697/") == (37.4275, -122.1697, None)
    assert parse_iso6709("-33.8688+151.2093+012.5/")[2] == 12.5
    assert parse_iso6709("garbage") is None
    assert parse_iso6709("+95.0+10.0/") is None


def test_fit_size_even_and_no_upscale():
    assert fit_size(1920, 1080, 1600) == (1600, 900)
    assert fit_size(640, 360, 1600) == (640, 360)
    w, h = fit_size(1080, 1920, 1599)
    assert w % 2 == 0 and h % 2 == 0 and max(w, h) <= 1600


def _stats(n=200, blur=(), pause=(80, 120), step=0.01):
    sharp = np.full(n, 100.0) + np.random.default_rng(0).normal(0, 2, n)
    sharp[list(blur)] = 30.0
    motion = np.full(n, step)
    motion[pause[0]:pause[1]] = 0.0
    motion[0] = 0
    return keyframes.FrameStats(sharp, motion, np.full(n, 200))


def test_select_skips_pause_and_blur():
    blur = [10, 11, 50, 51, 52, 150]
    sel = keyframes.select(_stats(blur=blur), target_step=0.02, min_keyframes=10,
                           max_keyframes=400, blur_reject=0.7, sharpness_window=15)
    assert not set(sel.indices) & set(blur)
    in_pause = [i for i in sel.indices if 81 <= i < 120]
    assert len(in_pause) <= 1
    # Evenly spaced in motion: ~2 frames per bin, so neighbours are <= 3 apart. A bin
    # made only of blurry frames is dropped, which leaves a wider (<= 5) gap there.
    moving = [i for i in sel.indices if i < 80]
    for a, b in zip(moving, moving[1:]):
        spans_blur = any(a < x < b for x in blur)
        assert b - a <= (5 if spans_blur else 3), (a, b)


def test_select_warns_on_fast_motion():
    st = _stats(pause=(0, 0))
    st.motion[100:110] = 0.5  # sudden whip pan
    sel = keyframes.select(st, target_step=0.02, min_keyframes=50, max_keyframes=400,
                           blur_reject=0.7, sharpness_window=15)
    assert any("too fast" in w for w in sel.warnings)


@needs_ffmpeg
@needs_cv2
def test_ingest_stage_on_synthetic_video(ctx, small_video):
    from memoryscapes_workers.stages import ingest

    video, gt = small_video
    ctx.storage.upload(video, f"{ctx.memory_id}/upload/{video.name}")
    summary = ingest.run(ctx)
    meta = json.loads(ctx.storage.read_bytes(ctx.prefix("ingest") + "meta.json"))

    assert summary["keyframes"] >= 20
    assert summary["geo"] == {"lat": 37.4275, "lon": -122.1697, "alt": 30.0}
    assert summary["captured_at"] == gt["captured_at"]
    # Original audio layout and rate are kept.
    assert meta["audio"]["sample_rate"] == 48000 and meta["audio"]["channels"] == 2
    assert ctx.storage.exists(ctx.prefix("ingest") + "audio.wav")
    # The audio stages' contract (services/workers/README.md).
    assert meta["mic"] == {"channels": 2, "layout": "stereo", "spacing_m": None}
    from memoryscapes_workers.models import MicInfo
    from memoryscapes_workers.storage import read_wav

    assert MicInfo.from_json(meta["mic"]).layout == "stereo"
    audio, sr = read_wav(ctx.storage, ctx.key("ingest", "audio.wav"))
    assert sr == 48000 and audio.shape[0] == 2 and audio.shape[1] > 3 * sr
    # Blurred frames are skipped; timestamps line up with the source frames.
    picked = {f["candidate"] for f in meta["keyframes"]["frames"]}
    assert not picked & set(gt["blurred_frames"])
    frames = ctx.storage.list(ctx.prefix("ingest") + "frames/")
    assert len(frames) == summary["keyframes"]


@needs_ffmpeg
@needs_cv2
def test_ingest_fails_early_on_static_camera(ctx, tmp_path):
    import subprocess

    video = tmp_path / "static.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=3",
                    "-vf", "boxblur=0", "-pix_fmt", "yuv420p", "-y", str(video)], check=True)
    # testsrc2 animates in place: no camera motion -> fail with a capture hint.
    ctx.storage.upload(video, f"{ctx.memory_id}/upload/static.mp4")
    from memoryscapes_workers.stages import ingest

    with pytest.raises(StageError) as err:
        ingest.run(ctx)
    assert err.value.hint


@needs_cv2
def test_image_io_handles_non_ascii_paths(tmp_path):
    """cv2.imread/imwrite can't open non-ASCII paths on Windows; our helpers can."""
    from memoryscapes_workers.visual.video import imread_rgb, imwrite_rgb

    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[..., 0], rgb[2:4, 3:5] = 200, (10, 220, 30)
    for name in ("frame.png", "frame.jpg"):
        path = tmp_path / "Fotos Ünïcødé 写真" / name
        path.parent.mkdir(exist_ok=True)
        imwrite_rgb(path, rgb)
        back = imread_rgb(path)
        assert back.shape == rgb.shape
        err = np.abs(back.astype(int) - rgb)
        if name.endswith(".png"):
            assert err.max() == 0
        else:  # lossy, and chroma subsampling smears the small block
            assert err.mean() < 20


@needs_ffmpeg
@needs_cv2
def test_separate_recording_is_aligned_to_the_video(ctx, small_video, tmp_path):
    """A recorder started 1.3 s before the camera, at 44.1 kHz with its own noise."""
    import subprocess

    import soundfile as sf
    from scipy import signal

    from memoryscapes_workers.storage import read_wav
    from memoryscapes_workers.visual.audio_align import find_offset

    video, _ = small_video
    track = tmp_path / "track.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-y", str(track)], check=True)
    audio, sr = sf.read(str(track), dtype="float32", always_2d=True)
    rng = np.random.default_rng(0)
    head = 0.05 * rng.standard_normal((int(1.3 * sr), audio.shape[1])).astype(np.float32)
    rec = signal.resample_poly(np.concatenate([head, audio]), 441, 480, axis=0)
    rec += 0.01 * rng.standard_normal(rec.shape)
    sep = tmp_path / "recorder.wav"
    sf.write(str(sep), rec.astype(np.float32), 44100, subtype="PCM_16")

    ctx.storage.upload(video, f"{ctx.memory_id}/upload/{video.name}")
    ctx.storage.upload(sep, f"{ctx.memory_id}/upload/recorder.wav")
    from memoryscapes_workers.stages import ingest

    ingest.run(ctx)
    meta = json.loads(ctx.storage.read_bytes(ctx.key("ingest", "meta.json")))
    al = meta["audio"]["alignment"]
    assert al["applied"] and abs(al["offset_s"] - 1.3) < 0.002, al
    assert not any("align" in w or "started together" in w for w in meta["warnings"])
    # audio.wav is now on the video's clock and as long as the video.
    aligned, a_sr = read_wav(ctx.storage, ctx.key("ingest", "audio.wav"))
    vid, v_sr = read_wav(ctx.storage, ctx.key("ingest", "video_audio.wav"))
    assert a_sr == 44100 and abs(aligned.shape[1] / a_sr - vid.shape[1] / v_sr) < 0.05
    assert abs(find_offset(vid, v_sr, aligned, a_sr).offset_s) < 0.002
