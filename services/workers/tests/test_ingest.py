import json

import numpy as np
import pytest

from memoryscapes_workers.stagekit import StageError
from memoryscapes_workers.visual import keyframes
from memoryscapes_workers.visual.video import fit_size, parse_iso6709, parse_probe

from .conftest import needs_cv2, needs_ffmpeg

PROBE_IPHONE = {
    "format": {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "12.5", "size": "1000",
        "tags": {
            "creation_time": "2026-06-02T01:31:00.000000Z",
            "com.apple.quicktime.location.ISO6709": "+37.4275-122.1697+030.000/",
            "com.apple.quicktime.creationdate": "2026-06-01T18:30:00-0700",
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
    meta = json.loads(ctx.storage.read_bytes(ctx.prefix("ingest") + "metadata.json"))

    assert summary["keyframes"] >= 20
    assert summary["geo"] == {"lat": 37.4275, "lon": -122.1697, "alt": 30.0}
    assert summary["captured_at"] == gt["captured_at"]
    # Original audio layout and rate are kept.
    assert meta["audio"]["sample_rate"] == 48000 and meta["audio"]["channels"] == 2
    assert ctx.storage.exists(ctx.prefix("ingest") + "audio.flac")
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
