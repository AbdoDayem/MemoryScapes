"""The runner command the API starts (app/pipeline.py)."""

from pathlib import Path

from app import pipeline
from app.settings import get_settings


def _max_seconds(cmd: list[str]) -> float:
    return float(cmd[cmd.index("--max-seconds") + 1])


def test_the_runner_accepts_videos_as_long_as_the_upload_page_allows():
    """The upload page allows 2 minutes. Without --max-seconds the runner used its own 60 s
    default, so every 1-2 minute upload failed at ingest."""
    cmd = pipeline.build_command("mem_x", Path("/data"))
    assert cmd[-4:-2] == ["--memory", "mem_x"]
    assert _max_seconds(cmd) == 120


def test_the_limit_is_a_setting(monkeypatch):
    monkeypatch.setenv("MAX_VIDEO_SECONDS", "90")
    get_settings.cache_clear()
    try:
        assert _max_seconds(pipeline.build_command("mem_x", Path("/data"))) == 90
    finally:
        get_settings.cache_clear()
