"""ffmpeg / ffprobe helpers for the ingest stage."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def find_tool(name: str) -> str:
    """ffmpeg/ffprobe from FFMPEG_DIR, PATH, or winget's links folder (a fresh winget
    install isn't on the PATH of processes that were already running)."""
    candidates = []
    if os.environ.get("FFMPEG_DIR"):
        candidates.append(Path(os.environ["FFMPEG_DIR"]) / name)
    found = shutil.which(name)
    if found:
        return found
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(Path(local) / "Microsoft" / "WinGet" / "Links" / name)
        packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if packages.exists():
            candidates += sorted(packages.glob(f"Gyan.FFmpeg*/**/bin/{name}*"))
    for c in candidates:
        for p in (c, c.with_suffix(".exe")):
            if p.exists():
                return str(p)
    raise MediaError(f"{name} not found. Install ffmpeg (e.g. winget install Gyan.FFmpeg) or set FFMPEG_DIR.")


def run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise MediaError(f"{Path(args[0]).name} failed ({proc.returncode}):\n{tail}")
    return proc.stdout


def probe(path: str | Path) -> dict:
    out = run([find_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    return json.loads(out)


def parse_iso6709(value: str) -> dict | None:
    """'+37.4275-122.1697+030.000/' -> {"lat": 37.4275, "lon": -122.1697, "alt": 30.0}"""
    import re

    m = re.match(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?/?$", value.strip())
    if not m:
        return None
    out = {"lat": float(m.group(1)), "lon": float(m.group(2))}
    if m.group(3):
        out["alt"] = float(m.group(3))
    return out


def location_from_tags(tags: dict) -> dict | None:
    for key in ("com.apple.quicktime.location.ISO6709", "location", "location-eng"):
        if key in tags:
            loc = parse_iso6709(tags[key])
            if loc:
                return loc
    return None
