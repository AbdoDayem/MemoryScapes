"""Spark LoD trees (.rad) for streaming, built with Spark's `build-lod` tool.

The viewer draws splats with Spark. Given a LoD tree, Spark streams the splat: the first
chunk is a coarse view of the whole scene, then it fetches only the detail the view needs
within a per-device splat budget. Without one, the browser builds the tree itself after
downloading the full .spz (about 3 s per 1.5M gaussians on an M1 Pro).

`build-lod` is a Rust program from Spark's repository (MIT). deploy/setup.py installs it
into services/workers/.tools/ with cargo, at the Spark version the web app uses, so the
.rad format matches the viewer's decoder. Output is "chunked" RAD: a small header
(`<name>-lod.rad`) and chunk files (`<name>-lod-<i>.radc`) next to it, which the viewer
fetches by name relative to the header's URL.

Looked up in order: $SPARK_BUILD_LOD (a path, or "off" to disable), then
services/workers/.tools/bin/build-lod[.exe], then build-lod on PATH.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TOOL_ENV = "SPARK_BUILD_LOD"
TOOLS_DIR = Path(__file__).resolve().parents[2] / ".tools"  # services/workers/.tools
VERSION_FILE = TOOLS_DIR / "build-lod.version"  # the Spark tag it was built from
INSTALL_HINT = ("Install it with deploy/setup.py (needs Rust: https://rustup.rs), or set "
                f"{TOOL_ENV} to a build-lod binary. Without it, memories publish without a "
                "streaming LoD tree and the browser builds one itself.")
METHODS = ("quality", "quick")  # build-lod's bhatt-lod (better merges) and tiny-lod (faster)


class LodTreeError(RuntimeError):
    pass


@dataclass
class LodTree:
    header: str  # file name of the .rad header
    chunks: list[str]  # chunk file names, in order
    count: int | None  # splats in the tree, interior nodes included
    bytes: int


def find_tool() -> list[str] | None:
    """The command that runs build-lod, or None when it isn't installed (or is disabled)."""
    env = os.environ.get(TOOL_ENV, "").strip()
    if env.lower() == "off":
        return None
    if env:
        return [env] if Path(env).is_file() or shutil.which(env) else None
    local = TOOLS_DIR / "bin" / ("build-lod.exe" if sys.platform == "win32" else "build-lod")
    if local.is_file():
        return [str(local)]
    found = shutil.which("build-lod")
    return [found] if found else None


def tool_version() -> str:
    """The Spark version build-lod was installed from (part of the publish fingerprint, so a
    Spark upgrade rebuilds trees), or "unknown"."""
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


def _chunk_index(path: Path) -> int:
    m = re.search(r"-(\d+)\.radc$", path.name)
    return int(m.group(1)) if m else -1


def build(tool: list[str], splat: Path, out_dir: Path, *, method: str = "quality", name: str = "splat",
          timeout_s: float = 3600.0) -> LodTree:
    """Build a chunked LoD tree from `splat` (.spz or .ply) into `out_dir` as `<name>-lod.rad`
    plus `<name>-lod-<i>.radc`. Raises LodTreeError when build-lod fails."""
    if method not in METHODS:
        raise ValueError(f"Unknown LoD tree method {method!r}")
    with tempfile.TemporaryDirectory(prefix="lodtree-") as tmp:
        work = Path(tmp)
        # build-lod writes next to its input, named after it.
        src = work / f"{name}{splat.suffix}"
        shutil.copyfile(splat, src)
        try:
            proc = subprocess.run([*tool, f"--{method}", "--rad-chunked", str(src)], capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LodTreeError(f"build-lod didn't run: {exc}") from exc
        header = work / f"{name}-lod.rad"
        chunks = sorted(work.glob(f"{name}-lod-*.radc"), key=_chunk_index)
        # build-lod reports some failures (an unreadable file, non-finite splats) on stderr
        # and still exits 0, so check for its output too.
        if proc.returncode != 0 or not header.is_file() or not chunks:
            tail = " | ".join((proc.stderr or proc.stdout or "").strip().splitlines()[-4:])
            raise LodTreeError(f"build-lod failed (exit {proc.returncode}): {tail or 'no output'}")
        out_dir.mkdir(parents=True, exist_ok=True)
        size = 0
        for f in (header, *chunks):
            size += f.stat().st_size
            shutil.move(str(f), out_dir / f.name)
        m = re.search(r'"final_splat_count":\s*(\d+)', proc.stdout or "")
        return LodTree(header=header.name, chunks=[c.name for c in chunks],
                       count=int(m.group(1)) if m else None, bytes=size)
