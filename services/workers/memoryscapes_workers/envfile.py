"""Load ``.env`` files into the environment (KEY=VALUE lines), for the worker CLI.

Looks in ``<repo>/.env`` and ``services/workers/.env``. Variables already set in the
environment win, so a deploy's real configuration always overrides a dev file. Values
are never logged. Supports comments, blank lines, ``export KEY=...`` and quoted values.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKERS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKERS_DIR.parents[1]
DEFAULT_FILES = (REPO_ROOT / ".env", WORKERS_DIR / ".env")


def parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif " #" in value:  # trailing comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load(files=DEFAULT_FILES) -> list[str]:
    """Set variables from the files that exist; returns the names it set (not values)."""
    loaded = []
    for path in files:
        path = Path(path)
        if not path.is_file():
            continue
        for key, value in parse(path.read_text(encoding="utf-8")).items():
            if key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded
