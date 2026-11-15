"""API settings, read from environment variables.

The names match what the rest of the repo already reads: ``STORAGE_BACKEND`` and
``STORAGE_ROOT`` (``app/storage.py``, the workers) and ``CORS_ORIGINS`` (``app/main.py``).
See docs/webapp-plan/backend.md § Settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

API_DIR = Path(__file__).resolve().parents[1]  # services/api
REPO_DIR = API_DIR.parents[1]

DEFAULT_DATABASE_URL = "sqlite:///" + (API_DIR / "data" / "memoryscapes.db").as_posix()
_VENV_PYTHON = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
DEFAULT_WORKERS_PYTHON = REPO_DIR.joinpath("services", "workers", ".venv", *_VENV_PYTHON).as_posix()


class Settings(BaseSettings):
    database_url: str = DEFAULT_DATABASE_URL
    dev_user_id: str = "usr_dev"
    dev_user_handle: str = "dev"
    max_upload_mb: int = 500
    # Longest video the pipeline accepts. The upload page says the same
    # (apps/web/src/components/upload/UploadPanel.tsx, MAX_SECONDS); keep the two equal.
    max_video_seconds: int = 120
    workers_python: str = DEFAULT_WORKERS_PYTHON


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
