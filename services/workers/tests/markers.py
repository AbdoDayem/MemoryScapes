"""Skip markers for tests that need optional tools (the CI install has none of these)."""

import importlib.util
import shutil

import pytest


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
needs_cv2 = pytest.mark.skipif(not has_module("cv2"), reason="opencv not installed")
needs_pycolmap = pytest.mark.skipif(not has_module("pycolmap"), reason="pycolmap not installed")
