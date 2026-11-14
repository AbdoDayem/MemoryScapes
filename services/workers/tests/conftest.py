import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.storage import LocalStorage


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
needs_cv2 = pytest.mark.skipif(not has_module("cv2"), reason="opencv not installed")
needs_pycolmap = pytest.mark.skipif(not has_module("pycolmap"), reason="pycolmap not installed")


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path / "storage")


@pytest.fixture
def ctx(storage, tmp_path) -> StageContext:
    return StageContext(memory_id="mem_test", bucket="memories", storage=storage,
                        work_root=tmp_path / "work")


@pytest.fixture(scope="session")
def small_video(tmp_path_factory) -> tuple[Path, dict]:
    """A short, low-res synthetic clip (~5 s to render) plus its ground truth."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from memoryscapes_workers.devtools.synthetic import SyntheticSpec, generate

    out = tmp_path_factory.mktemp("synthetic_small")
    video = generate(out, SyntheticSpec(seconds=4.0, fps=15, width=320, height=180))
    return video, json.loads((out / "ground_truth.json").read_text())
