import json
import shutil
import sys
from pathlib import Path

import pytest

# Tests import helpers flat: tests/ (synth, markers) and examples/ (scenes, evaluate).
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from memoryscapes_workers.context import StageContext  # noqa: E402
from memoryscapes_workers.storage import LocalStorage  # noqa: E402


@pytest.fixture
def storage(tmp_path) -> LocalStorage:
    return LocalStorage(tmp_path / "storage")


@pytest.fixture
def ctx(storage, tmp_path) -> StageContext:
    return StageContext(memory_id="mem_test", storage=storage, work_root=tmp_path / "work")


@pytest.fixture(scope="session")
def small_video(tmp_path_factory) -> tuple[Path, dict]:
    """A short, low-res synthetic clip (~5 s to render) plus its ground truth."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from memoryscapes_workers.devtools.synthetic import SyntheticSpec, generate

    out = tmp_path_factory.mktemp("synthetic_small")
    video = generate(out, SyntheticSpec(seconds=4.0, fps=15, width=320, height=180))
    return video, json.loads((out / "ground_truth.json").read_text())
