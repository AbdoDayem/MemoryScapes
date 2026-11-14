import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from memoryscapes_workers.context import StageContext  # noqa: E402
from memoryscapes_workers.storage import LocalStorage  # noqa: E402


@pytest.fixture
def ctx(tmp_path) -> StageContext:
    return StageContext(memory_id="mem_test", storage=LocalStorage(tmp_path))
