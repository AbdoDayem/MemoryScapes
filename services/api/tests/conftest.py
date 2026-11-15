import os

import pytest

# Never read the developer's <repo>/.env in tests (app/main.py would load it on import).
os.environ.setdefault("MEMORYSCAPES_SKIP_DOTENV", "1")

from app import db, settings, storage  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    """Every test gets its own SQLite file, so no test touches services/api/data/."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "isolated.db").as_posix())
    settings.get_settings.cache_clear()
    _reset_engine()
    yield
    _reset_engine()
    settings.get_settings.cache_clear()
    storage.get_store.cache_clear()


def _reset_engine():
    if db._engine is not None:
        db._engine.dispose()
    db._engine = None
