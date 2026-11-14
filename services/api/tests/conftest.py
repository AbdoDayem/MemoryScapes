import pytest

from app import db, settings, storage


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
