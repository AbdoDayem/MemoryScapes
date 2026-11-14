# MemoryScapes API

FastAPI service for memories, uploads, pipeline status and globe queries. Plan and API
contract: [docs/webapp-plan/](../../docs/webapp-plan/README.md).

```bash
cd services/api
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload      # http://localhost:8000/docs
.venv/bin/python -m pytest                   # tests
.venv/bin/python -m app.seed                 # fake memories for building the globe (safe to re-run)
```

**Database:** SQLite at `services/api/data/memoryscapes.db` (gitignored), created on startup
with the `usr_dev` user. Set `DATABASE_URL` to use another file or Postgres. All database
code is in `app/db.py`; the tables are in `app/schema.py`. Inspect it with
`sqlite3 data/memoryscapes.db`.

**Working now:** `GET /memories/{id}` and `GET /memories/{id}/files/{key}` (read from
storage, used by the viewer). The other endpoints return `501` until they're implemented.
