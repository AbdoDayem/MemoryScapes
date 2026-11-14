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

**Endpoints** (contract: [docs/webapp-plan/backend.md](../../docs/webapp-plan/backend.md)):

| Endpoint | Does |
|---|---|
| `POST /memories` | Create a draft and return an upload URL per file |
| `PUT /memories/{id}/upload/{name}` | Receive one file's bytes |
| `POST /memories/{id}/process` | Start the workers' pipeline (`app/pipeline.py`) |
| `GET /memories`, `GET /memories/{id}` | List memories; one memory with its manifest, stages and details |
| `PATCH /memories/{id}` | Title, description, location (rounded to ~100 m); private only for now |
| `GET /globe` | Memories that have a location, for the homepage globe |
| `GET /memories/{id}/files/{key}` | The viewer's files (from the database, or storage) |
| `GET /memories/{id}/events`, `POST /memories/{id}/report` | Not built yet (`501`) |

Uploads and pipeline outputs live under `services/workers/data/storage/<id>/` (`STORAGE_ROOT`);
the pipeline runs with `WORKERS_PYTHON` (default `services/workers/.venv`).
