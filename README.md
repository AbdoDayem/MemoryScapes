# MemoryScapes

Turn a video (or a set of panoramas) and an audio recording into a **memory**: a Gaussian splat you can walk through, with the recorded sound separated, placed in 3D and rebuilt into a soundscape by SATIE (Attractor Labs). Publish memories to a shared globe and explore other people's.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design, and [docs/webapp-plan/](docs/webapp-plan/README.md) for the web app (upload, API, database, storage, globe).

## Repository layout

```
apps/web/            Next.js client: the globe homepage, upload, memory viewer, spatial audio
services/api/        FastAPI: memories, uploads, starting the pipeline, progress, globe queries (SQLite)
services/workers/    Processing pipeline (ingest → poses → splat, separate → localize → soundscape → publish)
packages/manifest/   Memory manifest JSON Schema + TypeScript types
tools/debugger/      Local UI for inspecting a memory's pipeline outputs
infra/               Docker Compose for Postgres/PostGIS, Redis, MinIO (for later); DB schema
docs/                Architecture, the web app plan, splat output contract
```

## Local development

Three processes: the **API** (with its SQLite database), the **web app**, and, to process uploads, the **pipeline**, which the API starts by itself. No Docker needed: the database is a SQLite file and uploads are stored on local disk.

Prerequisites: Node 24, Python 3.12 with [uv](https://docs.astral.sh/uv/), and ffmpeg (`brew install ffmpeg` on macOS, `winget install Gyan.FFmpeg` on Windows) for processing. On Windows, use `.venv\Scripts\python` wherever this says `.venv/bin/python`.

### 1. API + database: http://localhost:8000 (interactive docs at `/docs`)

```bash
cd services/api
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m app.seed               # optional: 11 demo memories on the globe (safe to re-run)
.venv/bin/uvicorn app.main:app --reload
```

The database is created on first start at `services/api/data/memoryscapes.db` (gitignored; `DATABASE_URL` points elsewhere). Uploads and pipeline outputs go to `services/workers/data/storage/<memory id>/` (`STORAGE_ROOT`).

### 2. Web app: http://localhost:3000

```bash
npm install
npm run dev:web
```

`/` is the globe of memories, **+ New memory** uploads one, and `/memories/<id>` is the viewer. The web app talks to the API at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

### 3. Pipeline (to process uploads)

```bash
cd services/workers
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,visual,audio,satie]"
.venv/bin/memoryscapes-worker setup-models   # the sound models, about 5.6 GB, once
```

When you upload, the API runs `.venv/bin/python -m memoryscapes_workers.run` from here (set `WORKERS_PYTHON` to use another Python). Splat training needs an NVIDIA GPU to be fast, and SATIE needs a one-time `memoryscapes-worker satie-login`; see [services/workers/README.md](services/workers/README.md). Without the pipeline installed, uploads still work but processing reports that it can't start.

### Tests

```bash
cd services/api && .venv/bin/python -m pytest
npm test --workspace @memoryscapes/web        # also: npm run typecheck / build --workspace @memoryscapes/web
cd services/workers && .venv/bin/python -m pytest
```

### Later: Postgres, Redis, MinIO

`infra/docker-compose.yml` (`npm run infra:up`, needs Docker) is ready for when the API moves to Postgres and S3-style storage. See [docs/webapp-plan/database.md](docs/webapp-plan/database.md) and [storage.md](docs/webapp-plan/storage.md).

## Status

- **Pipeline:** all stages run, including SATIE soundscapes and publishing private memories.
- **API:** upload, process, progress, details, globe and the viewer's files work; live progress (SSE), reports and login come later.
- **Web app:** the low-poly globe homepage, the upload panel and the viewer work.
- **Not yet:** public sharing (needs the privacy pass), real login, Postgres and cloud storage.

Build order: [docs/ARCHITECTURE.md § 9](docs/ARCHITECTURE.md#9-build-phases) and [docs/webapp-plan/README.md](docs/webapp-plan/README.md#build-order).
