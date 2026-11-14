# Web app plan: upload → API → database → globe

Owner: Huijun. Written November 14, 2026, against `main` at `899d3df` (after the viewer, PR #5).

This folder plans the path a user takes **into** MemoryScapes and back **out** again: upload
a video, watch it being processed, then find it as a pin on the globe.

What `main` already has:
- **The pipeline** (`services/workers`): all stages, including `publish`, which writes the
  manifest for private memories. `python -m memoryscapes_workers.run` runs it on one upload.
- **The viewer** (Marko): `/memories/[id]` renders the 3D scene with its sound, and shows a
  simple list of finished stages while processing.
- **Reading memories in the API**: `GET /memories/{id}` (manifest, stages, `asset_base`) and
  `GET /memories/{id}/files/{key}`, read-only storage in `services/api/app/storage.py`, CORS,
  and `services/api/tests/test_memories.py`.

What's missing, and what this plan builds: the upload page and endpoints, starting the
pipeline from the API, a database (none is used yet), editing a memory's details, and the
globe. The `/upload` and `/` pages are still placeholders, and `POST /memories`,
`POST /process`, `GET /events`, `PATCH` and `GET /globe` still return `501`.

| File | What it covers |
|---|---|
| `README.md` (this file) | Goal, decisions, how the parts connect, build order, checklist |
| [`database.md`](database.md) | What we store, the tables, SQLite now and Postgres later |
| [`storage.md`](storage.md) | Where the files live (videos and pipeline outputs), the key layout, local disk now and R2/S3 later |
| [`backend.md`](backend.md) | The API: every endpoint (**the API contract**), upload, starting the pipeline, progress |
| [`frontend.md`](frontend.md) | Pages and components in `apps/web` |

Each fact lives in one file. The table definitions are in `database.md`. The endpoint
definitions are in `backend.md`, and `frontend.md` links to them instead of repeating them.

## Goal

1. The homepage `/` is a cute low-poly 3D globe showing every memory as a small pin blowing
   soap bubbles.
2. **+ New memory** opens an upload panel over the globe: pick a phone video (and optionally
   a separate audio recording), give it a title, click the globe where it was filmed, upload.
3. The memory appears on the globe right away and processes. Clicking it opens a preview
   card; **Enter memory** opens `/memories/{id}` (the viewer), which shows the pipeline's
   stages and then the 3D scene.

**Out of scope:** the 3D viewer itself (Marko's work); the pipeline stages (the workers
team's; splat training runs on GPU machines, not tested on laptops); real login; public sharing.

## Decisions

| Decision | Why |
|---|---|
| **SQLite** through **SQLAlchemy** | One file, nothing to install or run. SQLAlchemy lets us switch to Postgres later by changing `DATABASE_URL`. See `database.md`. |
| **File storage: local disk now**, behind the API's existing storage interface (`app/storage.py`, extended with writes); MinIO to test locally, then R2/S3 when deployed | The pipeline, the debugger and the API already use `services/workers/data/storage/{id}/`. The interface means switching to the cloud is a settings change. The database never stores videos, only facts about them. See `storage.md`. |
| **Extend `GET /memories/{id}`, don't replace it** | The viewer depends on its current fields (`status`, `stages`, `manifest`, `asset_base`). New fields are added next to them. |
| **Progress by polling first**, live updates (Server-Sent Events) later | The viewer already polls `GET /memories/{id}` every 5 s. Adding per-stage detail to that response is enough for the first version. |
| **API starts the pipeline as a subprocess** (`python -m memoryscapes_workers.run`) | The runner already exists on `main` and writes progress to `job/status.json`. A Redis job queue can replace it later. |
| **One fake dev user** (`usr_dev`) | No login until the rest works. Everything is private and local. |
| **No Docker yet** | Nothing above needs it. `infra/docker-compose.yml` stays ready for Postgres, Redis and MinIO later. |
| **Private memories only** | Nothing goes public before the privacy pass (face and plate blur), which doesn't exist yet. |

`docs/ARCHITECTURE.md` describes the long-term setup (Postgres + PostGIS, S3, a job queue).
This plan is the smaller first version of it. 

## How the parts connect

```
Browser  apps/web (Next.js, localhost:3000)
  /upload ─────────┐   /memories/{id} ──┐   / (globe) ──┐
                   ▼                    ▼               ▼
API  services/api (FastAPI, localhost:8000)
  new:    POST /memories   PUT /memories/{id}/upload/{name}   POST /memories/{id}/process
          PATCH /memories/{id}   GET /globe   GET /memories
  exists: GET /memories/{id} (extended)   GET /memories/{id}/files/{key}
       │                          │                           │
       ▼                          ▼                           ▼
  SQLite database          File storage                 Pipeline (subprocess)
  services/api/data/       services/workers/data/       python -m memoryscapes_workers.run
  memoryscapes.db          storage/{id}/...  ◄────────  reads upload/, writes stage
  (rows: memories,         (video, frames, splat,       outputs and job/status.json
   jobs, users)             audio, status.json)
```

The **memory id** ties everything together: it's the row's primary key in the database and
the folder name in file storage.

## Build order

Settle the contracts first, then build in vertical slices. Each slice goes through all three
parts and ends with something you can demo.

### Phase 0: Contracts (1–2 days)
- [ ] Agree on the tables (`database.md` § Tables)
- [ ] Agree on the endpoints (`backend.md` § API contract)
- [ ] Tell the team about SQLite and the folder layout

### Phase 1: Database layer (about 2 days)
- [x] `services/api/app/settings.py`, `schema.py`, `db.py` (PR #12)
- [x] Seed script with a dev user and 11 fake memories (`python -m app.seed`)
- [x] Database tests

### Phase 2: Slice 1, upload a video (3–4 days)
Split in PRs: **2a** backend for uploads; **2b** the globe homepage with the upload panel
(this pulls Phase 4's globe forward, since the globe is the homepage).
- [x] 2a: add writes (`writer`, `write_bytes`, `delete_prefix`) to the existing `app/storage.py` (PR #14)
- [x] 2a: `app/pipeline.py`, which starts the workers' runner and records the result
- [x] 2a: `POST /memories`, `PUT …/upload/{name}`, `POST …/process`, `GET /memories`; `GET /memories/{id}` reads the database row too
- [x] Design: globe homepage + upload panel (`frontend.md`)
- [ ] 2b-backend: `GET /globe`, `PATCH /memories/{id}`
- [x] 2b-frontend: low-poly globe with pin-and-bubble markers, preview card, "My memories", upload panel
- [ ] Demo: upload a phone video, place it on the globe, see it process, enter it

### Phase 3: Slice 2, progress (about 2 days)
- [x] Backend (done in 2a): `GET /memories/{id}` adds `stage_detail` from `job/status.json` and the
      `failed` status; stage results are copied into `jobs` when a run ends
- [ ] Frontend: stage list and stopping on `failed`: Marko's PR #11 covers most of this;
      check it against real API responses
- [ ] Demo: watch the 7 stages run and finish as `ready` (the 3D scene appears) or `failed`

### Phase 4: Globe polish
The globe itself moved into 2b. What's left: thumbnails or preview loops on pins, and (with
public memories) server-side clustering.

### Phase 5: Later
Postgres + PostGIS; the S3 version of the storage interface, with MinIO locally, R2 when
deployed, and presigned uploads; live progress with Server-Sent Events
(`GET /memories/{id}/events`); a Redis job queue; real login; public visibility once the
privacy pass exists; moderation reports; per-user quotas; thumbnails; lines between nearby
memories on the globe.

## Verification (end to end)

```bash
cd services/api && .venv/bin/uvicorn app.main:app --reload     # terminal 1
npm run dev:web                                                 # terminal 2, repo root
```

1. Upload a short phone video at `localhost:3000/upload`.
2. Watch the stages advance on `/memories/{id}`, then the 3D scene load.
3. Check the row: `sqlite3 services/api/data/memoryscapes.db "select * from memories"`.
4. Check the files: `ls services/workers/data/storage/<id>/`. The debugger
   (`cd tools/debugger && MEMORYSCAPES_DATA=../../services/workers/data/storage npm run dev`)
   shows the same memory's outputs.
5. Set a location and see the pin on `localhost:3000/`.
6. `cd services/api && .venv/bin/python -m pytest`, plus the frontend typecheck and build.
