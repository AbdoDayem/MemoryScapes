# Backend plan (API)

Part of the [web app plan](README.md). This file covers the FastAPI service in
`services/api`: settings, **the API contract**, saving uploads, starting the pipeline, and
streaming progress. Tables are defined in [`database.md`](database.md).

## What exists on `main`

Added with the viewer (PR #5, Marko). **Keep these working**: the viewer depends on them.
- `app/main.py`: the FastAPI app, routers at `/memories` and `/globe`, `/health`, and CORS
  from `CORS_ORIGINS` (default `http://localhost:3000`).
- `app/routers/memories.py`:
  - `GET /memories/{id}`: reads storage only (no database). Returns `id`, `status`
    (`ready` once `publish/manifest.json` exists, otherwise `processing`), `stages`
    (stage → finished, from each stage's marker file), `manifest`, and `asset_base`. `404`
    if there are no stage outputs and no upload.
  - `GET /memories/{id}/files/{key}`: serves one stage output (a local file with range
    support, or a redirect to a presigned S3 URL). Refuses `upload/` and `_stage.json`.
  - `POST /memories`, `POST /process`, `GET /events`, `PATCH`, `POST /report`: still `501`.
- `app/routers/globe.py`: `GET /globe` returns `501`.
- `app/storage.py`: read-only `LocalStore` / `S3Store` (see `storage.md`).
- `app/models.py`: Pydantic models to reuse (`Visibility`, `MemoryStatus`,
  `CreateMemoryRequest`, `CreateMemoryResponse`, `UploadTarget`, `UpdateMemoryRequest`,
  `MemorySummary`).
- `tests/test_memories.py` and `.github/workflows/api.yml` (CI runs the API tests).
- Environment: `services/api/.venv` (uv, Python 3.12).

## Settings

New file `app/settings.py` (pydantic-settings, already a dependency). Read from the
environment or `.env`:

| Variable | Default | Used for |
|---|---|---|
| `DATABASE_URL` | `sqlite:///<repo>/services/api/data/memoryscapes.db` | `database.md` |
| `STORAGE_BACKEND` | `local` | Which storage version to use (`local` now, `s3` later; `storage.md`) |
| `STORAGE_ROOT` | `<repo>/services/workers/data/storage` | Upload folder and pipeline outputs |
| `WORKERS_PYTHON` | `<repo>/services/workers/.venv/bin/python` | Starting the pipeline |
| `DEV_USER_ID` | `usr_dev` | Owner of every memory until real login |
| `MAX_UPLOAD_MB` | `500` | Upload size limit |
| `CORS_ORIGINS` | `http://localhost:3000` | The web app's origin (already read in `main.py`) |

`STORAGE_BACKEND`, `STORAGE_ROOT` and `CORS_ORIGINS` are already read directly from the
environment by `storage.py` and `main.py`. Keep the same names and defaults.

In `app/main.py`: add `PUT` to the CORS `allow_methods` (it's `GET, POST, PATCH` today, and
uploads are `PUT`), and call `db.init_db()` on startup.

## API contract

All JSON. Errors are `{"detail": "<message>"}` with the status code shown. Memory ids follow
`database.md` (`mem_…`); any other id gets `404`.

### `POST /memories`: create a draft
Request (existing `CreateMemoryRequest`):
```json
{ "title": "Golden Gate walk", "has_audio_track": true, "input_kind": "video",
  "file_names": ["IMG_1234.MOV", "recorder.wav"] }
```
Response `201` (existing `CreateMemoryResponse`):
```json
{ "id": "mem_k3j9x0a1b2c3",
  "uploads": [
    { "file_name": "IMG_1234.MOV", "url": "http://localhost:8000/memories/mem_k3j9x0a1b2c3/upload/IMG_1234.MOV" },
    { "file_name": "recorder.wav", "url": "http://localhost:8000/memories/mem_k3j9x0a1b2c3/upload/recorder.wav" } ] }
```
Errors: `422` if `file_names` is empty, has no video, or has more than one video or more than
one audio file; `422` if `input_kind` isn't `video` (panoramas come later).

The frontend PUTs each file to its `url`. Later, `url` becomes a presigned S3 URL, and the
frontend doesn't change.

### `PUT /memories/{id}/upload/{file_name}`: upload one file
Body: the raw file bytes (not a form). `Content-Type` is the file's type.
- Saves through the storage interface (`store.writer("{id}/upload/video.<ext>")`, streamed
  in chunks with the size checked as it arrives) or `audio.<ext>`, picked by extension (the names the debugger uses; the
  pipeline finds files by extension). Writes the original names to `upload/meta.json`.
  See [`storage.md`](storage.md).
- Status → `uploading`.

Response `200`: `{ "file_name": "IMG_1234.MOV", "stored_as": "video.mov", "bytes": 48213344 }`

Errors: `404` unknown memory; `409` while it's processing; `413` over `MAX_UPLOAD_MB`;
`422` if the name isn't one returned by `POST /memories` or has an unsupported extension.
The writer saves to a hidden temp file and renames it at the end, so a broken or too-large
upload never leaves half a video. Uploading the video again replaces the previous one.
`501` if storage isn't local (uploads go straight to S3 later).

### `POST /memories/{id}/process`: start the pipeline
No body. Starts the runner (§ Starting the pipeline). Status → `processing`. Allowed again
after `failed` or `ready` (re-processing). `503` if the pipeline isn't installed.

Response `202`: `{ "id": "mem_…", "status": "processing" }`

Errors: `404`; `409` if already processing; `422` if no video was uploaded.

### `GET /memories/{id}`: one memory (exists; we extend it)
**The existing fields keep their meaning**, because the viewer uses them. New fields are added
next to them:
```json
{ "id": "mem_…",
  "status": "processing",
  "stages": { "ingest": true, "poses": false, "…": false },
  "manifest": null,
  "asset_base": "http://localhost:8000/memories/mem_…/files/",

  "title": "…", "description": null, "visibility": "private",
  "lat": 37.81, "lon": -122.478, "location_source": "video_gps",
  "captured_at": "2026-11-10T18:30:00Z", "created_at": "…", "error": null,
  "stage_detail": { "state": "running", "current": "poses", "detail": "<latest log line>",
                    "stages": { "ingest": {"status": "done", "seconds": 12.3},
                                "poses":  {"status": "running"}, "…": {} } },
  "suggested_location": { "lat": 37.81, "lon": -122.478 } }
```
- Existing fields, unchanged: `id`, `stages` (stage → finished), `manifest`, `asset_base`.
- `status`: for a memory with a database row, **the row decides** (`draft`, `uploading`,
  `processing`, `ready`, `failed`), because a re-run can sit on top of an old manifest. For a
  CLI memory (no row), `ready` when the manifest exists, else `processing`, as before. The
  viewer stops polling on `failed` (Marko's PR #11).
- New fields come from the database row. `stage_detail` comes from `job/status.json`
  (or the `jobs` table if the file is gone), and is `null` before processing starts.
  `suggested_location` is the blurred GPS from `ingest/meta.json` when it exists and no
  location is set yet.
- **Memories without a database row** (made with the CLI, e.g. `memoryscapes-worker demo`)
  keep working exactly as today: storage-only fields, new fields `null`.
- `404` when there's no row and nothing in storage (as today).
- Keep `tests/test_memories.py` passing, and add tests for the new fields.

### `GET /memories/{id}/files/{key}`: one output file (exists, unchanged)
Used by the viewer to load the 3D scene and audio. Not part of this work. Never serves `upload/`.

### `GET /memories`: my memories
Response `200`: a list of `MemorySummary` for `DEV_USER_ID`, newest first.

### `GET /memories/{id}/events`: live progress (Server-Sent Events), later
**Not needed for the first version:** the viewer polls `GET /memories/{id}` every 5 s, and
`stage_detail` carries the progress. Build this when polling isn't good enough.

`Content-Type: text/event-stream`. Every second, read `job/status.json`. When it changes,
send:
```
event: progress
data: {"state":"running","current":"poses","detail":"<latest log line>","stages":{...}}
```
When the run ends, send `event: done` with `{"status":"ready"}` or
`{"status":"failed","error":"…"}` and close. If the memory isn't processing, send its current
state once and close. A comment line (`: ping`) every 15 s keeps the connection open.

### `PATCH /memories/{id}`: edit details
Request (existing `UpdateMemoryRequest`): any of `title`, `description`, `visibility`, `lat`, `lon`.
- `lat`/`lon` must come together, are validated, and are **blurred before saving**
  (`database.md` § Location). `location_source` = `user_pin`.
- `visibility` other than `private` → `422` "Public sharing needs the privacy pass".

Response `200`: the same shape as `GET /memories/{id}`.

### `GET /globe?bbox=minLon,minLat,maxLon,maxLat&zoom=2`: pins for the globe
Returns the dev user's memories that have a location inside `bbox`, grouped when zoomed out:
```json
{ "clusters": [ { "lat": 37.7, "lon": -122.4, "count": 4, "ids": ["mem_…", "…"] } ],
  "memories": [ { "id": "mem_…", "title": "…", "lat": 48.86, "lon": 2.35,
                  "status": "ready", "thumbnail_url": null } ] }
```
Grouping: snap each point to a grid whose cell size shrinks with zoom (e.g.
`cell_deg = 360 / 2**(zoom + 3)`). A cell with 1 memory goes in `memories`; more go in
`clusters`. Errors: `422` for a malformed `bbox` or `zoom` outside 0–20. A `bbox` where
`minLon > maxLon` crosses ±180° and is allowed.

### `GET /health`
Exists: `{"status": "ok"}`.

## Starting the pipeline

`POST /process` runs, in the background:

```
{WORKERS_PYTHON} -m memoryscapes_workers.run --root {STORAGE_ROOT} --memory {id}
```

with working directory `services/workers`, through `subprocess.Popen`, stdout and stderr
sent to `/dev/null` (the runner writes its own `job/log.txt`).

- The runner already exists (`services/workers/memoryscapes_workers/run.py`). It runs the
  stages in order, each in its own process, and keeps `job/status.json` up to date.
  `tools/debugger/server/jobs.ts` starts the same runner the same way, so use it as a reference.
- Keep a dict `{memory_id: Popen}` so only one run per memory happens at a time.
- A background thread watches for the run to finish (`Popen.wait()`): read the final
  `status.json`, call `db.upsert_jobs` and `db.set_status(ready|failed)`.
  How much of the video is processed is the runner's own default (`--max-seconds`, 60 s
  today); the API doesn't override it. The runner's stdout/stderr go to
  `job/runner-output.txt`, so a run that dies before logging still has a reason.
- On API startup, any memory still `processing` with no live process is marked `failed`
  ("interrupted").
- `ffmpeg` must be on `PATH` (installed with Homebrew).
- The runner works on a local folder (`--root`, the `LocalStore`'s `root`), so it needs
  `STORAGE_BACKEND=local`. With S3, the workers would run with `STORAGE_BACKEND=s3` and the
  same bucket, to be agreed with the workers team (`storage.md` § The pipeline and storage).

`job/status.json` as written by the runner:
```json
{ "memory_id": "mem_…", "state": "running | done | failed", "started_at": 1731600000.0,
  "finished_at": null, "current": "poses", "detail": "latest log line",
  "options": { "max_seconds": 60, "max_keyframes": 120, "detector": true },
  "stages": { "ingest": { "status": "done", "started_at": 1731600001.2, "seconds": 12.3 },
              "poses":  { "status": "running", "started_at": 1731600013.5 },
              "publish": { "status": "pending" } } }
```
Stage statuses: `pending` `running` `done` `failed` `skipped` `blocked` `not requested`.

## Security notes (even locally)

- Check ids against the regex before building any path, and never use the user's file name
  in the path (we save as `video.<ext>` / `audio.<ext>`).
- Enforce the size limit while streaming, not after.
- Never serve `upload/`. The original video can show faces and places. The existing files
  endpoint already refuses it.

## Files

| File | Change |
|---|---|
| `services/api/app/settings.py` | New |
| `services/api/app/storage.py` | Exists (read-only): add `save`, `write_bytes`, `delete_prefix` (see `storage.md`) |
| `services/api/app/db.py`, `schema.py` | New (see `database.md`) |
| `services/api/app/pipeline.py` | New: start the runner, track processes, read `status.json` |
| `services/api/app/routers/memories.py` | Fill in the `501` endpoints; extend `GET /memories/{id}`; leave the files endpoint alone |
| `services/api/app/routers/globe.py` | Fill in `GET /globe` |
| `services/api/app/models.py` | Add response models (`MemoryDetail`, `GlobeResponse`) |
| `services/api/app/main.py` | Add `PUT` to CORS methods; startup hook |
| `services/api/pyproject.toml` | Add `sqlalchemy>=2.0` |
| `services/api/tests/` | Exists (`test_memories.py`): add `test_upload.py`, `test_db.py`, `test_globe.py` |

## Tests

`services/api/tests/` with pytest and FastAPI's `TestClient` (httpx is already in `[dev]`).
Follow the existing fixture in `test_memories.py`: a temp `STORAGE_ROOT` and
`storage.get_store.cache_clear()`, plus a temp SQLite file.
- The existing `test_memories.py` keeps passing unchanged.
- Create → upload → process: rows, files and status change as expected. Replace the runner
  with a fake that writes a `status.json`, so tests never run the real pipeline.
- `GET /memories/{id}`: the new fields for a memory with a row; unchanged output for one
  without; `failed` status.
- Upload errors: wrong name, bad extension, too large, while processing.
- `PATCH`: location is blurred, `public` is refused.
- `/globe`: bbox filter, crossing ±180°, clustering by zoom.

Run with `cd services/api && .venv/bin/python -m pytest`. CI already runs them
(`.github/workflows/api.yml`).
