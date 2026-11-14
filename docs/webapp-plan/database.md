# Database plan

Part of the [web app plan](README.md). This file covers what the API stores, the
tables, and how we move from SQLite to Postgres later.

## SQLite now, Postgres later

We start with **SQLite** (the database, a single file) used through **SQLAlchemy 2.0**
(a Python library that talks to SQLite, Postgres and others with the same code).

SQLite is enough for now because:
- It needs no setup: it's built into Python, with no server and no Docker.
- Only the API writes to the database. The pipeline writes files, not rows. One writer at a
  time is fine.
- The globe query ("memories inside this rectangle") is a range filter on `lat`/`lon` with an
  index. That's fast for thousands of rows without PostGIS.

**Switch to Postgres** when there are many users uploading at once, when the API runs on more
than one machine, or when we deploy. Use a managed host (Supabase or Neon; both support
PostGIS) or `infra/docker-compose.yml` locally. See § Moving to Postgres.

Settings:
- `DATABASE_URL`, default `sqlite:///services/api/data/memoryscapes.db`. The `data/`
  folder is already gitignored.
- Turn on WAL mode (`PRAGMA journal_mode=WAL`) and foreign keys (`PRAGMA foreign_keys=ON`)
  on each connection.

## What lives where

**Videos are never stored in the database.** Big files go in file storage, and the database
holds facts about them. The memory id links the two.

| Data | Stored in | Format |
|---|---|---|
| Uploaded video and audio, frames, 3D scene, sound tracks, progress | File storage, `services/workers/data/storage/{id}/` | Ordinary files (`.mp4`/`.mov`, `.wav`, `.jpg`, `.ply`, `.json`) |
| Owner, title, status, location, dates | Database | Rows |

File layout and storage tech: see [`storage.md`](storage.md). File paths are never stored in
the database, because they follow from the id.

## Tables

Adapted from `infra/db/init.sql`, which uses Postgres-only types (`geography`, `ENUM`,
`TIMESTAMPTZ`). Here those become portable columns. Define them in
`services/api/app/schema.py` with SQLAlchemy `Table` objects, and create them on API startup
(`metadata.create_all`).

### `users`
| Column | Type | Notes |
|---|---|---|
| `id` | text, primary key | e.g. `usr_dev` |
| `handle` | text, unique, not null | e.g. `dev` |
| `created_at` | datetime (UTC), not null | |

Seeded with one row: `usr_dev` / `dev`.

### `memories`
| Column | Type | Notes |
|---|---|---|
| `id` | text, primary key | `mem_` + 12 random lowercase letters/digits. Must match `^[a-z0-9][a-z0-9_-]{0,63}$` (the debugger's rule), so it's safe as a folder name. |
| `owner_id` | text, not null, → `users.id` | Always `usr_dev` for now |
| `title` | text, nullable | |
| `description` | text, nullable | |
| `visibility` | text, not null, default `private` | `private` \| `unlisted` \| `public`. Only `private` is allowed until the privacy pass exists. |
| `status` | text, not null, default `draft` | See § Status values |
| `lat`, `lon` | real, nullable | **Already blurred** to about 100 m (§ Location) |
| `location_source` | text, nullable | `video_gps` \| `user_pin` |
| `captured_at` | datetime (UTC), nullable | From the video's metadata when available |
| `error` | text, nullable | Short reason when `status = failed` |
| `thumbnail_url` | text, nullable | Unused until thumbnails exist |
| `created_at`, `updated_at` | datetime (UTC), not null | |

Indexes: `(lat, lon)` for the globe; `(owner_id, created_at)` for "my memories".

### `jobs`
One row per pipeline stage per run, copied from `job/status.json` so the database can answer
"what's processing / what failed" without reading files.

| Column | Type | Notes |
|---|---|---|
| `id` | integer, primary key | |
| `memory_id` | text, not null, → `memories.id`, delete cascades | |
| `stage` | text, not null | `ingest` `poses` `splat` `separate` `localize` `soundscape` `publish` |
| `status` | text, not null | The runner's values: `pending` `running` `done` `failed` `skipped` `blocked` |
| `error` | text, nullable | |
| `seconds` | real, nullable | How long the stage took |
| `started_at`, `finished_at` | datetime (UTC), nullable | |

Unique on `(memory_id, stage)`: a new run overwrites the previous run's rows.

Not yet: `memory_assets` and `reports` from `init.sql`. Add them when versioned outputs and
moderation are built.

## Status values

Reuse `MemoryStatus` from `services/api/app/models.py`:

```
draft ──> uploading ──> processing ──> ready
                             └──────> failed ──(process again)──> processing
```

| Status | Set when |
|---|---|
| `draft` | `POST /memories` creates the row |
| `uploading` | The first file starts arriving |
| `processing` | `POST /memories/{id}/process` starts the pipeline |
| `ready` | The run ends with `state: done` in `job/status.json` |
| `failed` | The run ends with `state: failed`; `error` holds the first failed stage's message |

`ready` lines up with the viewer: `publish` (implemented on `main` for private memories)
writes `publish/manifest.json`, and `GET /memories/{id}` reports `ready` once that file
exists. If the runner ends `done` but no manifest exists (e.g. `publish` was skipped), treat
the memory as `failed` with error "no manifest".

Memories created with the worker CLI (`memoryscapes-worker demo`) have no row. The API still
serves them from storage alone (`backend.md`).

## Location

- Source: the video's GPS, if the phone recorded it (`ingest/meta.json` → `geo` as
  `{lat, lon, alt}`), otherwise a pin the user drops.
- Blur before saving: round to 3 decimal places of latitude (about 110 m), and scale the
  longitude rounding by `cos(lat)` so it's also about 110 m. Store only the blurred value.
  (Design doc § 8: don't reveal exact home locations.)
- Validate: `-90 ≤ lat ≤ 90`, `-180 ≤ lon ≤ 180`.

## `db.py` functions

`services/api/app/db.py` holds all database code. Nothing else imports SQLAlchemy.

| Function | Does |
|---|---|
| `init_db()` | Create the engine and tables, set the pragmas, seed `usr_dev` |
| `create_memory(owner_id, title, description) -> dict` | Insert a `draft` row with a new id |
| `get_user(id)`, `list_jobs(id)` | Reads |
| `get_memory(id) -> dict \| None` | One row |
| `list_memories(owner_id) -> list[dict]` | Newest first |
| `update_memory(id, **fields) -> dict` | Title, description, visibility (only `private` for now), lat/lon (rounded here), location_source, captured_at |
| `set_status(id, status, error=None)` | Status change, bumps `updated_at` |
| `upsert_jobs(id, stages: dict)` | Copy the `stages` block of `job/status.json` into `jobs` |
| `list_in_bbox(min_lon, min_lat, max_lon, max_lat, owner_id) -> list[dict]` | Memories with a location inside the box. Handles boxes that cross ±180° by splitting into two ranges. |

Tests use a temporary SQLite file per test (`tmp_path`).

## Seed data

`services/api/app/seed.py` (`python -m app.seed`) inserts `usr_dev` and about 10 memories in different cities
with status `ready`, so the globe can be built before real uploads exist. Re-running it
doesn't create duplicates.

## Moving to Postgres

1. Start Postgres + PostGIS (`npm run infra:up`, or Supabase/Neon).
2. Set `DATABASE_URL=postgresql+psycopg://...` (psycopg is already a dependency of the API).
3. Run the same `create_all`. Alembic migrations should replace it once the schema changes
   with real data in it.
4. Optional: add a PostGIS `geography(Point)` column generated from `lat`/`lon` with a GiST
   index, and rewrite `list_in_bbox` to use `ST_MakeEnvelope` and `ST_ClusterDBSCAN`.
5. Copy rows across with a one-off script if any local data is worth keeping.
