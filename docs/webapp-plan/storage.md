# File storage plan

Part of the [web app plan](README.md). This file covers where the big files live (uploaded
videos and everything the pipeline produces), the key layout, the storage interface the
API uses, and the path from local disk to cloud storage.

## What file storage is

File storage holds **big files, looked up by a name** (a "key"). There are no tables and no
queries. The standard kind is **object storage**: Amazon S3 and the services that copy its API
(Cloudflare R2, MinIO, Supabase Storage). Think of it as one huge folder:

```
bucket "memories"
  mem_k3j9x0a1b2c3/upload/video.mov          ← 200 MB phone video
  mem_k3j9x0a1b2c3/splat/lods/lod0.spz       ← 3D scene
  mem_k3j9x0a1b2c3/separate/stems/birdsong.wav
```

It works as a pair with the database:

| | Holds | Example |
|---|---|---|
| Database ([`database.md`](database.md)) | Facts about each memory | "mem_k3j9… is owned by usr_dev, titled *Golden Gate walk*, status ready, at 37.81, -122.478" |
| File storage (this file) | The bytes | Everything under keys starting with `mem_k3j9…/` |

The **memory id** links them: it's the row's primary key and the first part of every key.
No file paths are stored in the database, because they follow from the id.

## Who does what

| Piece | Status | Owner |
|---|---|---|
| Pipeline reading and writing files: `services/workers/memoryscapes_workers/storage.py` (`LocalStorage`, `S3Storage`, picked by `STORAGE_BACKEND`) | Exists | Workers team |
| The API's storage interface, `services/api/app/storage.py` (`LocalStore`, `S3Store`, `get_store()`) | Exists, **read-only** (added with the viewer) | Marko wrote it; Huijun adds writes |
| Serving pipeline outputs to the browser, `GET /memories/{id}/files/{key}` | Exists | Marko |
| Taking the upload from the browser and saving it to storage | To build | Huijun (API) |
| Choosing and setting up cloud storage | Later | Huijun, agreed with the team |

## Key layout

Keys are relative to the storage root (a folder locally, a bucket in the cloud). The API
writes `upload/`. The pipeline writes everything else.

```
{id}/
  upload/video.<ext>      the original video, unchanged (.mp4 .mov .m4v .mkv .webm .avi .3gp)
  upload/audio.<ext>      optional separate recording (.wav .flac .m4a .mp3 .aac .ogg ...)
  upload/meta.json        original file names
  ingest/  poses/  splat/  separate/  localize/  soundscape/  publish/   stage outputs
  job/status.json         pipeline progress (format in backend.md)
  job/log.txt             full pipeline log
```

`video.<ext>` and `audio.<ext>` are the names the debugger uses. The pipeline's ingest
stage finds the files by extension. Stage outputs are specified in
`services/workers/README.md` and `docs/SPLAT_OUTPUTS.md`.

## Stack

| When | Where files live | Notes |
|---|---|---|
| **Now** | **Local disk**, `services/workers/data/storage/` (setting `STORAGE_ROOT`) | Nothing to install. The pipeline, the demo and the debugger already use it. Gitignored (`data/`). |
| **Next**, testing the cloud path locally | **MinIO** in Docker (`infra/docker-compose.yml`, `npm run infra:up`) | Speaks the S3 API, so presigned uploads work exactly as in production. Console at `localhost:9001`. |
| **Deployed** | **Cloudflare R2** (recommended) or **AWS S3** | Same API, so it's a settings change. R2 doesn't charge for downloads, which matters because viewers download large 3D-scene and audio files. Supabase Storage is an option if Postgres is on Supabase. |

The S3 settings already exist in `.env.example` (`S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY`,
`S3_SECRET_KEY`), and the workers read the same ones.

## The storage interface

The API never touches the disk directly. Everything goes through
`services/api/app/storage.py`, which **already exists** on `main` with two versions picked
by `STORAGE_BACKEND=local|s3` through `get_store()`:
- **`LocalStore`**: a folder at `STORAGE_ROOT` (default `services/workers/data/storage`).
  `path(key)` refuses keys that escape the folder.
- **`S3Store`**: `boto3` with the `S3_*` settings. Has `presigned_url(key)` for downloads.

What exists, and what we add:

| Function | Status | Does |
|---|---|---|
| `exists(key) -> bool` | Exists | |
| `read_bytes(key) -> bytes` | Exists | For small files like `job/status.json` and `ingest/meta.json` |
| `list(prefix) -> list[str]` | Exists | Keys under a prefix |
| `LocalStore.path(key) -> Path` | Exists | The file's path; used for serving files and as `--root` for the pipeline |
| `S3Store.presigned_url(key)` | Exists | Short-lived download URL |
| `writer(key)` | **Added** (local) | A context manager for writing a file in chunks (an upload as it arrives). Writes to a hidden temp file and renames at the end, so a broken upload never leaves half a file. The router checks the size limit as chunks arrive. |
| `write_bytes(key, data)` | **Added** | Small files like `upload/meta.json` |
| `delete(key)`, `delete_prefix(prefix)` | **Added** | Delete a file, or a memory's files |
| `S3Store.presigned_put(key)` | **Add later** | Direct-to-storage uploads |

Also add `save` and friends to the `BlobStore` protocol. Keep the existing functions as they
are: the viewer's endpoints and `tests/test_memories.py` use them.

Why an interface instead of writing to disk directly: S3, R2 and MinIO aren't disks. Without
it, every endpoint that touches files would need rewriting at the switch. With it, only this
file changes. It also lets tests use a temporary folder, and it mirrors the workers'
`storage.py`, so both sides agree on keys. (The viewer's file endpoint already relies on it:
it serves a local file, or redirects to a presigned URL on S3.)

## Uploads: now and later

**Now:** the browser sends each file to the API, and the API streams it into storage.
```
browser ──PUT /memories/{id}/upload/{name}──> API ──store.writer──> {id}/upload/video.mov
```

**Later (S3/R2):** the API hands out a presigned URL and the browser uploads straight to
storage, so 200 MB doesn't pass through the API server. Big files use multipart upload.
```
browser ──POST /memories──> API returns { url: <presigned PUT> }
browser ──PUT <presigned url>──> R2/S3
```

The frontend code is the same in both cases: it always PUTs each file to the `url` returned
by `POST /memories` (contract in [`backend.md`](backend.md)).

## The pipeline and storage

The subprocess runner (`python -m memoryscapes_workers.run --root <folder>`) works on a
**local folder**, so for now the API and the pipeline share `services/workers/data/storage/`.
Moving to S3 needs the workers to run with `STORAGE_BACKEND=s3` and the same bucket, which
means agreeing with the workers team when we get there.

## Rules

- Enforce the size limit while streaming, not after (`MAX_UPLOAD_MB`, default 500).
- Never use the user's file name in a key. Save as `video.<ext>` or `audio.<ext>`, and keep the
  original name in `upload/meta.json`.
- Check memory ids against the id pattern (`database.md`) before building any key.
- **Never serve `upload/`.** The original video can show faces and places. The existing
  `GET /memories/{id}/files/{key}` already refuses it; keep it that way.
- In the cloud, the bucket is private. Files are only reachable through short-lived
  presigned URLs.
- Never commit storage contents (`data/` is gitignored) or S3 keys (`.env`).
