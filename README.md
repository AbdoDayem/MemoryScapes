# MemoryScapes

Turn a phone video into a **memory** you can walk back into. MemoryScapes builds a 3D Gaussian splat from the video's frames. It separates the recorded sound into its sources, places each source in the scene, and SATIE (Attractor Labs) turns them into a spatial soundscape. You then explore the memory in the browser, hearing each sound from where it happened. Memories sit on a shared globe, pinned where they were filmed.

## What it does

- **Upload a video** (up to 2 minutes) from the browser. Its capture time and GPS location come from the file's metadata.
- **Rebuild the scene.** Keyframes go through COLMAP (pycolmap) to get camera poses, then gsplat trains a Gaussian splat at metric scale. iPhone spatial videos also give true scale, extra right-eye views and stereo depth.
- **Take the sound apart.** CLAP tags which sounds are present, AudioSep separates each one, and HTDemucs pulls out speech. The filmer's own voice becomes narration, heard everywhere.
- **Put each sound in its place.** Grounding DINO finds each sound's source in the frames, and the camera poses lift it into the same metric world frame as the splat.
- **Compose with SATIE.** The placed sources go to SATIE's MCP server, which writes a soundscape from the user's own recordings. The viewer plays it with `satie-runtime`, with each source anchored at its position and the camera as the listener, so the sound follows you as you move.
- **Walk through it.** The viewer renders the splat with Spark. It streams level-of-detail trees on desktops and phones, and picks a quality tier from your frame rate.
- **Find memories on the globe.** The homepage is a low-poly globe of memories, each pinned where it was filmed.

## How it works

```
                 ┌─> poses ─┬──────────> splat ─────────────────┐
video ─> ingest ─┤          └─> localize ──> soundscape ───────┴─> publish ─> manifest ─> viewer
                 └─> separate ──┘                (SATIE)
```

Each stage runs in its own process, writes only under `{memory_id}/{stage}/` and is safe to re-run. `publish` validates and writes the **memory manifest** ([`packages/manifest`](packages/manifest/memory.schema.json)). The manifest is the contract between the pipeline and the viewer: it ties the splat, the sound sources and the location to one metric coordinate frame. The full design is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), with a diagrammed version in [docs/architecture.html](docs/architecture.html).

## Repository layout

| Path | What lives there |
|---|---|
| [`apps/web`](apps/web) | Next.js 15 + React 19 client: globe (`/`), upload (`/upload`), viewer (`/memories/[id]`) with Spark and `satie-runtime` |
| [`services/api`](services/api) | FastAPI: create, upload and process memories, track progress, globe queries, serve a memory's files (SQLite) |
| [`services/workers`](services/workers) | The pipeline: one module per stage in `stages/`, SATIE behind `providers/soundscape.py` |
| [`packages/manifest`](packages/manifest) | Memory manifest JSON Schema and its TypeScript mirror |
| [`tools/debugger`](tools/debugger) | Local Vite + three.js UI for inspecting each stage's outputs and checking the splat and audio agree |
| [`deploy`](deploy) | Setting up and running the site on a Windows PC with an NVIDIA GPU |
| [`infra`](infra) | Docker Compose for Postgres/PostGIS, Redis and MinIO, for a later move off SQLite |
| [`docs`](docs) | Architecture, stage contracts, deployment, render speed, spatial video |

## Running it locally

There are three parts: the **API** (with its SQLite database), the **web app**, and the **pipeline**, which the API starts by itself for each upload. You don't need Docker: the database is a SQLite file and files are stored on local disk.

You need Node 24, Python 3.12 with [uv](https://docs.astral.sh/uv/), and ffmpeg 7.1 or newer (`brew install ffmpeg` or `winget install Gyan.FFmpeg`). On Windows, use `.venv\Scripts\...` wherever this says `.venv/bin/...`.

### 1. API: http://localhost:8000 (interactive docs at `/docs`)

```bash
cd services/api
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m app.seed               # optional: demo memories on the globe (safe to re-run)
.venv/bin/uvicorn app.main:app --reload
```

The database is created at `services/api/data/memoryscapes.db` (set `DATABASE_URL` to put it elsewhere). Uploads and pipeline outputs go under `services/workers/data/storage/<memory id>/` (`STORAGE_ROOT`).

### 2. Web app: http://localhost:3000

```bash
npm install
npm run dev:web
```

The web app talks to the API at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

### 3. Pipeline

```bash
cd services/workers
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev,visual,audio,satie]"
.venv/bin/memoryscapes-worker setup-models   # the sound models, about 5.6 GB, once
.venv/bin/memoryscapes-worker satie-login    # sign in to SATIE, once per machine
```

For each upload, the API runs `python -m memoryscapes_workers.run` here (set `WORKERS_PYTHON` to use another Python). Splat training needs an NVIDIA GPU to be fast. On Windows, gsplat has to be built from source once: see [services/workers/README.md](services/workers/README.md#gpu-splat-training-on-windows). To try the pipeline without uploading anything, run `memoryscapes-worker demo`, which processes a synthetic video.

### Debugger

```bash
cd tools/debugger && npm install && npm run dev   # http://localhost:5178
```

It shows each stage's outputs for a memory (frames, poses, splat, training report, stems, source positions), checks that the splat and the poses share one frame, and can upload a video and run the pipeline on it.

### Tests

```bash
cd services/api && .venv/bin/python -m pytest
cd services/workers && .venv/bin/python -m pytest
npm test --workspace @memoryscapes/web        # also: npm run typecheck --workspace @memoryscapes/web
cd tools/debugger && npm test
```

The tests use fake models and synthetic scenes with known ground truth, so they don't need torch, a GPU or a network connection.

## Deploying

The site runs on one Windows PC with an NVIDIA GPU. Run `deploy\windows\setup.cmd` to set up the virtual environments, CUDA torch, the models, Spark's `build-lod` and the web build. Then `deploy\windows\start.cmd` runs the API and the web app on port 3000 and restarts them if they fail. See [docs/DEPLOY_WINDOWS.md](docs/DEPLOY_WINDOWS.md).

## SATIE

All sound in MemoryScapes comes from [SATIE](https://www.satie.world) by Attractor Labs, and the viewer shows the "Sound by SATIE" badge its licence requires. By default SATIE only arranges the user's own recordings: `SATIE_MAX_CENTS=0` and `SATIE_ALLOW_GENERATION=false`. Each `soundscape` run uses one composition from the account's monthly allowance. If SATIE fails or would go over budget, the memory still publishes without sound, and the viewer says why.

## Status

| Area | State |
|---|---|
| Pipeline | All seven stages run end to end on real phone video, including SATIE soundscapes and streaming LoD trees |
| API | Upload, process, progress, memory details, globe and file serving work. Live progress (SSE), editing (`PATCH`) and reports return `501` |
| Web app | Globe, upload and viewer work on desktop and phone |
| Not yet | Public sharing (it needs the face and number-plate blur and speech check in `publish`), login, Postgres and cloud storage |

Memories are private for now: nothing becomes public until the privacy pass exists.

## Documentation

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The whole system design and build phases |
| [services/workers/README.md](services/workers/README.md) | Stage data contracts, models, running stages, SATIE details |
| [SPLAT_OUTPUTS.md](docs/SPLAT_OUTPUTS.md) | What the poses and splat stages write |
| [SPATIAL_VIDEO.md](docs/SPATIAL_VIDEO.md) | iPhone spatial video: scale, right-eye views, stereo depth |
| [RENDER_SPEED.md](docs/RENDER_SPEED.md) | Viewer frame rate: quality tiers, LoD trees, measurements |
| [DEPLOY_WINDOWS.md](docs/DEPLOY_WINDOWS.md) | Setting up and running the site on the Windows PC |
| [webapp-plan/](docs/webapp-plan/README.md) | Web app plan: frontend, backend, database, storage |

## License

[MIT](LICENSE)
