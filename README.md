# MemoryScapes

Turn a video (or a set of panoramas) and an audio recording into a **memory**: a Gaussian splat you can walk through, with the recorded sound separated, placed in 3D and rebuilt into a soundscape by SATIE (Attractor Labs). Publish memories to a shared globe and explore other people's.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system design.

## Repository layout

```
apps/web/            Next.js client: globe, upload, memory viewer, spatial audio
services/api/        FastAPI: auth, memories, presigned uploads, SSE progress, globe queries
services/workers/    Processing pipeline (ingest → poses → splat, separate → localize → soundscape → publish)
packages/manifest/   Memory manifest JSON Schema + TypeScript types
infra/               Docker Compose for Postgres/PostGIS, Redis, MinIO; DB schema
docs/                Architecture
```

## Local development

Start the local dependencies (needs Docker):

```bash
cp .env.example .env
npm run infra:up
```

Run the API:

```bash
cd services/api && pip install -e ".[dev]" && uvicorn app.main:app --reload
```

Run the web app:

```bash
npm install && npm run dev:web
```

Everything beyond the skeleton is stubbed: API endpoints return `501` and pipeline stages raise `NotImplementedError`.

## Status

Scaffold only. Build order is in [docs/ARCHITECTURE.md § 9](docs/ARCHITECTURE.md#9-build-phases).
