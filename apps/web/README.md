# MemoryScapes web

Next.js app with three surfaces:

| Route | Component | Purpose |
|---|---|---|
| `/` | `components/globe/MemoryGlobe.tsx` | Globe of public memories |
| `/upload` | `components/upload/UploadForm.tsx` | Upload video or panoramas plus audio |
| `/memories/[id]` | `components/viewer/MemoryViewer.tsx` | Walk through a memory with spatial audio |

## Viewer

`MemoryViewer` loads the manifest from the API (`GET /memories/:id`) and resolves its
relative URLs against the returned `asset_base`.

- **Splat:** three.js + [Spark](https://sparkjs.dev) (`.spz`). LODs load coarsest first,
  each replacing the last. No transform is applied: the splat is already in the metric,
  +Y-up world frame that the poses and audio positions share.
- **Walking:** WASD/arrows (Shift runs, Q/E down/up), drag to look. Movement is clamped to
  `navigation.walkable_bounds` (box, corridor around the capture path, floor).
- **Replay capture:** follows `poses/path.json`.
- **Sound is always SATIE's.** `components/audio/SatieAudioEngine.ts` loads the memory's
  SATIE script and `samples.json` with `satie-runtime/three`. Each localized source is bound
  as a host at its position, and the camera is the listener.
  - The "Sound by SATIE" badge shows; SATIE's licence requires it. In development, SATIE's
    mixer opens from the badge (or the backquote key) with every voice as a dot in the world.
  - A memory without SATIE output plays no sound and shows why.
  - `satie-runtime` is installed from satie.world (pinned to 0.4.8 in `package.json`). It's
    proprietary, but its licence allows bundling it unmodified in our app.

## Try a memory locally

```bash
# 1. A memory: the synthetic demo, or your own video
cd services/workers
memoryscapes-worker demo                                # or: import mem_1 clip.mp4, run mem_1 --until splat
memoryscapes-worker run demo --stages publish           # writes publish/manifest.json

# 2. The API serves the manifest and files from services/workers/data/storage
cd services/api && uvicorn app.main:app --port 8000

# 3. The web app
npm install && npm run dev:web                          # from the repo root
# open http://localhost:3000/memories/demo
```

```bash
npm run typecheck --workspace @memoryscapes/web
npm test --workspace @memoryscapes/web                  # navigation tests (Node >= 23.6)
```
