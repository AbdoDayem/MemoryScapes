# Frontend plan (web app)

Part of the [web app plan](README.md). This file covers the pages and components in
`apps/web`. The endpoints they call are defined in [`backend.md`](backend.md) § API contract.
Don't repeat them here.

## What exists on `main`

Next.js 15, React 19, TypeScript. Run with `npm run dev:web` from the repo root
(`localhost:3000`). Dependencies already include `three` and `globe.gl`.

| Route | File | Today |
|---|---|---|
| `/` | `src/app/page.tsx` → `components/globe/MemoryGlobe.tsx` | "Memory globe (not implemented)" |
| `/upload` | `src/app/upload/page.tsx` → `components/upload/UploadForm.tsx` | "Upload form (not implemented)" |
| `/memories/[id]` | `src/app/memories/[id]/page.tsx` → `components/viewer/MemoryViewer.tsx` | **Works** (Marko, PR #5): the 3D scene with its sound. While processing it polls `getMemory` every 5 s and shows a simple ✓/… list of stages. |

Also: `src/lib/api.ts` (`MemoryResponse` type, `getMemory`, `queryGlobe`, pointing at
`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`), `components/audio/SpatialAudioEngine.ts`
and `components/viewer/navigation.ts` (the viewer's). Scripts: `npm run typecheck` and
`npm test --workspace @memoryscapes/web`. CI: `.github/workflows/web.yml`.

**Ours:** upload, the progress panel, the details form, globe. **Not ours:** `MemoryViewer`
and its 3D scene and sound (Marko). Our pieces on `/memories/[id]` plug into his page, so
agree the changes with him (§ Page: `/memories/[id]`).

## `src/lib/api.ts`

One typed function per endpoint in `backend.md`, plus the response types. Every component goes
through these.

| Function | Endpoint |
|---|---|
| `createMemory(req)` | `POST /memories` |
| `uploadFile(url, file, onProgress)` | `PUT` to the returned `url`. Uses `XMLHttpRequest`, because `fetch` can't report upload progress. Supports cancel. |
| `startProcessing(id)` | `POST /memories/{id}/process` |
| `getMemory(id)` | `GET /memories/{id}` (exists). Add the new fields to `MemoryResponse` as **optional** and widen `status`, so the viewer's code keeps compiling. |
| `listMemories()` | `GET /memories` |
| `updateMemory(id, patch)` | `PATCH /memories/{id}` |
| `queryGlobe(bbox, zoom)` | `GET /globe` (exists; add the return type) |
| `subscribeProgress(id, onEvent)` | Later: `GET /memories/{id}/events` through `EventSource` |

Errors: throw an `ApiError` with the status code and the server's `detail`, so the UI can show it.

## Page: `/upload` (`UploadForm.tsx`)

1. **Pick files:** one video (`accept="video/*"`), optionally one separate audio recording
   (`accept="audio/*"`), and an optional title. Drag and drop is a nice extra.
2. **Check in the browser** before uploading, using a hidden `<video>` element to read the
   file's metadata:
   - Duration ≤ 2 minutes (the pipeline only processes the first 60 s by default; say so).
   - Has an audio track, if no separate audio file was picked. Browsers don't expose this
     reliably; warn instead of blocking when unsure.
   - Size ≤ 500 MB (matches `MAX_UPLOAD_MB`).
   Show problems inline next to the file.
3. **Capture tips**, always visible: move slowly; walk around the scene instead of turning on
   the spot; keep sound sources in view; avoid wind on the mic.
4. **Upload:** `createMemory` → `uploadFile` for each file, with a progress bar per file and a
   cancel button → `startProcessing` → go to `/memories/{id}`.
5. **Errors:** show the server's message and keep the picked files so the user can retry.

## Page: `/memories/[id]` (progress, then the viewer)

This page is **the viewer** (`MemoryViewer.tsx`, Marko's). It already loads the memory,
polls every 5 s while processing, and shows the 3D scene once `status` is `ready`. We don't
build a second page. We add two pieces to it, agreed with Marko:

1. **Progress panel**, new component `components/memory/StageProgress.tsx`. It replaces the
   viewer's small `StageList` (one line in `MemoryViewer.tsx`) while processing:
   - The 7 stages in pipeline order (ingest, poses, splat, separate, localize, soundscape,
     publish) from `stage_detail`, each with an icon for `pending`/`running`/`done`/`failed`/
     `skipped`/`blocked`, its time when done, and its error when failed. Show `detail` (the
     latest log line) under the running stage.
   - Falls back to the plain `stages` booleans when `stage_detail` is missing (CLI memories).
   - "This takes several minutes. You can leave and come back." A refresh picks up where it
     was, because the viewer re-fetches.
2. **Handling `failed`** in the viewer's polling: stop polling, show which stage failed and
   why, and a "Try again" button (`startProcessing`). Today any status other than `ready`
   means "keep polling", so a failed run would poll forever.

Plus the details form below, opened from a "Details" button in the viewer's corner panel.
Keep these changes small and in separate components, so they don't get in the way of
Marko's work on the scene.

## Details form (`components/memory/MemoryDetailsForm.tsx`)

Shown on the memory page as a small panel over the viewer, while processing or after.
Fields: title, description, location.
- **Location:** if the API returns `suggested_location` (from the video's GPS), offer "Use the
  video's location". Otherwise the user clicks a small map to drop a pin. A small globe.gl
  instance or a lat/lon input is enough for the first version.
- Say clearly: "Locations are rounded to about 100 m."
- Visibility: show "Private" only. Public and unlisted come with the privacy pass.
- Save with `updateMemory`.

## Page: `/` (globe, `MemoryGlobe.tsx`)

- globe.gl (already a dependency). It uses WebGL and `window`, so load it only in the browser:
  `dynamic(() => import(...), { ssr: false })`.
- On load and when the camera stops moving (debounce ~300 ms), work out the visible
  `bbox` and a `zoom` level from the camera altitude, then call `queryGlobe`.
- Draw `memories` as pins and `clusters` as larger points labelled with their count.
- Hover: title (and thumbnail later). Click a pin → `/memories/{id}`. Click a cluster → zoom
  in on it.
- Empty state: "No memories yet. Upload one" with a link to `/upload`.
- While building, use the seed data (`database.md` § Seed data).
- **Later (not needed now):** lines connecting nearby memories, e.g. each to its 2–3 nearest
  within N km. Compute them in the browser from the pins `/globe` already returns, and draw
  them with globe.gl's arcs layer.

## Layout and navigation

`src/app/layout.tsx`: a small header with links to Globe (`/`) and Upload (`/upload`), hidden
on the viewer page, which is full screen. Keep the styling simple (plain CSS modules). The
globe page is full screen too.

## Checks

- `npm run typecheck --workspace @memoryscapes/web` (types)
- `npm test --workspace @memoryscapes/web` (existing viewer tests, plus ours: the bbox/zoom
  maths is a good first one)
- `npm run build --workspace @memoryscapes/web` (production build)
- CI runs these (`.github/workflows/web.yml`).
- By hand, in Chrome and Safari: upload a real phone video, refresh during processing, cancel
  an upload, try a too-long video, see a failed run stop with its error, see the 3D scene
  appear when ready, set a location, find the pin, click through.
- The viewer still works for a CLI memory: `memoryscapes-worker demo`, then
  `memoryscapes-worker run demo --stages publish`, then open `/memories/demo`.
