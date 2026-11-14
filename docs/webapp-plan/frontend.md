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
`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`), `components/audio/SatieAudioEngine.ts` (sound is always SATIE's)
and `components/viewer/navigation.ts` (the viewer's). Scripts: `npm run typecheck` and
`npm test --workspace @memoryscapes/web`. CI: `.github/workflows/web.yml`.

**Ours:** the globe homepage, the preview card, the upload panel. **Not ours:** `MemoryViewer`
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
| `queryGlobe(bbox?)` | `GET /globe` (exists; add the return type). The globe asks for the whole world for now. |
| `subscribeProgress(id, onEvent)` | Later: `GET /memories/{id}/events` through `EventSource` |

Errors: throw an `ApiError` with the status code and the server's `detail`, so the UI can show it.

## Stack

| Layer | Tech |
|---|---|
| Framework | Next.js 15, React 19, TypeScript |
| Globe | **globe.gl** (built on three.js): Earth, atmosphere, points, rings, labels, fly-to. Its three.js scene is open for our own effects (stars, glow). |
| Viewer (Marko's) | three.js + Spark |
| Styling | Plain CSS modules |

Why globe.gl: it is three.js, with the globe parts (lat/lon on a sphere, atmosphere, camera
flights) done. Raw three.js would rebuild all that; CesiumJS (terrain, GIS) is heavy for
pins on a globe; Mapbox's globe needs an API key and is priced by usage.

## Page: `/` (home: the globe)

The homepage is a full-screen 3D globe of every memory, with the upload entry on top of it.

```
┌───────────────────────────────────────────────────────────────┐
│ MemoryScapes                                  [+ New memory]  │
│                         ✦     ✦        ✦                      │
│               ✦     ╭─────────────╮                           │
│                   ╭─┘ · city lights └─╮      ┌──────────────┐ │
│                  │   ●  Golden Gate    │      │ Golden Gate  │ │
│                  │  ◉3      ·    ●     │      │ walk         │ │
│                   ╰─╮   ·  night side╭─╯      │ Oct 2, 2026  │ │
│                     ╰───────────────╯         │ ● Ready      │ │
│             ✦  (soft blue atmosphere glow)    │[Enter memory]│ │
│                                         ✦     └──────────────┘ │
│  [≡ My memories]                                              │
└───────────────────────────────────────────────────────────────┘
```

**Look (night Earth):**
- Black space with sparse stars.
- The Earth's night side with city lights: NASA Black Marble (public domain), with a subtle
  bump map.
- A soft blue atmosphere glow.
- Slow auto-rotation until the user interacts. It pauses while a panel is open.
- Textures live in `apps/web/public/globe/`.

**Memories:**

| Status | Looks like |
|---|---|
| `ready` | Warm amber point with a slow pulse ring |
| `draft` / `uploading` / `processing` | Dimmer cool-blue point, faster pulse |
| `failed` | Grey point. Its card says why and offers "Try again". |

- **Hover:** a small label with the title and capture date.
- **Click:** the camera flies to it (`pointOfView`) and the preview card opens (below).
- **Zoomed out:** nearby points merge into a **cluster** with a count; clicking one zooms in.
  Clustering runs in the browser (`components/globe/clusters.ts`): group by a grid whose cell
  size shrinks with camera altitude.
- **"My memories"** (bottom left): a list of every memory, including ones without a location.
  It's the keyboard / screen-reader / no-WebGL way in. Clicking one flies to it, or opens it
  if it has no location.
- **Empty:** "No memories yet. Upload your first one."

**Data:** `queryGlobe()` on load and after an upload or edit (whole world: few memories for
now). `globe.gl` touches `window`, so load the globe only in the browser:
`dynamic(() => import(...), { ssr: false })`.

## Preview card (`components/globe/MemoryCard.tsx`)

- Title, capture date, status.
- **Enter memory** opens the viewer at `/memories/{id}`. While processing, it says "View
  progress" instead.
- **Failed:** the reason (`error`) and "Try again" (`startProcessing`).
- **No location yet** and the video had GPS (`suggested_location`): "Use the video's location"
  (`updateMemory` with that lat/lon).
- "Edit": title, description, and "Move pin" (click the globe), saved with `updateMemory`.
  Note: "Locations are rounded to about 100 m."

## Upload panel (`components/upload/UploadPanel.tsx`)

**+ New memory** (top right), or opening `/upload`, slides a panel in over the globe, which
stays live behind it. On phones it's a bottom sheet.

1. **Choose files:** one video (`accept="video/*"`), optionally one separate audio file. Drag
   and drop is a nice extra. Checks in the browser, with a hidden `<video>` reading the
   metadata:
   - At most 2 minutes. Say the pipeline currently uses the first 60 s (its default).
   - At most 500 MB (`MAX_UPLOAD_MB`).
   - Problems shown next to the file.

   Capture tips beside the picker: move slowly; walk around the scene instead of turning on
   the spot; keep sound sources in view; avoid wind on the mic.
2. **Title** (optional).
3. **Place it:** "Click the globe where this was filmed." A pin drops where you click; click
   again to move it. "Skip" is allowed: the video's GPS can be used later from the card.
4. **Upload:** `createMemory` → `uploadFile` per file (progress bar each, cancel button) →
   `updateMemory` with the pin, if any → `startProcessing`.
5. **Done:** the memory appears on the globe as a blue processing point. The panel shows
   live stage progress (polling `getMemory` every few seconds) and "View progress"
   (opens `/memories/{id}`, Marko's viewer).

**Errors:** show the server's `detail`, keep the picked files and pin, and offer retry.
Cancelling an upload leaves a `draft` memory, which can be retried from "My memories".

## Page: `/memories/[id]` (progress, then the viewer)

This page is **the viewer** (`MemoryViewer.tsx`, Marko's). It already loads the memory,
polls every 5 s while processing, and shows the 3D scene once `status` is `ready`. We don't
build a second page. We add two pieces to it, agreed with Marko:

**Marko's PR #11 does both of these** (stage list from `stage_detail`, stopping on `failed`).
What's left for us is checking it against real API responses.

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

Editing title and location happens on the globe's preview card, not on this page.
Keep these changes small and in separate components, so they don't get in the way of
Marko's work on the scene.

## Layout and navigation

`src/app/layout.tsx`: no global header. The globe page has its own overlay (name, "+ New
memory", "My memories"). The viewer page stays full screen with a small "← Globe" link.

## Files

| File | What |
|---|---|
| `components/globe/MemoryGlobe.tsx` | The globe: textures, atmosphere, stars, points, rings, clusters, fly-to, auto-rotate, pin placement |
| `components/globe/clusters.ts` (+ `clusters.test.ts`) | Grouping by zoom, as plain functions |
| `components/globe/MemoryCard.tsx` | The preview card |
| `components/globe/MemoryList.tsx` | "My memories" |
| `components/upload/UploadPanel.tsx` | The upload steps |
| `src/lib/api.ts` | New functions; after Marko's PR #11, which also edits it |
| `src/app/page.tsx`, `src/app/upload/page.tsx` | Home, and `/upload` opening the panel |
| `public/globe/` | Night-Earth texture, bump map |

`components/upload/UploadForm.tsx` (the placeholder) is replaced by `UploadPanel.tsx`.

## Checks

- `npm run typecheck --workspace @memoryscapes/web` (types)
- `npm test --workspace @memoryscapes/web` (existing viewer tests, plus ours: clustering)
- `npm run build --workspace @memoryscapes/web` (production build)
- CI runs these (`.github/workflows/web.yml`).
- By hand, in Chrome and Safari: upload a real phone video, refresh during processing, cancel
  an upload, try a too-long video, see a failed run stop with its error, see the 3D scene
  appear when ready, set a location, find the pin, click through.
- The viewer still works for a CLI memory: `memoryscapes-worker demo`, then
  `memoryscapes-worker run demo --stages publish`, then open `/memories/demo`.
