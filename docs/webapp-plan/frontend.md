# Frontend plan (web app)

Part of the [web app plan](README.md). This file covers the pages and components in
`apps/web`. The endpoints they call are defined in [`backend.md`](backend.md) § API contract.
Don't repeat them here.

## What exists

Next.js 15, React 19, TypeScript. Run with `npm run dev:web` from the repo root
(`localhost:3000`), with the API on `localhost:8000` (`NEXT_PUBLIC_API_URL`).

| Route | File | State |
|---|---|---|
| `/` | `src/app/page.tsx` → `components/home/HomePage.tsx` | **Built (2b-frontend):** the low-poly globe of memories, preview card, "My memories", upload panel |
| `/upload` | `src/app/upload/page.tsx` → `HomePage` with the upload panel open | **Built** |
| `/memories/[id]` | `components/viewer/MemoryViewer.tsx` | **Marko's:** the 3D scene with SATIE sound; stage list and stopping on `failed` (PR #11) |

Scripts: `npm run typecheck`, `npm test` and `npm run build` (each with
`--workspace @memoryscapes/web`). CI: `.github/workflows/web.yml`.

**Ours:** the globe homepage, the preview card, the upload panel. **Not ours:** `MemoryViewer`
and its 3D scene and sound (Marko).

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
| Globe | **globe.gl** (built on three.js): land polygons, rings, HTML labels, custom three.js objects, fly-to. Its scene is open for our own objects (ocean, clouds, markers). |
| Viewer (Marko's) | three.js + Spark |
| Styling | One scoped stylesheet (`components/home/home.css`, everything under `.ms-home`, so the viewer isn't affected); Nunito via `@fontsource/nunito` (bundled, no Google Fonts at runtime) |

Why globe.gl: it is three.js, with the globe parts (lat/lon on a sphere, atmosphere, camera
flights) done. Raw three.js would rebuild all that; CesiumJS (terrain, GIS) is heavy for
pins on a globe; Mapbox's globe needs an API key and is priced by usage.

## Page: `/` (home: the globe)

The homepage is a full-screen 3D globe of every memory, with the upload entry on top of it.
The look was chosen from browser mockups (night Earth → light beams → **cute low-poly**),
because the team wanted it cute and fun to play with.

```
┌───────────────────────────────────────────────────────────────┐
│ MemoryScapes  immerse in your memory          [+ New memory]  │
│        ☁            (soft blue → peach sky)        ☁          │
│               ╭──────── faceted pastel ocean ───────╮         │
│             ╭─┘  🟩🟨 chunky country tiles 🟪🟩     └─╮       │
│            │    °o○  ← bubbles     ┌──────────────┐  │        │
│            │     📍 Golden Gate     │ Golden Gate  │  │        │
│            │  🟡3 (cluster)         │ walk         │  │        │
│             ╰─╮                     │ ● Ready      │╭─╯       │
│               ╰─────────────────────│[Enter memory]│╯         │
│  [≡ My memories (11)]               └──────────────┘          │
└───────────────────────────────────────────────────────────────┘
```

**Look** (`components/globe/world.ts`):
- A soft daytime sky gradient (blue to peach) behind a transparent canvas.
- A **faceted low-poly ocean** (pastel blue icosahedron with a slight wobble).
- **Chunky, flat-shaded country tiles** in soft greens, yellows and lavender, at a few heights.
  Each country always gets the same colour. No pinks or peaches: they'd blend with the pins.
  Shapes: `public/globe/countries-110m.geojson` (Natural Earth 1:110m, public domain).
- Little low-poly **clouds** drifting around the planet; a white atmosphere.
- Slow auto-rotation until the user grabs the globe.

**Memories: a small pin blowing soap bubbles** (`components/globe/markers.ts`):
- A small low-poly pin (45% size, short stick) with a white outline, a shine and a soft glow.
- Low-poly **bubbles** (80 flat facets, pastel rainbow, a blocky white shine, 25% opacity)
  float up from the pin's head, wobble and pop.

| Status | Pin | Bubbles |
|---|---|---|
| `ready` | Coral, gently bobbing, ripple rings | 6 |
| `draft` / `uploading` / `processing` | Blue, hopping and spinning | 3, slower |
| `failed` | Grey, drooping | none |
| cluster | Bigger and golden, a white count badge | 9 |

- **Findable at any zoom:** markers keep about the same size on screen, glow, and have a
  large invisible hit area. Name tags appear when zoomed in.
- **Play:** hover grows a marker; a click squashes it like jelly; tapping land or sea makes
  a white splash.
- **Clusters** (`components/globe/clusters.ts`): memories merge only when they'd overlap on
  screen (the merge distance shrinks with camera altitude), so they come apart exactly when
  you can pick them. Clicking a cluster zooms in just far enough to separate it. Memories at
  practically the same spot never separate: clicking opens a **"N memories here"** list.
- **"My memories"** (bottom left): every memory, including ones without a location. It's the
  keyboard / screen-reader / no-WebGL way in.
- **Empty:** "No memories yet. Tap “+ New memory” to add your first one."

**Data:** `queryGlobe()` + `listMemories()` on load and after any change. While a memory is
**processing**, the page refreshes every 8 s; markers are only rebuilt when something visible
changed. `scene.ts` (globe.gl) touches `window`, so `MemoryGlobe.tsx` imports it inside an
effect.

## Preview card (`components/globe/MemoryCard.tsx`)

- Title, capture (or added) date, location, status.
- **Enter memory** (ready) or **View progress** (processing) opens `/memories/{id}`.
- **Failed:** the reason (`error`) and **Try again** (`startProcessing`).
- **Edit title**, and **Move pin** / **Place on globe** (then tap the globe), saved with
  `updateMemory`. "Locations are rounded to about 100 m."
- **No location yet** and the video had GPS (`suggested_location`): **Use the video's
  location** (`updateMemory` with `location_source: "video_gps"`).

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
A retry reuses the same draft and skips files that already finished. Cancel (or closing the
panel) stops everything: nothing uploads or starts processing afterwards. A cancelled upload
leaves a `draft` memory in "My memories".

**Known limit:** retrying with a *different video file* creates a new draft and leaves the old
one, because the API has no `DELETE /memories/{id}` yet.

## Page: `/memories/[id]` (the viewer)

Marko's `MemoryViewer.tsx`. It loads the memory, polls while processing, shows the stage list
(from `stage_detail`), stops on `failed`, and shows the 3D scene once `ready` (PR #11). The
upload panel reuses its stage helpers (`components/viewer/status.ts`). Editing title and
location happens on the globe's preview card, not on this page.

Seed memories (`python -m app.seed`) have no files, so their viewer waits forever. Real
uploads don't have this problem.

## Layout and navigation

`src/app/layout.tsx`: no global header. The globe page has its own overlay (name, "+ New
memory", "My memories"). The viewer page stays full screen. **Later:** a small "← Globe"
link on it, agreed with Marko.

## Files

| File | What |
|---|---|
| `components/home/HomePage.tsx`, `home.css` | The page: state, polling, overlays, styles |
| `components/globe/MemoryGlobe.tsx` | React wrapper; loads `scene.ts` in the browser |
| `components/globe/scene.ts` | globe.gl: world, markers, clusters, tags, ripples, placing pin, animation, clean-up |
| `components/globe/markers.ts` | The pin-and-bubbles marker |
| `components/globe/world.ts` | Ocean, country tiles, clouds, lights |
| `components/globe/clusters.ts` (+ `clusters.test.ts`) | Clustering by on-screen distance, as plain functions |
| `components/globe/format.ts` | Status colours and labels, dates (no three.js) |
| `components/globe/MemoryCard.tsx`, `MemoryList.tsx` | The preview card, "My memories" (also the "memories here" picker) |
| `components/upload/UploadPanel.tsx` | The upload steps |
| `src/lib/api.ts` | One function per endpoint, `ApiError` |
| `public/globe/countries-110m.geojson` | Country shapes |

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
