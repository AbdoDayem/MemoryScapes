# Faster splat rendering: status and next steps

The viewer (`apps/web/src/components/viewer/`) draws a memory's splat with Spark 2.4. Before this change it always ended on the full lod0 (up to 1.5M gaussians, SH degree 3) at up to 2× pixel ratio, on every device. This page covers:
- what the viewer does now;
- what was measured;
- how to measure on the Windows PC and on phones;
- what to do next.

## What a frame costs

Every splat Spark draws is evaluated (position, colour from its spherical harmonics) and sorted, then rasterized over every pixel it covers. The levers are pixels and splat count. Pixels cost the most.

Measured on one frame of a 1.5M-gaussian scene (MacBook Pro M1 Pro, Chrome, 3024×1658 canvas), changing one lever at a time:

| Change from the full setting | Frame time |
|---|---|
| none (2× pixel ratio, 1.5M splats, SH3) | 47 ms |
| pixel ratio 2× → 1× | 25 ms |
| splats drawn 1.5M → 400k | 32 ms |
| SH3 → SH0 | 40 ms |

## What's built (viewer)

`renderQuality.ts` holds the logic and is unit-tested. `MemoryViewer.tsx` applies it.

1. **Desktops stream a prebuilt LoD tree** when the memory has one (`splat.rad`, built by the publish stage; see "Streaming LoD tree" below). The first chunk is a coarse view of the whole scene, then Spark fetches the detail the view needs. If the tree can't be fetched, or nothing draws within 15 s, the viewer falls back to the `.spz` levels.
2. **Otherwise the device picks the top `.spz` LOD.** Phones and tablets stop at the most detailed LOD with at most 600k gaussians (lod1, 500k at SH1), so they never download or draw lod0. Desktops load every LOD, coarsest first.
3. **The top `.spz` LOD loads as a Spark LoD tree** built in the browser (`new SplatMesh({ lod: true })`).
   - Spark builds the tree in a background worker.
   - It then draws at most a splat budget: 2.5M on desktop, 1.5M on iOS, 1M on Android, times `lodSplatScale`.
   - Distant detail merges into splats no smaller than `lodRenderScale` pixels.
   - The previous LOD stays on screen until the tree draws, so the scene never flashes empty.
4. **Quality tiers:**

   | Tier | Max pixel ratio | Splat budget × | Smallest splat | SH degree | maxStdDev |
   |---|---|---|---|---|---|
   | high | 2 | 1 | 1.5 px | 3 | √8 |
   | medium | 1.5 | 0.4 | 2 px | 3 | √8 |
   | low | 1 | 0.25 | 3 px | 1 | √7 |
   | lowest | 0.75 | 0.15 | 4 px | 0 | √6 |

   Desktops start at high and phones at medium.
5. **A governor picks the tier from frame times.**
   - **Stepping down:** below 45 fps (judged over 1 s windows, ignoring the slowest 10% of frames), it steps down a tier.
   - **Stepping back up:** after holding the display's full rate for 5 s, it steps back up.
   - **Limited retries:** a tier that failed waits 4× longer before it's retried, and isn't retried after failing twice.
   - **Slow displays:** it measures the display while the splat loads, so a browser capped at 30 fps (Chrome's battery saver) doesn't push it to the lowest tier.
6. **Blur matches the training** (`blurFor`, from the manifest's `splat.antialiased`).
   - Classic training (the trainer's default) is drawn with Spark's `preBlurAmount: 0.3, blurAmount: 0`, which is the fixed dilation the rasterizer applied in training.
   - Anti-aliased training gets Spark's default `blurAmount: 0.3`, which compensates opacity.
   - Spark's default on a classic splat dims and thins fine detail. A manifest without the flag counts as classic.
   - This affects image quality, not speed.
7. **URL switches** for measuring:
   - `?perf` shows:
     - frame rate and GPU time per frame (from `EXT_disjoint_timer_query_webgl2`; GPU time is the useful number when the frame rate sits at the display's cap);
     - the tier, splats drawn and pixel ratio;
     - when the first splat appeared and when the splat finished loading, counted from page start.
   - `?quality=high|medium|low|lowest` fixes a tier.
   - `?lod=N` forces the `.spz` path with lod N on top.
   - `?stream=1|0` forces streaming the tree on or off.

## Measured on the Mac

The scene is the synthetic 1.5M-gaussian street from `services/workers/examples/render_benchmark.py`. The machine is a MacBook Pro M1 Pro running Chrome on a 120 Hz display, with the window at 1512×829 CSS pixels.

| Setting | fps standing | fps walking | GPU ms | Splats drawn | Pixel ratio |
|---|---|---|---|---|---|
| before (main) | 25 | 23 | n/a | 1.5M | 2× |
| high | 25 | 22 | 38–51 | 1.5M | 2× |
| medium | 42 | 38 | 22–28 | 1.0M | 1.5× |
| low | 88 | 82 | 10–12 | 625k | 1× |
| lowest | 120 (display cap) | 120 | 6 | 375k | 0.75× |

Auto mode left high within 0.2 s of the full splat loading and settled on medium or low (45–90 fps) within 3 s.

**Load-time cost (`.spz` path):** the LoD tree takes about 3 s per 1.5M gaussians to build in the browser. Full detail arrived at 7.8 s instead of 4.9 s, with the 500k LOD on screen meanwhile. The prebuilt tree below removes this wait.

## Streaming LoD tree

The publish stage builds it with Spark's `build-lod`: Rust, MIT licence, installed by `deploy/setup.py` at the Spark version the web app pins. Details: `services/workers/README.md` § "Publish: the streaming LoD tree".

**Building it,** on the same 1.5M-gaussian scene (M1 Pro):

| Method | Time | Splats in the tree | Size |
|---|---|---|---|
| `quality` (bhatt-lod, the default) | 22 s | 2.31M (36 chunks) | 55 MB |
| `quick` (tiny-lod) | 2 s | 2.06M | 46 MB |

By comparison, the three `.spz` levels together are 43 MB (lod0 alone is 31 MB). The tree is bigger because it also stores the merged interior nodes. `build-lod --cluster-sh` (an SH codebook) ran for over 18 minutes on the CPU without finishing, so it isn't used. Its GPU build might be fast enough; untested.

**In the browser** (M1 Pro, Chrome, local server):

| | First splat | Full detail |
|---|---|---|
| `.spz` levels + tree built in the browser | 2.1 s (lod2) | 8.9 s |
| streamed tree | 2.1 s (coarse chunk) | about 5.6 s |

- **Frame rates** are the same either way; it's the same splat budget.
- **Phones keep the `.spz` levels.** Even at the lowest tier (375k splats drawn), Spark fetched 27 of the 36 chunks, and 31 after a 6 m walk. That's about 40 MB, against 12 MB for the lod2 + lod1 a phone loads. `?stream=1` streams on a phone anyway, for testing.
- **Checked on a real memory:** canada's lod0 (12k gaussians) became a 1-chunk, 0.3 MB tree. It streams through the API from the database copy, including the byte-range request Spark makes for the header.

## How to measure on the Windows PC (and phones)

```bat
cd services\workers
.venv\Scripts\python examples\render_benchmark.py
```

With `build-lod` installed, the script also builds the scene's LoD tree. Then, with the API and web app running:
1. Open `http://localhost:3000/memories/render_bench?perf`.
2. Note where auto mode settles, and the "first splat" and "loaded" times.
3. Open each fixed tier (`&quality=high` … `&quality=lowest`) and record the fps and GPU ms, standing and walking (hold W).
4. Repeat step 2 with `&stream=0` (the `.spz` levels with the tree built in the browser), to compare load times.

Also try a real memory. For phones, open the same URL from the phone on the same network; phones start at medium and lod1.

With `?perf`, `window.memoryscapesViewer` exposes `{ renderer, scene, camera, spark, setTier(i) }` in devtools, for timing frames by hand.

Record the numbers here. The tiers, the 45 fps threshold and the 600k phone cap in `renderQuality.ts` were set from the Mac alone.

## Next steps, in order

1. **Done: the streaming LoD tree** (above). Still open:
   - a smaller tree for phones, built from lod1 (500k, SH1), so they stream too;
   - GPU SH clustering (`build-lod` built with its `gpu` feature) to shrink trees.
2. **Train fewer, better gaussians.**
   - Real captures are capped at 1.5M only by `make_lod`.
   - Try gsplat's MCMC strategy with `cap_max` around 1M, or stronger opacity pruning.
   - Compare held-out PSNR against count with the existing A/B setup (`splat.train.holdout_every`).
3. **SH degree of lod0.** Try SH2 and SH1 on real phone captures. If they look the same, ship the lower degree: a smaller download and less work per splat.
4. **Done: match the trainer's blur** (`splat.antialiased`, above).
5. **Tune the tiers on real devices** from the measurements above.
