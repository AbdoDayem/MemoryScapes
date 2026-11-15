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

1. **The device picks the top LOD.** Phones and tablets stop at the most detailed LOD with at most 600k gaussians (lod1, 500k at SH1), so they never download or draw lod0. Desktops load every LOD, coarsest first, as before.
2. **The top LOD loads as a Spark LoD tree** (`new SplatMesh({ lod: true })`).
   - Spark builds the tree in a background worker.
   - It then draws at most a splat budget: 2.5M on desktop, 1.5M on iOS, 1M on Android, times `lodSplatScale`.
   - Distant detail merges into splats no smaller than `lodRenderScale` pixels.
   - The previous LOD stays on screen until the tree draws, so the scene never flashes empty.
3. **Quality tiers:**

   | Tier | Max pixel ratio | Splat budget × | Smallest splat | SH degree | maxStdDev |
   |---|---|---|---|---|---|
   | high | 2 | 1 | 1.5 px | 3 | √8 |
   | medium | 1.5 | 0.4 | 2 px | 3 | √8 |
   | low | 1 | 0.25 | 3 px | 1 | √7 |
   | lowest | 0.75 | 0.15 | 4 px | 0 | √6 |

   Desktops start at high and phones at medium.
4. **A governor picks the tier from frame times.**
   - **Stepping down:** below 45 fps (judged over 1 s windows, ignoring the slowest 10% of frames), it steps down a tier.
   - **Stepping back up:** after holding the display's full rate for 5 s, it steps back up.
   - **Limited retries:** a tier that failed waits 4× longer before it's retried, and isn't retried after failing twice.
   - **Slow displays:** it measures the display while the splat loads, so a browser capped at 30 fps (Chrome's battery saver) doesn't push it to the lowest tier.
5. **URL switches** for measuring:
   - `?perf` shows frame rate, GPU time per frame, the tier, splats drawn and pixel ratio. GPU time comes from `EXT_disjoint_timer_query_webgl2`; it's the useful number when the frame rate sits at the display's cap.
   - `?quality=high|medium|low|lowest` fixes a tier.
   - `?lod=N` forces the top LOD.

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

**Load-time cost:** the LoD tree takes about 3 s per 1.5M gaussians to build in the browser. Full detail arrived at 7.8 s instead of 4.9 s; the 500k LOD shows in the meantime. Building the tree offline (next steps, item 1) removes this wait.

## How to measure on the Windows PC (and phones)

```bat
cd services\workers
.venv\Scripts\python examples\render_benchmark.py
```

Then, with the API and web app running:
1. Open `http://localhost:3000/memories/render_bench?perf`.
2. Note where auto mode settles.
3. Open each fixed tier (`&quality=high` … `&quality=lowest`) and record the fps and GPU ms, standing and walking (hold W).

Also try a real memory. For phones, open the same URL from the phone on the same network; phones start at medium and lod1.

With `?perf`, `window.memoryscapesViewer` exposes `{ renderer, scene, camera, spark, setTier(i) }` in devtools, for timing frames by hand.

Record the numbers here. The tiers, the 45 fps threshold and the 600k phone cap in `renderQuality.ts` were set from the Mac alone.

## Next steps, in order

1. **Build the LoD tree offline in the splat stage.**
   - Use Spark's `build-lod` tool (Rust; `npm run build-lod -- lod0.ply --quality`, `--rad-chunked` for streaming). It writes a `.rad` file.
   - The viewer would load it with `new SplatMesh({ url, paged: true })`. That gives:
     - no ~3 s tree build in the browser;
     - streaming (the first chunk is a coarse whole scene);
     - a better merge (`--quality`).
   - This needs:
     - Rust on the deploy PC (`deploy/setup.py`);
     - an optional `splat.rad` field in the manifest (`memory.schema.json` and `index.ts` together);
     - a viewer branch for it.
   - Keep the `.spz` LODs for older viewers and the debugger.
2. **Train fewer, better gaussians.**
   - Real captures are capped at 1.5M only by `make_lod`.
   - Try gsplat's MCMC strategy with `cap_max` around 1M, or stronger opacity pruning.
   - Compare held-out PSNR against count with the existing A/B setup (`splat.train.holdout_every`).
3. **SH degree of lod0.** Try SH2 and SH1 on real phone captures. If they look the same, ship the lower degree: a smaller download and less work per splat.
4. **Match the trainer's blur.**
   - The trainer's default is gsplat "classic" (`antialiased: false`). For that, Spark should use `preBlurAmount: 0.3, blurAmount: 0`.
   - Spark's default `blurAmount: 0.3` is meant for anti-aliased training, and slightly dims fine detail.
   - This needs a `splat.antialiased` flag in the manifest. It improves quality, not speed.
5. **Tune the tiers on real devices** from the measurements above.
