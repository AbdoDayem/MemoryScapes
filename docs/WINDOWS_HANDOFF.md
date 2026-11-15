# Handoff: deploy and check the latest work on the Windows PC

This is for the agent (or person) taking over on the Windows + NVIDIA PC. Work on the Mac stopped on 2026-11-14 with everything merged to `main`. What's left needs the PC: the real deploy, the GPU and the real hardware numbers.

- Read [`CLAUDE.md`](../CLAUDE.md) first.
- [`DEPLOY_WINDOWS.md`](DEPLOY_WINDOWS.md) is the full setup guide.

This page covers what changed, what to run, what to check, and where to record results.

## What changed (all merged)

| PR | What | Details |
|---|---|---|
| #26 | **Spatial video → better splats.** An iPhone spatial video's second eye gives true scale, extra right-eye training views and stereo depth maps. | [`SPATIAL_VIDEO.md`](SPATIAL_VIDEO.md) |
| #27 | **Audio keeps the filmer's voice.** Wind cleaning no longer cuts the speech band on iPhone audio. Speech always gets a separation slot, and "too quiet" is now relative. Unplaced speech becomes narration, heard everywhere. | `services/workers/README.md` (separate, localize) |
| #28 | **Faster viewer.** Phones stop at lod1. Spark LoD with quality tiers is chosen from frame times, and there's a `?perf` readout. | [`RENDER_SPEED.md`](RENDER_SPEED.md) |
| #29 | **Streaming LoD trees.** Publish builds `.rad` trees with Spark's `build-lod` (Rust). Since #36, of lod0 for desktops and of lod1 for phones, so phones stream too. | [`RENDER_SPEED.md`](RENDER_SPEED.md), `services/workers/README.md` § "Publish: the streaming LoD trees" |
| #33 | **Blur matches the training.** The manifest's `splat.antialiased` picks Spark's blur. Classic training, the default, had been drawn slightly dimmed and thinned. | [`RENDER_SPEED.md`](RENDER_SPEED.md) |
| #38 | **Exposure compensation (opt-in).** `splat.train.exposure_comp` learns a colour transform per training frame, absorbing auto exposure and white balance. It's off until run D of the A/B (step 6) measures it. The train report gains `psnr_cc` and `exposure_gains`. | [`SPATIAL_VIDEO.md`](SPATIAL_VIDEO.md) § "To improve or fix" 5 |

## Rules that matter here

- **SATIE costs compositions and money.**
  - Keep `SATIE_MAX_CENTS=0` and `SATIE_ALLOW_GENERATION=false`, and never raise them. SATIE only arranges the recorded sounds.
  - Each run of the `soundscape` stage uses one composition (the Free plan has 3 a month). Few are left for November.
  - **Ask the owner before running `soundscape`.** Re-running `publish` doesn't call SATIE.
- **No paid services** (no cloud GPUs); the owner won't pay for compute.
- **Never commit** `.env`, `.satie\`, or pipeline outputs (`data\`, `*.ply`, `*.spz`, `*.rad`, `*.radc`).
- **Ask the owner before merging a PR**, and before making anything public. Nothing becomes public before the privacy pass (face and number-plate blur) exists.

## 1. Update and set up

From the repo folder, in a terminal:

```bat
git pull
winget install Rustlang.Rustup
```

Open a **new terminal** so `cargo` is on `PATH`, then:

```bat
ffmpeg -version
deploy\windows\setup.cmd --skip-models
```

- **`ffmpeg -version`:** must be 7.1 or newer, which decoding the second eye of a spatial video needs. Otherwise `winget upgrade Gyan.FFmpeg`.
- **First setup on this PC:** run `deploy\windows\setup.cmd --build-gsplat` instead, which also downloads the models. Add `--build-gsplat` again whenever torch or gsplat change version.

Setup now also builds Spark's `build-lod` into `services\workers\.tools\`. That takes about 2 minutes, once per Spark version.

The checks at the end should say:

```
  torch ..., CUDA yes: <your GPU>, splat trainer: gsplat
  build-lod: v2.4.0
  SATIE: signed in
```

| If it says | Do |
|---|---|
| `build-lod: not installed` | Rust is missing, or its linker is. Install the VS Build Tools C++ workload (`DEPLOY_WINDOWS.md` § 1), open a new terminal, and re-run setup. |
| trainer isn't `gsplat` | `deploy\windows\setup.cmd --build-gsplat --skip-models` |
| SATIE not signed in | `services\workers\.venv\Scripts\memoryscapes-worker satie-login` |

## 2. Start the site

```bat
deploy\windows\start.cmd
```

Open http://localhost:3000. The logs are `logs\api.log` and `logs\web.log`.

## 3. Give existing memories a streaming tree

Memories published before `build-lod` was installed don't have trees, and memories published before per-level trees have only lod0's, so phones can't stream them. Republishing fixes both, and records the training mode for the blur (`splat.antialiased`). New memories processed through the site get both trees automatically: publish builds them, and the API copies them into the database when the run finishes.

For each existing memory (the folders in `services\workers\data\storage\` with a `publish\manifest.json`):

```bat
cd services\workers
.venv\Scripts\memoryscapes-worker run <id> --stages publish
cd ..\api
.venv\Scripts\python -m app.memory_files store <id>
cd ..\..
```

- The first command rebuilds only publish: no training and no SATIE.
- The second updates the database copy that the site serves for memories uploaded through it. For a memory made with the worker CLI it says "no database row", which is fine: the site serves those straight from storage, so the first command is enough.
- A memory without ingest, poses or splat outputs (some demo folders) can't be republished; skip it.

**To check:** open `http://localhost:3000/memories/<id>?perf`. The status line should end in "streamed".

## 4. Measure render speed

The tiers in `apps/web/src/components/viewer/renderQuality.ts` were tuned on a MacBook only. Measure them on the PC:

```bat
cd services\workers
.venv\Scripts\python examples\render_benchmark.py
```

This writes a synthetic 1.5M-gaussian memory with its LoD tree. Follow [`RENDER_SPEED.md`](RENDER_SPEED.md) § "How to measure":
1. Open `http://localhost:3000/memories/render_bench?perf`.
2. Note where auto mode settles, and the "first splat" and "loaded" times.
3. For each fixed tier (`&quality=high` … `&quality=lowest`), record fps and GPU ms, standing and walking (hold W).
4. Compare load times with `&stream=0`.

Then do the same on a real memory, and on a phone on the same network. Start the site with `start.cmd --host 0.0.0.0` for that. Phones stream lod1's tree; compare with `&stream=0`, which loads the `.spz` levels. On the PC, `&device=mobile` tries the phone path.

**Record the results in `RENDER_SPEED.md`:**
- GPU, monitor resolution and refresh rate, browser;
- the numbers from the steps above.

If the PC holds 60 fps at high with headroom, or phones struggle at medium, adjust `TIERS`, `GOVERNOR_DEFAULTS.slowMs` or `MAX_LOD_SPLATS` and update `renderQuality.test.ts`.

## 5. Check the audio fix on the owner's clips

The owner reported "none of my words can be heard" on two iPhone 16 Plus clips. Both are at night, with the filmer talking throughout:
- `IMG_9660.MOV`, a normal video;
- `IMG_9661.MOV`, a spatial video.

They aren't in the repo; ask the owner for them.

```bat
cd services\workers
.venv\Scripts\memoryscapes-worker import clip9660 C:\path\to\IMG_9660.MOV
.venv\Scripts\memoryscapes-worker run clip9660 --until poses
.venv\Scripts\memoryscapes-worker run clip9660 --stages separate,localize
```

These steps make no SATIE calls. Do the same for `IMG_9661.MOV` as `clip9661`.

What it should show, going by the Mac runs:
- **Speech is separated.** In `separate\stems.json` there's a speech stem, about −53 dBFS on 9660 and −58 on 9661. Listen to `separate\stems\*speech*.wav`: the voice should be clear.
- **Speech becomes narration.** In `localize\sources.json`, the speech source has `"position": null` and `"method": "narration"`.
- **Other sounds are placed:**
  - 9660: construction, footsteps, music, bells;
  - 9661: train, bells.
  - Airplane, train and siren may be dropped as redundant.

Only with the owner's OK, because it uses a SATIE composition, finish one clip with `run clip9660 --stages soundscape,publish`. Then open `/memories/clip9660`, start the sound, and walk around: the voice should stay audible everywhere.

## 6. Spatial video: the splat A/B

This is still unmeasured; the Mac was too slow to train. Follow [`SPATIAL_VIDEO.md`](SPATIAL_VIDEO.md) § "What to run on the Windows machine" on `IMG_9661.MOV`:
- **A:** left eye only;
- **B:** both eyes;
- **C:** both eyes + depth loss;
- **D:** both eyes + per-image exposure compensation (`exposure_comp`). Also try it on `IMG_9660.MOV`: night clips change exposure between frames.

Compare held-out PSNR (`psnr_cc` for D) and `depth_error_vs_sfm`, record them on that page, and set the defaults (`stereo_views`, `depth_lambda`, `exposure_comp`) from the result. Its "To improve or fix" list is the next work after that.

## Known issues

- **No login yet:** anyone who can reach port 3000 can see every memory and start processing. Keep the site on this PC or a trusted network.
- **Chrome's battery saver** caps pages at 30 fps. When measuring, plug in, or go by the GPU ms in `?perf`.
- **A memory's SATIE sample list** stores `/api/...` links (`PUBLIC_API_URL=/api`). They work through the site on port 3000, but not when a dev web server talks to the API directly on port 8000.
- **4-channel spatial audio:** the iPhone's APAC track isn't used. It decodes only on macOS, and its directions didn't follow the camera in the test clips.

## When you're done

Put results and any tuning on a branch, open a PR, and ask the owner before merging. Update this page, or remove it, once it's out of date.
