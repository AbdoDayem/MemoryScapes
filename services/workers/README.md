# MemoryScapes workers

The processing pipeline that turns an upload into a memory. The DAG is defined in
[`memoryscapes_workers/pipeline.py`](memoryscapes_workers/pipeline.py):

```
ingest ─┬─> poses ─┬──────────> splat ─────────────────┐
        │          └─> localize ──> soundscape ──────┴─> publish
        └─> separate ──┘
```

| Stage | Module | Hardware | Status |
|---|---|---|---|
| Ingest | `stages/ingest.py` | CPU | implemented |
| Camera poses | `stages/poses.py` | CPU (GPU optional) | implemented (COLMAP via pycolmap) |
| Splat training | `stages/splat.py` | GPU | implemented (gsplat on CUDA; PyTorch fallback on Mac) |
| Separation + labeling | `stages/separate.py` | GPU | implemented |
| Localization | `stages/localize.py` | GPU | implemented |
| Soundscape (SATIE) | `stages/soundscape.py` + `providers/` | external | implemented |
| Privacy + publish | `stages/publish.py` | GPU/CPU | stub |

## Setup

```bash
cd services/workers
python -m venv .venv
.venv/Scripts/activate   # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"                   # core + tests (numpy, scipy, soundfile)
pip install -e ".[audio,visual,satie]"    # model backends and the SATIE MCP client
pytest
```

The visual stages also need the `ffmpeg` binary. Tests that need ffmpeg, OpenCV, pycolmap or
torch skip themselves when those aren't installed (as in CI).

## Running stages locally

`memoryscapes-worker` (or `python -m memoryscapes_workers`) runs stages against local storage:
`STORAGE_ROOT`, default `data/storage`, or `STORAGE_BACKEND=s3` for MinIO/S3.

```bash
memoryscapes-worker demo                                   # synthetic video -> ingest -> poses -> splat
memoryscapes-worker import mem_1 clip.mp4 [--audio rec.wav]
memoryscapes-worker run mem_1 --until poses                # real inputs for the audio stages
memoryscapes-worker run mem_1 --stages separate,localize,soundscape
memoryscapes-worker run mem_1 --until splat --set splat.trainer=points   # seconds, no training
memoryscapes-worker run mem_1 --until splat --set splat.trainer=torch    # Mac (MPS); slow
```

`--until poses` turns any phone video into the inputs listed below, so the audio stages can be
tried on real footage. The splat stage writes the files in `docs/SPLAT_OUTPUTS.md` (`meta.json`,
`metrics.json`, `point_cloud.ply`, `renders/`, `lods/`), which is what `tools/debugger` reads. The visual stages skip themselves when their inputs and config haven't
changed (`--force` re-runs them). `splat.trainer=auto` (the default) uses gsplat on an NVIDIA
GPU, otherwise the PyTorch rasterizer on Apple Silicon, otherwise `points`.

## Windows + NVIDIA: a complete memory with spatial sound

This is the deploy path. From `services/workers`, in PowerShell:

1. Install Python 3.12, an NVIDIA driver, and ffmpeg on `PATH` (for example `winget install Gyan.FFmpeg`).
2. Install CUDA torch **first**: PyPI's torch wheels for Windows are CPU-only. Pick the CUDA build for your driver at https://pytorch.org/get-started/locally/, for example:
   `python -m venv .venv; .venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
3. `.venv\Scripts\pip install -e ".[dev,visual,audio]"`. The torch from step 2 is kept.
4. Get gsplat's CUDA kernels: a prebuilt wheel matching your torch and CUDA (see https://docs.gsplat.studio). Without one, gsplat compiles on first use, which needs the CUDA toolkit and Visual Studio Build Tools.
5. `.venv\Scripts\memoryscapes-worker setup-models --check`. This takes about 5.6 GB once.
6. Build a memory:
   ```
   .venv\Scripts\memoryscapes-worker import mem_1 C:\path\to\clip.mp4
   .venv\Scripts\memoryscapes-worker run mem_1 --until publish
   ```
   That runs ingest → poses → splat (`trainer=auto` picks gsplat on CUDA) → separate → localize → soundscape → publish.
7. Serve and view it: `cd ..\api; uvicorn app.main:app --port 8000`, then `npm run dev:web` from the repo root. Open http://localhost:3000/memories/mem_1 and press **Start sound**. Placed sources pan and get louder as you walk.

Settings go in `<repo>\.env`, which is git-ignored and loaded by the CLI; real environment variables win. For example `STORAGE_ROOT=D:\memoryscapes`. Keep `SATIE_MAX_CENTS=0` and `SATIE_ALLOW_GENERATION=false` unless the team agrees otherwise: SATIE should only arrange the recorded sounds.

To use SATIE on the server:
1. `.venv\Scripts\pip install -e ".[satie]"`.
2. Sign the server in once with its own SATIE account: `.venv\Scripts\memoryscapes-worker satie-login`. A browser opens on satie.world; after you sign in, it returns to `localhost` and the worker saves a refresh token to `<repo>\.satie\worker-token.json`. That file is git-ignored; keep it private, and set `SATIE_TOKEN_FILE` to store it elsewhere. Over SSH, or with the browser on another machine, use `satie-login --paste` and paste the `localhost` address the browser lands on.
3. Check it with `memoryscapes-worker satie-login --status`. It shows the plan and balance, never the token.
4. SATIE is the default provider (`SOUNDSCAPE_PROVIDER=satie`). The worker refreshes its access token by itself. If the login is revoked, the memory publishes without sound, and `soundscape/scene.json` and the viewer say to run `satie-login` again. A `SATIE_MCP_TOKEN` in the environment overrides the saved login.
5. The viewer's SATIE runtime loads each stem from the URL the worker gave SATIE. Set `PUBLIC_API_URL` to the API's address as viewers reach it (for example `https://memories.example.org`, not `localhost`), and the stems are linked as `<PUBLIC_API_URL>/memories/<id>/files/...`.

Capture tips that matter most:
- One continuous take. Edited videos are split into shots, and only the shots that join up are kept.
- Walk slowly; don't stand still or only turn in place.
- Don't film mostly sky or open water.

## Example memories

`examples/` has five procedural memories with known ground truth. Each one has synthesized sounds (birdsong, fountain, traffic, speech, music, a dog, bells, waves, wind), a camera path, a point cloud, and imperfect stand-in models: separation leakage, tag noise, and missed or false detections.

| Scenario | What it exercises |
|---|---|
| `park_walk` | Walking and panning past birds, a fountain and a road. The normal case. |
| `street_cafe` | Close speech, a music speaker, traffic, a dog; light reverb |
| `courtyard_static` | The camera never moves: bells above the frame can't be ranged and are left in the ambient bed |
| `beach_mono` | Mono mic: vision only, with diffuse wind and crowd |
| `hard_mode` | Heavy leakage, half the detections missed, false boxes, a false tag, reverb, noise, wrong mic spacing in the metadata |

```bash
.venv/Scripts/python examples/run_example.py            # all; writes examples/out/
.venv/Scripts/python examples/run_example.py park_walk  # one
```

Each run writes the full stage output, so you can listen to `ingest/audio.wav` and `separate/stems/*.wav`. It also writes stand-ins for the splat pipeline per [docs/SPLAT_OUTPUTS.md](../../docs/SPLAT_OUTPUTS.md): rendered PNG keyframes, a 3DGS `point_cloud.ply`, training metrics, splat renders, and `debug/truth.json`. That's everything the debugger (`tools/debugger`) needs. `hard_mode`'s splat is deliberately at half scale, so the debugger has an alignment failure to show. It also writes `report.md` (per-sound error, method, confidence, verdict) and `map.svg` (a top-down view of the true and estimated positions). The script exits non-zero if anything is placed confidently in the wrong spot. `tests/test_examples.py` runs the same scenes as regression tests, and CI (`.github/workflows/workers.yml`) runs both on every PR.

## Running on a real video

```bash
python -m memoryscapes_workers.run --root <data root> --memory <id> [--max-seconds 30] [--no-detector] [--only ingest,poses]
```

This is the debugger's upload runner: the whole DAG for one memory, with progress the UI can show. It expects the uploaded files in `<root>/<id>/upload/` (the debugger's upload page puts them there, alongside a `meta.json` the stages ignore). Each stage runs in its own process through `memoryscapes-worker run --in-process --result-file`, for the OpenMP reason in `cli.py`, and with `--force`, because a re-upload replaces files under the same names. Stages run in DAG order. Unimplemented stages are marked skipped, and stages whose dependencies failed are marked blocked, while the others carry on. Progress goes to `<id>/job/status.json`, and the full log to `<id>/job/log.txt`.

The CPU defaults are set in `run.stage_config`: at most `--max-keyframes 120` keyframes and SIFT on the CPU. The splat trainer is `auto`, so on a machine without CUDA it uses `points`.

`examples/make_video.py park_walk out.mp4` renders an example memory to a video with stereo sound and known source positions, for testing the upload path.

`examples/run_visual.py [scenario ...] [--steps 300]` makes **real-pipeline versions of the audio examples**, as `<scenario>_real` in `examples/out/`:
- It renders each scene to video and runs the real ingest, COLMAP poses and `torch` splat training. Keep `--steps` low on a CPU.
- It fits the similarity transform from the scene's true cameras to the recovered ones (rotation from orientations, scale and offset from positions) and moves the ground truth into the pipeline frame.
- It then runs the audio stages with the stand-in models on the real outputs, and writes `debug/truth.json`, `report.md` and `map.svg`.

hard_mode's splat is scaled by 0.5 afterwards for the debugger's alignment check. A scene whose SfM fails (`courtyard_static` turns in place) is reported, not forced.

## Models

`memoryscapes-worker setup-models` installs them all (~5.6 GB), the same on Windows, macOS
and Linux. Add `--check` to load each one and run it once. Each runs on the CPU, so no GPU
is needed. CUDA is used when present.

| Model | Used for | Where it goes |
|---|---|---|
| CLAP `laion/clap-htsat-unfused` | tagging which sounds are present | Hugging Face cache |
| AudioSep, pinned commit, plus 2 checkpoints (3.6 GB, SHA-256 checked) | separating each tagged sound | `<repo>/.models/AudioSep` (git-ignored; `AUDIOSEP_DIR` overrides) |
| RoBERTa `roberta-base` | AudioSep's text encoder | Hugging Face cache |
| HTDemucs | separating speech | torch / Hugging Face cache |
| Grounding DINO tiny | finding a sound's source in the frames (localization) | Hugging Face cache |

Two compatibility shims in `audio/separation.py` let AudioSep run on current torch and transformers without editing its code. One allowlists the NumPy types its checkpoints pickle, so they load under `weights_only=True` instead of turning that safety off. The other drops a stale `position_ids` key from its CLAP checkpoint.

## Audio stages: data contracts

All keys are relative to `{memory_id}/` in storage (`LocalStorage` in dev/tests, `S3Storage` deployed). Conventions: world frame in metres with +Y up; cameras in OpenCV convention (`c2w`); a stereo mic lies along the camera's +x axis (channel 0 = left).

**Inputs the audio stages read, written by ingest and poses** (`tests/test_visual_to_audio.py` runs the audio stages on real ingest/poses output):

| Key | Contents |
|---|---|
| `ingest/audio.wav` | Original recording, all channels, original sample rate |
| `ingest/meta.json` | `{"mic": {"channels", "layout": "mono" \| "stereo" \| "foa" \| "multichannel", "spacing_m"?}}`, plus media info, geo, capture time and the keyframe timeline. Only `stereo` enables DOA. |
| `ingest/frames/<id>.jpg` | Keyframe images |
| `poses/cameras.json` | `[{"frame_id", "t", "width", "height", "K", "c2w", "image_key"}]`, with `t` on the audio clock. `image_key` is the undistorted image in `poses/images/`, which `K` describes. |
| `poses/points3d.npy` | `(N, 3)` scene points, world metres |

**separate**: tags the mix with CLAP over sliding windows, separates each detected class (AudioSep, with Demucs vocals for speech), then turns the estimates into masks that partition the mix. The masks are applied to every channel, so stems + ambient equals the recording and the inter-channel cues survive for DOA. Diffuse classes (wind, rain, insects, crowd) stay in the ambient bed. Writes `separate/stems/<id>.wav`, `separate/ambient.wav` and `separate/stems.json`.

**localize**: for each stem, in order of preference:
1. **Audio-visual:** Grounding DINO looks for the class's prompt in the frames where the stem is loudest. Each box is lifted to 3D using the scene points inside it, then the lifts are clustered (at least 2 views).
2. **Stereo DOA:** GCC-PHAT bearings per window, triangulated along the moving camera path. This chooses between AV clusters, or places the source alone.
3. **Diffuse** otherwise.

Writes `localize/sources.json` with position, spread, confidence, method, and a level corrected to 1 m. Moving sources are placed at their dominant location for now.

**soundscape**: hands the sources to the configured provider.
- Sound is always SATIE's: the viewer plays it with `satie-runtime`, binding each host at its source's position.
- `SOUNDSCAPE_PROVIDER=satie` (the default) drives SATIE's MCP server (`satie_inspect_scene` → `satie_brief` → `satie_compose` → `satie_audit_scene` → `satie_realize`). Our sources are hosts with unchanged world positions, and our stems are clips named `memory/<id>`. The output is `soundscape/satie/scene.satie` and `samples.json`, for `satie-runtime`.
- Spending is capped by `SATIE_MAX_CENTS`, default 0. Retries reuse the same `requestKey`, so a re-run resumes the purchase instead of buying again.
- `SOUNDSCAPE_PROVIDER=stub` is for tests and offline work. It keeps the sources but writes no SATIE script, so the memory has no sound.
- If SATIE fails, the memory still publishes without sound, and `soundscape/scene.json` records why. Re-run `--stages soundscape,publish` to try again.
- **Quota:** each memory takes one SATIE composition, or two when the script needs a revision. The Free plan has 3 a month.

### SATIE against the live server (checked 2026-11-14)

A real memory (the Canada Water walk) went through the provider's steps on the live MCP:
`satie_inspect_scene` → `satie_brief` → `satie_compose` → `satie_check` → `satie_audit_scene` → `satie_realize`. The result was 8 of 8 sounds from our own files (`source: "provided"`), 0 generated, 0 from the library, and 0¢ spent.

- **Verified:**
  - The scene manifest is `satie.scene/v1`: `objects` hosted `at` our source ids, and our recordings as scene `samples`.
  - Scripts play a clip as `loop <clip>` with `at <host>` as an indented property.
  - `satie_realize` accepts our stem URLs as `samples`, free.
  - The fields the guards read: `clipsToSupply`, `libraryClips`, `takes` and per-voice `generated` from check and audit, plus `generated`, `fromLibrary`, `maxSpentCents` and per-sample `source` from realize.
- **Original sound only** (`SATIE_ALLOW_GENERATION=false`, the default):
  - The brief answers SATIE's open questions up front: no events, no weather or time beds, and no crowd bed. SATIE's defaults would otherwise add a library crowd bed.
  - Before realize, the script must buy 0 takes, use no library clips, and play only our clips. One revision is allowed; then nothing is realized and the stage uses the stub.
  - Realize always runs with `maxCents: 0` and `preferLibrary: false`, even if `SATIE_MAX_CENTS` is raised. After realize, anything not `provided` is rejected.
- **`saveSoundscapes` defaults to `off`.** SATIE's own default is a public gallery entry; `SATIE_SAVE_SOUNDSCAPES` changes it.
- **Training:** on the Free plan, SATIE keeps the scene and any audio it can fetch for training, and Free can't opt out. Paid plans can, with `satie_training`. Decide this before passing real users' recordings.
- **Sign-in:** SATIE's MCP server is an OAuth 2.1 resource. Its metadata names Supabase Auth, which supports dynamic client registration, PKCE (S256), `refresh_token` and `offline_access`. `satie-login` uses exactly that: a public client with no secret, then refresh tokens. Without a token the server answers 401 `invalid_token`, and the worker reports that as "not signed in". Tested with fake endpoints; the first real sign-in happens on the server.
- **Samples load from our URLs.** Every scene sample needs a `url`, and the runtime fetches the sound from it. An upload doesn't replace that.
- **Uploading the stems** (`SATIE_UPLOAD_STEMS=true`, off by default). After the original-only checks pass, each recording is uploaded through `satie_asset` (begin → PUT → finalize) under its `samples.json` key (`Audio/memory/<id>`). SATIE keeps it as a provided sample. Uploads are capped at 16 MB, so longer stems go up as MP3; this was tested live with a 46 s stem. A failed upload only adds a note.
- **Listening needs a paid plan.** `satie_listen` (its review of a ≤30 s, 16-bit WAV) answers "needs the Vibecoder plan" on Free, so the worker doesn't call it.

The model adapters (`ClapTagger`, `AudioSepSeparator`, `DemucsVocalsSeparator`, `GroundingDinoDetector`) follow those libraries' documented APIs but haven't been run here. The tests use fakes and synthetic scenes with known ground truth.
