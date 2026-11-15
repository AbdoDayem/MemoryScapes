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

### GPU splat training on Windows

`pip install -e ".[visual]"` installs gsplat, but on Windows it doesn't train out of the box:
- gsplat's prebuilt Windows wheels are for Python 3.10 only, and this package needs 3.11+.
- The PyPI package compiles its CUDA kernels the first time they're used. With gsplat 1.5.3 on
  Windows that fails with `ninja: error: loading 'build.ninja': The system cannot find the path
  specified`, because it deletes its own build folder before building.

So build gsplat's CUDA extension once, from source. It then loads directly with no compiler
needed at run time, and `splat.trainer=auto` picks `gsplat`. Tested with an RTX 4070, CUDA 12.8,
torch 2.11 and Python 3.11.

1. Install the toolchain. Pick a CUDA toolkit with the same major.minor as your torch build
   (`python -c "import torch; print(torch.version.cuda)"`):

   ```bash
   winget install Gyan.FFmpeg
   winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   winget install Nvidia.CUDA -v 12.8
   ```

2. Install CUDA torch before the extras, so pip doesn't pull a CPU build:

   ```bash
   .venv/Scripts/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   .venv/Scripts/python -m pip install -e ".[dev,visual,audio,satie]"
   ```

3. Build gsplat from source inside the MSVC + CUDA environment. [`cuda_env.cmd`](cuda_env.cmd)
   finds both and targets RTX 40-series GPUs (`TORCH_CUDA_ARCH_LIST=8.9`). Set
   `TORCH_CUDA_ARCH_LIST` yourself for other cards, e.g. `8.6` for RTX 30. The build takes a few
   minutes. Run it from `cmd`, or through `cmd /c` from PowerShell:

   ```bash
   .venv/Scripts/python -m pip install wheel setuptools ninja
   cmd /c "cuda_env.cmd .venv\Scripts\python -m pip install --no-build-isolation --no-binary gsplat --force-reinstall --no-deps gsplat==1.5.3"
   ```

4. Check it:

   ```bash
   .venv/Scripts/python -c "from gsplat import csrc; from memoryscapes_workers.visual.train import detect_trainer; print(detect_trainer())"   # gsplat
   memoryscapes-worker demo
   ```

On an RTX 4070, the default 30,000 MCMC steps (1.5M gaussians) take about 13 minutes for 40–50
keyframes at 640 px. Re-run step 3 after upgrading torch or gsplat.

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

**iPhone spatial video** (iPhone 15 Pro and later, iPhone 16 / 16 Plus): the pipeline keeps both eyes. It measures true scale from them, and adds right-eye cameras (plus stereo depth maps) for splat training. Status, measurements, the GPU A/B to run, and what still needs work are in [docs/SPATIAL_VIDEO.md](../../docs/SPATIAL_VIDEO.md). It needs ffmpeg 7.1 or newer.

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

**separate**: first cleans the recording (`audio/denoise.py`, below), then tags the mix with CLAP over sliding windows, separates each detected class (AudioSep, with Demucs vocals for speech), then turns the estimates into masks that partition the mix. The masks are applied to every channel, so stems + ambient equals the cleaned recording and the inter-channel cues survive for DOA. Diffuse classes (wind, rain, insects, crowd) stay in the ambient bed. Writes `separate/stems/<id>.wav`, `separate/ambient.wav` and `separate/stems.json`.

Speech comes first. On real recordings CLAP scores "people speaking" well below broadband prompts (construction, train, airplane all hit 1.0 on a quiet night clip), so with at most 8 stems a voice could get no slot at all. Detected speech therefore always gets one. A stem counts as too quiet only when it's more than 30 dB under the recording's own level (`min_stem_below_mix_db`), with a floor of −70 dBFS. A fixed −55 dBFS had dropped a voice at −58 dBFS from a quiet phone clip. On the iPhone 16 Plus test clips, the voice was lost before both changes and is kept after.

Cleaning, before separation (`AUDIO_CLEAN=false` turns it off):
- **High-pass at 60 Hz** (zero phase) removes rumble and handling noise.
- **Wind suppression, for two-channel recordings.** Microphone wind is turbulence at each capsule, so it's uncorrelated between the mics; real sound reaches both. Bins where the channels disagree are turned down by at most ~14 dB, with the same gain on both channels so the direction cues survive.
  - Below 200 Hz this always applies, since real sound agrees there on any phone.
  - Above 200 Hz (fading out by 2 kHz) it applies only while that low band shows wind. Phones process their stereo: an iPhone 16 Plus's channels hardly agree above ~500 Hz even in calm air. An always-on gain removed 3–5 dB of speech-band sound there; gated, it removes 0.0 dB.
- **Skipped** for mono, for 3+ channels, and for near-identical channels (correlation > 0.99, typical of edited downloads), where wind can't be told apart.
- **Results, on a synthetic scene with mic wind (12 cm stereo):** SI-SDR improved by +6.0, +10.3 and +12.3 dB for light, medium and heavy wind, with birdsong and speech detail unchanged. `noisereduce` made it worse (−1.5 to −3.9 dB, it removes steady ambience), and asking AudioSep for "wind noise" didn't help (≈0 dB, 12–23 s a clip). On the example memories, localization was as good or better.
- **What it did** is recorded in `stems.json` → `cleaning`: `highpass_hz`, `wind` ("suppressed", "off", or why it was skipped), `wind_like_frames`, and `removed_db` per band.

**localize**: for each stem, in order of preference:
1. **Audio-visual:** Grounding DINO looks for the class's prompt in the frames where the stem is loudest. Each box is lifted to 3D using the scene points inside it, or, when a sparse SfM cloud leaves too few there, the points just around it. Then the lifts are clustered (at least 2 views).
2. **Stereo DOA:** GCC-PHAT bearings per window, triangulated along the moving camera path. This chooses between AV clusters, or places the source alone. Candidates steeply above or below the camera (over 50° of elevation, none past 80°) are faded out: a horizontal mic pair can't tell them apart from anything else broadside, so stereo with no time difference between the channels, which many phones record, would otherwise pin every sound under the camera.
3. **Diffuse** otherwise.

Speech is placed only on strong evidence (confidence ≥ `speech_place_min_confidence`, 0.75). Otherwise it's **narration**: no position, heard everywhere at its recorded level, and the SATIE brief asks for it to stay clearly audible. Whoever is filming talks from right at the moving mic, which can't be triangulated, and a person in frame may not be the one talking. On an iPhone clip, the filmer's voice was pinned to a bystander 4 m away (confidence 0.68), so it faded as the visitor walked away. Stereo can't tell a voice behind the phone from a person straight ahead.

When several stems land on one point (their prompts found the same object, like a siren and a car horn on one vehicle), the most confident keeps it, and each other moves to its best placement elsewhere if it has a good enough one on its own. Otherwise it shares the point exactly, as one object making both sounds, and `shares_with` names the keeper.

Writes `localize/sources.json` with position, spread, confidence, method, `shares_with` (or null), and a level corrected to 1 m. Moving sources are placed at their dominant location for now.

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
- **The worker, end to end (2026-11-14):** after `satie-login` on a Mac, `run canada --stages soundscape,publish` composed, checked, audited and realized 8 of 8 provided sounds (0 generated, 0 from the library, 0¢), and the viewer played them through `satie-runtime`. What that run showed:
  - `satie_compose` pauses at its time budget, about 45 s for a worker call, and returns `status: "paused"`. The provider repeats the identical call until it finishes, which resumes the same run (SATIE counted one composition). Compose carries no `requestKey`, because a keyless call resumes by its arguments.
  - `satie_audit_scene` returns the quote nested, as `quote.quoteId`.
  - When SATIE fails, everything it answered is kept in `soundscape/satie/failed_raw.json`.
- **Sign-in:** SATIE's MCP server is an OAuth 2.1 resource. Its metadata names Supabase Auth, which supports dynamic client registration, PKCE (S256), `refresh_token` and `offline_access`. `satie-login` uses exactly that: a public client with no secret, then refresh tokens. Without a token the server answers 401 `invalid_token`, and the worker reports that as "not signed in". Signed in for real on 2026-11-14, from a Mac.
- **Samples load from our URLs.** Every scene sample needs a `url`, and the runtime fetches the sound from it. An upload doesn't replace that.
- **Uploading the stems** (`SATIE_UPLOAD_STEMS=true`, off by default). After the original-only checks pass, each recording is uploaded through `satie_asset` (begin → PUT → finalize) under its `samples.json` key (`Audio/memory/<id>`). SATIE keeps it as a provided sample. Uploads are capped at 16 MB, so longer stems go up as MP3; this was tested live with a 46 s stem. A failed upload only adds a note.
- **Listening needs a paid plan.** `satie_listen` (its review of a ≤30 s, 16-bit WAV) answers "needs the Vibecoder plan" on Free, so the worker doesn't call it.

The model adapters (`ClapTagger`, `AudioSepSeparator`, `DemucsVocalsSeparator`, `GroundingDinoDetector`) follow those libraries' documented APIs but haven't been run here. The tests use fakes and synthetic scenes with known ground truth.
