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

Each run writes the full stage output, so you can listen to `ingest/audio.wav` and `separate/stems/*.wav`. It also writes `report.md` (per-sound error, method, confidence, verdict) and `map.svg` (a top-down view of the true and estimated positions). The script exits non-zero if anything is placed confidently in the wrong spot. `tests/test_examples.py` runs the same scenes as regression tests, and CI (`.github/workflows/workers.yml`) runs both on every PR.

AudioSep isn't on PyPI. Clone [Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep), download its checkpoint, put the repo on `PYTHONPATH`, and set `AUDIOSEP_CONFIG` and `AUDIOSEP_CHECKPOINT`.

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
- `SOUNDSCAPE_PROVIDER=stub` passes them through as object-based audio for the web app's `SpatialAudioEngine`.
- `SOUNDSCAPE_PROVIDER=satie` drives SATIE's MCP server (`satie_inspect_scene` → `satie_brief` → `satie_compose` → `satie_audit_scene` → `satie_realize`). Our sources are hosts with unchanged world positions, and our stems are clips named `memory/<id>`. The output is `soundscape/satie/scene.satie` and `samples.json`, for `satie-runtime`.
- Spending is capped by `SATIE_MAX_CENTS`, default 0. Retries reuse the same `requestKey`, so a re-run resumes the purchase instead of buying again.
- If SATIE fails, the stage falls back to the stub and records why in `soundscape/scene.json`.

### Unverified against the live SATIE server

These are isolated in `providers/soundscape.py` and `providers/mcp_transport.py`:
- The scene manifest shape. The server publishes the real one as the resource `satie://schema/scene`.
- Whether `satie_realize` `samples` accepts our own stem URLs, and whether SATIE's 30 s audio limit applies to them.
- How a headless worker authenticates. The MCP uses OAuth sign-in, and the code currently sends `SATIE_MCP_TOKEN` as a bearer token.
- The exact response fields of each tool. They're parsed defensively.

The model adapters (`ClapTagger`, `AudioSepSeparator`, `DemucsVocalsSeparator`, `GroundingDinoDetector`) follow those libraries' documented APIs but haven't been run here. The tests use fakes and synthetic scenes with known ground truth.
