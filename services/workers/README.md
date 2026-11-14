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
| Ingest | `stages/ingest.py` | CPU | stub |
| Camera poses | `stages/poses.py` | GPU | stub |
| Splat training | `stages/splat.py` | GPU | stub |
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

AudioSep isn't on PyPI. Clone [Audio-AGI/AudioSep](https://github.com/Audio-AGI/AudioSep), download its checkpoint, put the repo on `PYTHONPATH`, and set `AUDIOSEP_CONFIG` and `AUDIOSEP_CHECKPOINT`.

## Audio stages: data contracts

All keys are relative to `{memory_id}/` in storage (`LocalStorage` in dev/tests, `S3Storage` deployed). Conventions: world frame in metres with +Y up; cameras in OpenCV convention (`c2w`); a stereo mic lies along the camera's +x axis (channel 0 = left).

**Inputs the audio stages expect from ingest and poses** (those stages are still stubs):

| Key | Contents |
|---|---|
| `ingest/audio.wav` | Original recording, all channels, original sample rate |
| `ingest/meta.json` | `{"mic": {"channels", "layout": "mono" \| "stereo" \| "foa", "spacing_m"?}}` |
| `ingest/frames/<id>.jpg` | Keyframe images |
| `poses/cameras.json` | `[{"frame_id", "t", "width", "height", "K", "c2w", "image_key"}]`, with `t` on the audio clock |
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
