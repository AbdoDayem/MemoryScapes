# MemoryScapes debugger

A local, read-only web UI for inspecting one memory's pipeline outputs: the audio stages (`separate`, `localize`, `soundscape`) and the splat stages (`poses`, `splat`). Above all, it checks that the two agree. It reads the storage layout straight from disk, so it works on any run, whether a worker's `LocalStorage` root or a bucket synced to a folder.

```bash
cd tools/debugger
npm install
npm run dev                                      # http://localhost:5178, reads ../../services/workers/examples/out
MEMORYSCAPES_DATA=/path/to/root npm run dev      # any folder with one sub-folder per memory
```

For sample data, generate the example memories first. They now include synthetic splat outputs:

```bash
cd services/workers && .venv/Scripts/python examples/run_example.py
```

## Tabs

| Tab | What it shows |
|---|---|
| **Checks** | Integration checks, worst first (see below) |
| **Audio** | Tag scores against the selection thresholds; the recording, every stem and the ambient bed, with waveform, activity envelope and playback; a spectrogram of the selected track; dropped stems and why |
| **Localization** | Top-down map of the camera path, scene points, placed sounds with their spread and ground truth; a table of method, confidence, spread, level at 1 m and error |
| **Splat** | Trainer metadata, Gaussian stats (count, bounds, opacity and scale histograms, alignment against the poses), training curves, and each keyframe beside its splat render with a difference image and PSNR |
| **3D scene** | The pose point cloud, splat Gaussian centres, camera frustums, sounds, spread spheres and ground truth in the one shared world frame, each a toggleable layer. **Listen from this viewpoint** loops every stem through HRTF panners with the orbit camera as the listener. |
| **Files** | Every file in the memory, with a link to the raw file |

## Checks

They live in [`src/lib/checks.ts`](src/lib/checks.ts) and are pure functions with unit tests. Each check is fail, warn, info or ok.

- **Contract:** each stage wrote the files [docs/SPLAT_OUTPUTS.md](../../docs/SPLAT_OUTPUTS.md) and `services/workers/README.md` promise; `stems.json` and `cameras.json` don't point at missing files; every JSON and `.npy` file parses.
- **Timing:** camera `t` values are ordered, inside the recording, and cover most of it. If they don't, the poses and the audio are on different clocks.
- **Frame:** cameras are upright in a +Y-up world, and the camera moves at walking pace (i.e. the poses are in metres).
- **Splat alignment:** the splat's Gaussian centres sit on the pose point cloud. If not, the check reports the scale ratio and offset. The audio positions use the poses' frame, so a misaligned splat puts every sound in the wrong place. `hard_mode` has a deliberate half-scale splat that this catches.
- **Splat health:** `meta.json` units and up-axis match the contract, metrics contain no NaN or Inf, final PSNR is reasonable, loss isn't rising at the end, and renders exist for each keyframe.
- **Audio:** placed sounds are inside the scene bounds and near some scene surface; low-confidence placements; dropped stems; a SATIE fallback and its reason; `scene.json` agrees with `sources.json`.
- **Ground truth:** for synthetic memories with `debug/truth.json`, each sound's error.

## Layout

```
server/dataPlugin.ts   Vite middleware: /api/memories, /api/memories/:id/files, /data/:id/<key> (read-only, no path escapes)
src/lib/               parsers (.npy, 3DGS .ply, FFT), checks, data loading
src/views/             one component per tab
```

```bash
npm test          # parsers, checks, data server
npm run build     # type-check + production bundle (npm run preview serves it with the same data API)
```
