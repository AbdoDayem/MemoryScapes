# Splat pipeline outputs

What the poses and splat stages write for each memory, so the audio stages, the publish stage and the debugger (`tools/debugger`) can read them. Keys are relative to `{memory_id}/` in storage. Every file is optional to the debugger, which shows what exists and flags what doesn't line up.

## Conventions

These are shared with the audio pipeline (see `services/workers/memoryscapes_workers/models.py`):

- **World frame:** metres, +Y up. The splat, the camera poses and the audio source positions must all be in this one frame. The debugger's alignment check compares them.
- **Cameras:** OpenCV convention (x right, y down, z forward). `c2w` maps camera coordinates to world coordinates.
- **Time:** `t` on cameras is seconds from the start of the recording, on the same clock as `ingest/audio.wav`.

## poses/

| Key | Contents |
|---|---|
| `cameras.json` | `[{"frame_id", "t", "width", "height", "K": 3x3, "c2w": 4x4, "image_key"}]` |
| `points3d.npy` | `(N, 3)` float32 SfM or dense points in world metres |
| `points3d_rgb.npy` | optional `(N, 3)` uint8 colours for the points |

## splat/

| Key | Contents |
|---|---|
| `meta.json` | `{"trainer", "trainer_version", "iterations", "sh_degree", "units": "meters", "up": "+y", "eval_frames": [frame_id, ...]}` |
| `metrics.json` | Training curves, parallel arrays: `{"iteration": [], "loss": [], "psnr": [], "num_gaussians": [], "elapsed_s": []}`. Any of `ssim` and `lpips` too. |
| `point_cloud.ply` | Trained Gaussians, standard 3DGS binary little-endian PLY: `x y z`, `f_dc_0..2`, optional `f_rest_*`, `opacity` (logit), `scale_0..2` (log), `rot_0..3` (quaternion w, x, y, z) |
| `renders/<frame_id>.png` | The trained splat rendered from that keyframe's camera, at the frame's size or proportionally smaller. Used for side-by-side checks against `ingest/frames/`. |
| `lods/lod<n>.spz` | Compressed deliverables for the viewer. The debugger lists them but doesn't parse them. |

## debug/ (examples only)

`debug/truth.json` holds ground truth for synthetic memories: `{"sources": [{"label", "position"}], "splat_transform": {...}}`. The debugger overlays it when present. Real memories don't have it.
