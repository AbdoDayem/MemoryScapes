# Splat pipeline outputs

What the poses and splat stages write for each memory, so the audio stages, the publish stage and the debugger (`tools/debugger`) can read them. Keys are relative to `{memory_id}/` in storage. Every file is optional to the debugger, which shows what exists and flags what doesn't line up.

## Conventions

These are shared with the audio pipeline (see `services/workers/memoryscapes_workers/models.py`):

- **World frame:** metres, +Y up. The splat, the camera poses and the audio source positions must all be in this one frame. The debugger's alignment check compares them.
- **Cameras:** OpenCV convention (x right, y down, z forward). `c2w` maps camera coordinates to world coordinates.
- **Time:** `t` on cameras is seconds from the start of the recording, on the same clock as `ingest/audio.wav`.

## ingest/ (spatial video)

An iPhone spatial video (MV-HEVC, two lenses 17.7 mm apart) keeps both eyes:

| Key | Contents |
|---|---|
| `frames/frame_*.jpg` | Keyframes from the **left** eye (any video: the keyframes) |
| `frames_right/frame_*.jpg` | The right eye at the same instants, same names |
| `meta.json` → `stereo` | `{"baseline_m", "hfov_deg", "left_view", "right_view", "right_frames"}`, or `null` for ordinary video. `keyframes.focal_px` comes from the exact field of view. |

## poses/

| Key | Contents |
|---|---|
| `cameras.json` | `[{"frame_id", "t", "width", "height", "K": 3x3, "c2w": 4x4, "image_key"}]` (the left eye for spatial video) |
| `points3d.npy` | `(N, 3)` float32 SfM or dense points in world metres |
| `points3d_rgb.npy` | optional `(N, 3)` uint8 colours for the points |
| `alignment.json` | The SfM → world transform: `scale_method` (`stereo` for spatial video, else `eye_height`, `walking_speed`, `scene_depth` or `fixed`), `scale_confidence`, and `stereo_scale` (`{scale, confidence, points, frames, disparity_offset_px, vertical_offset_px, spread}` or `null`) |
| `stereo_right.json` | Spatial video with a stereo scale only: `{"baseline_m", "disparity_offset_px", "vertical_offset_px", "frames": [{"frame_id": "<left>_right", "left_frame_id", "width", "height", "K", "c2w", "image": "images_right/…"}], "depth": {"dir", "scale", "units": "1/m", "valid_fraction", "bias_removed_px", "noise_px"}}`. The right-eye cameras: undistorted like the left, each `baseline_m` along its left partner's +x. |
| `images_right/frame_*_right.jpg` | The right eye, undistorted to match `K` in `stereo_right.json` |
| `stereo_depth/<left frame_id>.png` | 16-bit inverse depth from the stereo pair: value / `depth.scale` = 1/metres, 0 = unknown |

## splat/

| Key | Contents |
|---|---|
| `meta.json` | `{"trainer", "trainer_version", "iterations", "sh_degree", "units": "meters", "up": "+y", "eval_frames": [frame_id, ...]}` |
| `train_report.json` | Training metrics, including `psnr` (`psnr_split`: `holdout` or `train`), `train_images`, `right_eye_images` (spatial video: right-eye training views), `depth_supervised_images`, `depth_error_vs_sfm` (median relative error of the rendered depth against `points3d` on the scored frames), `psnr_cc` (PSNR after the best affine colour fit per scored frame: structure, not exposure), `exposure_comp`, and `exposure_gains` (`{frame_id: gain}`, the learned brightness per training frame, 1 = the splat's own; `null` without `exposure_comp`) |
| `metrics.json` | Training curves, parallel arrays: `{"iteration": [], "loss": [], "psnr": [], "num_gaussians": [], "elapsed_s": []}`. Any of `ssim` and `lpips` too. |
| `point_cloud.ply` | Trained Gaussians, standard 3DGS binary little-endian PLY: `x y z`, `f_dc_0..2`, optional `f_rest_*`, `opacity` (logit), `scale_0..2` (log), `rot_0..3` (quaternion w, x, y, z) |
| `renders/<frame_id>.png` | The trained splat rendered from that keyframe's camera, at the frame's size or proportionally smaller. Used for side-by-side checks against `ingest/frames/`. |
| `lods/lod<n>.spz` | Compressed deliverables for the viewer. The debugger lists them but doesn't parse them. |
| `lods.json` | `{"order": "most detailed first", "format": "spz v3", "lods": [{"file", "count", "sh_degree", "bytes"}], "antialiased", "bounds", "cleanup"}`. `antialiased` is the training mode (gsplat `antialiased` vs classic): publish copies it into the manifest so the viewer's blur matches. |

## debug/ (examples only)

`debug/truth.json` holds ground truth for synthetic memories: `{"sources": [{"label", "position"}], "splat_transform": {...}}`. The debugger overlays it when present. Real memories don't have it.
