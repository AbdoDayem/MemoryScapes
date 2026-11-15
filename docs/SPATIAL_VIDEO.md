# Spatial video → better splats: status and handoff

An iPhone **spatial video** (iPhone 15 Pro and later, and the iPhone 16 / 16 Plus) records two
views at once from two lenses 17.7 mm apart, as MV-HEVC. The pipeline now uses that second view.
This page says what's built, what was measured, and what the Windows/NVIDIA machine should do next.
The splat comparison couldn't be finished on the Mac, where training runs at about 1.4 steps/s.

Output formats are in [SPLAT_OUTPUTS.md](SPLAT_OUTPUTS.md). The code is in
`services/workers/memoryscapes_workers/`: `visual/video.py`, `stages/ingest.py`,
`visual/stereo.py`, `stages/poses.py`, `visual/train.py` and `visual/rasterize_torch.py`.

## What's built

1. **Ingest keeps both eyes.**
   - ffprobe reports which view is left or right, the lens baseline and the field of view.
   - Keyframes come from the left eye, and the right eye's frames at the same instants go to `ingest/frames_right/`.
   - `meta.json → stereo` records the layout, and `focal_px` comes from the exact field of view.
   - This needs **ffmpeg 7.1 or newer** to decode one view (`0:v:view:N`). With an older ffmpeg, ingest warns and uses one eye.
2. **True scale from the two eyes.**
   - COLMAP still reconstructs the left eye. Then, for COLMAP's own 3D points, the stage measures the disparity between the eyes (template matching along the row).
   - It fits `d = f·B / (s·z_colmap) + d0` (RANSAC, then least squares). The slope gives the metric scale s. The intercept d0 absorbs a constant shift Apple puts between the views.
   - This replaces the eye-height and walking-speed guesses: `alignment.json` → `scale_method: "stereo"`, plus `stereo_scale`.
3. **Right-eye cameras.**
   - The right frames are undistorted with the left eye's lens model.
   - Each right camera is placed `baseline_m` along its left partner's +x axis. The world is metric by then.
   - Its principal point is shifted by the measured view offsets.
   - Output: `poses/stereo_right.json` and `poses/images_right/`.
4. **Stereo depth maps.**
   - OpenCV SGBM on each undistorted pair, after shifting the right image by the view offsets, with a left-right consistency check.
   - The remaining constant disparity bias is measured against COLMAP's points and removed.
   - Output: `poses/stereo_depth/<frame>.png`, a 16-bit inverse depth.
5. **Splat training.**
   - It loads the right-eye cameras as extra views (`splat.train.stereo_views`, on by default).
   - Held-out evaluation is per instant: both eyes of a held-out instant stay out of training. PSNR is scored on the left eye, so runs with and without the right eye compare directly.
   - There's an optional L1 loss on inverse depth against the stereo maps (`splat.train.depth_lambda`, **0 = off by default** until it's tuned).
   - `train_report.json` adds `right_eye_images`, `depth_supervised_images` and `depth_error_vs_sfm` (the rendered depth's median relative error against COLMAP's points, on the scored frames).
   - The PyTorch rasterizer supports `render_mode="RGB+ED"`, like gsplat.

Ordinary videos are unaffected: no stereo data, so the old behaviour applies.

## Measured on the Mac

Test clip: `IMG_9661.MOV`, a 35 s spatial video from an iPhone 16 Plus at night, outdoors (110 keyframes at 1600×900). It isn't in the repo; get it from Marko. `IMG_9660.MOV` is the same scene as a normal video.

| What | Result |
|---|---|
| Scale | 0.985 and 0.990 m per COLMAP unit (two COLMAP runs), from 7,800 points in 110 frames. 8% spread, confidence 0.82. All frames registered (1.05 px reprojection). The walk came out at 22.9 m, 0.66 m/s. |
| View offsets | +1.1 px horizontal (d0) and ≈ −1.2 px vertical, at 1600 px |
| Right cameras | Patches around COLMAP points matched between the eyes with correlation **0.858**. Comparison cameras scored worse: no lens offset 0.658, offset on the wrong side 0.441, no pixel-shift correction 0.785. |
| Stereo depth | ~50% of pixels valid. Noise ±0.9 px of disparity (dark footage), and a +0.5 px bias, now removed. At these disparities (2–10 px) that's 17–37% depth error by distance before the bias fix. |
| Colour | The right eye averages ~3% brighter than the left |
| **Splat quality** | **Not measured.** The A/B was stopped because the Mac is too slow. |

## What to run on the Windows machine

```bat
git pull
deploy\windows\setup.cmd --skip-models
ffmpeg -version            :: must be 7.1 or newer for the two views

cd services\workers
.venv\Scripts\memoryscapes-worker import spatial C:\path\to\IMG_9661.MOV
.venv\Scripts\memoryscapes-worker run spatial --until poses
```

Check `poses\alignment.json` (`scale_method` should be `stereo`; see `stereo_scale`) and `poses\stereo_right.json` (the number of frames, and `depth.valid_fraction`, `bias_removed_px`, `noise_px`).

Then the splat A/B, with gsplat on the GPU at the default 30,000 steps. Each run overwrites `splat\`, so copy `splat\train_report.json` away after each one:

```bat
:: A: left eye only
.venv\Scripts\memoryscapes-worker run spatial --stages splat --force --set splat.train.holdout_every=8 --set splat.train.stereo_views=false
:: B: both eyes (the default)
.venv\Scripts\memoryscapes-worker run spatial --stages splat --force --set splat.train.holdout_every=8
:: C: both eyes + depth loss; also try 0.05 and 0.5
.venv\Scripts\memoryscapes-worker run spatial --stages splat --force --set splat.train.holdout_every=8 --set splat.train.depth_lambda=0.2
:: D: B + per-image exposure compensation (also worth trying on IMG_9660, the normal video)
.venv\Scripts\memoryscapes-worker run spatial --stages splat --force --set splat.train.holdout_every=8 --set splat.train.exposure_comp=true
```

Compare `psnr` (held-out frames) and `depth_error_vs_sfm`. Use `psnr_cc` (PSNR after fitting each frame's colour) to compare D with the others: its splat takes the average frame's exposure, which plain PSNR on a differently exposed held-out frame penalizes. D's `exposure_gains` show how much the exposure varied per frame. Also walk each result in the viewer, away from the capture path, where better geometry shows most. Then set the defaults (`stereo_views`, `depth_lambda`, `exposure_comp`) from the results.

## To improve or fix, in order

1. **Run the A/B above** and set the defaults from it. Nothing about splat quality is measured yet.
2. **Verify gsplat's `RGB+ED` path on CUDA.** The depth loss was trained only with the PyTorch rasterizer. The gsplat call is checked only against gsplat's function signature (`test_gsplat_backend_calls_match_gsplat_api`), not run.
3. **Tune the depth loss.** Things to try:
   - a weight schedule (strong early, then decaying);
   - a robust loss (Huber);
   - per-pixel confidence (larger disparity means more reliable);
   - leaving out sky.
4. **Stereo depth quality.** It's ±0.9 px on night footage, which is the limit with a 17.7 mm baseline. Things to try:
   - a daylight clip, to see how much is noise from the dark;
   - fusing depth across neighbouring frames;
   - a learned stereo matcher (check its licence).
5. **Exposure: built, to be measured** (run D above). The right eye is ~3% brighter, and night clips change exposure and white balance from frame to frame. `splat.train.exposure_comp` learns a 3×3 colour matrix + offset per training image, applied to the render before the loss, with the average held at identity. It's off by default until measured.
   - **On the synthetic test scene:** with frames re-exposed at 0.55–0.95× and tinted, the learned per-frame gains followed the applied ones (correlation 0.98), and held-out PSNR rose 0.9 dB (`psnr_cc` +0.3 dB).
   - **Without it:** the colour-fitted score got worse the longer it trained, which means the splat was baking the exposure changes in.
6. **Stronger poses.** SfM runs on the left eye only, and the right cameras are derived from it. Registering both eyes as a rig with a fixed baseline (COLMAP rigs) could tighten the poses and the scale.
7. **Dense initialisation** from the stereo depth (back-projected points) for bland surfaces SfM misses.
8. **Debugger.** Show the right-eye frames and the depth maps, and the stereo scale next to the alignment check.

## Audio on these clips (fixed)

On the iPhone 16 Plus test clips, the filmer's voice was lost on the way to playback, for three reasons. All three are now fixed (see `services/workers/README.md`, separate and localize):
1. **Cleaning cut the speech band.** The wind step removed 3–5 dB at 0.5–2 kHz, because the phone's processed stereo hardly agrees there. Above 200 Hz it now acts only while wind blows, and removes 0.0 dB there on both clips.
2. **Separation dropped the voice.** Speech came 11th behind false CLAP tags (only 8 stems get separated), or was cut as "too quiet" by a fixed −55 dBFS. Speech now always gets a slot, and "too quiet" is relative to the recording.
3. **Localization pinned it to a bystander 4 m away**, so it faded as the visitor walked. Speech without strong evidence of one visible spot is now narration, heard everywhere.

The iPhone's 4-channel spatial audio (APAC) can be decoded on a Mac (AVFoundation), but its directions didn't follow the camera's turning in these clips. It isn't used yet; a controlled test (a speaker at a known side) would show whether it can be.
