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
```

Compare `psnr` (held-out frames) and `depth_error_vs_sfm`. Also walk each result in the viewer, away from the capture path, where better geometry shows most. Then set the defaults (`stereo_views`, `depth_lambda`) from the results.

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
5. **Exposure.** The right eye is ~3% brighter. Add per-image colour/exposure compensation to the trainer (an affine colour per camera), so the extra views don't teach a colour mismatch.
6. **Stronger poses.** SfM runs on the left eye only, and the right cameras are derived from it. Registering both eyes as a rig with a fixed baseline (COLMAP rigs) could tighten the poses and the scale.
7. **Dense initialisation** from the stereo depth (back-projected points) for bland surfaces SfM misses.
8. **Debugger.** Show the right-eye frames and the depth maps, and the stereo scale next to the alignment check.

## Known issue elsewhere (audio)

The audio cleaning step (`audio/denoise.py`, PR #23) removes **3–5 dB of real sound at 0.5–2 kHz** on iPhone 16 Plus recordings, even with no wind.
- **Cause:** the phone's processed stereo channels hardly agree above ~500 Hz, and the step reads that as wind.
- **Workaround:** `AUDIO_CLEAN=false`.
- **Fix, designed and measured but not built:**
  - below 200 Hz, keep the current gain;
  - above 200 Hz, apply it only in frames where the 50–200 Hz band shows wind: weight = clip((0.8 − mean coherent share) / 0.4, 0, 1), smoothed over 0.3 s, fading out by 1 kHz.
  - **Measured:** the iPhone clips lost 0.0 dB at 0.5–2 kHz, and synthetic wind improved by +5.9 / +10.3 / +12.3 dB (light / medium / heavy).
