"""Audio source localization in the splat's world frame.

Strategies, in order:
1. Audio-visual grounding (default): match stem labels to objects detected in frames
   (Grounding DINO + SAM), lift to 3D with camera poses and splat depth.
2. Multichannel DOA (stereo/ambisonic/multi-mic): SRP-PHAT or MUSIC per window,
   triangulated across the known camera trajectory.
3. Fallback: unplaceable stems become diffuse sources or join the ambient bed.

Output per stem: {stem_id, label, position | trajectory, spread, confidence, gain}.
"""

from memoryscapes_workers.context import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError
