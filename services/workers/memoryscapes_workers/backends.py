"""Builds the real ML backends from environment variables.

Stages accept injected backends (tests pass fakes); when none is given they call these.
Model weights load lazily on first use, so importing this module is cheap.

    AUDIO_TAGGER       clap (default)
    SEPARATOR          audiosep (default)
    SEPARATOR_SPEECH   demucs (default) | audiosep
    AUDIOSEP_DIR       the AudioSep checkout (holds config/ and checkpoint/)
    AUDIOSEP_CONFIG, AUDIOSEP_CHECKPOINT   paths inside that checkout
    DETECTOR           grounding-dino (default) | none
"""

from __future__ import annotations

import os

from memoryscapes_workers.audio.separation import (
    AudioSepSeparator,
    DemucsVocalsSeparator,
    RoutingSeparator,
    Separator,
)
from memoryscapes_workers.audio.tagging import ClapTagger, Tagger
from memoryscapes_workers.vision.grounding import Detector, GroundingDinoDetector


def get_tagger() -> Tagger:
    name = os.environ.get("AUDIO_TAGGER", "clap")
    if name == "clap":
        return ClapTagger()
    raise ValueError(f"Unknown AUDIO_TAGGER: {name}")


def get_separator() -> Separator:
    name = os.environ.get("SEPARATOR", "audiosep")
    if name != "audiosep":
        raise ValueError(f"Unknown SEPARATOR: {name}")
    default = AudioSepSeparator(
        config_yaml=os.environ.get("AUDIOSEP_CONFIG", "config/audiosep_base.yaml"),
        checkpoint_path=os.environ.get("AUDIOSEP_CHECKPOINT", "checkpoint/audiosep_base_4M_steps.ckpt"),
        repo_dir=os.environ.get("AUDIOSEP_DIR"),
    )
    speech = os.environ.get("SEPARATOR_SPEECH", "demucs")
    by_kind = {"speech": DemucsVocalsSeparator()} if speech == "demucs" else {}
    return RoutingSeparator(default, by_kind)


def get_detector() -> Detector | None:
    name = os.environ.get("DETECTOR", "grounding-dino")
    if name == "none":
        return None
    if name == "grounding-dino":
        return GroundingDinoDetector()
    raise ValueError(f"Unknown DETECTOR: {name}")
