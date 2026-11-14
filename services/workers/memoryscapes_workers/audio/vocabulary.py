"""The sound classes the pipeline knows how to tag, separate and place.

One table drives all three steps:
- `clap_prompt`: text prompt for zero-shot tagging (CLAP).
- `query`: text query for query-based separation (AudioSep).
- `detect_prompt`: Grounding DINO prompt used to find the visual source in frames.
- `diffuse`: sounds with no single location (wind, rain). These are not separated or
  placed; they stay in the ambient bed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SoundClass:
    label: str
    clap_prompt: str
    query: str
    detect_prompt: str | None
    kind: str = "event"  # "event" | "speech" | "music"
    diffuse: bool = False


VOCABULARY: tuple[SoundClass, ...] = (
    SoundClass("speech", "people talking", "people speaking", "person.", kind="speech"),
    SoundClass("laughter", "people laughing", "laughter", "person."),
    SoundClass("footsteps", "footsteps walking", "footsteps", "person. feet."),
    SoundClass("birdsong", "birds chirping and singing", "birds chirping", "bird. tree."),
    SoundClass("dog", "a dog barking", "dog barking", "dog."),
    SoundClass("traffic", "cars driving past in traffic", "car traffic", "car. bus. truck. motorcycle."),
    SoundClass("car_horn", "a car horn honking", "car horn", "car. truck."),
    SoundClass("siren", "an emergency siren", "siren", "ambulance. police car. fire truck."),
    SoundClass("train", "a train passing", "train", "train. railway."),
    SoundClass("water", "flowing water from a stream or fountain", "flowing water", "fountain. river. stream. waterfall."),
    SoundClass("waves", "ocean waves on a beach", "ocean waves", "ocean. wave. shoreline."),
    SoundClass("bells", "church bells ringing", "bells ringing", "bell tower. church."),
    SoundClass("music", "music playing", "music", "speaker. guitar. piano. musician.", kind="music"),
    SoundClass("construction", "construction machinery and hammering", "construction noise", "excavator. construction site. crane."),
    SoundClass("airplane", "an airplane flying overhead", "airplane", "airplane."),
    SoundClass("wind", "wind blowing", "wind", None, diffuse=True),
    SoundClass("rain", "rain falling", "rain", None, diffuse=True),
    SoundClass("insects", "crickets and insects buzzing", "insects", None, diffuse=True),
    SoundClass("crowd", "a crowd of people murmuring", "crowd murmur", None, diffuse=True),
)

BY_LABEL: dict[str, SoundClass] = {c.label: c for c in VOCABULARY}
