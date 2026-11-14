"""SoundscapeProvider adapter.

SATIE (Attractor Labs) sits behind this interface so its API can change without touching
the pipeline, and so development can run against a stub.

NOTE: SATIE's real input/output formats are not yet confirmed. The request/response
shapes below are this project's own contract; adapt SatieProvider once the API is known.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LocalizedStem:
    stem_id: str
    label: str
    url: str
    position: tuple[float, float, float] | None  # None = diffuse
    confidence: float = 0.0
    gain: float = 1.0


@dataclass
class SoundscapeRequest:
    memory_id: str
    stems: list[LocalizedStem]
    ambient_bed_url: str | None
    scene_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    mood_prompt: str | None = None


@dataclass
class SoundscapeResult:
    """Prefer object-based output (sources + positions) so the client spatializes at
    runtime. If only rendered audio is available, request ambisonics (FOA/HOA)."""

    sources: list[LocalizedStem]
    ambient_bed_url: str | None
    ambient_bed_format: str = "stereo"  # "stereo" | "ambisonic_foa" | "ambisonic_hoa"
    provider_job_id: str | None = None
    raw: dict = field(default_factory=dict)


class SoundscapeProvider(Protocol):
    name: str

    def generate(self, request: SoundscapeRequest) -> SoundscapeResult: ...


class StubProvider:
    """Passes the localized stems straight through. Use for local development."""

    name = "stub"

    def generate(self, request: SoundscapeRequest) -> SoundscapeResult:
        return SoundscapeResult(sources=request.stems, ambient_bed_url=request.ambient_bed_url)


class SatieProvider:
    name = "satie"

    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url
        self.api_key = api_key

    def generate(self, request: SoundscapeRequest) -> SoundscapeResult:
        raise NotImplementedError("Implement once SATIE's API contract is confirmed")


def get_provider(name: str) -> SoundscapeProvider:
    import os

    if name == "stub":
        return StubProvider()
    if name == "satie":
        return SatieProvider(os.environ["SATIE_API_URL"], os.environ["SATIE_API_KEY"])
    raise ValueError(f"Unknown soundscape provider: {name}")
