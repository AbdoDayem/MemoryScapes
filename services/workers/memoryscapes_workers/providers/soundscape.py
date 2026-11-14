"""SoundscapeProvider adapter: turns localized stems into the memory's soundscape.

Providers:
- StubProvider: passes the localized stems through as object-based audio for the web
  app's own SpatialAudioEngine. Always works; used in development and as the fallback.
- SatieMcpProvider: drives SATIE (Attractor Labs) through its MCP server, following the
  workflow its skill prescribes (github.com/Attractor-Laboratories/satie-skills):
      satie_inspect_scene -> satie_brief -> satie_compose -> satie_audit_scene -> satie_realize
  The output is a .satie script plus samples.json, played in the viewer by satie-runtime.

Not documented publicly, so isolated here and worth checking against the live server:
- the scene manifest shape (the server publishes it as resource satie://schema/scene),
- whether satie_realize's `samples` accepts our own stem URLs, and the 30 s audio limit,
- the exact response fields of each tool (parsed defensively below).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from memoryscapes_workers.models import LocalizedSource
from memoryscapes_workers.providers.mcp_transport import McpTransport


@dataclass
class SoundscapeSource:
    source: LocalizedSource
    url: str  # fetchable URL of the stem WAV

    @property
    def clip(self) -> str:
        return f"memory/{self.source.id}"


@dataclass
class SoundscapeRequest:
    memory_id: str
    sources: list[SoundscapeSource]
    ambient_key: str
    ambient_url: str
    ambient_channels: int
    bounds_min: list[float]
    bounds_max: list[float]
    mood_prompt: str | None = None


@dataclass
class SatieOutput:
    script: str
    samples: dict[str, Any]
    host_map: dict[str, str]  # SATIE host id -> manifest source id
    contract_id: str
    realize_id: str | None = None


@dataclass
class SoundscapeResult:
    provider: str
    sources: list[dict[str, Any]]  # manifest-shaped audio sources (object-based fallback)
    ambient_bed: dict[str, Any]
    satie: SatieOutput | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class SoundscapeProvider(Protocol):
    name: str

    def generate(self, request: SoundscapeRequest) -> SoundscapeResult: ...


def object_based_sources(request: SoundscapeRequest) -> list[dict[str, Any]]:
    """Manifest audio sources. ref_distance = how far the mic was from the source, so a
    listener standing where the recording was made hears it at the recorded level."""
    out = []
    for s in request.sources:
        src = s.source
        ref = 1.0
        if src.level_db_at_1m is not None:
            ref = round(10 ** ((src.level_db_at_1m - src.level_dbfs) / 20), 3)
        out.append(
            {
                "id": src.id,
                "label": src.label,
                "key": src.stem_key,
                "position": src.position,
                "trajectory": src.trajectory,
                "ref_distance": ref,
                "rolloff": 1.0,
                "gain": 1.0,
                "loop": True,
                "confidence": src.confidence,
                "method": src.method,
            }
        )
    return out


def ambient_bed(request: SoundscapeRequest) -> dict[str, Any]:
    return {"key": request.ambient_key, "format": "mono" if request.ambient_channels == 1 else "stereo"}


class StubProvider:
    name = "stub"

    def generate(self, request: SoundscapeRequest) -> SoundscapeResult:
        return SoundscapeResult(provider=self.name, sources=object_based_sources(request), ambient_bed=ambient_bed(request))


# ---------------------------------------------------------------- SATIE


class SatieError(RuntimeError):
    pass


class SatieBudgetExceeded(SatieError):
    pass


@dataclass(frozen=True)
class SatieConfig:
    max_cents: int = 0  # spend cap per memory; 0 = only free reuse (our stems, library)
    allow_generation: bool = False  # may SATIE add sounds that weren't recorded?
    max_extra_voices: int = 3
    poll_attempts: int = 20
    poll_interval_s: float = 15.0


def build_scene_manifest(request: SoundscapeRequest) -> dict[str, Any]:
    """Hosts are the placed sources, ids equal to our source ids. Metres, +Y up, the
    world frame unchanged (SATIE: "use the game's coordinates unchanged")."""
    hosts = [
        {
            "id": s.source.id,
            "name": s.source.label,
            "position": s.source.position,
            "motion": "static",
            "radius": s.source.spread_m,
        }
        for s in request.sources
        if s.source.position is not None
    ]
    return {
        "units": "meters",
        "up": "+y",
        "hosts": hosts,
        "rooms": [{"id": "memory", "bounds": {"min": request.bounds_min, "max": request.bounds_max}}],
    }


def build_brief(request: SoundscapeRequest, cfg: SatieConfig) -> str:
    placed = [s for s in request.sources if s.source.position is not None]
    diffuse = [s for s in request.sources if s.source.position is None]
    lines = [
        "This scene is a real place someone recorded, rebuilt as a walkable 3D memory.",
        "Recreate its soundscape from their own recordings. Use exactly the clip names below.",
        "",
        "Placed voices (loop each one `at` its host):",
    ]
    for s in placed:
        spl = f", about {s.source.level_db_at_1m:.0f} dBFS at 1 m" if s.source.level_db_at_1m is not None else ""
        lines.append(f"- host {s.source.id}: clip {s.clip} ({s.source.label}{spl})")
    lines += ["", "Ambient bed (loop as `bed` in room memory):", "- clip memory/ambient (the rest of the recording)"]
    for s in diffuse:
        lines.append(f"- clip {s.clip} ({s.source.label}, no single location)")
    lines.append("")
    if cfg.allow_generation:
        lines.append(
            f"You may add up to {cfg.max_extra_voices} quiet voices that fit the place. "
            "Never replace or cover the recordings."
        )
    else:
        lines.append("Do not add library or generated sounds. Only the recordings above.")
    if request.mood_prompt:
        lines.append(f"Mood: {request.mood_prompt}")
    lines.append("This runs unattended: where you have open questions, use your stated assumptions.")
    return "\n".join(lines)


def samples_map(request: SoundscapeRequest) -> dict[str, str]:
    m = {s.clip: s.url for s in request.sources}
    m["memory/ambient"] = request.ambient_url
    return m


_FENCE = re.compile(r"```(?:satie)?\s*\n(.*?)```", re.DOTALL)


def extract_script(result: dict[str, Any]) -> str:
    if isinstance(result.get("script"), str) and result["script"].strip():
        return result["script"].strip()
    m = _FENCE.search(result.get("text", ""))
    if not m:
        raise SatieError("satie_compose returned no script")
    return m.group(1).strip()


def _get(d: dict[str, Any], *names: str) -> Any:
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return None


def _http_get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (URL comes from SATIE)
        return json.loads(resp.read())


class SatieMcpProvider:
    name = "satie"

    def __init__(
        self,
        transport: McpTransport,
        cfg: SatieConfig = SatieConfig(),
        http_get_json: Callable[[str], dict[str, Any]] = _http_get_json,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport
        self.cfg = cfg
        self.http_get_json = http_get_json
        self.sleep = sleep

    def generate(self, request: SoundscapeRequest) -> SoundscapeResult:
        raw: dict[str, Any] = {}
        notes: list[str] = []
        call = self.transport.call_tool

        raw["inspect"] = inspect = call("satie_inspect_scene", {"scene": build_scene_manifest(request), "unitsPerMeter": 1})
        contract_id = _get(inspect, "contractId", "contract_id")
        if not contract_id:
            raise SatieError("satie_inspect_scene returned no contractId")

        brief = build_brief(request, self.cfg)
        raw["brief"] = call("satie_brief", {"brief": brief, "contractId": contract_id})

        raw["compose"] = compose = call("satie_compose", {"brief": brief, "contractId": contract_id})
        script = extract_script(compose)

        missing = [c for c in samples_map(request) if c not in script]
        if missing:
            revision = "Every recording must be used. Missing clips: " + ", ".join(missing)
            raw["compose_revision"] = compose = call(
                "satie_compose", {"brief": revision, "contractId": contract_id, "script": script}
            )
            script = extract_script(compose)
            still = [c for c in missing if c not in script]
            if still:
                notes.append("SATIE script does not use: " + ", ".join(still))

        raw["audit"] = audit = call("satie_audit_scene", {"script": script, "contractId": contract_id})
        quote_id = _get(audit, "quoteId", "quote_id")
        if not quote_id:
            raise SatieError("satie_audit_scene returned no quoteId (audit failed?)")

        realize = self._realize(request, script, quote_id)
        raw["realize"] = realize
        samples_url = _get(realize, "samplesUrl", "samples_url")
        if not samples_url:
            raise SatieError("satie_realize returned no samplesUrl")
        samples = self.http_get_json(samples_url)  # signed URL lasts an hour: save it now
        for take in realize.get("suspectTakes") or []:
            notes.append(f"SATIE flagged a suspect take: {take}")

        host_map = {s.source.id: s.source.id for s in request.sources if s.source.position is not None}
        return SoundscapeResult(
            provider=self.name,
            sources=object_based_sources(request),
            ambient_bed=ambient_bed(request),
            satie=SatieOutput(
                script=script,
                samples=samples,
                host_map=host_map,
                contract_id=contract_id,
                realize_id=_get(realize, "realizeId", "realize_id"),
            ),
            raw=raw,
            notes=notes,
        )

    def _realize(self, request: SoundscapeRequest, script: str, quote_id: str) -> dict[str, Any]:
        # Same script -> same requestKey, so a retried pipeline run resumes the same
        # paid attempt instead of buying again (SATIE's documented recovery rule).
        request_key = f"{request.memory_id}-{hashlib.sha256(script.encode()).hexdigest()[:16]}"
        args = {
            "script": script,
            "maxCents": self.cfg.max_cents,
            "confirm": {"quoteId": quote_id},
            "samples": samples_map(request),
            "preferLibrary": True,
            "requestKey": request_key,
        }
        for _ in range(self.cfg.poll_attempts):
            result = self.transport.call_tool("satie_realize", args)
            status = result.get("status")
            if status == "input_required":
                raise SatieBudgetExceeded(
                    f"SATIE needs more than the {self.cfg.max_cents} cent cap for this memory. Raise SATIE_MAX_CENTS to allow it."
                )
            if status != "in_progress":
                return result
            self.sleep(self.cfg.poll_interval_s)
        raise SatieError("satie_realize still in progress after polling; resume later with the same requestKey")


def get_provider(name: str) -> SoundscapeProvider:
    import os

    if name == "stub":
        return StubProvider()
    if name == "satie":
        from memoryscapes_workers.providers.mcp_transport import HttpMcpTransport

        transport = HttpMcpTransport(
            os.environ.get("SATIE_MCP_URL", "https://www.satie.world/mcp"), os.environ.get("SATIE_MCP_TOKEN")
        )
        cfg = SatieConfig(
            max_cents=int(os.environ.get("SATIE_MAX_CENTS", "0")),
            allow_generation=os.environ.get("SATIE_ALLOW_GENERATION", "false").lower() == "true",
        )
        return SatieMcpProvider(transport, cfg)
    raise ValueError(f"Unknown soundscape provider: {name}")
