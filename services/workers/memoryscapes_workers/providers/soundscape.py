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


class SatieNotOriginal(SatieError):
    """The script still used sounds that weren't recorded (generated, synthesized or
    library audio) after one revision. Nothing is realized."""


@dataclass(frozen=True)
class SatieConfig:
    # Spend cap per memory. Only used when allow_generation is on: an original-only
    # soundscape reuses our own recordings, which costs nothing, so realize then always
    # runs with maxCents 0 and anything SATIE would buy is refused.
    max_cents: int = 0
    allow_generation: bool = False  # may SATIE add sounds that weren't recorded?
    max_extra_voices: int = 3
    # SATIE saves each soundscape to the account's gallery, public by default. Memories
    # are private, so don't, unless asked ("private" needs a paid plan).
    save_soundscapes: str = "off"
    world: str = "memoryscapes"
    poll_attempts: int = 20
    poll_interval_s: float = 15.0


def build_scene_manifest(request: SoundscapeRequest) -> dict[str, Any]:
    """The satie.scene/v1 manifest (satie://schema/scene). Placed sources are objects,
    hosted `at` our source ids; every recording is a sample the script plays by name.
    Metres, +Y up, our world frame unchanged. No rooms: a recorded outdoor place isn't
    an enclosure, and a room would add reverb the recording already has."""
    objects = [
        {
            "id": s.source.id,
            "label": s.source.label,
            "position": s.source.position,
            "motion": "static",
            "radiusM": max(float(s.source.spread_m or 0.0), 0.5),
            "occludes": False,
            "notes": f"where the recorded {s.source.label} was heard",
        }
        for s in request.sources
        if s.source.position is not None
    ]
    samples = [
        {
            "name": s.clip,
            "url": s.url,
            "loop": True,
            "role": "stationary-texture" if s.source.position is not None else "diffuse-background",
            "notes": f"recorded {s.source.label}, separated from the original recording",
        }
        for s in request.sources
    ]
    samples.append({"name": "memory/ambient", "url": request.ambient_url, "loop": True,
                    "role": "diffuse-background", "notes": "the rest of the original recording"})
    return {
        "environment": {"setting": "a real place someone recorded and walked through",
                        "notes": "Everything audible must come from the supplied recordings."},
        "listener": "first_person",
        "objects": objects,
        "samples": samples,
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
    lines += ["", "Ambient bed (loop each as `bed`, heard everywhere):", "- clip memory/ambient (the rest of the recording)"]
    for s in diffuse:
        lines.append(f"- clip {s.clip} ({s.source.label}, no single location)")
    lines.append("")
    if cfg.allow_generation:
        lines.append(
            f"You may add up to {cfg.max_extra_voices} quiet voices that fit the place. "
            "Never replace or cover the recordings."
        )
    else:
        lines.append("Use only the recordings above: every voice plays one of these clips. "
                     "No `gen`, `synth` or library (`community/`) voices, and nothing added.")
        # satie_brief's open questions, answered for every memory (it runs unattended, and
        # its defaults would add a library crowd bed and weather/time beds).
        lines += [
            "No events: the visitor's own footsteps and actions are not sounded, so no `on` voices.",
            "The ground is not voiced. Time of day and weather are whatever the recordings hold: "
            "add no rain, wind, insect or bird beds.",
            "People are heard only as recorded (the speech clip, if listed); no crowd bed of any kind.",
            "One state: nothing is gated by `zone`.",
        ]
    if request.mood_prompt:
        lines.append(f"Mood: {request.mood_prompt}")
    lines.append("This runs unattended: where you have open questions, use your stated assumptions.")
    return "\n".join(lines)


_GEN_OR_SYNTH = re.compile(
    r"^\s*(?:\d+\s*\*\s*)?(?:(?:loop|oneshot)\s+|on\s+\S+\s+)?(?:gen|synth)\b", re.MULTILINE)


def original_only_violations(report: dict[str, Any], script: str, allowed: set[str]) -> list[str]:
    """Why a script isn't original-only, from satie_check's report and the script text.
    Empty means every voice plays one of our recordings and realize buys nothing."""
    out = []
    takes = int(((report.get("takes") or {}).get("total")) or 0)
    if takes:
        out.append(f"{takes} new take(s) SATIE would generate")
    library = report.get("libraryClips") or []
    if library:
        out.append("library recordings: " + ", ".join(map(str, library)))
    foreign = [c for c in (report.get("clipsToSupply") or []) if c not in allowed]
    if foreign:
        out.append("clips that aren't our recordings: " + ", ".join(foreign))
    voices = report.get("voices") or []  # per voice, when the report includes them
    odd = sorted({str(v.get("clip") or v.get("label")) for v in voices
                  if v.get("generated") or (v.get("clip") and v["clip"] not in allowed)})
    if odd:
        out.append("voices that don't play our recordings: " + ", ".join(odd))
    if _GEN_OR_SYNTH.search(script):
        out.append("generated or synthesized voices in the script")
    return out


def realized_violations(realize: dict[str, Any], samples: dict[str, Any]) -> list[str]:
    """After realize: anything SATIE delivered that isn't one of our recordings.
    samples.json entries carry `source` ("provided" for our files)."""
    out = []
    if int(realize.get("generated") or 0):
        out.append(f"{realize['generated']} generated sound(s)")
    if int(realize.get("fromLibrary") or 0):
        out.append(f"{realize['fromLibrary']} library recording(s)")
    entries = samples.get("samples", samples) if isinstance(samples, dict) else {}
    foreign = sorted(k for k, v in entries.items()
                     if isinstance(v, dict) and v.get("source") not in (None, "provided"))
    if foreign:
        out.append("clips that aren't our files: " + ", ".join(foreign))
    return out


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

        raw["inspect"] = inspect = call("satie_inspect_scene", {"scene": build_scene_manifest(request)})
        contract_id = _get(inspect, "contractId", "contract_id")
        if not contract_id:
            raise SatieError("satie_inspect_scene returned no contractId")

        brief = build_brief(request, self.cfg)
        raw["brief"] = call("satie_brief", {"brief": brief, "contractId": contract_id})

        raw["compose"] = compose = call("satie_compose", {"brief": brief, "contractId": contract_id})
        script = extract_script(compose)

        # One revision at most (the free plan has few compositions a month): it fixes
        # missing recordings and, unless generation is allowed, anything not recorded.
        allowed = set(samples_map(request))
        hosts = [s.source.id for s in request.sources if s.source.position is not None]
        missing = [c for c in allowed if c not in script]
        raw["check"] = report = call("satie_check", {"script": script, "hosts": hosts})
        foreign = [] if self.cfg.allow_generation else original_only_violations(report, script, allowed)
        if missing or foreign:
            asks = []
            if missing:
                asks.append("Every recording must be used. Missing clips: " + ", ".join(sorted(missing)) + ".")
            if foreign:
                asks.append("Use only the supplied recordings. Remove " + "; ".join(foreign) + ".")
            raw["compose_revision"] = compose = call(
                "satie_compose", {"brief": " ".join(asks), "contractId": contract_id, "script": script}
            )
            script = extract_script(compose)
            raw["check_revision"] = report = call("satie_check", {"script": script, "hosts": hosts})
            still = [c for c in missing if c not in script]
            if still:
                notes.append("SATIE script does not use: " + ", ".join(sorted(still)))
            if not self.cfg.allow_generation:
                foreign = original_only_violations(report, script, allowed)
                if foreign:
                    raise SatieNotOriginal(
                        "SATIE kept sounds that weren't recorded after a revision (" + "; ".join(foreign)
                        + "); nothing was realized or bought")

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
        if not self.cfg.allow_generation:
            wrong = realized_violations(realize, samples)
            if wrong:
                raise SatieNotOriginal("SATIE delivered sounds that weren't recorded (" + "; ".join(wrong)
                                       + "); the soundscape isn't used")
        if int(realize.get("maxSpentCents") or 0):
            notes.append(f"SATIE spent {realize['maxSpentCents']} cents")
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
        original_only = not self.cfg.allow_generation
        args = {
            "script": script,
            # Original only: our recordings are reused for free, so there's nothing to buy
            # and any purchase would be a sound that wasn't recorded. Never spend then.
            "maxCents": 0 if original_only else self.cfg.max_cents,
            "confirm": {"quoteId": quote_id},
            "samples": samples_map(request),
            "preferLibrary": not original_only,  # no library stand-ins for our recordings
            "saveSoundscapes": self.cfg.save_soundscapes,
            "world": self.cfg.world,
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
            save_soundscapes=os.environ.get("SATIE_SAVE_SOUNDSCAPES", "off"),
            world=os.environ.get("SATIE_WORLD", "memoryscapes"),
        )
        return SatieMcpProvider(transport, cfg)
    raise ValueError(f"Unknown soundscape provider: {name}")
