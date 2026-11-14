"""SoundscapeProvider adapter: turns localized stems into the memory's soundscape.

Providers:
- StubProvider: passes the localized sources through (ids, positions, levels) without a
  SATIE script. For tests and offline work, and what's kept when SATIE fails; the viewer
  plays no sound for it, since sound is always SATIE's.
- SatieMcpProvider: drives SATIE (Attractor Labs) through its MCP server, following the
  workflow its skill prescribes (github.com/Attractor-Laboratories/satie-skills):
      satie_inspect_scene -> satie_brief -> satie_compose -> satie_audit_scene -> satie_realize
  The output is a .satie script plus samples.json, played in the viewer by satie-runtime.

Checked against the live server (see services/workers/README.md, "SATIE against the live
server"): the satie.scene/v1 manifest, the DSL, the check/audit/realize fields, and that our
stems come back as `provided` samples at 0 cents. SATIE's runtime loads every sample from
the `url` we give, so the stems must be reachable by the viewer. With
SATIE_UPLOAD_STEMS=true the stems are also uploaded (satie_asset with a clipPath), which
SATIE keeps as provided samples; its listening review (satie_listen) needs a paid plan.
"""

from __future__ import annotations

import hashlib
import io
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
    # Reads a storage key (the stem WAVs), for uploading the recordings to SATIE.
    read_bytes: Callable[[str], bytes] | None = None


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
    # Upload the stems to SATIE (kept there as provided samples, and for training on the
    # Free plan). Playback doesn't need it: the runtime loads each sample from its url.
    upload_recordings: bool = False
    # SATIE pauses a long call at its time budget (about 45 s for compose) and resumes it
    # when the identical call is repeated; each repeat waits out the next budget itself.
    poll_attempts: int = 20
    poll_interval_s: float = 3.0


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


def recording_keys(request: SoundscapeRequest) -> dict[str, str]:
    """Clip name -> storage key of the WAV it plays."""
    keys = {s.clip: s.source.stem_key for s in request.sources}
    keys["memory/ambient"] = request.ambient_key
    return keys


def clip_path(clip: str) -> str:
    """The key a clip has in samples.json (`Audio/` + its name), which satie_asset uses."""
    return f"Audio/{clip}"


MAX_AUDIO_UPLOAD = 16 * 1024 * 1024  # satie_asset kind "audio"


def encode_for_upload(wav: bytes, max_bytes: int = MAX_AUDIO_UPLOAD) -> tuple[bytes, str]:
    """The WAV as is when it fits SATIE's audio limit, else MP3 (libsndfile 1.1+, which
    soundfile's wheels bundle). Not OGG: libsndfile's Vorbis encoder crashes the process
    on long in-memory files."""
    if len(wav) <= max_bytes:
        return wav, "audio/wav"
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(wav), dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="MP3")
    data = buf.getvalue()
    if len(data) > max_bytes:
        raise SatieError(f"{len(data) / 1e6:.1f} MB even as MP3, over SATIE's {max_bytes / 1e6:.0f} MB limit")
    return data, "audio/mpeg"


def samples_map(request: SoundscapeRequest) -> dict[str, str]:
    m = {s.clip: s.url for s in request.sources}
    m["memory/ambient"] = request.ambient_url
    return m


_UNFINISHED = {"in_progress", "paused", "running", "queued"}

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


def _http_put(url: str, data: bytes, content_type: str) -> None:
    req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310 (signed URL from SATIE)
        resp.read()


class SatieMcpProvider:
    name = "satie"

    def __init__(
        self,
        transport: McpTransport,
        cfg: SatieConfig = SatieConfig(),
        http_get_json: Callable[[str], dict[str, Any]] = _http_get_json,
        sleep: Callable[[float], None] = time.sleep,
        http_put: Callable[[str, bytes, str], None] = _http_put,
    ) -> None:
        self.transport = transport
        self.cfg = cfg
        self.http_get_json = http_get_json
        self.sleep = sleep
        self.http_put = http_put
        self.last_raw: dict[str, Any] = {}

    def generate(self, request: SoundscapeRequest) -> SoundscapeResult:
        raw: dict[str, Any] = {}
        self.last_raw = raw  # every response so far, kept by the stage if this fails
        notes: list[str] = []
        call = self.transport.call_tool

        raw["inspect"] = inspect = call("satie_inspect_scene", {"scene": build_scene_manifest(request)})
        contract_id = _get(inspect, "contractId", "contract_id")
        if not contract_id:
            raise SatieError("satie_inspect_scene returned no contractId")

        brief = build_brief(request, self.cfg)
        raw["brief"] = call("satie_brief", {"brief": brief, "contractId": contract_id})

        raw["compose"] = compose = self._until_done("satie_compose", {"brief": brief, "contractId": contract_id})
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
            raw["compose_revision"] = compose = self._until_done(
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
        # Live shape (2026-11-14): {"ok": true, "quote": {"quoteId", "maxCents", "expiresAt"}, ...}
        quote_id = _get(audit, "quoteId", "quote_id") or _get(audit.get("quote") or {}, "quoteId", "quote_id")
        if not quote_id:
            issues = "; ".join(map(str, audit.get("issues") or [])) or audit.get("text", "")[:300]
            raise SatieError(f"satie_audit_scene gave no quote: {issues or 'no reason given'}")

        if self.cfg.upload_recordings:  # only once the script is known to be original-only
            raw["uploads"] = self._upload_recordings(request, notes)
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

    def _upload_recordings(self, request: SoundscapeRequest, notes: list[str]) -> dict[str, Any]:
        """satie_asset begin -> PUT -> finalize for each recording, under its samples.json
        key. A failed upload is a note: playback doesn't depend on it."""
        if request.read_bytes is None:
            notes.append("recordings not uploaded to SATIE: no storage reader given")
            return {}
        call = self.transport.call_tool
        out: dict[str, Any] = {}
        for clip, key in recording_keys(request).items():
            path = clip_path(clip)
            try:
                data, media = encode_for_upload(request.read_bytes(key))
                begin = call("satie_asset", {"action": "begin", "kind": "audio", "clipPath": path})
                if not begin.get("signedUrl") or not begin.get("uploadId"):
                    raise SatieError("satie_asset begin returned no signedUrl/uploadId")
                self.http_put(begin["signedUrl"], data, media)
                done = call("satie_asset", {"action": "finalize", "kind": "audio",
                                            "uploadId": begin["uploadId"], "clipPath": path})
                out[clip] = {"audioId": done.get("audioId"), "clipPath": path, "bytes": len(data), "type": media}
            except Exception as exc:  # noqa: BLE001
                notes.append(f"couldn't upload {clip} to SATIE: {type(exc).__name__}: {exc}")
        return out

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
        result = self._until_done("satie_realize", args)
        if result.get("status") == "input_required":
            raise SatieBudgetExceeded(
                f"SATIE needs more than the {self.cfg.max_cents} cent cap for this memory. Raise SATIE_MAX_CENTS to allow it."
            )
        return result

    def _until_done(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Call `tool`, repeating the identical call while SATIE reports the run paused or
        in progress. That resumes the same run: nothing finished is redone, and compose
        doesn't count a new composition. Compose carries no requestKey on purpose: a
        keyless call resumes by its arguments, so a run paused by an older worker still
        resumes, where adding a key would start a new attempt."""
        for _ in range(self.cfg.poll_attempts):
            result = self.transport.call_tool(tool, args)
            if result.get("status") not in _UNFINISHED:
                return result
            self.sleep(self.cfg.poll_interval_s)
        raise SatieError(f"{tool} still running after {self.cfg.poll_attempts} calls; "
                         "re-run the stage to resume it (the identical call continues the same run)")


def get_provider(name: str) -> SoundscapeProvider:
    import os

    if name == "stub":
        return StubProvider()
    if name == "satie":
        from memoryscapes_workers.providers.mcp_transport import HttpMcpTransport
        from memoryscapes_workers.providers.satie_auth import bearer_from_env

        # SATIE_MCP_TOKEN if set, else the login saved by `memoryscapes-worker satie-login`.
        transport = HttpMcpTransport(os.environ.get("SATIE_MCP_URL", "https://www.satie.world/mcp"), bearer_from_env())
        cfg = SatieConfig(
            max_cents=int(os.environ.get("SATIE_MAX_CENTS", "0")),
            allow_generation=os.environ.get("SATIE_ALLOW_GENERATION", "false").lower() == "true",
            save_soundscapes=os.environ.get("SATIE_SAVE_SOUNDSCAPES", "off"),
            world=os.environ.get("SATIE_WORLD", "memoryscapes"),
            upload_recordings=os.environ.get("SATIE_UPLOAD_STEMS", "false").lower() == "true",
        )
        return SatieMcpProvider(transport, cfg)
    raise ValueError(f"Unknown soundscape provider: {name}")
