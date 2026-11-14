"""Soundscape generation: SATIE arranges the localized stems (SOUNDSCAPE_PROVIDER, default
"satie"; "stub" only for tests and offline work). The viewer plays sound only through
SATIE's runtime, so a memory without SATIE output has no sound.

Inputs:  localize/sources.json, separate/stems.json
Outputs:
    soundscape/scene.json          provider, object-based sources, ambient bed, SATIE refs
    soundscape/satie/scene.satie   SATIE script        (SATIE provider only)
    soundscape/satie/samples.json  SATIE samples        (saved as returned, never edited)
    soundscape/satie/raw.json      every tool response, for debugging and re-rendering

If SATIE fails and SOUNDSCAPE_FALLBACK=stub (the default), the stage still finishes, so
the memory publishes with its splat and source positions but without sound. scene.json
records why SATIE was skipped, and the viewer shows it. Re-run the stage to try SATIE
again; the same script resumes its realize instead of buying again.

SATIE's runtime loads each stem from the URL it's given, in the viewer's browser. Set
PUBLIC_API_URL (the API's address as viewers reach it) so those URLs point at the API's
`/memories/<id>/files/...`; otherwise they're storage URLs (file:// for local storage).
"""

from __future__ import annotations

import os
from urllib.parse import quote

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.models import LocalizedSource, StemsIndex
from memoryscapes_workers.providers.soundscape import (
    SoundscapeProvider,
    SoundscapeRequest,
    SoundscapeResult,
    SoundscapeSource,
    StubProvider,
    get_provider,
)
from memoryscapes_workers.storage import read_json, write_json


def stem_url(ctx: StageContext, key: str) -> str:
    """Where a viewer's browser can fetch a stored file."""
    base, prefix = os.environ.get("PUBLIC_API_URL", "").rstrip("/"), f"{ctx.memory_id}/"
    if base and key.startswith(prefix):
        return f"{base}/memories/{quote(ctx.memory_id)}/files/{quote(key[len(prefix):])}"
    return ctx.storage.url_for(key)


def build_request(ctx: StageContext, mood_prompt: str | None = None) -> SoundscapeRequest:
    storage = ctx.storage
    localized = read_json(storage, ctx.key("localize", "sources.json"))
    index = StemsIndex.from_json(read_json(storage, ctx.key("separate", "stems.json")))
    sources = [LocalizedSource.from_json(s) for s in localized["sources"]]
    return SoundscapeRequest(
        memory_id=ctx.memory_id,
        sources=[SoundscapeSource(s, stem_url(ctx, s.stem_key)) for s in sources],
        ambient_key=index.ambient_key,
        ambient_url=stem_url(ctx, index.ambient_key),
        ambient_channels=index.channels,
        bounds_min=localized["bounds"]["min"],
        bounds_max=localized["bounds"]["max"],
        mood_prompt=mood_prompt,
        read_bytes=storage.read_bytes,
    )


def run(
    ctx: StageContext,
    provider: SoundscapeProvider | None = None,
    fallback: SoundscapeProvider | None = None,
    mood_prompt: str | None = None,
) -> dict:
    provider = provider or get_provider(os.environ.get("SOUNDSCAPE_PROVIDER", "satie"))
    if fallback is None and os.environ.get("SOUNDSCAPE_FALLBACK", "stub") == "stub":
        fallback = StubProvider()
    request = build_request(ctx, mood_prompt)

    error: str | None = None
    try:
        result = provider.generate(request)
    except Exception as exc:  # noqa: BLE001 (any provider failure falls back)
        partial = getattr(provider, "last_raw", None)
        if partial:  # what SATIE answered before it failed, for debugging
            write_json(ctx.storage, ctx.key("soundscape", "satie/failed_raw.json"), partial)
        if fallback is None or fallback is provider:
            raise
        error = f"{type(exc).__name__}: {exc}"
        result = fallback.generate(request)

    scene = _write(ctx, result)
    if error:
        scene["provider"]["fallback_reason"] = f"{provider.name} failed: {error}"
        write_json(ctx.storage, ctx.key("soundscape", "scene.json"), scene)
    return scene


def _write(ctx: StageContext, result: SoundscapeResult) -> dict:
    storage = ctx.storage
    scene: dict = {
        "provider": {"name": result.provider, "notes": result.notes},
        "ambient_bed": result.ambient_bed,
        "sources": result.sources,
        "satie": None,
    }
    if result.satie is not None:
        s = result.satie
        script_key = ctx.key("soundscape", "satie/scene.satie")
        samples_key = ctx.key("soundscape", "satie/samples.json")
        storage.write_bytes(script_key, s.script.encode(), "text/plain")
        write_json(storage, samples_key, s.samples)
        write_json(storage, ctx.key("soundscape", "satie/raw.json"), result.raw)
        scene["satie"] = {
            "script_key": script_key,
            "samples_key": samples_key,
            "host_map": s.host_map,
            "contract_id": s.contract_id,
            "realize_id": s.realize_id,
        }
        scene["provider"]["realize_id"] = s.realize_id
    write_json(storage, ctx.key("soundscape", "scene.json"), scene)
    return scene
