"""Soundscape generation through the configured SoundscapeProvider (SATIE or stub).

Inputs:  localize/sources.json, separate/stems.json
Outputs:
    soundscape/scene.json          provider, object-based sources, ambient bed, SATIE refs
    soundscape/satie/scene.satie   SATIE script        (SATIE provider only)
    soundscape/satie/samples.json  SATIE samples        (saved as returned, never edited)
    soundscape/satie/raw.json      every tool response, for debugging and re-rendering

If SATIE fails and SOUNDSCAPE_FALLBACK=stub (the default), the memory still gets the
object-based soundscape, and scene.json records why SATIE was skipped.
"""

from __future__ import annotations

import os

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


def build_request(ctx: StageContext, mood_prompt: str | None = None) -> SoundscapeRequest:
    storage = ctx.storage
    localized = read_json(storage, ctx.key("localize", "sources.json"))
    index = StemsIndex.from_json(read_json(storage, ctx.key("separate", "stems.json")))
    sources = [LocalizedSource.from_json(s) for s in localized["sources"]]
    return SoundscapeRequest(
        memory_id=ctx.memory_id,
        sources=[SoundscapeSource(s, storage.url_for(s.stem_key)) for s in sources],
        ambient_key=index.ambient_key,
        ambient_url=storage.url_for(index.ambient_key),
        ambient_channels=index.channels,
        bounds_min=localized["bounds"]["min"],
        bounds_max=localized["bounds"]["max"],
        mood_prompt=mood_prompt,
    )


def run(
    ctx: StageContext,
    provider: SoundscapeProvider | None = None,
    fallback: SoundscapeProvider | None = None,
    mood_prompt: str | None = None,
) -> dict:
    provider = provider or get_provider(os.environ.get("SOUNDSCAPE_PROVIDER", "stub"))
    if fallback is None and os.environ.get("SOUNDSCAPE_FALLBACK", "stub") == "stub":
        fallback = StubProvider()
    request = build_request(ctx, mood_prompt)

    error: str | None = None
    try:
        result = provider.generate(request)
    except Exception as exc:  # noqa: BLE001 (any provider failure falls back)
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
