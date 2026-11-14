"""Soundscape generation through the configured SoundscapeProvider (SATIE or stub)."""

import os

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.providers.soundscape import get_provider


def run(ctx: StageContext) -> None:
    provider = get_provider(os.environ.get("SOUNDSCAPE_PROVIDER", "stub"))
    # TODO: load localized stems from ctx.prefix("localize"), call provider.generate(...),
    # and store the raw provider output under ctx.prefix("soundscape") so it can be
    # re-rendered later without regenerating.
    raise NotImplementedError(f"soundscape stage with provider {provider.name!r}")
