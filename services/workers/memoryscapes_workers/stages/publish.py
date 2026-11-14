"""Privacy pass, previews and manifest publish.

- Before public visibility: blur faces and license plates (retrain or mask the splat);
  optionally detect and mute or flag speech.
- Render a thumbnail and a 5-second fly-through preview for globe pins.
- Build the manifest, validate it against packages/manifest/memory.schema.json,
  write it and warm the CDN.
"""

from memoryscapes_workers.context import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError
