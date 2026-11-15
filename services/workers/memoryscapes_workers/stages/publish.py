"""Privacy pass, previews and manifest publish.

Implemented: build the manifest from the stage outputs, validate it against
packages/manifest/memory.schema.json and write it, for **private** memories. When Spark's
build-lod tool is installed (visual/lod_tree.py), also build the splat's streaming LoD tree
from lod0 (``lod_tree``: auto | quality | quick | off). It's built here, not in the splat
stage, because publishing is cheap to redo (installing the tool later rebuilds only this
stage) and because the privacy pass will change the splat here first.

Not yet (needed before ``unlisted``/``public``, which this stage refuses until then):
- Blur faces and license plates (retrain or mask the splat); optionally detect and mute
  or flag speech.
- Render a thumbnail and a 5-second fly-through preview for globe pins; warm the CDN.

Reads   splat/lods.json, poses/navigation.json, ingest/meta.json,
        soundscape/scene.json (optional: without it the memory has no sound, since sound
        always comes from SATIE after separation and localization; the original
        recording is still listed as the ambient bed)
Writes  publish/manifest.json
        publish/lod/splat-lod.rad, splat-lod-<i>.radc  (the LoD tree, when built; listed in
        the manifest as splat.rad)

Asset URLs in the manifest are keys relative to the memory root, e.g.
"splat/lods/lod0.spz". The API returns an ``asset_base`` next to the manifest and the
viewer resolves URLs against it, so one manifest works on local disk, S3 or a CDN and
never stores expiring presigned URLs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from memoryscapes_workers.context import StageContext
from memoryscapes_workers.stagekit import StageError, StageIO, run_stage, write_json
from memoryscapes_workers.visual import lod_tree

STAGE = "publish"
VERSION = "1"
MANIFEST = "manifest.json"

_BED_FORMATS = {"mono": "mono", "stereo": "stereo", "foa": "ambisonic_foa"}


class PublishConfig(BaseModel):
    visibility: Literal["private", "unlisted", "public"] = "private"
    title: str | None = None
    owner: str | None = None
    # The streaming LoD tree: "auto" builds one ("quality") when build-lod is installed.
    # Resolved before fingerprinting, so installing the tool republishes.
    lod_tree: Literal["auto", "quality", "quick", "off"] = "auto"


def run(ctx: StageContext) -> dict:
    cfg = PublishConfig(**ctx.stage_config(STAGE))
    if cfg.visibility != "private":
        raise StageError(
            STAGE, f"Can't publish as {cfg.visibility!r} yet: the privacy pass (face and "
            "license-plate blur, speech check) isn't implemented.",
            "Publish as private for now.")
    scene_key = ctx.key("soundscape", "scene.json")
    has_scene = ctx.storage.exists(scene_key)
    # The audio stages write no stage marker, so fingerprint their output directly.
    digest = hashlib.sha256(ctx.storage.read_bytes(scene_key)).hexdigest()[:16] if has_scene else None
    version = _previous_version(ctx) + 1
    strict = cfg.lod_tree != "auto"
    tool = lod_tree.find_tool() if cfg.lod_tree != "off" else None
    if cfg.lod_tree == "auto":
        cfg = cfg.model_copy(update={"lod_tree": "quality" if tool else "off"})
    elif cfg.lod_tree != "off" and tool is None:
        raise StageError(STAGE, f"lod_tree={cfg.lod_tree!r} needs Spark's build-lod tool, which "
                         "isn't installed.", lod_tree.INSTALL_HINT)
    extra = {"soundscape": digest} if digest else {}
    if cfg.lod_tree != "off":
        extra["lod_tree_tool"] = lod_tree.tool_version()
    return run_stage(ctx, STAGE, version=VERSION, config=cfg,
                     upstream=["ingest", "poses", "splat"],
                     extra_inputs=extra or None,
                     body=lambda io: _publish(io, cfg, version, has_scene, tool, strict))


def _previous_version(ctx: StageContext) -> int:
    key = ctx.key(STAGE, MANIFEST)
    if not ctx.storage.exists(key):
        return 0
    try:
        return int(json.loads(ctx.storage.read_bytes(key)).get("version", 0))
    except (ValueError, TypeError):
        return 0


def _publish(io: StageIO, cfg: PublishConfig, version: int, has_scene: bool,
             tool: list[str] | None = None, strict: bool = False) -> dict:
    ctx = io.ctx
    lods = io.read_upstream_json("splat", "lods.json")
    nav = io.read_upstream_json("poses", "navigation.json")
    meta = io.read_upstream_json("ingest", "meta.json")
    scene = io.read_upstream_json("soundscape", "scene.json") if has_scene else None
    tree, tree_bytes, tree_note = None, 0, None
    if cfg.lod_tree != "off" and tool and lods.get("lods"):
        try:
            tree, tree_bytes = _build_lod_tree(io, tool, lods["lods"][0]["file"], cfg.lod_tree)
        except lod_tree.LodTreeError as exc:
            if strict:
                raise StageError(STAGE, f"Couldn't build the streaming LoD tree: {exc}") from exc
            # Optional: the viewer falls back to the .spz levels.
            tree_note = str(exc)
    manifest = build_manifest(ctx.memory_id, version=version, lods=lods, navigation=nav,
                              ingest_meta=meta, scene=scene, cfg=cfg,
                              has_audio=ctx.storage.exists(ctx.key("ingest", "audio.wav")),
                              rad=tree)
    validate_manifest(manifest)
    write_json(io.out_dir / MANIFEST, manifest)
    audio = manifest["audio"]
    summary = {
        "version": version,
        "visibility": cfg.visibility,
        "lods": len(manifest["splat"]["lods"]),
        "audio_sources": len(audio["sources"]),
        "placed_sources": sum(1 for s in audio["sources"] if s.get("position") is not None),
        "soundscape": audio.get("soundscape_provider", {}).get("name"),
    }
    if tree is not None:
        summary["lod_tree"] = {"chunks": len(tree["chunks"]), "count": tree.get("count"),
                               "mb": round(tree_bytes / 1e6, 1)}
    elif tree_note:
        summary["lod_tree_error"] = tree_note
    return summary


def _build_lod_tree(io: StageIO, tool: list[str], lod0_file: str, method: str) -> tuple[dict, int]:
    """Build publish/lod/splat-lod.rad (+ chunks) from the splat's lod0. Returns the
    manifest's splat.rad entry and the tree's size in bytes."""
    ctx = io.ctx
    io.progress(0.1, "building the streaming LoD tree")
    src = io.work_dir / "lod_tree_src" / Path(lod0_file).name
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(ctx.storage.read_bytes(ctx.key("splat", lod0_file)))  # just lod0, not all of splat/
    built = lod_tree.build(tool, src, io.out_dir / "lod", method=method)
    entry: dict[str, Any] = {"url": f"{STAGE}/lod/{built.header}",
                             "chunks": [f"{STAGE}/lod/{c}" for c in built.chunks]}
    if built.count is not None:
        entry["count"] = built.count
    return entry, built.bytes


def build_manifest(memory_id: str, *, version: int, lods: dict, navigation: dict,
                   ingest_meta: dict, scene: dict | None, cfg: PublishConfig,
                   has_audio: bool, rad: dict | None = None) -> dict[str, Any]:
    def rel(key: str) -> str:
        """Storage key -> URL relative to the memory root."""
        prefix = f"{memory_id}/"
        return key[len(prefix):] if key.startswith(prefix) else key

    coord: dict[str, Any] = {"units": "meters", "up": "+y"}
    geo = ingest_meta.get("geo")
    if geo and geo.get("lat") is not None and geo.get("lon") is not None:
        coord["origin_geo"] = [geo["lat"], geo["lon"], geo.get("alt") or 0.0]

    nav: dict[str, Any] = {"original_path": "poses/path.json"}
    for key in ("spawn", "walkable_bounds", "floor_y", "eye_height_m"):
        if navigation.get(key) is not None:
            nav[key] = navigation[key]

    audio: dict[str, Any]
    if scene is not None:
        audio = {"sources": [_source(s, rel) for s in scene.get("sources", [])]}
        bed = scene.get("ambient_bed")
        if bed and bed.get("key"):
            audio["ambient_bed"] = {"url": rel(bed["key"]),
                                    "format": _BED_FORMATS.get(bed.get("format"), "stereo")}
        provider = scene.get("provider", {})
        audio["soundscape_provider"] = {"name": provider.get("name", "unknown")}
        if provider.get("realize_id"):
            audio["soundscape_provider"]["job_id"] = provider["realize_id"]
        if provider.get("fallback_reason"):
            audio["soundscape_provider"]["fallback_reason"] = provider["fallback_reason"]
        satie = scene.get("satie")
        audio["satie"] = None if not satie else {
            "script_url": rel(satie["script_key"]), "samples_url": rel(satie["samples_key"]),
            "host_map": satie.get("host_map", {})}
    else:
        audio = {"sources": [], "soundscape_provider": {
            "name": "none",
            "fallback_reason": "The audio stages haven't run yet. Sound comes from SATIE once "
                               "the recording is separated and localized."}}
        if has_audio:
            layout = (ingest_meta.get("mic") or {}).get("layout", "stereo")
            audio["ambient_bed"] = {"url": "ingest/audio.wav",
                                    "format": _BED_FORMATS.get(layout, "stereo")}

    meta_out: dict[str, Any] = {"visibility": cfg.visibility}
    if cfg.title:
        meta_out["title"] = cfg.title
    if cfg.owner:
        meta_out["owner"] = cfg.owner
    if ingest_meta.get("captured_at"):
        meta_out["captured_at"] = ingest_meta["captured_at"]

    # Most detailed first; the viewer loads the last (coarsest) one first.
    splat: dict[str, Any] = {"lods": [{"url": f"splat/{lod['file']}", "count": int(lod["count"])}
                                      for lod in lods["lods"]]}
    if rad is not None:
        splat["rad"] = rad
    return {
        "id": memory_id,
        "version": version,
        "coord_frame": coord,
        "splat": splat,
        "navigation": nav,
        "audio": audio,
        "meta": meta_out,
    }


def _source(s: dict, rel) -> dict[str, Any]:
    out = {"id": s["id"], "label": s["label"], "url": rel(s["key"]), "position": s.get("position")}
    for key in ("trajectory", "ref_distance", "rolloff", "gain", "loop", "confidence", "method"):
        if s.get(key) is not None:
            out[key] = s[key]
    return out


def schema_path() -> Path:
    """packages/manifest/memory.schema.json: MANIFEST_SCHEMA, else found from this file."""
    if os.environ.get("MANIFEST_SCHEMA"):
        return Path(os.environ["MANIFEST_SCHEMA"])
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "packages" / "manifest" / "memory.schema.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("memory.schema.json not found; set MANIFEST_SCHEMA")


def validate_manifest(manifest: dict) -> None:
    import jsonschema

    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    try:
        jsonschema.Draft202012Validator(schema).validate(manifest)
    except jsonschema.ValidationError as err:
        path = "/".join(str(p) for p in err.absolute_path) or "(root)"
        raise StageError(STAGE, f"The manifest doesn't match the schema at {path}: {err.message}") from err
