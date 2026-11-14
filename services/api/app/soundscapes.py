"""Copy a memory's soundscape from file storage into the ``soundscapes`` table.

The workers' soundscape stage writes (services/workers/README.md):

    {id}/soundscape/scene.json          provider, notes, fallback reason, SATIE refs
    {id}/soundscape/satie/scene.satie   the SATIE script
    {id}/soundscape/satie/samples.json  SATIE's samples, as returned
    {id}/soundscape/satie/raw.json      every SATIE answer (stays a file: debugging only)

``sync(memory_id)`` reads those and stores one row, replacing the previous run's; the
pipeline calls it when a run finishes. The files stay where they are: the viewer keeps
loading the scene through the manifest, so nothing the frontend uses changes.

Backfill a memory made with the worker CLI (it has no database row yet):

    python -m app.soundscapes canada --register
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app import db
from app.models import MemoryStatus
from app.settings import get_settings
from app.storage import BlobStore, get_store

SCENE = "soundscape/scene.json"
RAW = "soundscape/satie/raw.json"


def _json(store: BlobStore, key: str) -> Any:
    return json.loads(store.read_bytes(key))


def _own_key(memory_id: str, key: str) -> str:
    """scene.json names the script and samples by storage key; only accept this memory's."""
    if not isinstance(key, str) or not key.startswith(f"{memory_id}/soundscape/") or ".." in key.split("/"):
        raise ValueError(f"unexpected soundscape file {key!r}")
    return key


def read_soundscape(memory_id: str, store: BlobStore | None = None) -> dict[str, Any] | None:
    """The soundscape record for ``db.upsert_soundscape``, or None if the stage hasn't run."""
    store = store or get_store()
    scene_key = f"{memory_id}/{SCENE}"
    if not store.exists(scene_key):
        return None
    scene = _json(store, scene_key)
    provider = scene.get("provider") or {}
    record: dict[str, Any] = {
        "provider": provider.get("name") or "unknown",
        "fallback_reason": provider.get("fallback_reason"),
        "notes": list(provider.get("notes") or []),
        "realize_id": provider.get("realize_id"),
    }
    satie = scene.get("satie")
    if satie:
        record.update(
            script=store.read_bytes(_own_key(memory_id, satie["script_key"])).decode("utf-8"),
            samples=_json(store, _own_key(memory_id, satie["samples_key"])),
            host_map=satie.get("host_map") or {},
            contract_id=satie.get("contract_id"),
            realize_id=satie.get("realize_id") or record["realize_id"],
        )
        raw_key = f"{memory_id}/{RAW}"
        realize = (_json(store, raw_key).get("realize") or {}) if store.exists(raw_key) else {}
        for field, name in (("spent_cents", "maxSpentCents"), ("generated", "generated"),
                            ("from_library", "fromLibrary")):
            if realize.get(name) is not None:
                record[field] = int(realize[name])
    return record


def sync(memory_id: str, store: BlobStore | None = None) -> dict[str, Any] | None:
    """Store the memory's current soundscape (or remove the row when there's none).
    Needs the memory's database row. Returns the stored row."""
    record = read_soundscape(memory_id, store)
    if record is None:
        db.delete_soundscape(memory_id)
        return None
    return db.upsert_soundscape(memory_id, record)


def register(memory_id: str, store: BlobStore | None = None) -> dict[str, Any]:
    """Give a memory made with the worker CLI a database row (the dev user's), marked
    ``ready`` when its manifest exists, so its soundscape can be stored."""
    store = store or get_store()
    row = db.get_memory(memory_id)
    if row is not None:
        return row
    manifest_key = f"{memory_id}/publish/manifest.json"
    manifest = _json(store, manifest_key) if store.exists(manifest_key) else None
    title = ((manifest or {}).get("meta") or {}).get("title") or memory_id
    db.create_memory(get_settings().dev_user_id, title=title, memory_id=memory_id)
    if manifest is not None:
        db.set_status(memory_id, MemoryStatus.ready)
    return db.get_memory(memory_id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.soundscapes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("memory_ids", nargs="+")
    ap.add_argument("--register", action="store_true",
                    help="create the database row for memories made with the worker CLI")
    args = ap.parse_args(argv)
    db.init_db()
    failed = 0
    for memory_id in args.memory_ids:
        if db.get_memory(memory_id) is None:
            if not args.register:
                print(f"{memory_id}: no database row (made with the worker CLI?); use --register")
                failed += 1
                continue
            register(memory_id)
        row = sync(memory_id)
        if row is None:
            print(f"{memory_id}: no soundscape yet (the soundscape stage hasn't run)")
            continue
        voices = len(row["samples"] or {})
        print(f"{memory_id}: stored {row['provider']} soundscape"
              + (f", {voices} sounds, {row['spent_cents'] or 0}¢" if row["script"] else f": {row['fallback_reason']}"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
