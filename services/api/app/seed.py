"""Fill the database with fake memories for building the globe before real uploads exist.

    cd services/api && .venv/bin/python -m app.seed

Safe to run again: each seed memory has a fixed id. A finished one is left alone; one left
half-made by an interrupted run is completed. Seed
memories have no files in storage, so opening one in the viewer shows nothing.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app import db
from app.models import MemoryStatus
from app.settings import get_settings

SEED_PREFIX = "seed_"

# (id suffix, title, lat, lon, captured)
SEED_MEMORIES = [
    ("golden-gate", "Golden Gate walk", 37.8199, -122.4783, "2026-10-02T16:10:00"),
    ("stanford-oval", "Stanford Oval at dusk", 37.4294, -122.1697, "2026-10-19T18:40:00"),
    ("central-park", "Central Park birds", 40.7812, -73.9665, "2026-09-14T08:05:00"),
    ("paris-seine", "Along the Seine", 48.8584, 2.2945, "2026-07-21T20:15:00"),
    ("london-market", "Borough Market", 51.5055, -0.0910, "2026-08-03T12:30:00"),
    ("tokyo-crossing", "Shibuya crossing", 35.6595, 139.7005, "2026-06-11T21:00:00"),
    ("kyoto-bamboo", "Arashiyama bamboo grove", 35.0170, 135.6713, "2026-06-15T07:20:00"),
    ("sydney-harbour", "Sydney harbour ferry", -33.8568, 151.2153, "2026-05-02T10:45:00"),
    ("cape-town", "Cape Town waterfront", -33.9036, 18.4207, "2026-04-18T17:00:00"),
    ("rio-beach", "Copacabana waves", -22.9711, -43.1822, "2026-03-09T15:30:00"),
    ("fiji-reef", "Fiji reef boat", -17.7134, 178.0650, "2026-02-01T11:00:00"),  # near ±180°
]


def seed() -> int:
    """Add or complete any seed memories that aren't finished. Returns how many it touched."""
    db.init_db()
    owner = get_settings().dev_user_id
    added = 0
    for suffix, title, lat, lon, captured in SEED_MEMORIES:
        memory_id = SEED_PREFIX + suffix
        existing = db.get_memory(memory_id)
        if existing and existing["status"] == MemoryStatus.ready.value and existing["lat"] is not None:
            continue
        if existing is None:
            db.create_memory(owner, title=title, memory_id=memory_id)
        db.update_memory(
            memory_id, lat=lat, lon=lon, location_source="user_pin",
            captured_at=datetime.fromisoformat(captured).replace(tzinfo=timezone.utc))
        db.set_status(memory_id, MemoryStatus.ready)
        added += 1
    return added


if __name__ == "__main__":
    n = seed()
    print(f"Added {n} seed memories ({len(SEED_MEMORIES) - n} were already there).")
