from fastapi import APIRouter, HTTPException, Query

from app import db
from app.models import GlobeMemory, GlobeResponse
from app.settings import get_settings

router = APIRouter()

WORLD = (-180.0, -90.0, 180.0, 90.0)


@router.get("", response_model=GlobeResponse)
def query_globe(
    bbox: str | None = Query(None, description="minLon,minLat,maxLon,maxLat (default: the whole world). "
                                                "minLon > maxLon crosses the ±180° line."),
) -> GlobeResponse:
    """The dev user's memories that have a location, newest first, for the homepage globe.

    Grouping nearby memories into clusters happens in the browser for now
    (docs/webapp-plan/backend.md § GET /globe). Server-side clustering (PostGIS) comes back
    with many more memories, or public ones from other users.
    """
    min_lon, min_lat, max_lon, max_lat = _parse_bbox(bbox) if bbox else WORLD
    try:
        rows = db.list_in_bbox(min_lon, min_lat, max_lon, max_lat, owner_id=get_settings().dev_user_id)
    except ValueError as err:
        raise HTTPException(422, str(err)) from None
    return GlobeResponse(memories=[GlobeMemory(**{k: r[k] for k in GlobeMemory.model_fields}) for r in rows])


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    parts = bbox.split(",")
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        values = ()
    if len(values) != 4 or not all(v == v and abs(v) != float("inf") for v in values):
        raise HTTPException(422, "bbox must be four numbers: minLon,minLat,maxLon,maxLat")
    return values  # type: ignore[return-value]
