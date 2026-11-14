from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("")
def query_globe(
    bbox: str = Query(..., description="minLon,minLat,maxLon,maxLat"),
    zoom: int = Query(2, ge=0, le=20),
) -> dict:
    """Return clustered public memories inside the bounding box.

    Planned: PostGIS ST_ClusterDBSCAN on memories.location (GiST index),
    or precomputed per-zoom tiles once volume grows.
    """
    raise HTTPException(501, "Not implemented")
