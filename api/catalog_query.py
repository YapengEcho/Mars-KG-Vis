"""One catalog query implementation shared by HTTP and Agent tools."""
import math
from api.schemas import Catalog, Slide

Bounds = tuple[float, float, float, float]


def parse_bbox(value: str | None) -> Bounds | None:
    """west,south,east,north in degrees; west > east crosses the dateline."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("bbox must be west,south,east,north")
    try:
        parts = tuple(float(part.strip()) for part in value.split(','))
    except ValueError:
        raise ValueError("bbox must contain four finite numbers") from None
    if len(parts) != 4 or not all(math.isfinite(n) for n in parts):
        raise ValueError("bbox must contain four finite numbers")
    west, south, east, north = parts
    if not (-180 <= west <= 180 and -180 <= east <= 180
            and -90 <= south <= north <= 90):
        raise ValueError("bbox longitude must be within [-180,180], latitude within [-90,90], south <= north")
    return west, south, east, north


def filter_records(catalog: Catalog, q: str | None = None,
                   slide_type: str | None = None, bbox: Bounds | None = None) -> list[Slide]:
    term = (q or '').strip().casefold()
    result = []
    for record in catalog.records:
        if slide_type and record.type != slide_type:
            continue
        if term and term not in ' '.join((record.id,
                str(record.sourceId) if record.sourceId is not None else '',
                record.name, record.summary)).casefold():
            continue
        if bbox is not None:
            west, south, east, north = bbox
            # -180 and +180 represent the same meridian.
            longitudes = (record.longitude, -record.longitude) if abs(record.longitude) == 180 else (record.longitude,)
            within = any(west <= lon <= east if west <= east else lon >= west or lon <= east
                         for lon in longitudes)
            if not within or not south <= record.latitude <= north:
                continue
        result.append(record)
    return result
