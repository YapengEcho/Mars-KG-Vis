"""Mars latitude conventions and the planetographic -> planetocentric conversion.

The handoff inventory stores *planetographic* latitudes (老师 2026-10-31 确认：
「纬度用行星面」), while ``api.schemas.CoordinateSystem`` only accepts
``planetocentric, degrees`` and the published basemaps (MOLA 463 m, MDIM 2.1)
are planetocentric. The two differ by up to 0.338 deg (about 20 km) around
latitude 45 deg, so the export converts explicitly instead of relabelling.

IAU 2000 Mars ellipsoid: a = 3396190 m, b = 3376200 m (1/f = 169.894).
``tan(phi_c) = (b/a)^2 * tan(phi_g)``.
"""

from __future__ import annotations

import math

MARS_SEMI_MAJOR_M = 3396190.0
MARS_SEMI_MINOR_M = 3376200.0
AXIS_RATIO = MARS_SEMI_MINOR_M / MARS_SEMI_MAJOR_M
SQUARED_AXIS_RATIO = AXIS_RATIO**2
FLATTENING = 1.0 - AXIS_RATIO
KM_PER_DEGREE = math.pi * MARS_SEMI_MAJOR_M / 180.0 / 1000.0  # 59.27 km per degree of arc


def planetographic_to_planetocentric(latitude_deg: float) -> float:
    """Convert one planetographic latitude to planetocentric, preserving the sign."""
    if abs(latitude_deg) >= 90.0:
        return latitude_deg
    return math.degrees(math.atan(SQUARED_AXIS_RATIO * math.tan(math.radians(latitude_deg))))


def planetocentric_to_planetographic(latitude_deg: float) -> float:
    """Inverse of :func:`planetographic_to_planetocentric` (for round-trip checks)."""
    if abs(latitude_deg) >= 90.0:
        return latitude_deg
    return math.degrees(math.atan(math.tan(math.radians(latitude_deg)) / SQUARED_AXIS_RATIO))


def shift_deg(latitude_deg: float) -> float:
    """Absolute offset between the two conventions at a given latitude, in degrees."""
    return abs(planetographic_to_planetocentric(latitude_deg) - latitude_deg)


def max_shift_deg() -> float:
    """Largest offset over the sphere (0.338 deg near 45 deg)."""
    best = 0.0
    for step in range(-900, 901):
        best = max(best, shift_deg(step / 10.0))
    return best


def describe() -> dict[str, float | str]:
    """Machine-readable record for manifests."""
    return {
        "from": "planetographic",
        "to": "planetocentric",
        "ellipsoid": "IAU_2000_Mars",
        "semiMajorM": MARS_SEMI_MAJOR_M,
        "semiMinorM": MARS_SEMI_MINOR_M,
        "axisRatio": round(AXIS_RATIO, 9),
        "flattening": round(FLATTENING, 9),
        "maxShiftDeg": round(max_shift_deg(), 4),
    }
