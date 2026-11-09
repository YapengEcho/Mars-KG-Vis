"""Mars latitude conventions, and the optional planetographic conversion.

Background
----------
The handoff inventory does **not** document which latitude convention it uses.
The two candidates differ by up to 0.338 deg (about 20 km) near latitude 45 deg:

* planetographic (行星面) — the angle of the surface normal against the equator;
* planetocentric (行星心) — the angle at the centre of the ellipsoid.

``api.schemas.CoordinateSystem`` only accepts ``planetocentric, degrees`` and the
published basemaps (MOLA 463 m, MDIM 2.1) are planetocentric, so sooner or later
the question has to be answered. Until it is answered with a source, the default
here is ``unverified``: **values are passed through untouched** and the export
stays in ``staging``.

Evidence against converting by default
--------------------------------------
The 1083 preview crops record their own ``manual_box_world_m`` and
``native_window_world_m`` in the local CTX mosaic projection, which is
planetocentric. Those windows were cut around the same points, and the crop
centres match the catalog latitudes exactly. Converting the catalog to
planetocentric while leaving the frames alone pushes **736 of the 3 points that
sit outside their manual box** out to 739: i.e. 736 points that were inside the
manual frame leave it. ``admission_report.manual_box_evidence`` reproduces that
count, and ``scripts/compare_latitude_conventions.py`` prints the full table.

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

CONVENTIONS = ("unverified", "planetocentric", "planetographic")
DEFAULT_CONVENTION = "unverified"


def resolve_convention(convention: str) -> str:
    """Normalise ``convention`` and reject anything outside :data:`CONVENTIONS`."""
    value = (convention or "").strip().lower()
    if value not in CONVENTIONS:
        raise ValueError(f"未知纬度约定：{convention!r}（可选 {', '.join(CONVENTIONS)}）")
    return value


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


def to_export_latitude(latitude_deg: float, convention: str) -> float:
    """Latitude to publish for a source latitude, given the source convention.

    Only an explicit ``planetographic`` source is converted. ``unverified`` and
    ``planetocentric`` pass the value through unchanged.
    """
    if resolve_convention(convention) == "planetographic":
        return planetographic_to_planetocentric(latitude_deg)
    return latitude_deg


def export_label(convention: str) -> str:
    """Value for ``coordinateSystem.latitude`` in an export payload."""
    convention = resolve_convention(convention)
    return "unverified" if convention == "unverified" else "planetocentric, degrees"


def shift_deg(latitude_deg: float) -> float:
    """Absolute offset between the two conventions at a given latitude, in degrees."""
    return abs(planetographic_to_planetocentric(latitude_deg) - latitude_deg)


def max_shift_deg() -> float:
    """Largest offset over the sphere (0.338 deg near 45 deg)."""
    best = 0.0
    for step in range(-900, 901):
        best = max(best, shift_deg(step / 10.0))
    return best


def describe(convention: str = DEFAULT_CONVENTION) -> dict[str, float | str | bool]:
    """Machine-readable record for manifests, reports and mappings."""
    convention = resolve_convention(convention)
    return {
        "from": convention,
        "to": "planetocentric",
        "applied": convention == "planetographic",
        "label": export_label(convention),
        "ellipsoid": "IAU_2000_Mars",
        "semiMajorM": MARS_SEMI_MAJOR_M,
        "semiMinorM": MARS_SEMI_MINOR_M,
        "axisRatio": round(AXIS_RATIO, 9),
        "flattening": round(FLATTENING, 9),
        "maxShiftDeg": round(max_shift_deg(), 4),
    }
