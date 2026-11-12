"""Country outlines for the world map: Natural Earth admin-0 polygons in the Robinson projection,
matched to the countries of the prompts by ISO A2 code.

Land areas come from the Equal Earth projection (Šavrič, Patterson and Jenny 2018): it is
equal-area, so the area of a polygon projected on a sphere of radius R in kilometres is its area in
square kilometres.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib import path as mpath

A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
SQRT3 = np.sqrt(3.0)
# Robinson's table (Snyder 1993): the length of each parallel and its distance from the equator,
# every 5 degrees of latitude from 0 to 90.
ROBINSON_LAT = np.arange(0.0, 91.0, 5.0)
ROBINSON_X = np.array(
    [1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962]
    + [0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322]
)
ROBINSON_Y = np.array(
    [0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571]
    + [0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000]
)


def equal_earth(
    lon: np.ndarray | float, lat: np.ndarray | float, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """``x`` and ``y`` of longitudes and latitudes in degrees, on a sphere of ``radius``."""
    lam, phi = np.radians(lon), np.radians(lat)
    theta = np.arcsin(SQRT3 / 2 * np.sin(phi))
    t2 = theta**2
    t6 = t2**3
    x = 2 * SQRT3 * lam * np.cos(theta) / (3 * (9 * A4 * t6 * t2 + 7 * A3 * t6 + 3 * A2 * t2 + A1))
    y = theta * (A4 * t6 * t2 + A3 * t6 + A2 * t2 + A1)
    return radius * x, radius * y


def robinson(
    lon: np.ndarray | float, lat: np.ndarray | float, radius: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """``x`` and ``y`` of longitudes and latitudes in degrees: the table interpolated linearly."""
    lam, phi = np.radians(lon), np.asarray(lat, dtype=float)
    length = np.interp(np.abs(phi), ROBINSON_LAT, ROBINSON_X)
    height = np.interp(np.abs(phi), ROBINSON_LAT, ROBINSON_Y)
    return radius * 0.8487 * length * lam, radius * 1.3523 * height * np.sign(phi)


def signed_area(ring: np.ndarray) -> float:
    """Shoelace area of a closed or open ring of ``(n, 2)`` points; positive if anticlockwise."""
    x, y = ring[:, 0], ring[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2


def _oriented(ring: np.ndarray, anticlockwise: bool) -> np.ndarray:
    return ring if (signed_area(ring) > 0) == anticlockwise else ring[::-1]


def _centroid(ring: np.ndarray) -> tuple[float, float]:
    x, y = ring[:, 0], ring[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
    return float(cx), float(cy)


@dataclass(frozen=True, slots=True)
class Country:
    """One admin-0 country: its Natural Earth code, its ISO A2 code if it is a country of the
    prompts (else empty), its polygons in the Robinson projection, each an anticlockwise exterior
    ring followed by clockwise holes, and its land area (exteriors minus holes, equal-area)."""

    code: str
    iso2: str
    polygons: tuple[tuple[np.ndarray, ...], ...]
    area: float

    @property
    def centroid(self) -> tuple[float, float]:
        """The centroid of the largest polygon's exterior."""
        largest = max(self.polygons, key=lambda polygon: signed_area(polygon[0]))
        return _centroid(largest[0])


def _clip(ring: np.ndarray, at: float, above: bool, axis: int = 0) -> np.ndarray:
    """The part of a ring whose coordinate ``axis`` (0 longitude, 1 latitude) is at least (or at
    most) ``at`` (Sutherland-Hodgman, one half-plane)."""
    inside = ring[:, axis] >= at if above else ring[:, axis] <= at
    points = []
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        if inside[k]:
            points.append(p)
        if inside[k] != inside[(k + 1) % len(ring)]:
            t = (at - p[axis]) / (q[axis] - p[axis])
            crossing = p + t * (q - p)
            crossing[axis] = at
            points.append(crossing)
    return np.asarray(points).reshape(-1, 2)


def _open(ring: Sequence[Sequence[float]]) -> np.ndarray:
    """A ring without the repeated first vertex that closes it in GeoJSON."""
    points = np.asarray(ring, dtype=float)
    return points[:-1] if (points[0] == points[-1]).all() else points


def _closed(ring: np.ndarray) -> np.ndarray:
    return np.vstack([ring, ring[:1]])


def _joined(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The union of two open rings that share one run of vertices, their common border: the run
    dropped and ``b``'s own vertices put in its place."""
    a, b = _oriented(a, anticlockwise=True), _oriented(b, anticlockwise=True)
    shared = {tuple(p) for p in a} & {tuple(p) for p in b}
    in_a = np.array([tuple(p) in shared for p in a])
    in_b = np.array([tuple(p) in shared for p in b])
    for mask in (in_a, in_b):
        if len(shared) < 2 or np.count_nonzero(mask != np.roll(mask, 1)) != 2:
            raise ValueError("the rings do not share a single run of vertices")
    # ``a`` from its first own vertex: its own vertices, then the run; ``b`` runs it backwards.
    start = int(np.flatnonzero(~in_a & np.roll(in_a, 1))[0])
    a, in_a = np.roll(a, -start, axis=0), np.roll(in_a, -start)
    first, last = a[in_a][0], a[-1]
    start = int(np.flatnonzero((b == first).all(axis=1))[0])
    b, in_b = np.roll(b, -start, axis=0), np.roll(in_b, -start)
    return np.vstack([a[~in_a], first, b[~in_b], last])


def _polygons(geometry: Mapping[str, Any]) -> list[list[np.ndarray]]:
    parts = geometry["coordinates"]
    polygons = parts if geometry["type"] == "MultiPolygon" else [parts]
    return [[np.asarray(ring, dtype=float) for ring in polygon] for polygon in polygons]


def ceded(
    features: Sequence[Mapping[str, Any]],
    source: str,
    target: str,
    at: Sequence[float] | None = None,
    south_of: float | None = None,
) -> list[dict[str, Any]]:
    """``features`` with a part of country ``source`` joined to its neighbour ``target`` (Natural
    Earth ADM0_A3 codes) across their common border: the polygon of ``source`` that holds the
    point ``at`` (longitude, latitude), or the part of ``source`` south of the parallel
    ``south_of``. The part and ``target`` are single polygons without holes."""
    if (at is None) == (south_of is None):
        raise ValueError("give either a point or a parallel")
    by_code = {feature["properties"]["ADM0_A3"]: feature for feature in features}
    kept = _polygons(by_code[source]["geometry"])
    (target_polygon,) = _polygons(by_code[target]["geometry"])
    if at is not None:
        (k,) = [k for k, p in enumerate(kept) if mpath.Path(p[0]).contains_point(tuple(at))]
        part = _open(kept.pop(k)[0])
    else:
        (polygon,) = kept
        ring = _open(polygon[0])
        kept = [[_closed(_clip(ring, south_of, above=True, axis=1))]]
        part = _clip(ring, south_of, above=False, axis=1)
    joined = [[_closed(_joined(_open(target_polygon[0]), part))]]
    geometry = {source: kept, target: joined}
    return [
        {**feature, "geometry": {"type": "MultiPolygon", "coordinates": geometry[code]}}
        if (code := feature["properties"]["ADM0_A3"]) in geometry
        else dict(feature)
        for feature in features
    ]


def _recentred(
    polygon: Sequence[Sequence[Sequence[float]]], central_lon: float
) -> list[list[np.ndarray]]:
    """A polygon's rings with longitudes relative to ``central_lon``, split where they cross the
    new antimeridian: one polygon per side. A centre west of Greenwich puts the seam at +180, so
    longitudes are mirrored while splitting to keep it at -180."""
    mirror = np.array([-1.0 if central_lon < 0 else 1.0, 1.0])
    rings = [(np.asarray(ring, dtype=float) - [central_lon, 0.0]) * mirror for ring in polygon]
    if all((r[:, 0] >= -180).all() for r in rings):
        sides = [rings]
    elif all((r[:, 0] <= -180).all() for r in rings):
        sides = [[r + [360.0, 0.0] for r in rings]]
    else:
        sides = []
        for east, shift in ((True, 0.0), (False, 360.0)):
            # The exterior has points on both sides; a hole may lie on one side only.
            parts = [_clip(r, -180.0, east) + [shift, 0.0] for r in rings]
            sides.append([part for part in parts if len(part) >= 3])
    return [[part * mirror for part in side] for side in sides]


Projection = Callable[..., tuple[np.ndarray, np.ndarray]]


def _project(
    rings: Iterable[np.ndarray], radius: float, projection: Projection
) -> tuple[np.ndarray, ...]:
    projected = []
    for k, ring in enumerate(rings):
        x, y = projection(ring[:, 0], ring[:, 1], radius)
        projected.append(_oriented(np.column_stack([x, y]), anticlockwise=k == 0))
    return tuple(projected)


def iso2_of(
    properties: Mapping[str, Any], study: Collection[str], overrides: Mapping[str, str]
) -> str:
    """The study's ISO A2 code of a feature: by ``overrides`` (ISO A2 to Natural Earth ADM0_A3)
    for the codes Natural Earth lacks, else by its ISO_A2; empty for a country outside the study."""
    by_code = {code: iso2 for iso2, code in overrides.items()}
    iso2 = by_code.get(properties["ADM0_A3"], properties["ISO_A2"])
    return iso2 if iso2 in study else ""


def countries(
    features: Sequence[Mapping[str, Any]],
    study: Collection[str],
    overrides: Mapping[str, str],
    drop_continents: Collection[str],
    radius: float,
    central_lon: float = 0.0,
) -> list[Country]:
    """Every country not on a dropped continent, in the Robinson projection about ``central_lon``,
    in the order of its Natural Earth code; each country of the study matched once."""
    shapes = []
    for feature in features:
        properties, geometry = feature["properties"], feature["geometry"]
        if properties["CONTINENT"] in drop_continents:
            continue
        parts = geometry["coordinates"]
        polygons = parts if geometry["type"] == "MultiPolygon" else [parts]
        recentred = [part for polygon in polygons for part in _recentred(polygon, central_lon)]
        area = sum(
            signed_area(ring) for part in recentred for ring in _project(part, radius, equal_earth)
        )
        shapes.append(
            Country(
                properties["ADM0_A3"],
                iso2_of(properties, study, overrides),
                tuple(_project(part, radius, robinson) for part in recentred),
                area,
            )
        )
    matched = [c.iso2 for c in shapes if c.iso2]
    unmatched = sorted(set(study) - set(matched))
    twice = sorted({c for c in matched if matched.count(c) > 1})
    if unmatched or twice:
        raise ValueError(f"countries not matched: {unmatched}; matched twice: {twice}")
    return sorted(shapes, key=lambda c: c.code)
