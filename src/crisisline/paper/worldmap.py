"""Country outlines for the world map: Natural Earth admin-0 polygons in the Equal Earth projection
(Šavrič, Patterson and Jenny 2018), matched to the countries of the prompts by ISO A2 code.

Equal Earth is an equal-area projection, so the area of a projected polygon on a sphere of radius R
in kilometres is its area in square kilometres.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
SQRT3 = np.sqrt(3.0)


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
    prompts (else empty), and its projected polygons, each an anticlockwise exterior ring followed
    by clockwise holes."""

    code: str
    iso2: str
    polygons: tuple[tuple[np.ndarray, ...], ...]

    @property
    def area(self) -> float:
        """Projected area: exteriors minus holes."""
        return sum(signed_area(ring) for polygon in self.polygons for ring in polygon)

    @property
    def centroid(self) -> tuple[float, float]:
        """The centroid of the largest polygon's exterior."""
        largest = max(self.polygons, key=lambda polygon: signed_area(polygon[0]))
        return _centroid(largest[0])


def _clip(ring: np.ndarray, east: bool) -> np.ndarray:
    """The part of a ring east (or west) of longitude -180 (Sutherland-Hodgman, one half-plane)."""
    inside = ring[:, 0] >= -180 if east else ring[:, 0] <= -180
    points = []
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        if inside[k]:
            points.append(p)
        if inside[k] != inside[(k + 1) % len(ring)]:
            t = (-180 - p[0]) / (q[0] - p[0])
            points.append(np.array([-180.0, p[1] + t * (q[1] - p[1])]))
    return np.asarray(points).reshape(-1, 2)


def _recentred(
    polygon: Sequence[Sequence[Sequence[float]]], central_lon: float
) -> list[list[np.ndarray]]:
    """A polygon's rings with longitudes relative to ``central_lon``, split where they cross the
    new antimeridian: one polygon per side."""
    rings = [np.asarray(ring, dtype=float) - [central_lon, 0.0] for ring in polygon]
    if all((r[:, 0] >= -180).all() for r in rings):
        return [rings]
    if all((r[:, 0] <= -180).all() for r in rings):
        return [[r + [360.0, 0.0] for r in rings]]
    sides = []
    for east, shift in ((True, 0.0), (False, 360.0)):
        parts = [_clip(r, east) + [shift, 0.0] for r in rings]
        if len(parts[0]) >= 3:
            sides.append([part for part in parts if len(part) >= 3])
    return sides


def _project(rings: Iterable[np.ndarray], radius: float) -> tuple[np.ndarray, ...]:
    projected = []
    for k, ring in enumerate(rings):
        x, y = equal_earth(ring[:, 0], ring[:, 1], radius)
        projected.append(_oriented(np.column_stack([x, y]), anticlockwise=k == 0))
    return tuple(projected)


def iso2_of(
    properties: Mapping[str, Any], study: Collection[str], overrides: Mapping[str, str]
) -> str:
    """The study's ISO A2 code of a feature: by ``overrides`` (ISO A2 to Natural Earth ADM0_A3)
    for the codes Natural Earth lacks, else by its ISO_A2; empty for a country outside the study."""
    by_code = {code: iso2 for iso2, code in overrides.items()}
    if properties["ADM0_A3"] in by_code:
        return by_code[properties["ADM0_A3"]]
    return properties["ISO_A2"] if properties["ISO_A2"] in study else ""


def countries(
    features: Sequence[Mapping[str, Any]],
    study: Collection[str],
    overrides: Mapping[str, str],
    drop_continents: Collection[str],
    radius: float,
    central_lon: float = 0.0,
) -> list[Country]:
    """Every country not on a dropped continent, projected about ``central_lon``, in the order of
    its Natural Earth code; each country of the study matched once."""
    shapes = []
    for feature in features:
        properties, geometry = feature["properties"], feature["geometry"]
        if properties["CONTINENT"] in drop_continents:
            continue
        parts = geometry["coordinates"]
        polygons = parts if geometry["type"] == "MultiPolygon" else [parts]
        shapes.append(
            Country(
                properties["ADM0_A3"],
                iso2_of(properties, study, overrides),
                tuple(
                    _project(part, radius)
                    for polygon in polygons
                    for part in _recentred(polygon, central_lon)
                ),
            )
        )
    matched = [c.iso2 for c in shapes if c.iso2]
    unmatched = sorted(set(study) - set(matched))
    twice = sorted({c for c in matched if matched.count(c) > 1})
    if unmatched or twice:
        raise ValueError(f"countries not matched: {unmatched}; matched twice: {twice}")
    return sorted(shapes, key=lambda c: c.code)
