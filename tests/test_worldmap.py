"""Tests of the Equal Earth projection and the matching of Natural Earth countries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from crisisline.cli import load_config, read_csv
from crisisline.paper import worldmap


def test_equal_earth_reference_points() -> None:
    # The projection's own extents on the unit sphere: the equator at 180 degrees, the pole.
    x, y = worldmap.equal_earth(180.0, 0.0)
    assert float(x) == pytest.approx(2.7066299836, abs=1e-8) and float(y) == 0.0
    x, y = worldmap.equal_earth(0.0, 90.0)
    assert float(x) == 0.0 and float(y) == pytest.approx(1.3173627592, abs=1e-8)
    # Symmetric in longitude and latitude; scaled by the radius.
    x, y = worldmap.equal_earth(np.array([-40.0, 40.0]), np.array([-25.0, 25.0]), radius=2.0)
    assert x[0] == -x[1] and y[0] == -y[1]
    assert x[1] == pytest.approx(2 * float(worldmap.equal_earth(40.0, 25.0)[0]))


def test_equal_earth_keeps_areas() -> None:
    """A band of 60 by 30 degrees has the area it has on the sphere wherever it lies."""
    for lat in (0.0, 30.0, 50.0):
        edge = np.linspace(0, 1, 400)
        lon = np.concatenate([60 * edge, np.full(400, 60.0), 60 * edge[::-1], np.zeros(400)])
        lats = lat + 30 * np.concatenate([np.zeros(400), edge, np.ones(400), edge[::-1]])
        x, y = worldmap.equal_earth(lon, lats)
        sphere = np.radians(60) * (np.sin(np.radians(lat + 30)) - np.sin(np.radians(lat)))
        assert abs(worldmap.signed_area(np.column_stack([x, y]))) == pytest.approx(sphere, rel=1e-4)


def test_robinson_takes_its_table_at_the_nodes() -> None:
    x, y = worldmap.robinson(180.0, 0.0)
    assert float(x) == pytest.approx(0.8487 * np.pi) and float(y) == 0.0
    x, y = worldmap.robinson(0.0, 90.0)
    assert float(x) == 0.0 and float(y) == pytest.approx(1.3523)
    x, y = worldmap.robinson(np.array([-90.0, 90.0]), np.array([-45.0, 45.0]), radius=2.0)
    assert x[1] == pytest.approx(2 * 0.8487 * 0.8962 * np.pi / 2)
    assert y[1] == pytest.approx(2 * 1.3523 * 0.5571)
    assert x[0] == -x[1] and y[0] == -y[1]
    # Between two nodes, linear: at 47.5 degrees, halfway between 45 and 50.
    _, y = worldmap.robinson(0.0, 47.5)
    assert float(y) == pytest.approx(1.3523 * (0.5571 + 0.6176) / 2)


def _square(lon: float, lat: float, side: float) -> list[list[float]]:
    # Clockwise on purpose: the projection orients rings itself.
    return [[lon, lat], [lon, lat + side], [lon + side, lat + side], [lon + side, lat]]


def _feature(code: str, iso2: str, continent: str, *polygons: list) -> dict[str, Any]:
    geometry = (
        {"type": "Polygon", "coordinates": polygons[0]}
        if len(polygons) == 1
        else {"type": "MultiPolygon", "coordinates": list(polygons)}
    )
    return {
        "properties": {"ADM0_A3": code, "ISO_A2": iso2, "CONTINENT": continent},
        "geometry": geometry,
    }


FEATURES = [
    _feature("FRA", "-99", "Europe", [_square(0, 45, 2)], [_square(-53, 4, 1)]),
    _feature("DEU", "DE", "Europe", [_square(8, 50, 4), _square(9, 51, 1)]),  # with a hole
    _feature("ATA", "AQ", "Antarctica", [_square(0, -80, 5)]),
    _feature("NAM", "NA", "Africa", [_square(15, -22, 3)]),
]


def test_countries_match_by_iso2_and_overrides_and_drop_continents() -> None:
    shapes = worldmap.countries(FEATURES, {"FR", "DE"}, {"FR": "FRA"}, {"Antarctica"}, 1.0)
    assert [(c.code, c.iso2) for c in shapes] == [("DEU", "DE"), ("FRA", "FR"), ("NAM", "")]
    germany = shapes[0]
    exterior, hole = germany.polygons[0]
    assert worldmap.signed_area(exterior) > 0 > worldmap.signed_area(hole)

    # Land area on the unit sphere: the square minus its hole.
    def band(west: float, east: float, south: float, north: float) -> float:
        return np.radians(east - west) * (np.sin(np.radians(north)) - np.sin(np.radians(south)))

    assert germany.area == pytest.approx(band(8, 12, 50, 54) - band(9, 10, 51, 52), rel=1e-2)
    # France's centroid is that of its larger polygon, in Europe.
    assert shapes[1].centroid[1] == pytest.approx(float(worldmap.robinson(1.0, 46.0)[1]), rel=1e-2)


def test_recentring_splits_a_country_across_the_new_antimeridian() -> None:
    """Centred on 10 degrees east, the seam is at 170 degrees west: a square from 175 to 165
    degrees west becomes two polygons at the two edges of the map, with the same total area."""
    crossing = [_feature("XXX", "XX", "Asia", [_square(-175, 60, 10)])]
    whole = worldmap.countries(crossing, {"XX"}, {}, (), 1.0)[0]
    split = worldmap.countries(crossing, {"XX"}, {}, (), 1.0, central_lon=10.0)[0]
    assert len(split.polygons) == 2
    edges = sorted(float(polygon[0][:, 0].mean()) for polygon in split.polygons)
    assert edges[0] < -1.5 and edges[1] > 1.5
    assert split.area == pytest.approx(whole.area, rel=1e-9)


def test_recentring_west_of_greenwich_splits_at_the_seam_in_the_east() -> None:
    """Centred on 10 degrees west, the seam is at 170 degrees east: a square from 165 to 175
    degrees east becomes two polygons inside the map's outline, with the same total area."""
    crossing = [_feature("XXX", "XX", "Asia", [_square(165, 60, 10)])]
    whole = worldmap.countries(crossing, {"XX"}, {}, (), 1.0)[0]
    split = worldmap.countries(crossing, {"XX"}, {}, (), 1.0, central_lon=-10.0)[0]
    assert len(split.polygons) == 2
    edge = float(worldmap.robinson(180.0, 0.0)[0])
    assert max(float(np.abs(polygon[0][:, 0]).max()) for polygon in split.polygons) <= edge
    assert split.area == pytest.approx(whole.area, rel=1e-9)


def test_an_override_outside_the_study_matches_no_country() -> None:
    france = {"ADM0_A3": "FRA", "ISO_A2": "-99"}
    assert worldmap.iso2_of(france, {"FR"}, {"FR": "FRA"}) == "FR"
    assert worldmap.iso2_of(france, {"DE"}, {"FR": "FRA"}) == ""


def test_countries_fail_on_a_code_they_cannot_match() -> None:
    with pytest.raises(ValueError, match=r"not matched: \['FR'\]"):
        worldmap.countries(FEATURES, {"FR", "DE"}, {}, {"Antarctica"}, 1.0)


def test_the_committed_map_matches_every_country_of_the_prompts() -> None:
    config = load_config(Path("configs/experiment.yaml"))
    world = config["paper"]["world_map"]
    study = {r["iso2"] for r in read_csv(Path(config["paths"]["countries"]))}
    features = json.loads(Path(config["paths"]["world_map"]).read_text(encoding="utf-8"))
    shapes = worldmap.countries(
        features["features"],
        study,
        world["iso2_to_adm0"],
        world["drop_continents"],
        world["earth_radius_km"],
        world["central_longitude"],
    )
    assert {c.iso2 for c in shapes if c.iso2} == study
    lebanon = next(c for c in shapes if c.iso2 == "LB")
    assert lebanon.area == pytest.approx(10_450, rel=0.05)  # km²
