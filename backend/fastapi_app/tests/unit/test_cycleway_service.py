# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from backend.fastapi_app.app.services import cycleway_service


def test_load_cycleways_merges_local_overrides(monkeypatch, tmp_path):
    base_path = tmp_path / "base.geojson"
    override_path = tmp_path / "overrides.geojson"
    base_path.write_text(
        json.dumps({"type": "FeatureCollection", "name": "base", "features": []}),
        encoding="utf-8",
    )
    override_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "overrides",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"local_id": "local-victor-lamas-cycleway", "source": "Corrección local"},
                        "geometry": {"type": "LineString", "coordinates": [[-73.05, -36.83], [-73.04, -36.82]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cycleway_service, "DEFAULT_CYCLEWAYS_PATH", base_path)
    monkeypatch.setattr(cycleway_service, "DEFAULT_MINVU_PATH", tmp_path / "missing_minvu.geojson")
    monkeypatch.setattr(cycleway_service, "DEFAULT_OVERRIDES_PATH", override_path)
    cycleway_service.load_cycleways.cache_clear()

    payload = cycleway_service.load_cycleways()

    assert payload["features"][0]["properties"]["local_id"] == "local-victor-lamas-cycleway"
    assert payload["features"][0]["properties"]["source"] == "Corrección local"
    cycleway_service.load_cycleways.cache_clear()


def test_load_cycleways_merges_minvu_source(monkeypatch, tmp_path):
    base_path = tmp_path / "base.geojson"
    minvu_path = tmp_path / "minvu.geojson"
    override_path = tmp_path / "missing_overrides.geojson"
    base_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "base",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"osm_id": 123, "source": "OpenStreetMap/Overpass"},
                        "geometry": {"type": "LineString", "coordinates": [[-73.07, -36.83], [-73.06, -36.82]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    minvu_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "name": "minvu",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"minvu_id": 456, "source": "MINVU GeoIDE"},
                        "geometry": {"type": "LineString", "coordinates": [[-73.06, -36.83], [-73.05, -36.82]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cycleway_service, "DEFAULT_CYCLEWAYS_PATH", base_path)
    monkeypatch.setattr(cycleway_service, "DEFAULT_MINVU_PATH", minvu_path)
    monkeypatch.setattr(cycleway_service, "DEFAULT_OVERRIDES_PATH", override_path)
    cycleway_service.load_cycleways.cache_clear()

    payload = cycleway_service.load_cycleways()

    assert [feature["properties"].get("source") for feature in payload["features"]] == [
        "OpenStreetMap/Overpass",
        "MINVU GeoIDE",
    ]
    assert payload["features"][1]["properties"]["minvu_id"] == 456
    cycleway_service.load_cycleways.cache_clear()


def test_estimate_route_coverage_uses_osm_overpass_geojson(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "name": "gran_concepcion_cycleways",
        "features": [
            {
                "type": "Feature",
                "properties": {"source": "OpenStreetMap/Overpass"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-73.00005, -36.0], [-73.00005, -36.01]],
                },
            }
        ],
    }
    path = tmp_path / "cycleways.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    coverage = cycleway_service.estimate_route_coverage(
        [{"lat": -36.0, "lon": -73.0}, {"lat": -36.01, "lon": -73.0}],
        path=str(path),
        nearby_buffer_m=80.0,
    )

    assert coverage["available"] is True
    assert coverage["has_high_coverage"] is True
    assert coverage["coverage_ratio"] > 0.8
    assert coverage["data_source"] == "OpenStreetMap/Overpass"


def test_estimate_route_coverage_stays_low_when_cycleway_is_far(tmp_path):
    geojson = {
        "type": "FeatureCollection",
        "name": "gran_concepcion_cycleways",
        "features": [
            {
                "type": "Feature",
                "properties": {"source": "OpenStreetMap/Overpass"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-73.01, -36.0], [-73.01, -36.01]],
                },
            }
        ],
    }
    path = tmp_path / "cycleways.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    coverage = cycleway_service.estimate_route_coverage(
        [{"lat": -36.0, "lon": -73.0}, {"lat": -36.01, "lon": -73.0}],
        path=str(path),
        nearby_buffer_m=40.0,
    )

    assert coverage["available"] is True
    assert coverage["has_high_coverage"] is False
    assert coverage["coverage_ratio"] == 0.0
