# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.fastapi_app.app import main


def test_metadata_cycleways_returns_geojson(monkeypatch):
    monkeypatch.setattr(
        main.cycleway_service,
        "load_cycleways",
        lambda: {
            "type": "FeatureCollection",
            "name": "gran_concepcion_cycleways",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"osm_id": 123, "category": "cycle_lane"},
                    "geometry": {"type": "LineString", "coordinates": [[-73.05, -36.82], [-73.04, -36.81]]},
                }
            ],
        },
    )

    with TestClient(main.app) as client:
        response = client.get("/metadata/cycleways")

    payload = response.json()
    assert response.status_code == 200
    assert payload["type"] == "FeatureCollection"
    assert payload["features"][0]["properties"]["category"] == "cycle_lane"
