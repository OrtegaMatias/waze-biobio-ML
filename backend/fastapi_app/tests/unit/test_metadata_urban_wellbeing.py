# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.fastapi_app.app import main


def test_metadata_urban_wellbeing_returns_geojson(monkeypatch):
    monkeypatch.setattr(
        main.urban_wellbeing_service,
        "load_wellbeing_features",
        lambda: {
            "type": "FeatureCollection",
            "name": "gran_concepcion_urban_wellbeing",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"feature_id": "park-1", "category": "green_space"},
                    "geometry": {"type": "Point", "coordinates": [-73.05, -36.82]},
                }
            ],
        },
    )

    with TestClient(main.app) as client:
        response = client.get("/metadata/urban-wellbeing")

    payload = response.json()
    assert response.status_code == 200
    assert payload["features"][0]["properties"]["category"] == "green_space"
