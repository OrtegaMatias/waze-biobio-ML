from __future__ import annotations

import json

from backend.fastapi_app.app.services import urban_wellbeing_service
from backend.fastapi_app.app.services.urban_wellbeing_service import UrbanWellbeingService


def test_route_only_scores_urban_features_adjacent_to_the_path(tmp_path):
    payload = {
        "type": "FeatureCollection",
        "name": "test",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "feature_id": "park-next-to-route",
                    "name": "Parque junto a la ruta",
                    "category": "green_space",
                    "subtype": "park",
                    "base_weight": 1.0,
                    "source": "test",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-73.0001, -36.0001], [-73.0001, -35.9999], [-72.9999, -35.9999], [-72.9999, -36.0001], [-73.0001, -36.0001]]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "feature_id": "park-two-blocks-away",
                    "name": "Parque a dos cuadras",
                    "category": "green_space",
                    "subtype": "park",
                    "base_weight": 1.0,
                    "source": "test",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-72.9981, -36.0001], [-72.9981, -35.9999], [-72.9979, -35.9999], [-72.9979, -36.0001], [-72.9981, -36.0001]]],
                },
            },
        ],
    }
    path = tmp_path / "wellbeing.geojson"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = UrbanWellbeingService(str(path)).evaluate_route(
        [{"lat": -36.0, "lon": -73.001}, {"lat": -36.0, "lon": -72.999}],
    )

    assert result["nearby_buffer_m"] == 30.0
    assert [feature["feature_id"] for feature in result["top_features"]] == ["park-next-to-route"]


def test_cycleways_can_be_used_as_urban_wellbeing_evidence(monkeypatch):
    monkeypatch.setattr(
        urban_wellbeing_service,
        "load_wellbeing_features",
        lambda _path=None: {"type": "FeatureCollection", "name": "empty", "features": []},
    )
    monkeypatch.setattr(
        urban_wellbeing_service,
        "load_cycleways",
        lambda: {
            "type": "FeatureCollection",
            "name": "cycleways",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "local_id": "local-victor-lamas-cycleway",
                        "name": "Victor Lamas",
                        "category": "local_verified_cycleway",
                        "source": "Correccion local",
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-73.001, -36.0], [-72.999, -36.0]],
                    },
                }
            ],
        },
    )

    result = UrbanWellbeingService().evaluate_route(
        [{"lat": -36.0, "lon": -73.001}, {"lat": -36.0, "lon": -72.999}],
    )

    assert result["cycleway_ratio"] == 1.0
    assert result["top_features"][0]["feature_id"] == "cycleway-local-local-victor-lamas-cycleway"
    assert result["top_features"][0]["category"] == "cycleway"


def test_short_cycleway_segments_do_not_count_as_urban_wellbeing_evidence(monkeypatch):
    monkeypatch.setattr(
        urban_wellbeing_service,
        "load_wellbeing_features",
        lambda _path=None: {"type": "FeatureCollection", "name": "empty", "features": []},
    )
    monkeypatch.setattr(
        urban_wellbeing_service,
        "load_cycleways",
        lambda: {
            "type": "FeatureCollection",
            "name": "cycleways",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "local_id": "short-cycleway",
                        "name": "Ciclovia corta",
                        "category": "local_verified_cycleway",
                        "source": "Correccion local",
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-73.001, -36.0], [-73.0009, -36.0]],
                    },
                }
            ],
        },
    )

    result = UrbanWellbeingService().evaluate_route(
        [{"lat": -36.0, "lon": -73.001}, {"lat": -36.0, "lon": -72.999}],
    )

    assert result["cycleway_ratio"] == 0.0
    assert result["top_features"] == []
