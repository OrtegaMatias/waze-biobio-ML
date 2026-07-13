# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from backend.fastapi_app.app import main


def test_build_hotspot_points_handles_missing_times(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "tipo_evento": "Congestion",
                "lat": -36.8,
                "lon": -73.0,
                "hora_inicio": "",
                "hora_fin": None,
                "dia_semana": "Monday",
                "franja_horaria": "Punta AM (06-09h)",
                "segment_id": "seg-1",
                "velocidad_kmh": 15,
            }
        ]
    )

    monkeypatch.setattr(main.data_loader, "load_congestion_events", lambda: df)

    points = main._build_hotspot_points()

    assert len(points) == 1
    assert points[0]["hora_inicio_float"] is None
    assert points[0]["hora_fin_float"] is None


def test_metadata_hotspots_filters_bbox_day_and_hour(monkeypatch):
    monkeypatch.setattr(
        main,
        "_cached_hotspots",
        lambda _limit: [
            {
                "lat": -36.82,
                "lon": -73.05,
                "weight": 0.5,
                "day": "Wednesday",
                "bucket": "Punta AM (06-09h)",
                "segment_id": "seg-a",
                "hora_inicio_float": 7.0,
                "hora_fin_float": 8.0,
            },
            {
                "lat": -36.75,
                "lon": -73.15,
                "weight": 0.3,
                "day": "Friday",
                "bucket": "Punta PM (18-21h)",
                "segment_id": "seg-b",
                "hora_inicio_float": 18.0,
                "hora_fin_float": 19.0,
            },
        ],
    )

    with TestClient(main.app) as client:
        response = client.get(
            "/metadata/hotspots",
            params={
                "bbox": "-73.10,-36.90,-73.00,-36.70",
                "day_of_week": "Wednesday",
                "departure_hour": 8,
                "limit": 200,
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert len(payload["points"]) == 1
    assert payload["points"][0]["segment_id"] == "seg-a"


def test_metadata_congestion_dates_reports_available_and_missing_days(monkeypatch):
    df = pd.DataFrame(
        [
            {"fecha": "2025-03-13"},
            {"fecha": "2025-03-13"},
            {"fecha": "2025-03-15"},
        ]
    )
    monkeypatch.setattr(main.data_loader, "load_congestion_events", lambda: df)
    monkeypatch.setattr(main, "CONGESTION_COVERAGE_FILES", {})

    with TestClient(main.app) as client:
        response = client.get("/metadata/congestion/dates")

    payload = response.json()
    assert response.status_code == 200
    assert payload["start"] == "2025-03-13"
    assert payload["end"] == "2025-03-15"
    assert payload["available_dates"] == ["2025-03-13", "2025-03-15"]
    assert payload["missing_dates"] == ["2025-03-14"]
    assert payload["available_days"] == 2
    assert payload["calendar_days"] == 3
