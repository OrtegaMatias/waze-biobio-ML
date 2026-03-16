# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from backend.fastapi_app.app.services import data_quality_service


def test_inspect_data_quality_flags_duplicate_sources(monkeypatch):
    incidents = pd.DataFrame(
        [
            {"fecha": "2025-07-01", "via": "Ruta Uno", "lat": -36.8, "lon": -73.0},
            {"fecha": "2025-07-02", "via": "Ruta Dos", "lat": -36.81, "lon": -73.01},
        ]
    )
    road_network = pd.DataFrame({"comuna": ["Concepción", "Q-44-P"]})

    monkeypatch.setattr(data_quality_service.data_loader, "data_version", lambda: ("concepcion", 1.0, 1.0, 1.0))
    monkeypatch.setattr(data_quality_service.data_loader, "get_data_profile", lambda: "concepcion")
    monkeypatch.setattr(data_quality_service, "_load_profiled_raw", lambda _path, _profile: incidents.copy())
    monkeypatch.setattr(data_quality_service.pd, "read_csv", lambda *_args, **_kwargs: road_network.copy())

    result = data_quality_service.inspect_data_quality()

    assert result["status"] == "warning"
    assert result["duplicate_incident_sources"] is True
    assert result["date_range"]["days"] == 2
    assert result["anomalous_communes"] == ["Q-44-P"]
