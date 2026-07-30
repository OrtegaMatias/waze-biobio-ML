# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from backend.fastapi_app.app.services import data_quality_service


class FakePath:
    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists


def test_inspect_data_quality_reports_congestion_only(monkeypatch):
    data_quality_service._inspect_data_quality.cache_clear()
    incidents = pd.DataFrame(
        [
            {"fecha": "2025-07-01", "via": "Ruta Uno", "comuna": "Concepcion"},
            {"fecha": "2025-07-02", "via": "Ruta Dos", "comuna": "Talcahuano"},
        ]
    )
    road_network = pd.DataFrame({"comuna": ["Concepcion", "Q-44-P"]})

    monkeypatch.setattr(
        data_quality_service.data_loader,
        "data_version",
        lambda: ("gran_concepcion", 1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(data_quality_service.data_loader, "get_data_profile", lambda: "gran_concepcion")
    monkeypatch.setattr(data_quality_service.data_loader, "incident_sources_are_duplicates", lambda: False)
    monkeypatch.setattr(data_quality_service, "_load_profiled_raw", lambda _path, _profile: incidents.copy())
    monkeypatch.setattr(data_quality_service.data_loader, "load_reference_network", lambda: road_network.copy())
    monkeypatch.setattr(data_quality_service.data_loader, "ROAD_NETWORK_PATH", FakePath(True))
    monkeypatch.setattr(data_quality_service.data_loader, "ACCIDENT_PATH", FakePath(False))

    result = data_quality_service.inspect_data_quality()

    assert result["status"] == "warning"
    assert result["duplicate_incident_sources"] is False
    assert result["date_range"]["days"] == 2
    assert result["anomalous_communes"] == ["Q-44-P"]
    assert result["raw_counts"]["accidents"] == 0
    assert result["raw_counts"]["congestions"] == 2


def test_inspect_data_quality_warns_when_road_network_is_missing(monkeypatch):
    data_quality_service._inspect_data_quality.cache_clear()
    incidents = pd.DataFrame(
        [
            {"fecha": "2025-07-01", "via": "Ruta Uno", "comuna": "Concepcion"},
        ]
    )

    monkeypatch.setattr(
        data_quality_service.data_loader,
        "data_version",
        lambda: ("gran_concepcion", 1.0, 1.0, 0.0),
    )
    monkeypatch.setattr(data_quality_service.data_loader, "get_data_profile", lambda: "gran_concepcion")
    monkeypatch.setattr(data_quality_service.data_loader, "incident_sources_are_duplicates", lambda: False)
    monkeypatch.setattr(data_quality_service, "_load_profiled_raw", lambda _path, _profile: incidents.copy())
    monkeypatch.setattr(data_quality_service.data_loader, "load_reference_network", lambda: pd.DataFrame())
    monkeypatch.setattr(data_quality_service.data_loader, "ROAD_NETWORK_PATH", FakePath(False))
    monkeypatch.setattr(data_quality_service.data_loader, "ACCIDENT_PATH", FakePath(False))

    result = data_quality_service.inspect_data_quality()

    assert result["status"] == "warning"
    assert result["anomalous_communes"] == []
    assert any("road_network.csv" in warning for warning in result["warnings"])


def test_missing_via_ratio_counts_each_row_once(monkeypatch):
    data_quality_service._inspect_data_quality.cache_clear()
    incidents = pd.DataFrame(
        [
            {"fecha": "2025-07-01", "via": None, "comuna": "Concepcion"},
            {"fecha": "2025-07-01", "via": "", "comuna": "Concepcion"},
            {"fecha": "2025-07-01", "via": "nan", "comuna": "Concepcion"},
            {"fecha": "2025-07-01", "via": "Ruta Uno", "comuna": "Concepcion"},
        ]
    )
    monkeypatch.setattr(data_quality_service.data_loader, "data_version", lambda: ("test", 1, 1, 1, 1))
    monkeypatch.setattr(data_quality_service.data_loader, "get_data_profile", lambda: "gran_concepcion")
    monkeypatch.setattr(data_quality_service.data_loader, "incident_sources_are_duplicates", lambda: False)
    monkeypatch.setattr(data_quality_service, "_load_profiled_raw", lambda _path, _profile: incidents.copy())
    monkeypatch.setattr(data_quality_service.data_loader, "ROAD_NETWORK_PATH", FakePath(True))
    monkeypatch.setattr(data_quality_service.data_loader, "ACCIDENT_PATH", FakePath(False))
    monkeypatch.setattr(data_quality_service, "_load_profiled_network_communes", lambda _profile: pd.Series(dtype=str))

    result = data_quality_service.inspect_data_quality()

    assert result["missing_via_ratio"] == 0.75


def test_duplicate_accident_source_is_not_counted_twice(monkeypatch):
    data_quality_service._inspect_data_quality.cache_clear()
    incidents = pd.DataFrame(
        [{"fecha": "2025-07-01", "via": "Ruta Uno", "comuna": "Concepcion"}]
    )
    monkeypatch.setattr(data_quality_service.data_loader, "data_version", lambda: ("test", 1, 1, 1, 1))
    monkeypatch.setattr(data_quality_service.data_loader, "get_data_profile", lambda: "gran_concepcion")
    monkeypatch.setattr(data_quality_service.data_loader, "incident_sources_are_duplicates", lambda: True)
    monkeypatch.setattr(data_quality_service, "_load_profiled_raw", lambda _path, _profile: incidents.copy())
    monkeypatch.setattr(data_quality_service.data_loader, "ROAD_NETWORK_PATH", FakePath(True))
    monkeypatch.setattr(data_quality_service.data_loader, "ACCIDENT_PATH", FakePath(True))
    monkeypatch.setattr(data_quality_service, "_load_profiled_network_communes", lambda _profile: pd.Series(dtype=str))

    result = data_quality_service.inspect_data_quality()

    assert result["duplicate_incident_sources"] is True
    assert result["raw_counts"] == {"accidents": 0, "congestions": 1, "combined": 1}
