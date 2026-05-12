# -*- coding: utf-8 -*-
from __future__ import annotations

from backend.fastapi_app.app.schemas.routes import RoutePoint
from backend.fastapi_app.app.services.air_quality_service import AirQualityService


def test_air_quality_service_estimates_route_from_nearest_station(tmp_path):
    hourly_path = tmp_path / "pm25.csv"
    summary_path = tmp_path / "stations.csv"
    hourly_path.write_text(
        "\n".join(
            [
                "timestamp,station_id,station_name,PM25",
                "2026-01-01 08:00:00,1,Centro,10",
                "2026-01-02 08:00:00,1,Centro,20",
                "2026-01-01 08:00:00,2,Costa,40",
            ]
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                "1,Centro,-36.8200,-73.0400",
                "2,Costa,-36.9000,-73.1500",
            ]
        ),
        encoding="utf-8",
    )

    service = AirQualityService(hourly_path=hourly_path, station_summary_path=summary_path)
    exposure = service.estimate_route_exposure(
        geometry=[RoutePoint(lat=-36.821, lon=-73.041), RoutePoint(lat=-36.822, lon=-73.042)],
        departure_hour=8,
    )

    assert exposure is not None
    assert exposure.average_pm25 == 15.0
    assert exposure.category == "Media"
    assert exposure.stations[0].station_name == "Centro"


def test_air_quality_service_returns_none_when_files_are_missing(tmp_path):
    service = AirQualityService(
        hourly_path=tmp_path / "missing_pm25.csv",
        station_summary_path=tmp_path / "missing_stations.csv",
    )

    assert service.estimate_route_exposure([RoutePoint(lat=-36.82, lon=-73.04)], departure_hour=8) is None
