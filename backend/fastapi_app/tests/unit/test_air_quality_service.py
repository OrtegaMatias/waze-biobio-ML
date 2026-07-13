# -*- coding: utf-8 -*-
from __future__ import annotations

from backend.fastapi_app.app.schemas.routes import RoutePoint
from backend.fastapi_app.app.services.air_quality_service import AirQualityService


def test_air_quality_service_estimates_route_with_idw_interpolation(tmp_path):
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
    assert exposure.category == "Baja"
    assert exposure.stations[0].station_name == "Centro"
    assert "IDW" in exposure.method
    assert "costo ambiental" in exposure.method


def test_air_quality_service_interpolates_between_nearby_stations(tmp_path):
    hourly_path = tmp_path / "pm25.csv"
    summary_path = tmp_path / "stations.csv"
    hourly_path.write_text(
        "\n".join(
            [
                "timestamp,station_id,station_name,PM25",
                "2026-01-01 08:00:00,1,Oeste,10",
                "2026-01-01 08:00:00,2,Centro,30",
                "2026-01-01 08:00:00,3,Este,50",
            ]
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                "1,Oeste,0.0000,0.0000",
                "2,Centro,0.0000,0.1000",
                "3,Este,0.0000,0.2000",
            ]
        ),
        encoding="utf-8",
    )

    service = AirQualityService(hourly_path=hourly_path, station_summary_path=summary_path)

    interpolated = service.estimate_point_pm25(0.0, 0.05, departure_hour=8)

    assert interpolated is not None
    assert 10.0 < interpolated < 30.0


def test_air_quality_service_builds_route_cost_factor(tmp_path):
    hourly_path = tmp_path / "pm25.csv"
    summary_path = tmp_path / "stations.csv"
    hourly_path.write_text(
        "\n".join(
            [
                "timestamp,station_id,station_name,PM25",
                "2026-01-01 08:00:00,1,Centro,10",
                "2026-01-01 08:00:00,2,Costa,45",
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

    assert service.route_cost_factor(-36.821, -73.041, departure_hour=8) == 1.0
    assert service.route_cost_factor(-36.901, -73.151, departure_hour=8) > 1.5


def test_air_quality_service_returns_real_station_snapshot_for_selected_hour(tmp_path):
    hourly_path = tmp_path / "pm25.csv"
    summary_path = tmp_path / "stations.csv"
    hourly_path.write_text(
        "\n".join(
            [
                "timestamp,station_id,station_name,PM25",
                "2024-01-01 08:00:00,1,Centro,99",
                "2025-01-01 08:00:00,1,Centro,12",
                "2025-01-01 08:00:00,2,Costa,38",
                "2025-01-01 09:00:00,1,Centro,20",
                "2026-01-01 08:00:00,1,Centro,99",
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
    snapshot = service.station_snapshot("2025-01-01", 8)

    assert snapshot.available is True
    assert snapshot.requested_at == "2025-01-01 08:00:00"
    assert snapshot.average_pm25 == 25.0
    assert [station.station_name for station in snapshot.stations] == ["Centro", "Costa"]
    assert [station.category for station in snapshot.stations] == ["Baja", "Media"]
    assert snapshot.date_range == {"start": "2025-01-01", "end": "2025-01-01"}
    assert "2025" in snapshot.method


def test_air_quality_service_rejects_snapshot_dates_outside_2025(tmp_path):
    hourly_path = tmp_path / "pm25.csv"
    summary_path = tmp_path / "stations.csv"
    hourly_path.write_text(
        "\n".join(
            [
                "timestamp,station_id,station_name,PM25",
                "2025-01-01 08:00:00,1,Centro,12",
            ]
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        "\n".join(
            [
                "station_id,station_name,latitude,longitude",
                "1,Centro,-36.8200,-73.0400",
            ]
        ),
        encoding="utf-8",
    )

    service = AirQualityService(hourly_path=hourly_path, station_summary_path=summary_path)

    try:
        service.station_snapshot("2026-01-01", 8)
    except ValueError as exc:
        assert "2025" in str(exc)
    else:
        raise AssertionError("Expected dates outside 2025 to be rejected")


def test_air_quality_service_returns_none_when_files_are_missing(tmp_path):
    service = AirQualityService(
        hourly_path=tmp_path / "missing_pm25.csv",
        station_summary_path=tmp_path / "missing_stations.csv",
    )

    assert service.estimate_route_exposure([RoutePoint(lat=-36.82, lon=-73.04)], departure_hour=8) is None
