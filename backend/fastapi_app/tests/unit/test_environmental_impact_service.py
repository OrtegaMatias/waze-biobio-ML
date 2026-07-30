# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd
from shapely.geometry import shape

from backend.fastapi_app.app.services import environmental_impact_service
from backend.fastapi_app.app.services.environmental_impact_service import EnvironmentalImpactService
from backend.fastapi_app.app.services.environmental_impact_service import WeatherSnapshot
from backend.fastapi_app.app.schemas.routes import EnvironmentalImpactPoint


def test_environmental_impact_score_uses_55_45_weights_without_weather():
    congestion_only = EnvironmentalImpactService._score(
        congestion_score=1.0,
        pm25=20.0,
        rain_mm=None,
        wind_speed=None,
    )
    pm25_only = EnvironmentalImpactService._score(
        congestion_score=0.0,
        pm25=80.0,
        rain_mm=None,
        wind_speed=None,
    )

    assert congestion_only == 55.0
    assert pm25_only == 45.0


def test_environmental_memory_responds_to_wind_and_rain():
    calm_dry = EnvironmentalImpactService._memory_weights(
        WeatherSnapshot(rain_mm=0.0, wind_speed=1.0, wet_station_count=0, global_radiation=None)
    )
    rainy = EnvironmentalImpactService._memory_weights(
        WeatherSnapshot(rain_mm=0.5, wind_speed=1.0, wet_station_count=1, global_radiation=None)
    )
    ventilated = EnvironmentalImpactService._memory_weights(
        WeatherSnapshot(rain_mm=0.0, wind_speed=2.0, wet_station_count=0, global_radiation=None)
    )

    assert calm_dry == {0: 1.0, 1: 0.25, 2: 0.10}
    assert rainy == {0: 1.0, 1: 0.10}
    assert ventilated == {0: 1.0, 1: 0.10}


def test_environmental_zones_combine_overlaps_into_one_exclusive_level(tmp_path):
    service = EnvironmentalImpactService(
        congestion_path=tmp_path / "missing.csv",
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )
    rows = pd.DataFrame(
        [
            {"segment_id": "a", "lat": -36.8200, "lon": -73.0400, "indice_coord": 0},
            {"segment_id": "a", "lat": -36.8190, "lon": -73.0400, "indice_coord": 1},
            {"segment_id": "b", "lat": -36.8200, "lon": -73.0400, "indice_coord": 0},
            {"segment_id": "b", "lat": -36.8190, "lon": -73.0400, "indice_coord": 1},
        ]
    )
    points = [
        EnvironmentalImpactPoint(
            lat=-36.8195,
            lon=lon,
            score=30.0,
            level="low",
            congestion_score=0.3,
            congestion_level="low",
            segment_id=segment_id,
            via=f"Calle {segment_id.upper()}",
            message="test",
        )
        for segment_id, lon in [("a", -73.0400), ("b", -73.0400)]
    ]

    zones = service._build_zones(rows, points)
    by_level = {feature["properties"]["level"]: feature for feature in zones["features"]}
    zone_ids = [feature["properties"]["zone_id"] for feature in zones["features"]]

    assert set(by_level) == {"low", "medium"}
    assert len(zones["features"]) >= 2
    assert len(zone_ids) == len(set(zone_ids))
    assert 35.0 <= by_level["medium"]["properties"]["score_max"] <= 39.0
    assert by_level["medium"]["properties"]["segment_ids"] == ["a", "b"]
    assert by_level["medium"]["properties"]["vias"] == ["Calle A", "Calle B"]
    assert by_level["medium"]["properties"]["segment_count"] == 2
    assert by_level["medium"]["properties"]["overlap_count_max"] == 2
    assert (
        by_level["medium"]["properties"]["composition"]
        == "background_plus_distance_weighted_congestion"
    )
    assert shape(by_level["low"]["geometry"]).intersection(shape(by_level["medium"]["geometry"])).area < 1e-12


def test_environmental_impact_weather_adjusts_pm25_and_congestion_separately():
    dry_score = EnvironmentalImpactService._score(
        congestion_score=0.6,
        pm25=80.0,
        rain_mm=0.0,
        wind_speed=None,
    )
    heavy_rain_score = EnvironmentalImpactService._score(
        congestion_score=0.6,
        pm25=80.0,
        rain_mm=12.0,
        wind_speed=None,
    )

    assert dry_score == 78.0
    assert heavy_rain_score == 77.6


class DummyAirQualityService:
    def local_pm25_range(self):
        return (8.0, 55.0)

    def station_snapshot(self, _date: str, _hour: int):
        return SimpleNamespace(
            average_pm25=27.0,
            stations=[
                SimpleNamespace(lat=-36.82, lon=-73.04, pm25=42.0),
                SimpleNamespace(lat=-36.90, lon=-73.15, pm25=12.0),
            ]
        )


def test_environmental_impact_uses_only_real_congestion_rows(tmp_path, monkeypatch):
    congestion_path = tmp_path / "congestion.csv"
    rain_path = tmp_path / "rain.csv"
    wind_path = tmp_path / "wind.csv"
    congestion_path.write_text(
        "\n".join(
            [
                "segment_id,lat,lon,datetime_inicio,datetime_fin,velocidad_kmh,duracion_min,via,comuna",
                "seg-1,-36.8200,-73.0400,2025-01-01 08:10:00,2025-01-01 08:40:00,8,30,Centro,Concepcion",
                "seg-1,-36.8210,-73.0410,2025-01-01 08:10:00,2025-01-01 08:40:00,8,30,Centro,Concepcion",
            ]
        ),
        encoding="utf-8",
    )
    rain_path.write_text(
        "\n".join(
            [
                "timestamp,rain_mm_mean,rain_mm_max,wet_station_count,station_count",
                "2025-01-01 08:00:00,0,0,0,1",
            ]
        ),
        encoding="utf-8",
    )
    wind_path.write_text(
        "\n".join(
            [
                "timestamp,wind_speed_mean",
                "2025-01-01 08:00:00,0.5",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(environmental_impact_service, "get_air_quality_service", lambda: DummyAirQualityService())

    service = EnvironmentalImpactService(congestion_path=congestion_path, rain_path=rain_path, wind_path=wind_path)
    snapshot = service.build_snapshot("2025-01-01", 8)
    residual_snapshot = service.build_snapshot("2025-01-01", 9)
    empty_snapshot = service.build_snapshot("2025-01-01", 11)

    assert snapshot.summary.available is True
    assert snapshot.summary.point_count == 1
    assert snapshot.summary.weather.pm25 == 27.0
    assert snapshot.summary.weather.pm25_min == 3.0
    assert snapshot.summary.weather.pm25_max == 36.0
    assert snapshot.summary.weather.wind_speed_min == 0.8
    assert snapshot.summary.weather.wind_speed_max == 3.46
    assert snapshot.summary.weather.wind_speed_kmh == 1.8
    assert snapshot.summary.weather.has_rain is False
    assert snapshot.points[0].level == "high"
    assert snapshot.points[0].congestion_level == "medium"
    assert snapshot.points[0].pm25 == 42.0
    assert snapshot.points[0].message == "Alta presion ambiental por congestion y baja ventilacion."
    assert snapshot.zones["type"] == "FeatureCollection"
    assert snapshot.zones["features"]
    assert snapshot.zones["features"][0]["geometry"]["type"] in {"Polygon", "MultiPolygon"}
    assert all(feature["properties"]["recency_weight"] == 1.0 for feature in snapshot.zones["features"])
    assert snapshot.congestion_lines["features"][0]["properties"]["recency"] == "actual"
    assert snapshot.congestion_lines["features"][0]["properties"]["observation_count"] == 1
    assert snapshot.congestion_lines["features"][0]["properties"]["level"] == "medium"
    assert snapshot.congestion_lines["features"][0]["properties"]["environmental_level"] == "high"
    assert snapshot.congestion_lines["features"][0]["properties"]["speed_kmh"] == 8.0
    assert snapshot.congestion_lines["features"][0]["properties"]["duration_min"] == 30.0
    assert residual_snapshot.summary.available is True
    assert residual_snapshot.points[0].score < snapshot.points[0].score
    assert all(
        0.0 < feature["properties"]["recency_weight"] < 1.0
        for feature in residual_snapshot.zones["features"]
    )
    assert residual_snapshot.congestion_lines["features"] == []
    assert empty_snapshot.summary.available is False
    assert empty_snapshot.points == []
    assert empty_snapshot.zones["features"] == []


def test_environmental_impact_consolidates_identical_congestion_lines(tmp_path, monkeypatch):
    congestion_path = tmp_path / "congestion.csv"
    congestion_path.write_text(
        "\n".join(
            [
                "segment_id,lat,lon,datetime_inicio,datetime_fin,velocidad_kmh,duracion_min,via,comuna",
                "seg-recent,-36.8200,-73.0400,2025-01-01 08:05:00,2025-01-01 08:35:00,5,30,Centro,Concepcion",
                "seg-recent,-36.8210,-73.0410,2025-01-01 08:05:00,2025-01-01 08:35:00,5,30,Centro,Concepcion",
                "seg-current,-36.8200,-73.0400,2025-01-01 09:05:00,2025-01-01 09:35:00,20,30,Centro,Concepcion",
                "seg-current,-36.8210,-73.0410,2025-01-01 09:05:00,2025-01-01 09:35:00,20,30,Centro,Concepcion",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(environmental_impact_service, "get_air_quality_service", lambda: DummyAirQualityService())

    service = EnvironmentalImpactService(
        congestion_path=congestion_path,
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )
    snapshot = service.build_snapshot("2025-01-01", 9)

    assert snapshot.summary.point_count == 2
    assert len(snapshot.congestion_lines["features"]) == 1
    properties = snapshot.congestion_lines["features"][0]["properties"]
    assert properties["segment_id"] == "seg-current"
    assert properties["recency"] == "actual"
    assert properties["observation_count"] == 1


def test_environmental_impact_orders_congestion_line_coordinates(tmp_path, monkeypatch):
    congestion_path = tmp_path / "congestion.csv"
    congestion_path.write_text(
        "\n".join(
            [
                "segment_id,segment_seq,indice_coord,lat,lon,datetime_inicio,datetime_fin,velocidad_kmh,duracion_min,via,comuna",
                "seg-ordered,4,0,-36.8200,-73.0400,2025-01-01 08:05:00,2025-01-01 08:35:00,10,30,Centro,Concepcion",
                "seg-ordered,4,2,-36.8220,-73.0420,2025-01-01 08:05:00,2025-01-01 08:35:00,10,30,Centro,Concepcion",
                "seg-ordered,4,1,-36.8210,-73.0410,2025-01-01 08:05:00,2025-01-01 08:35:00,10,30,Centro,Concepcion",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(environmental_impact_service, "get_air_quality_service", lambda: DummyAirQualityService())

    service = EnvironmentalImpactService(
        congestion_path=congestion_path,
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )
    snapshot = service.build_snapshot("2025-01-01", 8)

    assert snapshot.congestion_lines["features"][0]["geometry"]["coordinates"] == [
        [-73.04, -36.82],
        [-73.041, -36.821],
        [-73.042, -36.822],
    ]


def test_environmental_impact_levels_use_fixed_historical_reference(tmp_path, monkeypatch):
    congestion_path = tmp_path / "congestion.csv"
    rain_path = tmp_path / "rain.csv"
    wind_path = tmp_path / "wind.csv"
    congestion_path.write_text(
        "\n".join(
                [
                    "segment_id,lat,lon,datetime_inicio,datetime_fin,velocidad_kmh,duracion_min,via,comuna",
                    "seg-low,-36.8200,-73.0400,2025-01-01 08:10:00,2025-01-01 08:20:00,30,5,Centro,Concepcion",
                    "seg-low,-36.8210,-73.0410,2025-01-01 08:10:00,2025-01-01 08:20:00,30,5,Centro,Concepcion",
                    "seg-high,-36.9000,-73.1500,2025-01-01 08:10:00,2025-01-01 08:40:00,20,30,Costa,Concepcion",
                    "seg-high,-36.9010,-73.1510,2025-01-01 08:10:00,2025-01-01 08:40:00,20,30,Costa,Concepcion",
                ]
        ),
        encoding="utf-8",
    )
    rain_path.write_text(
        "\n".join(
            [
                "timestamp,rain_mm_mean,rain_mm_max,wet_station_count,station_count",
                "2025-01-01 08:00:00,0,0,0,1",
            ]
        ),
        encoding="utf-8",
    )
    wind_path.write_text(
        "\n".join(
            [
                "timestamp,wind_speed_mean",
                "2025-01-01 08:00:00,2.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(environmental_impact_service, "get_air_quality_service", lambda: DummyAirQualityService())

    service = EnvironmentalImpactService(congestion_path=congestion_path, rain_path=rain_path, wind_path=wind_path)
    snapshot = service.build_snapshot("2025-01-01", 8)

    points_by_segment = {point.segment_id: point for point in snapshot.points}
    assert points_by_segment["seg-low"].score == 58.7
    assert points_by_segment["seg-low"].level == "medium"
    assert points_by_segment["seg-high"].level in {"low", "medium"}
    assert points_by_segment["seg-low"].score > points_by_segment["seg-high"].score
    assert points_by_segment["seg-low"].congestion_level == "low"
    assert points_by_segment["seg-high"].congestion_level == "low"
    assert points_by_segment["seg-high"].congestion_score > points_by_segment["seg-low"].congestion_score
    line_levels = {
        feature["properties"]["segment_id"]: feature["properties"]["level"]
        for feature in snapshot.congestion_lines["features"]
    }
    assert line_levels == {"seg-low": "low", "seg-high": "low"}
    assert "referencia historica fija" in snapshot.summary.method
    assert snapshot.summary.weather.pm25_min == 3.0
    assert snapshot.summary.weather.pm25_max == 36.0


def test_environmental_reference_maps_p10_and_p90_to_bounds(tmp_path):
    service = EnvironmentalImpactService(
        congestion_path=tmp_path / "missing.csv",
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )
    pm25_range = service._normalization_range("pm25")
    wind_range = service._normalization_range("wind_speed")
    speed_range = service._normalization_range("congestion_speed_kmh")
    duration_range = service._normalization_range("congestion_duration_min")

    assert environmental_impact_service._local_component(3.0, pm25_range) == 0.0
    assert environmental_impact_service._local_component(36.0, pm25_range) == 1.0
    assert environmental_impact_service._local_component(-100.0, pm25_range) == 0.0
    assert environmental_impact_service._local_component(1000.0, pm25_range) == 1.0
    assert environmental_impact_service._local_component(wind_range[0], wind_range, invert=True) == 1.0
    assert environmental_impact_service._local_component(wind_range[1], wind_range, invert=True) == 0.0
    assert environmental_impact_service._local_component(speed_range[0], speed_range, invert=True) == 1.0
    assert environmental_impact_service._local_component(speed_range[1], speed_range, invert=True) == 0.0
    assert environmental_impact_service._local_component(duration_range[0], duration_range) == 0.0
    assert environmental_impact_service._local_component(duration_range[1], duration_range) == 1.0


def test_congestion_score_combines_slow_speed_and_long_duration(tmp_path):
    service = EnvironmentalImpactService(
        congestion_path=tmp_path / "missing.csv",
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )

    slow_short = service._congestion_score(speed_kmh=10.54, duration_min=15.0)
    fast_long = service._congestion_score(speed_kmh=23.66, duration_min=109.8)
    slow_long = service._congestion_score(speed_kmh=10.54, duration_min=109.8)
    faster_long = service._congestion_score(speed_kmh=18.48, duration_min=109.8)
    slow_medium = service._congestion_score(speed_kmh=10.54, duration_min=30.0)

    assert slow_short == 0.5
    assert fast_long == 0.5
    assert slow_long == 1.0
    assert slow_long > slow_short
    assert slow_long > fast_long
    assert slow_long > faster_long
    assert slow_long > slow_medium


def test_congestion_score_uses_available_dimension_when_the_other_is_missing(tmp_path):
    service = EnvironmentalImpactService(
        congestion_path=tmp_path / "missing.csv",
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )

    assert service._congestion_score(speed_kmh=10.54, duration_min=None) == 1.0
    assert service._congestion_score(speed_kmh=None, duration_min=109.8) == 1.0
    assert service._congestion_score(speed_kmh=None, duration_min=None) == 0.0


def test_environmental_reference_fallback_is_fixed_and_reported(tmp_path, monkeypatch):
    congestion_path = tmp_path / "congestion.csv"
    congestion_path.write_text(
        "\n".join(
            [
                "segment_id,lat,lon,datetime_inicio,datetime_fin,velocidad_kmh,duracion_min,via,comuna",
                "seg-1,-36.82,-73.04,2025-01-01 08:00:00,2025-01-01 08:30:00,10,30,Centro,Concepcion",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(environmental_impact_service, "get_air_quality_service", lambda: DummyAirQualityService())
    service = EnvironmentalImpactService(
        congestion_path=congestion_path,
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
        normalization_path=tmp_path / "missing_reference.json",
    )

    snapshot = service.build_snapshot("2025-01-01", 8)

    assert snapshot.summary.weather.pm25_min == 3.0
    assert snapshot.summary.weather.pm25_max == 36.0
    assert "normalization_reference=builtin_fixed_v1" in snapshot.summary.data_source
    assert "wind=historical_p50" in snapshot.summary.data_source
    assert "rain=no_relief" in snapshot.summary.data_source


class DailyExtremeAirQualityService:
    def station_snapshot(self, snapshot_date: str, _hour: int):
        daily_extreme = 5.0 if snapshot_date == "2025-01-01" else 500.0
        return SimpleNamespace(
            average_pm25=25.0,
            stations=[
                SimpleNamespace(lat=-36.82, lon=-73.04, pm25=25.0),
                SimpleNamespace(lat=-38.00, lon=-75.00, pm25=daily_extreme),
            ],
        )


def test_same_conditions_keep_same_score_across_dates_despite_daily_extremes(tmp_path, monkeypatch):
    congestion_path = tmp_path / "congestion.csv"
    congestion_path.write_text(
        "\n".join(
            [
                "segment_id,lat,lon,datetime_inicio,datetime_fin,velocidad_kmh,duracion_min,via,comuna",
                "seg-day-1,-36.82,-73.04,2025-01-01 08:00:00,2025-01-01 08:30:00,10,30,Centro,Concepcion",
                "seg-day-2,-36.82,-73.04,2025-02-01 08:00:00,2025-02-01 08:30:00,10,30,Centro,Concepcion",
            ]
        ),
        encoding="utf-8",
    )
    wind_path = tmp_path / "wind.csv"
    wind_path.write_text(
        "\n".join(
            [
                "timestamp,wind_speed_mean",
                "2025-01-01 08:00:00,1.5",
                "2025-02-01 08:00:00,1.5",
            ]
        ),
        encoding="utf-8",
    )
    rain_path = tmp_path / "rain.csv"
    rain_path.write_text(
        "\n".join(
            [
                "timestamp,rain_mm_mean,wet_station_count",
                "2025-01-01 08:00:00,0,0",
                "2025-02-01 08:00:00,0,0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        environmental_impact_service,
        "get_air_quality_service",
        lambda: DailyExtremeAirQualityService(),
    )
    service = EnvironmentalImpactService(
        congestion_path=congestion_path,
        rain_path=rain_path,
        wind_path=wind_path,
    )

    first = service.build_snapshot("2025-01-01", 8)
    second = service.build_snapshot("2025-02-01", 8)

    assert first.points[0].pm25 == second.points[0].pm25 == 25.0
    assert first.points[0].score == second.points[0].score


def test_environmental_impact_rejects_dates_outside_2025(tmp_path):
    service = EnvironmentalImpactService(
        congestion_path=tmp_path / "missing.csv",
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )

    try:
        service.build_snapshot("2026-01-01", 8)
    except ValueError as exc:
        assert "2025" in str(exc)
    else:
        raise AssertionError("Expected dates outside 2025 to be rejected")


def test_environmental_impact_skips_radiation_by_default_and_caches_snapshots(tmp_path, monkeypatch):
    service = EnvironmentalImpactService(
        congestion_path=tmp_path / "missing.csv",
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )
    monkeypatch.setattr(
        service,
        "_load_radiation_network",
        lambda: (_ for _ in ()).throw(AssertionError("Radiation should not load by default")),
    )

    first = service.build_snapshot("2025-01-01", 8)
    second = service.build_snapshot("2025-01-01", 8)

    assert first is second
    assert first.summary.weather.global_radiation is None
    assert first.summary.weather.sky_label == "Sin dato"


def test_congestion_hour_index_preserves_each_hour_touched_by_an_interval(tmp_path):
    congestion_path = tmp_path / "congestion.csv"
    congestion_path.write_text(
        "\n".join(
            [
                "segment_id,lat,lon,datetime_inicio,datetime_fin,velocidad_kmh,duracion_min,via,comuna",
                "seg-1,-36.82,-73.04,2025-01-01 08:10:00,2025-01-01 09:10:00,10,60,Centro,Concepcion",
            ]
        ),
        encoding="utf-8",
    )
    service = EnvironmentalImpactService(
        congestion_path=congestion_path,
        rain_path=tmp_path / "missing_rain.csv",
        wind_path=tmp_path / "missing_wind.csv",
    )

    assert service._congestion_hour_index() == {
        datetime(2025, 1, 1, 8),
        datetime(2025, 1, 1, 9),
    }
    assert service._congestion_hour_index(9) == {datetime(2025, 1, 1, 9)}
