# -*- coding: utf-8 -*-
from __future__ import annotations

from backend.fastapi_app.app.schemas.routes import CyclewayCoverage, IncidentExposure, Pm25Exposure
from backend.fastapi_app.app.services import active_mobility_messages


def _incident_exposure(matched: int = 0, minutes: float = 0.0) -> IncidentExposure:
    return IncidentExposure(
        total_incident_segments=matched,
        matched_incident_segments=matched,
        congestion_segments=matched,
        accident_segments=0,
        exposure_minutes=minutes,
    )


def _pm25(category: str = "Baja", average: float = 11.0) -> Pm25Exposure:
    return Pm25Exposure(
        available=True,
        average_pm25=average,
        category=category,  # type: ignore[arg-type]
        method="fixture",
        data_source="test",
    )


def _cycleways(high: bool = True) -> CyclewayCoverage:
    return CyclewayCoverage(
        available=True,
        coverage_ratio=0.55 if high else 0.1,
        nearby_cycleway_km=1.1,
        route_km=2.0,
        nearby_buffer_m=80.0,
        has_high_coverage=high,
        data_source="test",
    )


def test_prioritizes_bike_when_air_congestion_and_cycleways_are_favorable():
    estimate = active_mobility_messages.estimate_active_travel_times(
        distance_km=3.0,
        auto_min=8.0,
        cycleway_coverage=_cycleways(high=True),
    )
    messages = active_mobility_messages.build_route_messages(
        route_key="healthiest",
        distance_km=3.0,
        delay_min=0.0,
        risk_level="low",
        incident_exposure=_incident_exposure(),
        pm25_exposure=_pm25("Baja", 10.0),
        cycleway_coverage=_cycleways(high=True),
        active_mobility_estimate=estimate,
    )

    assert messages
    assert messages[0].mode == "bike"
    assert "bicicleta" in messages[0].title.lower()
    assert "min mas" in messages[0].detail


def test_estimates_active_travel_times_from_route_distance_and_cycleway_signal():
    estimate = active_mobility_messages.estimate_active_travel_times(
        distance_km=3.6,
        auto_min=9.0,
        cycleway_coverage=_cycleways(high=True),
    )

    assert estimate.auto_min == 9.0
    assert estimate.bike_speed_kmh == 18.0
    assert estimate.bike_min == 12.0
    assert estimate.bike_extra_min == 3.0


def test_prioritizes_walking_for_short_low_exposure_routes_without_cycleway_signal():
    messages = active_mobility_messages.build_route_messages(
        route_key="base",
        distance_km=1.2,
        delay_min=0.0,
        risk_level="low",
        incident_exposure=_incident_exposure(),
        pm25_exposure=_pm25("Baja", 9.0),
        cycleway_coverage=_cycleways(high=False),
    )

    assert messages
    assert messages[0].mode == "walk"


def test_does_not_suggest_active_modes_when_congestion_is_high():
    messages = active_mobility_messages.build_route_messages(
        route_key="base",
        distance_km=2.0,
        delay_min=9.0,
        risk_level="high",
        incident_exposure=_incident_exposure(matched=4, minutes=10.0),
        pm25_exposure=_pm25("Baja", 10.0),
        cycleway_coverage=_cycleways(high=True),
    )

    assert messages == []


def test_does_not_suggest_active_modes_when_pm25_is_high():
    messages = active_mobility_messages.build_route_messages(
        route_key="base",
        distance_km=1.0,
        delay_min=0.0,
        risk_level="low",
        incident_exposure=_incident_exposure(),
        pm25_exposure=_pm25("Alta", 55.0),
        cycleway_coverage=_cycleways(high=True),
    )

    assert messages == []


def test_does_not_suggest_walking_when_distance_is_excessive():
    messages = active_mobility_messages.build_route_messages(
        route_key="base",
        distance_km=4.0,
        delay_min=0.0,
        risk_level="low",
        incident_exposure=_incident_exposure(),
        pm25_exposure=_pm25("Baja", 9.0),
        cycleway_coverage=_cycleways(high=False),
    )

    assert all(message.mode != "walk" for message in messages)
