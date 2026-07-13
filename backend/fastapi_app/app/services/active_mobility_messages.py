# -*- coding: utf-8 -*-
from __future__ import annotations

from ..schemas.routes import (
    ActiveMobilityEstimate,
    ContextualMobilityMessage,
    CyclewayCoverage,
    IncidentExposure,
    Pm25Exposure,
    UserRouteCard,
)

WALK_MAX_KM = 2.0
BIKE_MAX_KM = 7.5
HIGH_CONGESTION_SEGMENTS = 3
HIGH_CONGESTION_MINUTES = 8.0
HIGH_PM25_UG_M3 = 50.0
MODERATE_PM25_UG_M3 = 20.0
BIKE_SPEED_WITH_HIGH_CYCLEWAY_KMH = 18.0
BIKE_SPEED_WITH_CYCLEWAY_SIGNAL_KMH = 15.0
BIKE_SPEED_BASE_KMH = 12.0
WALK_SPEED_SHORT_KMH = 4.8
WALK_SPEED_BASE_KMH = 4.5


def _air_is_favorable(exposure: Pm25Exposure | None) -> bool:
    if exposure is None or not exposure.available:
        return False
    return exposure.category in {"Baja", "Media"} and exposure.average_pm25 < HIGH_PM25_UG_M3


def _air_is_low(exposure: Pm25Exposure | None) -> bool:
    if exposure is None or not exposure.available:
        return False
    return exposure.category == "Baja" or exposure.average_pm25 < MODERATE_PM25_UG_M3


def _environmental_exposure_is_high(exposure: Pm25Exposure | None) -> bool:
    if exposure is None or not exposure.available:
        return False
    return exposure.category == "Alta" or exposure.average_pm25 >= HIGH_PM25_UG_M3


def _congestion_is_low(exposure: IncidentExposure, delay_min: float, risk_level: str) -> bool:
    return (
        risk_level == "low"
        and exposure.matched_incident_segments <= 1
        and exposure.exposure_minutes <= 3.0
        and delay_min <= 3.0
    )


def _congestion_is_high(exposure: IncidentExposure, delay_min: float, risk_level: str) -> bool:
    return (
        risk_level == "high"
        or exposure.matched_incident_segments >= HIGH_CONGESTION_SEGMENTS
        or exposure.exposure_minutes >= HIGH_CONGESTION_MINUTES
        or delay_min >= HIGH_CONGESTION_MINUTES
    )


def _has_cycleway_signal(coverage: CyclewayCoverage | None) -> bool:
    return bool(coverage and coverage.available and coverage.has_high_coverage)


def _extra_minutes(active_min: float, auto_min: float) -> float:
    return round(max(0.0, active_min - auto_min), 1)


def estimate_active_travel_times(
    *,
    distance_km: float,
    auto_min: float,
    cycleway_coverage: CyclewayCoverage | None,
) -> ActiveMobilityEstimate:
    if cycleway_coverage and cycleway_coverage.available and cycleway_coverage.has_high_coverage:
        bike_speed = BIKE_SPEED_WITH_HIGH_CYCLEWAY_KMH
    elif cycleway_coverage and cycleway_coverage.available:
        bike_speed = BIKE_SPEED_WITH_CYCLEWAY_SIGNAL_KMH
    else:
        bike_speed = BIKE_SPEED_BASE_KMH
    walk_speed = WALK_SPEED_SHORT_KMH if distance_km <= WALK_MAX_KM else WALK_SPEED_BASE_KMH
    bike_min = round((distance_km / bike_speed) * 60.0, 1) if distance_km > 0 else 0.0
    walk_min = round((distance_km / walk_speed) * 60.0, 1) if distance_km > 0 else 0.0
    return ActiveMobilityEstimate(
        auto_min=round(auto_min, 1),
        bike_min=bike_min,
        walk_min=walk_min,
        bike_extra_min=_extra_minutes(bike_min, auto_min),
        walk_extra_min=_extra_minutes(walk_min, auto_min),
        bike_speed_kmh=bike_speed,
        walk_speed_kmh=walk_speed,
    )


def _extra_time_phrase(extra_min: float, mode: str) -> str:
    if extra_min <= 1.0:
        return f"Con un tiempo similar al auto, puedes considerar {mode}."
    return f"Por cerca de {extra_min:.0f} min mas que en auto, puedes considerar {mode}."


def _detail(distance_km: float, pm25: Pm25Exposure | None, exposure: IncidentExposure) -> str:
    air = "PM2.5 bajo o moderado"
    if pm25 and pm25.available:
        air = f"PM2.5 {pm25.category.lower()} ({pm25.average_pm25:.1f} ug/m3)"
    return (
        f"Ruta de {distance_km:.1f} km, {air}, "
        f"y {exposure.matched_incident_segments} zonas de congestion historica relevantes."
    )


def build_route_messages(
    *,
    route_key: str,
    distance_km: float,
    delay_min: float,
    risk_level: str,
    incident_exposure: IncidentExposure,
    pm25_exposure: Pm25Exposure | None,
    cycleway_coverage: CyclewayCoverage | None,
    active_mobility_estimate: ActiveMobilityEstimate | None = None,
) -> list[ContextualMobilityMessage]:
    messages: list[ContextualMobilityMessage] = []
    high_congestion = _congestion_is_high(incident_exposure, delay_min, risk_level)
    high_environmental = _environmental_exposure_is_high(pm25_exposure)
    favorable_air = _air_is_favorable(pm25_exposure)
    low_air = _air_is_low(pm25_exposure)
    low_congestion = _congestion_is_low(incident_exposure, delay_min, risk_level)
    has_cycleway = _has_cycleway_signal(cycleway_coverage)

    can_suggest_bike = (
        favorable_air
        and low_congestion
        and has_cycleway
        and distance_km <= BIKE_MAX_KM
        and not high_congestion
        and not high_environmental
    )
    can_suggest_walk = (
        low_air
        and low_congestion
        and distance_km <= WALK_MAX_KM
        and not high_congestion
        and not high_environmental
    )
    can_suggest_general = (
        favorable_air
        and low_congestion
        and distance_km <= BIKE_MAX_KM
        and not high_congestion
        and not high_environmental
    )

    if can_suggest_bike:
        coverage_percent = round((cycleway_coverage.coverage_ratio if cycleway_coverage else 0.0) * 100)
        bike_extra = active_mobility_estimate.bike_extra_min if active_mobility_estimate else 0.0
        messages.append(
            ContextualMobilityMessage(
                id=f"{route_key}-bike",
                title="Buenas condiciones para bicicleta",
                detail=(
                    f"{_extra_time_phrase(bike_extra, 'bicicleta')} "
                    "Esta zona presenta baja congestion y buena cobertura ciclista cercana "
                    f"({coverage_percent}% del trayecto). Revisa las ciclovias marcadas con lineas celestes sobre el mapa."
                ),
                mode="bike",
                priority="high",
            )
        )

    if can_suggest_walk:
        walk_extra = active_mobility_estimate.walk_extra_min if active_mobility_estimate else 0.0
        messages.append(
            ContextualMobilityMessage(
                id=f"{route_key}-walk",
                title="Trayecto favorable para caminar",
                detail=(
                    f"{_extra_time_phrase(walk_extra, 'caminar')} "
                    "La calidad del aire es favorable para caminar en trayectos cortos."
                ),
                mode="walk",
                priority="high" if not can_suggest_bike else "medium",
            )
        )

    if can_suggest_general and not messages:
        messages.append(
            ContextualMobilityMessage(
                id=f"{route_key}-active",
                title="Condiciones favorables para movilidad activa",
                detail=_detail(distance_km, pm25_exposure, incident_exposure),
                mode="active_mobility",
                priority="medium",
            )
        )

    if favorable_air and low_congestion and distance_km <= BIKE_MAX_KM:
        messages.append(
            ContextualMobilityMessage(
                id=f"{route_key}-selected-low-impact",
                title="Baja exposicion en la ruta seleccionada",
                detail="La ruta tiene baja exposicion ambiental y baja congestion para el horario elegido.",
                mode="active_mobility",
                priority="medium",
            )
        )

    return messages[:2]


def select_plan_messages(routes: list[UserRouteCard], selected_route_key: str) -> list[ContextualMobilityMessage]:
    selected = next((route for route in routes if route.key == selected_route_key), None)
    selected_messages = selected.contextual_messages if selected else []
    if selected_messages:
        return selected_messages[:2]

    ranked = sorted(
        (message for route in routes for message in route.contextual_messages),
        key=lambda message: {"high": 0, "medium": 1, "low": 2}[message.priority],
    )
    return ranked[:2]
