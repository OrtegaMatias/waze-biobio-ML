# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class RegionBounds(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class MetadataResponse(BaseModel):
    event_types: List[str]
    communes: List[str]
    franjas: List[str]
    durations: List[str]
    velocities: List[str]
    vias: List[str]
    total_events: int
    total_vias: int
    accident_ratio: float
    bounds: RegionBounds


class HotspotPoint(BaseModel):
    lat: float
    lon: float
    weight: float = Field(1.0, ge=0.0)
    day: str | None = None
    bucket: str | None = None
    segment_id: str | None = None
    hora_inicio_float: float | None = None
    hora_fin_float: float | None = None


class HotspotResponse(BaseModel):
    points: List[HotspotPoint]


class RoutePoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class ViaPreference(BaseModel):
    via: str
    weight: float = Field(..., ge=0.0, le=1.0)


class RouteRequest(BaseModel):
    origin: RoutePoint
    destination: RoutePoint
    preferences: List[ViaPreference] = Field(default_factory=list)
    ubcf_preferences: List[ViaPreference] = Field(default_factory=list)
    ibcf_preferences: List[ViaPreference] = Field(default_factory=list)
    day_of_week: str = Field("Monday")
    departure_hour: float = Field(8.0, ge=0.0, le=24.0)
    avoid_congestion: bool = True
    avoid_accidents: bool = False


class RouteStepResponse(BaseModel):
    node_id: str
    lat: float
    lon: float
    via: str
    comuna: str
    cumulative_cost: float


class IncidentExposure(BaseModel):
    total_incident_segments: int = 0
    matched_incident_segments: int = 0
    congestion_segments: int = 0
    accident_segments: int = 0
    exposure_minutes: float = 0.0


class Pm25StationExposure(BaseModel):
    station_id: str
    station_name: str
    distance_km: float
    pm25: float
    sample_points: int = 0


class Pm25Exposure(BaseModel):
    available: bool = True
    average_pm25: float
    category: Literal["Baja", "Media", "Alta"]
    stations: List[Pm25StationExposure] = Field(default_factory=list)
    method: str
    data_source: str


class SegmentImpact(BaseModel):
    segment_id: str
    via: str
    comuna: str
    event_type: str
    impact_score: float
    reason: str


class PreferredViaImpact(BaseModel):
    via: str
    factor: float
    reason: str


class RouteVariant(BaseModel):
    distance_km: float
    estimated_duration_min: float
    steps: List[RouteStepResponse]
    geometry: List[RoutePoint]
    extra_delay_min: float = 0.0
    risk_score: float = 0.0
    incident_exposure: IncidentExposure = Field(default_factory=IncidentExposure)
    pm25_exposure: Pm25Exposure | None = None
    why_changed: List[str] = Field(default_factory=list)
    top_penalized_segments: List[SegmentImpact] = Field(default_factory=list)
    top_preferred_vias: List[PreferredViaImpact] = Field(default_factory=list)


class RouteDelta(BaseModel):
    variant: str
    distance_delta_km: float = 0.0
    total_duration_delta_min: float = 0.0
    risk_delta: float = 0.0
    exposure_delta: float = 0.0


class RouteComparison(BaseModel):
    fastest_variant: str
    safest_variant: str
    lowest_exposure_variant: str
    best_balance_variant: str
    deltas: List[RouteDelta] = Field(default_factory=list)


class RouteResponse(BaseModel):
    reference: RouteVariant
    ubcf: RouteVariant
    ibcf: RouteVariant
    personalized: RouteVariant | None = None
    comparison: RouteComparison


class PlaceResult(BaseModel):
    id: str
    label: str
    lat: float
    lon: float
    bbox: list[float] | None = None


class PlaceSearchResponse(BaseModel):
    results: List[PlaceResult] = Field(default_factory=list)


class PlaceReverseResponse(BaseModel):
    result: PlaceResult | None = None


class PlanRouteRequest(BaseModel):
    origin: RoutePoint
    destination: RoutePoint
    day_of_week: str = Field("Monday")
    departure_hour: float = Field(8.0, ge=0.0, le=24.0)
    travel_style: Literal["safe", "balanced", "fast"] = "balanced"
    avoid_congestion: bool = True
    avoid_accidents: bool = False


class RouteBadge(BaseModel):
    key: Literal["base", "least_congestion", "healthiest", "recommended", "fastest", "least_exposure"]
    label: str


class UserRouteAlert(BaseModel):
    title: str
    detail: str
    severity: Literal["low", "medium", "high"] = "medium"


class UserRouteSummary(BaseModel):
    eta_total_min: float
    distance_km: float
    delay_min: float
    alerts_on_route: int
    main_reason: str


class UserRouteCard(BaseModel):
    key: str
    label: str
    badges: List[RouteBadge] = Field(default_factory=list)
    duration_min: float
    distance_km: float
    delay_min: float
    risk_level: Literal["low", "medium", "high"]
    summary: str
    geometry: List[RoutePoint] = Field(default_factory=list)
    top_alerts: List[UserRouteAlert] = Field(default_factory=list)
    why_changed: List[str] = Field(default_factory=list)
    top_penalized_segments: List[SegmentImpact] = Field(default_factory=list)
    top_preferred_vias: List[PreferredViaImpact] = Field(default_factory=list)
    incident_exposure: IncidentExposure = Field(default_factory=IncidentExposure)
    pm25_exposure: Pm25Exposure | None = None


class PlanRouteResponse(BaseModel):
    selected_route_key: str
    routes: List[UserRouteCard] = Field(default_factory=list)
    summary: UserRouteSummary
    alerts: List[UserRouteAlert] = Field(default_factory=list)
    hotspots: List[HotspotPoint] = Field(default_factory=list)
    map_bounds: RegionBounds
