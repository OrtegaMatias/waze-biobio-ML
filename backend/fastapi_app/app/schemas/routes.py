# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

RouteType = Literal["fastest", "least_congested", "healthiest"]


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


class EnvironmentalImpactPoint(BaseModel):
    lat: float
    lon: float
    score: float = Field(..., ge=0.0, le=100.0)
    level: Literal["low", "medium", "high"]
    congestion_score: float = Field(..., ge=0.0, le=1.0)
    congestion_level: Literal["low", "medium", "high"]
    pm25: float | None = None
    rain_mm: float | None = None
    wind_speed: float | None = None
    segment_id: str
    via: str | None = None
    comuna: str | None = None
    message: str


class EnvironmentalWeatherSummary(BaseModel):
    pm25: float | None = None
    pm25_min: float | None = None
    pm25_max: float | None = None
    rain_mm: float | None = None
    has_rain: bool | None = None
    rain_label: Literal["Sin lluvia", "Llovizna", "Lluvia", "Lluvia fuerte", "Sin dato"] = "Sin dato"
    wind_speed: float | None = None
    wind_speed_kmh: float | None = None
    wind_speed_min: float | None = None
    wind_speed_max: float | None = None
    wind_speed_min_kmh: float | None = None
    wind_speed_max_kmh: float | None = None
    wind_label: Literal["Viento suave", "Viento moderado", "Viento fuerte", "Sin dato"] = "Sin dato"
    global_radiation: float | None = None
    sky_label: Literal["Despejado", "Parcial", "Nublado", "Oscuro", "Sin dato"] = "Sin dato"


class EnvironmentalImpactSummary(BaseModel):
    available: bool
    requested_at: str
    point_count: int = 0
    dominant_level: Literal["low", "medium", "high", "none"] = "none"
    weather: EnvironmentalWeatherSummary
    messages: List[str] = Field(default_factory=list)
    method: str
    data_source: str


class EnvironmentalImpactResponse(BaseModel):
    summary: EnvironmentalImpactSummary
    points: List[EnvironmentalImpactPoint] = Field(default_factory=list)
    zones: dict = Field(default_factory=lambda: {"type": "FeatureCollection", "features": []})
    congestion_lines: dict = Field(default_factory=lambda: {"type": "FeatureCollection", "features": []})


class CongestionDateCoverageResponse(BaseModel):
    start: str | None = None
    end: str | None = None
    available_dates: List[str] = Field(default_factory=list)
    missing_dates: List[str] = Field(default_factory=list)
    rain_dates: List[str] = Field(default_factory=list)
    available_days: int = 0
    calendar_days: int = 0
    data_source: str


class CongestionHourAvailabilityResponse(BaseModel):
    date: str
    available_hours: List[int] = Field(default_factory=list)
    count: int = 0
    data_source: str


class CyclewayResponse(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    name: str = "gran_concepcion_cycleways"
    features: List[dict] = Field(default_factory=list)


class UrbanWellbeingResponse(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    name: str = "gran_concepcion_urban_wellbeing"
    features: List[dict] = Field(default_factory=list)


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
    congestion_date: str | None = None
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


class Pm25StationCondition(BaseModel):
    station_id: str
    station_name: str
    lat: float
    lon: float
    pm25: float
    category: Literal["Baja", "Media", "Alta"]


class Pm25SnapshotResponse(BaseModel):
    available: bool
    requested_at: str
    stations: List[Pm25StationCondition] = Field(default_factory=list)
    average_pm25: float | None = None
    date_range: dict[str, str | None] = Field(default_factory=dict)
    method: str
    data_source: str


class CyclewayCoverage(BaseModel):
    available: bool = False
    coverage_ratio: float = Field(0.0, ge=0.0, le=1.0)
    nearby_cycleway_km: float = 0.0
    route_km: float = 0.0
    nearby_buffer_m: float = 80.0
    has_high_coverage: bool = False
    data_source: str = "OpenStreetMap/Overpass"


class WellbeingFeatureImpact(BaseModel):
    feature_id: str
    name: str
    category: Literal["green_space", "blue_space", "tree_cover", "public_space", "sustainability", "cycleway"]
    subtype: str
    distance_m: float = 0.0
    source: str
    base_weight: float = 1.0


class UrbanWellbeingAnalysis(BaseModel):
    available: bool = False
    score: float = Field(0.0, ge=0.0, le=100.0)
    green_ratio: float = Field(0.0, ge=0.0, le=1.0)
    blue_ratio: float = Field(0.0, ge=0.0, le=1.0)
    tree_ratio: float = Field(0.0, ge=0.0, le=1.0)
    public_space_ratio: float = Field(0.0, ge=0.0, le=1.0)
    sustainability_ratio: float = Field(0.0, ge=0.0, le=1.0)
    cycleway_ratio: float = Field(0.0, ge=0.0, le=1.0)
    nearby_feature_count: int = 0
    nearby_buffer_m: float = 30.0
    top_features: List[WellbeingFeatureImpact] = Field(default_factory=list)
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


class RouteCongestionCoverage(BaseModel):
    route_m: float = 0.0
    congested_m: float = 0.0
    high_m: float = 0.0
    medium_m: float = 0.0
    low_m: float = 0.0
    congested_pct: float = 0.0
    high_pct: float = 0.0
    medium_pct: float = 0.0
    low_pct: float = 0.0
    primary_via: str | None = None


class RouteOptimizationTrace(BaseModel):
    objective: Literal["fastest", "fluent", "environmental"]
    cost_model_version: str = "direct-edge-cost-v1"
    logical_segment_count: int = 0
    base_time_min: float = 0.0
    congestion_delay_min: float = 0.0
    congestion_penalty_min: float = 0.0
    stop_penalty_min: float = 0.0
    pm25_penalty_min: float = 0.0
    adverse_environment_penalty_min: float = 0.0
    urban_benefit_min: float = 0.0
    optimization_cost_min: float = 0.0
    pm25_data_available: bool | None = None
    urban_data_available: bool | None = None


class RouteVariant(BaseModel):
    distance_km: float
    estimated_duration_min: float
    steps: List[RouteStepResponse]
    geometry: List[RoutePoint]
    road_geometry: List[RoutePoint] = Field(default_factory=list)
    access_geometry: List[List[RoutePoint]] = Field(default_factory=list)
    extra_delay_min: float = 0.0
    risk_score: float = 0.0
    incident_exposure: IncidentExposure = Field(default_factory=IncidentExposure)
    pm25_exposure: Pm25Exposure | None = None
    urban_wellbeing: UrbanWellbeingAnalysis | None = None
    healthy_route_score: float | None = Field(None, ge=0.0, le=100.0)
    why_changed: List[str] = Field(default_factory=list)
    top_penalized_segments: List[SegmentImpact] = Field(default_factory=list)
    top_preferred_vias: List[PreferredViaImpact] = Field(default_factory=list)
    congestion_coverage: RouteCongestionCoverage = Field(default_factory=RouteCongestionCoverage)
    optimization_trace: RouteOptimizationTrace | None = None


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
    least_congestion: RouteVariant | None = None
    ubcf: RouteVariant
    ibcf: RouteVariant
    healthiest: RouteVariant | None = None
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
    congestion_date: str | None = None
    day_of_week: str = Field("Monday")
    departure_hour: float = Field(8.0, ge=0.0, le=24.0)
    avoid_congestion: bool = True
    avoid_accidents: bool = False


class RouteBadge(BaseModel):
    key: Literal["base", "least_congestion", "healthiest", "recommended", "fastest", "least_exposure"]
    label: str


class UserRouteAlert(BaseModel):
    title: str
    detail: str
    severity: Literal["low", "medium", "high"] = "medium"


class ContextualMobilityMessage(BaseModel):
    id: str
    title: str
    detail: str
    mode: Literal["walk", "bike", "active_mobility"]
    priority: Literal["low", "medium", "high"] = "medium"


class ActiveMobilityEstimate(BaseModel):
    auto_min: float
    bike_min: float
    walk_min: float
    bike_extra_min: float
    walk_extra_min: float
    bike_speed_kmh: float
    walk_speed_kmh: float


class UserRouteSummary(BaseModel):
    eta_total_min: float
    distance_km: float
    delay_min: float
    alerts_on_route: int
    main_reason: str


class UserRouteCard(BaseModel):
    key: RouteType
    label: str
    badges: List[RouteBadge] = Field(default_factory=list)
    duration_min: float
    distance_km: float
    delay_min: float
    congestion_score: float = 0.0
    risk_level: Literal["low", "medium", "high"]
    summary: str
    geometry: List[RoutePoint] = Field(default_factory=list)
    road_geometry: List[RoutePoint] = Field(default_factory=list)
    access_geometry: List[List[RoutePoint]] = Field(default_factory=list)
    top_alerts: List[UserRouteAlert] = Field(default_factory=list)
    why_changed: List[str] = Field(default_factory=list)
    top_penalized_segments: List[SegmentImpact] = Field(default_factory=list)
    top_preferred_vias: List[PreferredViaImpact] = Field(default_factory=list)
    congestion_coverage: RouteCongestionCoverage = Field(default_factory=RouteCongestionCoverage)
    incident_exposure: IncidentExposure = Field(default_factory=IncidentExposure)
    pm25_exposure: Pm25Exposure | None = None
    urban_wellbeing: UrbanWellbeingAnalysis | None = None
    healthy_route_score: float | None = Field(None, ge=0.0, le=100.0)
    optimization_trace: RouteOptimizationTrace | None = None
    cycleway_coverage: CyclewayCoverage | None = None
    bicycle_suggestion: str | None = None
    contextual_messages: List[ContextualMobilityMessage] = Field(default_factory=list)
    active_mobility_estimate: ActiveMobilityEstimate | None = None


class PlanRouteResponse(BaseModel):
    selected_route_key: RouteType
    routes: List[UserRouteCard] = Field(default_factory=list)
    routes_by_type: dict[RouteType, UserRouteCard] = Field(default_factory=dict)
    summary: UserRouteSummary
    alerts: List[UserRouteAlert] = Field(default_factory=list)
    contextual_messages: List[ContextualMobilityMessage] = Field(default_factory=list)
    hotspots: List[HotspotPoint] = Field(default_factory=list)
    map_bounds: RegionBounds
