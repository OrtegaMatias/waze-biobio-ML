# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

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


class RouteVariant(BaseModel):
    distance_km: float
    estimated_duration_min: float
    steps: List[RouteStepResponse]
    geometry: List[RoutePoint]
    extra_delay_min: float = 0.0


class RouteResponse(BaseModel):
    reference: RouteVariant
    personalized: RouteVariant
