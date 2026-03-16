# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    key: str
    label: str


class DatasetStatus(BaseModel):
    current: str
    current_label: str
    available: List[DatasetInfo] = Field(default_factory=list)


class DatasetChangeRequest(BaseModel):
    profile: str


class DateCoverage(BaseModel):
    start: str | None = None
    end: str | None = None
    days: int = 0


class RawCounts(BaseModel):
    accidents: int = 0
    congestions: int = 0
    combined: int = 0


class DataQualitySummary(BaseModel):
    status: Literal["ok", "warning", "error"] = "ok"
    dataset_profile: str
    duplicate_incident_sources: bool = False
    date_range: DateCoverage = Field(default_factory=DateCoverage)
    missing_via_ratio: float = 0.0
    anomalous_communes: List[str] = Field(default_factory=list)
    raw_counts: RawCounts = Field(default_factory=RawCounts)
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class BootstrapStatus(BaseModel):
    status: Literal["idle", "running", "completed", "error"] = "idle"
    message: str
    percent: int = 0
    routing_nodes: int = 0
    routing_segments: int = 0
    duration_ms: float = 0.0
    dataset_profile: str
    quality: DataQualitySummary | None = None


class ReadinessStatus(BaseModel):
    status: Literal["ready", "warming", "error"] = "warming"
    ready: bool = False
    message: str
    dataset_profile: str
    bootstrap: BootstrapStatus


class DemoScenarioPoint(BaseModel):
    lat: float
    lon: float


class DemoScenario(BaseModel):
    id: str
    title: str
    description: str
    origin: DemoScenarioPoint
    destination: DemoScenarioPoint
    day_of_week: str
    departure_hour: float
    profile: str
    recommended_focus: str


class DemoScenarioList(BaseModel):
    scenarios: List[DemoScenario] = Field(default_factory=list)
