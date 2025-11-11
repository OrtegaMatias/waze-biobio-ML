# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class AssociationRequest(BaseModel):
    event_types: List[str] = Field(default_factory=list)
    communes: List[str] = Field(default_factory=list)
    franjas: List[str] = Field(default_factory=list)
    durations: List[str] = Field(default_factory=list)
    velocities: List[str] = Field(default_factory=list)
    day_type: Optional[str] = None
    via_preferred: Optional[str] = None
    min_support: float = Field(0.02, ge=0.001, le=0.5)
    min_confidence: float = Field(0.4, ge=0.1, le=0.95)
    limit: int = Field(5, ge=1, le=15)


class AssociationRecommendation(BaseModel):
    via: str
    communes: List[str]
    events: int
    accident_ratio: float
    duration_avg: float
    speed_avg: float
    confidence: float
    lift: float
    support: float
    rule: str
    lat: Optional[float] = None
    lon: Optional[float] = None


class AssociationResponse(BaseModel):
    recommendations: List[AssociationRecommendation]


class CollaborativeRequest(BaseModel):
    user_id: str
    known_vias: List[str] = Field(default_factory=list)
    strategy: Literal["ubcf", "ibcf"] = "ubcf"
    limit: int = Field(5, ge=1, le=20)


class CollaborativeRecommendation(BaseModel):
    via: str
    estimated_rating: float
    strategy: str


class CollaborativeResponse(BaseModel):
    recommendations: List[CollaborativeRecommendation]
