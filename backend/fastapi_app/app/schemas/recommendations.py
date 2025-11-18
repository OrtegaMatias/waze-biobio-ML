# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class CollaborativeRequest(BaseModel):
    user_id: str
    known_vias: List[str] = Field(default_factory=list)
    strategy: Literal["ubcf", "ibcf"] = "ubcf"
    limit: int = Field(5, ge=1, le=200)


class CollaborativeRecommendation(BaseModel):
    via: str
    estimated_rating: float
    strategy: str


class CollaborativeResponse(BaseModel):
    recommendations: List[CollaborativeRecommendation]


class PlaygroundRequest(BaseModel):
    user_id: str
    known_vias: List[str] = Field(default_factory=list)
    limit: int = Field(5, ge=1, le=200)
    strategies: List[Literal["ubcf", "ibcf"]] = Field(default_factory=list)


class PlaygroundResponse(BaseModel):
    ubcf: List[CollaborativeRecommendation] = Field(default_factory=list)
    ibcf: List[CollaborativeRecommendation] = Field(default_factory=list)
