# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    key: str
    label: str


class DatasetStatus(BaseModel):
    current: str
    available: List[DatasetInfo] = Field(default_factory=list)


class DatasetChangeRequest(BaseModel):
    profile: str
