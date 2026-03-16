# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass(frozen=True)
class BackendClient:
    base_url: str = os.getenv("BACKEND_URL", "http://localhost:8000")
    timeout: float = float(os.getenv("BACKEND_TIMEOUT", "180"))

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get(self, path: str, **kwargs) -> Dict[str, Any]:
        timeout = kwargs.pop("timeout", self.timeout)
        response = requests.get(self._url(path), timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, payload: dict | None = None, **kwargs) -> Dict[str, Any]:
        timeout = kwargs.pop("timeout", self.timeout)
        response = requests.post(self._url(path), json=payload or {}, timeout=timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        return self.get("/health", timeout=5)

    def ready(self) -> Dict[str, Any]:
        return self.get("/readyz", timeout=10)

    def bootstrap(self) -> Dict[str, Any]:
        return self.post("/system/bootstrap", timeout=15)

    def bootstrap_status(self) -> Dict[str, Any]:
        return self.get("/system/bootstrap/status", timeout=10)

    def metadata(self) -> Dict[str, Any]:
        return self.get("/metadata/options")

    def hotspots(self, limit: int = 2000) -> Dict[str, Any]:
        return self.get("/metadata/hotspots", params={"limit": limit})

    def dataset_status(self) -> Dict[str, Any]:
        return self.get("/system/dataset")

    def set_dataset(self, profile: str) -> Dict[str, Any]:
        return self.post("/system/dataset", payload={"profile": profile}, timeout=20)

    def demo_scenarios(self) -> Dict[str, Any]:
        return self.get("/system/demo-scenarios")

    def playground(self, payload: dict) -> Dict[str, Any]:
        return self.post("/recommendations/playground", payload=payload)

    def optimal_route(self, payload: dict, timeout: float | None = None) -> Dict[str, Any]:
        return self.post("/routes/optimal", payload=payload, timeout=timeout or self.timeout)
