# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..schemas.routes import PlaceResult


class GeocodingConfigError(RuntimeError):
    pass


class GeocodingLookupError(RuntimeError):
    pass


class GeocodingService:
    def __init__(self) -> None:
        self._token = os.getenv("MAPBOX_TOKEN") or os.getenv("VITE_MAPBOX_TOKEN")
        self._base_url = os.getenv("MAPBOX_GEOCODING_BASE_URL", "https://api.mapbox.com/search/geocode/v6")

    @property
    def configured(self) -> bool:
        return bool(self._token)

    def search(self, query: str, limit: int = 5) -> list[PlaceResult]:
        self._ensure_configured()
        params = {
            "q": query,
            "limit": max(1, min(limit, 10)),
            "language": "es",
            "country": "cl",
            "autocomplete": "true",
            "access_token": self._token,
        }
        payload = self._request_json(f"{self._base_url}/forward?{urlencode(params)}")
        return self._normalize_features(payload)

    def reverse(self, lat: float, lon: float) -> PlaceResult | None:
        self._ensure_configured()
        params = {
            "longitude": lon,
            "latitude": lat,
            "limit": 1,
            "language": "es",
            "access_token": self._token,
        }
        payload = self._request_json(f"{self._base_url}/reverse?{urlencode(params)}")
        results = self._normalize_features(payload)
        return results[0] if results else None

    def _ensure_configured(self) -> None:
        if not self.configured:
            raise GeocodingConfigError(
                "Geocoding no configurado. Define MAPBOX_TOKEN o VITE_MAPBOX_TOKEN para habilitar busqueda y reverse geocoding."
            )

    @staticmethod
    def _extract_label(feature: dict[str, Any]) -> str:
        properties = feature.get("properties") or {}
        context = properties.get("context") or {}
        candidates = [
            properties.get("full_address"),
            properties.get("place_formatted"),
            properties.get("name_preferred"),
            feature.get("place_name"),
            feature.get("name"),
        ]
        for value in candidates:
            if value:
                return str(value)
        parts = [feature.get("name"), context.get("place", {}).get("name"), context.get("region", {}).get("name")]
        return ", ".join(str(part) for part in parts if part)

    @staticmethod
    def _extract_coordinates(feature: dict[str, Any]) -> tuple[float, float] | None:
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            return float(coords[1]), float(coords[0])
        properties = feature.get("properties") or {}
        point = properties.get("coordinates") or {}
        if point.get("latitude") is not None and point.get("longitude") is not None:
            return float(point["latitude"]), float(point["longitude"])
        return None

    def _normalize_features(self, payload: dict[str, Any]) -> list[PlaceResult]:
        features = payload.get("features") or []
        results: list[PlaceResult] = []
        for feature in features:
            coords = self._extract_coordinates(feature)
            if not coords:
                continue
            lat, lon = coords
            bbox = feature.get("bbox")
            results.append(
                PlaceResult(
                    id=str(feature.get("id") or feature.get("mapbox_id") or f"{lat:.5f},{lon:.5f}"),
                    label=self._extract_label(feature) or f"{lat:.5f}, {lon:.5f}",
                    lat=lat,
                    lon=lon,
                    bbox=[float(value) for value in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None,
                )
            )
        return results

    @staticmethod
    def _request_json(url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": "waze-biobio-ml/0.3"})
        try:
            with urlopen(request, timeout=8) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
            raise GeocodingLookupError(f"Proveedor de geocoding respondio con error: {detail}") from exc
        except URLError as exc:
            raise GeocodingLookupError("No fue posible contactar al proveedor de geocoding.") from exc
        except TimeoutError as exc:
            raise GeocodingLookupError("El proveedor de geocoding excedio el tiempo de espera.") from exc
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise GeocodingLookupError("La respuesta del proveedor de geocoding no fue JSON valido.") from exc


@lru_cache(maxsize=1)
def get_geocoding_service() -> GeocodingService:
    return GeocodingService()
