# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from shapely.geometry import LineString
from shapely.ops import unary_union

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CYCLEWAYS_PATH = PROJECT_ROOT / "data_processed" / "gran_concepcion_cycleways.geojson"
DEFAULT_MINVU_PATH = PROJECT_ROOT / "data_processed" / "gran_concepcion_cycleways_minvu.geojson"
DEFAULT_OVERRIDES_PATH = PROJECT_ROOT / "data_processed" / "gran_concepcion_cycleway_overrides.geojson"
DEFAULT_NEARBY_BUFFER_M = 80.0
HIGH_COVERAGE_THRESHOLD = 0.35
NON_ROUTABLE_CYCLEWAY_CATEGORIES = {"minvu_planned_cycleway"}


def _empty_collection() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": "gran_concepcion_cycleways",
        "features": [],
    }


def _load_collection(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_collection()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_collection()
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        return _empty_collection()
    return payload


@lru_cache(maxsize=1)
def load_cycleways(path: str | None = None) -> dict[str, Any]:
    source_path = Path(path) if path else DEFAULT_CYCLEWAYS_PATH
    payload = _load_collection(source_path)
    if path:
        return payload

    minvu = _load_collection(DEFAULT_MINVU_PATH)
    overrides = _load_collection(DEFAULT_OVERRIDES_PATH)
    extra_features = [*minvu.get("features", []), *overrides.get("features", [])]
    if not extra_features:
        return payload
    merged = dict(payload)
    merged["features"] = [*payload.get("features", []), *extra_features]
    return merged


def _project_points(points: list[tuple[float, float]], reference_lat: float) -> list[tuple[float, float]]:
    scale_x = 111_320.0 * math.cos(math.radians(reference_lat))
    scale_y = 110_540.0
    return [(lon * scale_x, lat * scale_y) for lon, lat in points]


def _route_pairs(geometry: list[Any]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for point in geometry:
        lat = getattr(point, "lat", None)
        lon = getattr(point, "lon", None)
        if lat is None and isinstance(point, dict):
            lat = point.get("lat")
            lon = point.get("lon")
        if lat is None or lon is None:
            continue
        pairs.append((float(lon), float(lat)))
    return pairs


def _cycleway_lines(payload: dict[str, Any], reference_lat: float) -> list[LineString]:
    lines: list[LineString] = []
    for feature in payload.get("features", []):
        properties = feature.get("properties") if isinstance(feature, dict) else None
        if isinstance(properties, dict) and properties.get("category") in NON_ROUTABLE_CYCLEWAY_CATEGORIES:
            continue
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry or geometry.get("type") != "LineString":
            continue
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        try:
            raw_points = [(float(lon), float(lat)) for lon, lat in coordinates]
        except (TypeError, ValueError):
            continue
        line = LineString(_project_points(raw_points, reference_lat))
        if line.length > 0:
            lines.append(line)
    return lines


def _data_source_label(payload: dict[str, Any]) -> str:
    sources = {
        str((feature.get("properties") or {}).get("source"))
        for feature in payload.get("features", [])
        if isinstance(feature, dict) and isinstance(feature.get("properties"), dict)
    }
    sources.discard("None")
    if "MINVU GeoIDE" in sources and "OpenStreetMap/Overpass" in sources:
        return "OpenStreetMap/Overpass + MINVU GeoIDE"
    if "MINVU GeoIDE" in sources:
        return "MINVU GeoIDE"
    return "OpenStreetMap/Overpass"


def estimate_route_coverage(
    geometry: list[Any],
    *,
    path: str | None = None,
    nearby_buffer_m: float = DEFAULT_NEARBY_BUFFER_M,
) -> dict[str, Any]:
    route_points = _route_pairs(geometry)
    if len(route_points) < 2:
        return {
            "available": False,
            "coverage_ratio": 0.0,
            "nearby_cycleway_km": 0.0,
            "route_km": 0.0,
            "nearby_buffer_m": nearby_buffer_m,
            "has_high_coverage": False,
            "data_source": "OpenStreetMap/Overpass",
        }

    payload = load_cycleways(path)
    data_source_label = _data_source_label(payload)
    reference_lat = sum(lat for _lon, lat in route_points) / len(route_points)
    route_line = LineString(_project_points(route_points, reference_lat))
    route_length_m = route_line.length
    if route_length_m <= 0:
        return {
            "available": False,
            "coverage_ratio": 0.0,
            "nearby_cycleway_km": 0.0,
            "route_km": 0.0,
            "nearby_buffer_m": nearby_buffer_m,
            "has_high_coverage": False,
            "data_source": data_source_label,
        }

    corridor = route_line.buffer(nearby_buffer_m)
    nearby_segments = []
    for line in _cycleway_lines(payload, reference_lat):
        if not line.intersects(corridor):
            continue
        nearby_segments.append(line.intersection(corridor))

    nearby_length_m = unary_union(nearby_segments).length if nearby_segments else 0.0
    coverage_ratio = min(1.0, nearby_length_m / route_length_m)
    return {
        "available": bool(payload.get("features")),
        "coverage_ratio": round(coverage_ratio, 3),
        "nearby_cycleway_km": round(nearby_length_m / 1000.0, 2),
        "route_km": round(route_length_m / 1000.0, 2),
        "nearby_buffer_m": nearby_buffer_m,
        "has_high_coverage": coverage_ratio >= HIGH_COVERAGE_THRESHOLD,
        "data_source": data_source_label,
    }
