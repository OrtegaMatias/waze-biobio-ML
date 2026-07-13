# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, Point, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from .cycleway_service import NON_ROUTABLE_CYCLEWAY_CATEGORIES, load_cycleways

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WELLBEING_PATH = PROJECT_ROOT / "data_processed" / "gran_concepcion_urban_wellbeing.geojson"
DEFAULT_NEARBY_BUFFER_M = 30.0
CYCLEWAY_COVERAGE_BUFFER_M = 15.0
MIN_CYCLEWAY_ROUTE_RATIO = 0.35
WAYPOINT_SEARCH_BUFFER_M = 1_200.0
WAYPOINT_MIN_OFFSET_M = 120.0
WAYPOINT_MAX_DIRECT_RATIO = 1.20
REFERENCE_LAT = -36.82

CATEGORY_WEIGHTS = {
    "green_space": 0.42,
    "blue_space": 0.18,
    "tree_cover": 0.12,
    "public_space": 0.10,
    "sustainability": 0.05,
    "cycleway": 0.13,
}


def _empty_collection() -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": "gran_concepcion_urban_wellbeing",
        "features": [],
    }


@lru_cache(maxsize=2)
def load_wellbeing_features(path: str | None = None) -> dict[str, Any]:
    source_path = Path(path) if path else DEFAULT_WELLBEING_PATH
    if not source_path.exists():
        return _empty_collection()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_collection()
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        return _empty_collection()
    return payload


def _project_xy(x: float, y: float, z: float | None = None) -> tuple[float, float]:
    del z
    scale_x = 111_320.0 * math.cos(math.radians(REFERENCE_LAT))
    scale_y = 110_540.0
    return x * scale_x, y * scale_y


def _route_pairs(geometry: list[Any]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in geometry:
        lat = getattr(item, "lat", None)
        lon = getattr(item, "lon", None)
        if isinstance(item, dict):
            lat = item.get("lat", lat)
            lon = item.get("lon", lon)
        if lat is None or lon is None:
            continue
        pairs.append((float(lon), float(lat)))
    return pairs


def _feature_name(properties: dict[str, Any]) -> str:
    return str(properties.get("name") or properties.get("subtype") or "Elemento urbano")


def _cycleway_feature_id(properties: dict[str, Any], index: int) -> str:
    if properties.get("local_id"):
        return f"cycleway-local-{properties['local_id']}"
    if properties.get("minvu_id") is not None:
        suffix = properties.get("minvu_path_index", index)
        return f"cycleway-minvu-{properties['minvu_id']}-{suffix}"
    if properties.get("osm_id") is not None:
        return f"cycleway-osm-{properties['osm_id']}"
    return f"cycleway-feature-{index}"


def _cycleway_name(properties: dict[str, Any]) -> str:
    return str(properties.get("name") or properties.get("project") or "Ciclovia")


def _cycleway_weight(properties: dict[str, Any]) -> float:
    category = str(properties.get("category") or "")
    if category in {"segregated_cycleway", "cycle_track", "minvu_existing_cycleway", "local_verified_cycleway"}:
        return 1.0
    if category in {"shared_lane", "bicycle_access"}:
        return 0.75
    return 0.85


def _cycleway_features() -> list[dict[str, Any]]:
    payload = load_cycleways()
    features: list[dict[str, Any]] = []
    for index, feature in enumerate(payload.get("features", [])):
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
        if not geometry or geometry.get("type") != "LineString":
            continue
        if properties.get("category") in NON_ROUTABLE_CYCLEWAY_CATEGORIES:
            continue
        merged_properties = {
            "feature_id": _cycleway_feature_id(properties, index),
            "name": _cycleway_name(properties),
            "category": "cycleway",
            "subtype": str(properties.get("category") or "cycleway"),
            "base_weight": _cycleway_weight(properties),
            "source": str(properties.get("source") or "OpenStreetMap/Overpass"),
        }
        features.append({"type": "Feature", "properties": merged_properties, "geometry": geometry})
    return features


class UrbanWellbeingService:
    def __init__(self, path: str | None = None) -> None:
        self.payload = load_wellbeing_features(path)
        self.features: list[dict[str, Any]] = []
        self.geometries: list[Any] = []
        self.waypoint_points: list[Point] = []
        raw_features = list(self.payload.get("features", []))
        if path is None:
            raw_features.extend(_cycleway_features())
        for raw_feature in raw_features:
            geometry_payload = raw_feature.get("geometry") if isinstance(raw_feature, dict) else None
            properties = raw_feature.get("properties") if isinstance(raw_feature, dict) else None
            if not geometry_payload or not isinstance(properties, dict):
                continue
            category = str(properties.get("category") or "")
            if category not in CATEGORY_WEIGHTS:
                continue
            try:
                raw_geometry = shape(geometry_payload)
                geometry = transform(_project_xy, raw_geometry)
            except Exception:
                continue
            if geometry.is_empty:
                continue
            self.features.append(raw_feature)
            self.geometries.append(geometry)
            self.waypoint_points.append(raw_geometry.representative_point())
        self.tree = STRtree(self.geometries) if self.geometries else None

    def route_cost_factor(self, lat: float, lon: float, nearby_buffer_m: float = DEFAULT_NEARBY_BUFFER_M) -> float:
        if self.tree is None:
            return 1.0
        point = transform(_project_xy, Point(float(lon), float(lat)))
        nearby = self.tree.query(point.buffer(nearby_buffer_m))
        benefit = 0.0
        for raw_index in nearby:
            index = int(raw_index)
            geometry = self.geometries[index]
            distance = geometry.distance(point)
            if distance > nearby_buffer_m:
                continue
            properties = self.features[index].get("properties") or {}
            category = str(properties.get("category") or "")
            if category == "cycleway":
                continue
            base_weight = float(properties.get("base_weight") or 1.0)
            proximity = max(0.0, 1.0 - distance / nearby_buffer_m)
            benefit = max(benefit, CATEGORY_WEIGHTS.get(category, 0.0) * base_weight * proximity)
        # The factor only generates candidates. Final selection evaluates complete routes.
        return round(max(0.65, 1.0 - benefit * 0.70), 4)

    def candidate_waypoints(
        self,
        origin: Any,
        destination: Any,
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        if self.tree is None or limit <= 0:
            return []
        origin_point = transform(_project_xy, Point(float(origin.lon), float(origin.lat)))
        destination_point = transform(_project_xy, Point(float(destination.lon), float(destination.lat)))
        direct_route = LineString([origin_point, destination_point])
        direct_length = direct_route.length
        if direct_length <= 0:
            return []

        ranked: list[tuple[float, int, float]] = []
        search_area = direct_route.buffer(WAYPOINT_SEARCH_BUFFER_M)
        for raw_index in self.tree.query(search_area):
            index = int(raw_index)
            geometry = self.geometries[index]
            waypoint = transform(_project_xy, self.waypoint_points[index])
            offset = geometry.distance(direct_route)
            if offset < WAYPOINT_MIN_OFFSET_M or offset > WAYPOINT_SEARCH_BUFFER_M:
                continue
            progress = direct_route.project(waypoint) / direct_length
            if progress < 0.15 or progress > 0.85:
                continue
            direct_via_length = origin_point.distance(waypoint) + waypoint.distance(destination_point)
            if direct_via_length > direct_length * WAYPOINT_MAX_DIRECT_RATIO:
                continue
            properties = self.features[index].get("properties") or {}
            category = str(properties.get("category") or "")
            if category == "cycleway":
                continue
            base_weight = float(properties.get("base_weight") or 1.0)
            category_weight = CATEGORY_WEIGHTS.get(category, 0.0)
            midpoint_bonus = 1.0 - abs(progress - 0.5)
            offset_bonus = min(offset, 600.0) / 600.0
            score = category_weight * base_weight * (0.7 + 0.2 * midpoint_bonus + 0.1 * offset_bonus)
            ranked.append((score, index, offset))

        selected: list[dict[str, Any]] = []
        selected_points: list[Point] = []
        for _, index, offset in sorted(ranked, reverse=True):
            point = self.waypoint_points[index]
            projected = transform(_project_xy, point)
            if any(projected.distance(existing) < 500.0 for existing in selected_points):
                continue
            properties = self.features[index].get("properties") or {}
            selected.append(
                {
                    "lat": float(point.y),
                    "lon": float(point.x),
                    "name": _feature_name(properties),
                    "category": str(properties.get("category") or ""),
                    "offset_m": round(offset, 1),
                }
            )
            selected_points.append(projected)
            if len(selected) >= limit:
                break
        return selected

    def evaluate_route(
        self,
        geometry: list[Any],
        nearby_buffer_m: float = DEFAULT_NEARBY_BUFFER_M,
    ) -> dict[str, Any]:
        route_pairs = _route_pairs(geometry)
        if len(route_pairs) < 2:
            return self._empty_analysis(nearby_buffer_m)
        route = transform(_project_xy, LineString(route_pairs))
        route_length = route.length
        if route_length <= 0 or self.tree is None:
            return self._empty_analysis(nearby_buffer_m)

        corridor = route.buffer(nearby_buffer_m)
        category_geometries: dict[str, list[Any]] = {key: [] for key in CATEGORY_WEIGHTS}
        nearby_features: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for raw_index in self.tree.query(corridor):
            index = int(raw_index)
            feature = self.features[index]
            feature_geometry = self.geometries[index]
            if not feature_geometry.intersects(corridor):
                continue
            properties = feature.get("properties") or {}
            category = str(properties.get("category") or "")
            influence_buffer = CYCLEWAY_COVERAGE_BUFFER_M if category == "cycleway" else nearby_buffer_m
            influence = feature_geometry.buffer(influence_buffer)
            covered_route = route.intersection(influence)
            if covered_route.is_empty:
                continue
            category_geometries[category].append(covered_route)
            feature_id = str(properties.get("feature_id") or f"feature-{index}")
            if feature_id in seen_ids:
                continue
            seen_ids.add(feature_id)
            nearby_features.append(
                {
                    "feature_id": feature_id,
                    "name": _feature_name(properties),
                    "category": category,
                    "subtype": str(properties.get("subtype") or category),
                    "distance_m": round(feature_geometry.distance(route), 1),
                    "source": str(properties.get("source") or "OpenStreetMap/Overpass"),
                    "base_weight": float(properties.get("base_weight") or 1.0),
                }
            )

        ratios: dict[str, float] = {}
        for category, intersections in category_geometries.items():
            covered_length = unary_union(intersections).length if intersections else 0.0
            ratios[category] = min(1.0, covered_length / route_length)

        if ratios["cycleway"] < MIN_CYCLEWAY_ROUTE_RATIO:
            ratios["cycleway"] = 0.0
            nearby_features = [feature for feature in nearby_features if feature["category"] != "cycleway"]

        score = sum(CATEGORY_WEIGHTS[key] * ratios[key] for key in CATEGORY_WEIGHTS)
        top_features = sorted(
            nearby_features,
            key=lambda item: (
                -(CATEGORY_WEIGHTS.get(item["category"], 0.0) * item["base_weight"]),
                item["distance_m"],
            ),
        )[:5]
        return {
            "available": bool(self.features),
            "score": round(score * 100.0, 1),
            "green_ratio": round(ratios["green_space"], 3),
            "blue_ratio": round(ratios["blue_space"], 3),
            "tree_ratio": round(ratios["tree_cover"], 3),
            "public_space_ratio": round(ratios["public_space"], 3),
            "sustainability_ratio": round(ratios["sustainability"], 3),
            "cycleway_ratio": round(ratios["cycleway"], 3),
            "nearby_feature_count": len(nearby_features),
            "nearby_buffer_m": nearby_buffer_m,
            "top_features": top_features,
            "method": "Cobertura del corredor de ruta por elementos de bienestar urbano adyacentes al recorrido.",
            "data_source": "OpenStreetMap/Overpass, MINVU GeoIDE y fuentes abiertas complementarias",
        }

    @staticmethod
    def _empty_analysis(nearby_buffer_m: float) -> dict[str, Any]:
        return {
            "available": False,
            "score": 0.0,
            "green_ratio": 0.0,
            "blue_ratio": 0.0,
            "tree_ratio": 0.0,
            "public_space_ratio": 0.0,
            "sustainability_ratio": 0.0,
            "cycleway_ratio": 0.0,
            "nearby_feature_count": 0,
            "nearby_buffer_m": nearby_buffer_m,
            "top_features": [],
            "method": "Cobertura del corredor de ruta por elementos de bienestar urbano adyacentes al recorrido.",
            "data_source": "OpenStreetMap/Overpass, MINVU GeoIDE y fuentes abiertas complementarias",
        }


@lru_cache(maxsize=1)
def get_urban_wellbeing_service() -> UrbanWellbeingService:
    return UrbanWellbeingService()
