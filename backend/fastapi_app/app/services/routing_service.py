# -*- coding: utf-8 -*-
from __future__ import annotations

import gc
import json
import logging
import math
import pickle
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from algorithms.recommenders import data_loader, routing

from ..schemas.routes import (
    IncidentExposure,
    PreferredViaImpact,
    RoutePoint,
    RouteComparison,
    RouteCongestionCoverage,
    RouteOptimizationTrace,
    RouteDelta,
    RouteRequest,
    RouteResponse,
    RouteStepResponse,
    RouteVariant,
    SegmentImpact,
)
from .air_quality_service import get_air_quality_service
from .environmental_impact_service import get_environmental_impact_service
from .routing_costs import RouteCostTotals, RoutingCostModel, sum_breakdowns
from .urban_wellbeing_service import get_urban_wellbeing_service

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "cache"
GRAPH_CACHE_VERSION = 8
MAX_POINT_SNAP_KM = 0.5
ON_STREET_SNAP_TOLERANCE_KM = 0.008
GEOMETRY_SIMPLIFY_TOLERANCE_M = 12.0
MAX_GEOMETRY_STEP_KM = 0.12
HEALTHY_PM25_WEIGHT = 0.25
HEALTHY_CONGESTION_WEIGHT = 0.35
HEALTHY_WELLBEING_WEIGHT = 0.30
HEALTHY_TRAVEL_WEIGHT = 0.10
HEALTHY_MAX_DISTANCE_RATIO = 2.0
HEALTHY_MAX_EXTRA_MIN = 15.0
ROUTE_ASSUMED_SPEED_KMH = 35.0
HEALTHY_PM25_MEANINGFUL_DELTA = 2.0
HEALTHY_CONGESTION_RISK_TOLERANCE = 12.0
HEALTHY_CONGESTION_SEGMENT_TOLERANCE = 1
HEALTHY_HIGH_CONGESTION_PCT_TOLERANCE = 5.0
HEALTHY_ENVIRONMENT_WAYPOINT_LIMIT = 1
HEALTHY_WAYPOINT_SEARCH_EXTRA_KM = 1.5
HEALTHY_MAX_BACKTRACK_RATIO = 0.20
ALTERNATIVE_OVERLAP_PENALTY = 3.0
ALTERNATIVE_MAX_DISTANCE_RATIO = 2.0
ALTERNATIVE_MAX_EXTRA_MIN = 15.0
ALTERNATIVE_MAX_TIME_RATIO = 1.50
ACTIVE_CONGESTION_MATCH_TOLERANCE_M = 45.0
ACTIVE_CONGESTION_NODE_TOLERANCE_M = 55.0
ACTIVE_CONGESTION_FULL_IMPACT_M = 180.0
ACTIVE_CONGESTION_MIN_IMPACT_FACTOR = 0.12
ENVIRONMENT_LOW_ZONE_PENALTY = (0.04, 0.10)
ENVIRONMENT_MEDIUM_ZONE_PENALTY = (0.90, 1.50)
ENVIRONMENT_HIGH_ZONE_PENALTY = (2.00, 3.00)
DAY_ALIASES = {
    "lunes": "Monday",
    "martes": "Tuesday",
    "miércoles": "Wednesday",
    "miercoles": "Wednesday",
    "jueves": "Thursday",
    "viernes": "Friday",
    "sábado": "Saturday",
    "sabado": "Saturday",
    "domingo": "Sunday",
}

# Límite simplificado del Parque Metropolitano Cerro Caracol (OSM way 1061161704).
# El corredor de Víctor Lamas se permite explícitamente porque bordea el parque
# y constituye la alternativa vial indicada para atravesar este sector.
CERRO_CARACOL_PROTECTED_POLYGON: Tuple[Tuple[float, float], ...] = (
    (-36.8355807, -73.0555244),
    (-36.8380489, -73.0549166),
    (-36.8360418, -73.0496085),
    (-36.8383344, -73.0490452),
    (-36.8382292, -73.0443004),
    (-36.8406227, -73.0414385),
    (-36.8436837, -73.0391049),
    (-36.8436923, -73.0379999),
    (-36.8420180, -73.0322117),
    (-36.8386478, -73.0331934),
    (-36.8374899, -73.0336365),
    (-36.8365613, -73.0335260),
    (-36.8359473, -73.0356985),
    (-36.8330443, -73.0398319),
    (-36.8327551, -73.0395425),
    (-36.8320432, -73.0406373),
    (-36.8305391, -73.0415440),
    (-36.8308884, -73.0420446),
    (-36.8312384, -73.0433494),
    (-36.8320678, -73.0445917),
    (-36.8327758, -73.0452263),
    (-36.8339353, -73.0472987),
    (-36.8352312, -73.0493902),
    (-36.8343773, -73.0500710),
    (-36.8344362, -73.0524658),
    (-36.8351270, -73.0534724),
)
CERRO_CARACOL_APPROACH_BUFFER_KM = 0.60
CERRO_CARACOL_APPROACH_PENALTY = 80.0


def _point_in_polygon(lat: float, lon: float, polygon: Tuple[Tuple[float, float], ...]) -> bool:
    inside = False
    previous_lat, previous_lon = polygon[-1]
    for current_lat, current_lon in polygon:
        crosses_latitude = (current_lat > lat) != (previous_lat > lat)
        if crosses_latitude:
            crossing_lon = (
                (previous_lon - current_lon) * (lat - current_lat) / (previous_lat - current_lat)
                + current_lon
            )
            if lon < crossing_lon:
                inside = not inside
        previous_lat, previous_lon = current_lat, current_lon
    return inside


def _is_victor_lamas(via: str) -> bool:
    normalized = unicodedata.normalize("NFKD", str(via or ""))
    ascii_name = "".join(char for char in normalized if not unicodedata.combining(char)).lower()
    return "victor lamas" in ascii_name


def _point_to_segment_distance_km(
    lat: float,
    lon: float,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> float:
    reference_lat = math.radians((lat + start[0] + end[0]) / 3)

    def project(point_lat: float, point_lon: float) -> Tuple[float, float]:
        return (
            math.radians(point_lon - lon) * routing.EARTH_RADIUS_KM * math.cos(reference_lat),
            math.radians(point_lat - lat) * routing.EARTH_RADIUS_KM,
        )

    start_x, start_y = project(*start)
    end_x, end_y = project(*end)
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    denominator = delta_x * delta_x + delta_y * delta_y
    if denominator <= 0:
        return math.hypot(start_x, start_y)
    position = max(0.0, min(1.0, -(start_x * delta_x + start_y * delta_y) / denominator))
    return math.hypot(start_x + position * delta_x, start_y + position * delta_y)


def _distance_to_cerro_caracol_km(lat: float, lon: float) -> float:
    if _point_in_polygon(lat, lon, CERRO_CARACOL_PROTECTED_POLYGON):
        return 0.0
    distances = []
    previous = CERRO_CARACOL_PROTECTED_POLYGON[-1]
    for current in CERRO_CARACOL_PROTECTED_POLYGON:
        distances.append(_point_to_segment_distance_km(lat, lon, previous, current))
        previous = current
    return min(distances)


def _cerro_caracol_edge_allowed(source: routing.GraphNode, target: routing.GraphNode) -> bool:
    midpoint_lat = (source.lat + target.lat) / 2
    midpoint_lon = (source.lon + target.lon) / 2
    touches_protected_area = any(
        _point_in_polygon(lat, lon, CERRO_CARACOL_PROTECTED_POLYGON)
        for lat, lon in (
            (source.lat, source.lon),
            (target.lat, target.lon),
            (midpoint_lat, midpoint_lon),
        )
    )
    if not touches_protected_area:
        return True
    return _is_victor_lamas(source.via) or _is_victor_lamas(target.via)


def _cerro_caracol_edge_cost_factor(source: routing.GraphNode, target: routing.GraphNode) -> float:
    if _is_victor_lamas(source.via) or _is_victor_lamas(target.via):
        return 1.0
    midpoint_lat = (source.lat + target.lat) / 2
    midpoint_lon = (source.lon + target.lon) / 2
    closest_distance = min(
        _distance_to_cerro_caracol_km(lat, lon)
        for lat, lon in (
            (source.lat, source.lon),
            (target.lat, target.lon),
            (midpoint_lat, midpoint_lon),
        )
    )
    if closest_distance <= CERRO_CARACOL_APPROACH_BUFFER_KM:
        return CERRO_CARACOL_APPROACH_PENALTY
    return 1.0


def _vehicle_edge_allowed(source: routing.GraphNode, target: routing.GraphNode) -> bool:
    blocked_classes = {"bridleway", "cycleway", "footway", "path", "pedestrian", "steps"}
    blocked_access = {"no", "private"}
    for node in (source, target):
        if str(getattr(node, "road_class", "") or "").strip().lower() in blocked_classes:
            return False
        if str(getattr(node, "motor_vehicle", "") or "").strip().lower() in blocked_access:
            return False
    return True


@dataclass(frozen=True)
class RoadSnap:
    point: Dict[str, float]
    projected_point: Dict[str, float]
    distance_km: float
    source_node_costs: Dict[str, float]
    target_node_costs: Dict[str, float]


def _graph_cache_paths() -> Tuple[Path, Path]:
    profile = data_loader.get_data_profile()
    return (
        CACHE_DIR / f"route_graph.{profile}.pkl",
        CACHE_DIR / f"route_graph.{profile}.meta.json",
    )


def _load_graph_cache(signature):
    graph_cache, graph_meta = _graph_cache_paths()
    if not graph_cache.exists() or not graph_meta.exists():
        return None
    try:
        meta = json.loads(graph_meta.read_text())
    except Exception:
        return None
    if meta.get("signature") != [GRAPH_CACHE_VERSION, *list(signature)]:
        return None
    try:
        with graph_cache.open("rb") as fh:
            return pickle.load(fh)
    except Exception as exc:
        logger.warning("No se pudo cargar cache de grafo %s: %s", graph_cache.name, exc)
        return None


def _store_graph_cache(signature, bundle):
    graph_cache, graph_meta = _graph_cache_paths()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with graph_cache.open("wb") as fh:
        pickle.dump(bundle, fh)
    graph_meta.write_text(json.dumps({"signature": [GRAPH_CACHE_VERSION, *list(signature)]}))


def _normalize_day(value: str | None) -> str:
    if not value:
        return "Monday"
    cleaned = value.strip().lower()
    return DAY_ALIASES.get(cleaned, value.strip().title())


class RoutingService:
    def __init__(self, progress=None) -> None:
        self._data_version = None
        self.events = None
        self.graph = None
        self.segment_lookup = {}
        self.segment_lengths_km: Dict[str, float] = {}
        self._progress = progress
        self._build_lock = Lock()
        logger.info("Inicializando RoutingService (lazy build)")

    def _build_structures(self, progress=None) -> None:
        if not self._build_lock.acquire(blocking=False):
            # Otra hebra ya está construyendo; espera a que termine y salir.
            with self._build_lock:
                return
        try:
            logger.info("Construyendo estructuras de ruta...")
            signature = data_loader.data_version()
            self._data_version = signature
            cached = _load_graph_cache(signature)
            if cached:
                if progress:
                    progress("Cargando grafo desde cache", 0.2)
                self.events = None
                self.graph = cached["graph"]
                self.segment_lookup = cached["segment_lookup"]
                self.segment_lengths_km = self._build_segment_lengths(self.segment_lookup)
                logger.info(
                    "Grafo cargado desde cache: %d nodos, %d segmentos",
                    len(self.graph.nodes),
                    len(self.segment_lookup),
                )
                if progress:
                    progress("Grafo listo (cache)", 1.0)
                return
            if progress:
                progress("Cargando red vial", 0.2)
            events = data_loader.load_route_network()
            if events.empty:
                raise ValueError(
                    "No se encontro una red vial utilizable. Genera data/processed/road_network.csv "
                    "desde data/processed/road_network_parts antes de iniciar el backend."
                )
            if progress:
                progress("Red vial cargada", 0.4)

            def rg_progress(stage: str, ratio: float) -> None:
                base = {"nodes": 0.4, "segments": 0.8, "junctions": 0.95}.get(stage, 0.4)
                span = 0.4 if stage == "nodes" else (0.15 if stage == "segments" else 0.05)
                percent = min(0.99, base + span * ratio)
                if progress:
                    progress(f"Construyendo grafo ({stage})", percent)

            self.graph = routing.RouteGraph.from_events(events, progress=rg_progress)
            self.segment_lookup = self._build_segment_lookup(events)
            self.segment_lengths_km = self._build_segment_lengths(self.segment_lookup)
            self.events = None
            logger.info(
                "Grafo cargado: %d nodos, %d segmentos",
                len(self.graph.nodes),
                len(self.segment_lookup),
            )
            _store_graph_cache(signature, {"graph": self.graph, "segment_lookup": self.segment_lookup})
            del events
            gc.collect()
            if progress:
                progress("Grafo listo", 1.0)
        finally:
            self._build_lock.release()

    def _ensure_fresh_data(self) -> None:
        current_version = data_loader.data_version()
        if self._data_version is None or self.graph is None:
            try:
                self._build_structures(self._progress)
            finally:
                self._data_version = data_loader.data_version()
            return
        if current_version != self._data_version:
            logger.info("Detectado cambio en datos (%s), reconstruyendo grafo...", current_version)
            try:
                self._build_structures(self._progress)
            finally:
                self._data_version = data_loader.data_version()

    def build(self, progress=None) -> None:
        self._data_version = data_loader.data_version()
        self._build_structures(progress)

    @staticmethod
    def _build_segment_lookup(events) -> Dict[str, Dict[int, Tuple[float, float]]]:
        lookup: Dict[str, Dict[int, Tuple[float, float]]] = {}
        df = events.loc[:, ["segment_id", "segment_seq", "lat", "lon"]].copy()
        df["segment_seq"] = pd.to_numeric(df["segment_seq"], errors="coerce").fillna(0).astype(int)
        for segment_id, group in df.groupby("segment_id"):
            seq_map = {
                int(row.segment_seq): (float(row.lat), float(row.lon))
                for row in group.sort_values("segment_seq").itertuples()
            }
            lookup[segment_id] = seq_map
        return lookup

    @staticmethod
    def _build_segment_lengths(
        segment_lookup: Dict[str, Dict[int, Tuple[float, float]]],
    ) -> Dict[str, float]:
        lengths: Dict[str, float] = {}
        for segment_id, seq_map in segment_lookup.items():
            ordered = [coord for _, coord in sorted(seq_map.items())]
            lengths[segment_id] = sum(
                routing.haversine_km(start[0], start[1], end[0], end[1])
                for start, end in zip(ordered, ordered[1:])
            )
        return lengths

    def _segment_between(self, segment_id: str, start_seq: int, end_seq: int) -> List[Tuple[float, float]]:
        seq_map = self.segment_lookup.get(segment_id)
        if not seq_map:
            return []
        if start_seq == end_seq:
            return []
        step = 1 if end_seq > start_seq else -1
        coords: List[Tuple[float, float]] = []
        for seq in range(start_seq + step, end_seq, step):
            point = seq_map.get(seq)
            if point:
                coords.append(point)
        return coords

    @classmethod
    def _nearest_point_on_polyline(
        cls,
        point: RoutePoint,
        coordinates: List[Tuple[float, float]],
    ) -> Dict[str, float]:
        if not coordinates:
            return {"lat": point.lat, "lon": point.lon}
        if len(coordinates) == 1:
            lat, lon = coordinates[0]
            return {"lat": lat, "lon": lon}

        reference_lat = point.lat
        target = {"lat": point.lat, "lon": point.lon}
        px, py = cls._project_point(target, reference_lat)
        best_distance = float("inf")
        best_point = {"lat": coordinates[0][0], "lon": coordinates[0][1]}

        for start_raw, end_raw in zip(coordinates, coordinates[1:]):
            start = {"lat": start_raw[0], "lon": start_raw[1]}
            end = {"lat": end_raw[0], "lon": end_raw[1]}
            ax, ay = cls._project_point(start, reference_lat)
            bx, by = cls._project_point(end, reference_lat)
            dx = bx - ax
            dy = by - ay
            if dx == 0 and dy == 0:
                t = 0.0
                nearest_x = ax
                nearest_y = ay
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
                nearest_x = ax + t * dx
                nearest_y = ay + t * dy
            distance = math.hypot(px - nearest_x, py - nearest_y)
            if distance < best_distance:
                best_distance = distance
                best_point = {
                    "lat": start["lat"] + (end["lat"] - start["lat"]) * t,
                    "lon": start["lon"] + (end["lon"] - start["lon"]) * t,
                }
        return best_point

    def _snap_point_to_step_segment(self, point: RoutePoint, step: routing.RouteStep) -> Dict[str, float]:
        seq_map = self.segment_lookup.get(step.segment_id)
        if not seq_map:
            return {"lat": step.lat, "lon": step.lon}
        coordinates = [coord for _, coord in sorted(seq_map.items())]
        return self._nearest_point_on_polyline(point, coordinates)

    def _nearest_road_snap(
        self,
        point: RoutePoint,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RoadSnap:
        if self.graph is None:
            fallback = {"lat": point.lat, "lon": point.lon}
            return RoadSnap(fallback, fallback, float("inf"), {}, {})

        target = {"lat": point.lat, "lon": point.lon}
        reference_lat = point.lat
        px, py = self._project_point(target, reference_lat)
        best: dict | None = None

        for segment_index, (segment_id, seq_map) in enumerate(self.segment_lookup.items()):
            if segment_index % 256 == 0 and should_cancel is not None and should_cancel():
                raise routing.RouteSearchCancelled()
            ordered = sorted(seq_map.items())
            for (start_seq, start_raw), (end_seq, end_raw) in zip(ordered, ordered[1:]):
                start = {"lat": start_raw[0], "lon": start_raw[1]}
                end = {"lat": end_raw[0], "lon": end_raw[1]}
                ax, ay = self._project_point(start, reference_lat)
                bx, by = self._project_point(end, reference_lat)
                dx = bx - ax
                dy = by - ay
                if dx == 0 and dy == 0:
                    t = 0.0
                    nearest_x = ax
                    nearest_y = ay
                else:
                    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
                    nearest_x = ax + t * dx
                    nearest_y = ay + t * dy
                distance_m = math.hypot(px - nearest_x, py - nearest_y)
                if best is not None and distance_m >= best["distance_m"]:
                    continue
                projected = {
                    "lat": start["lat"] + (end["lat"] - start["lat"]) * t,
                    "lon": start["lon"] + (end["lon"] - start["lon"]) * t,
                }
                start_node_id = f"{segment_id}::{int(start_seq)}"
                end_node_id = f"{segment_id}::{int(end_seq)}"
                if start_node_id not in self.graph.nodes or end_node_id not in self.graph.nodes:
                    continue
                best = {
                    "segment_id": segment_id,
                    "start_node_id": start_node_id,
                    "end_node_id": end_node_id,
                    "start": start,
                    "end": end,
                    "projected": projected,
                    "t": t,
                    "distance_m": distance_m,
                }

        if best is None:
            if not hasattr(self.graph, "nearest_nodes"):
                fallback = {"lat": point.lat, "lon": point.lon}
                return RoadSnap(fallback, fallback, 0.0, {}, {})
            nearest = self.graph.nearest_nodes(point.lat, point.lon, limit=1)
            if not nearest:
                fallback = {"lat": point.lat, "lon": point.lon}
                return RoadSnap(fallback, fallback, float("inf"), {}, {})
            node, distance_km = nearest[0]
            fallback = {"lat": node.lat, "lon": node.lon}
            return RoadSnap(
                point=fallback,
                projected_point=fallback,
                distance_km=distance_km,
                source_node_costs={node.node_id: distance_km},
                target_node_costs={node.node_id: distance_km},
            )

        start_node = self.graph.nodes[best["start_node_id"]]
        end_node = self.graph.nodes[best["end_node_id"]]
        is_oneway = bool(start_node.oneway or end_node.oneway)
        t = float(best["t"])
        segment_length_km = routing.haversine_km(
            best["start"]["lat"],
            best["start"]["lon"],
            best["end"]["lat"],
            best["end"]["lon"],
        )
        start_to_snap_km = segment_length_km * t
        snap_to_end_km = segment_length_km * (1.0 - t)
        eps = 1e-6

        if is_oneway:
            if t <= eps:
                source_costs = {best["start_node_id"]: 0.0}
                target_costs = {best["start_node_id"]: 0.0}
            elif t >= 1.0 - eps:
                source_costs = {best["end_node_id"]: 0.0}
                target_costs = {best["end_node_id"]: 0.0}
            else:
                source_costs = {best["end_node_id"]: snap_to_end_km}
                target_costs = {best["start_node_id"]: start_to_snap_km}
        else:
            source_costs = {
                best["start_node_id"]: start_to_snap_km,
                best["end_node_id"]: snap_to_end_km,
            }
            target_costs = dict(source_costs)

        projected = best["projected"]
        display_point = (
            {"lat": point.lat, "lon": point.lon}
            if best["distance_m"] / 1000 <= ON_STREET_SNAP_TOLERANCE_KM
            else projected
        )
        return RoadSnap(
            point=display_point,
            projected_point=projected,
            distance_km=best["distance_m"] / 1000,
            source_node_costs=source_costs,
            target_node_costs=target_costs,
        )

    def compute_route(
        self,
        payload: RouteRequest,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RouteResponse:
        """Calculate the three product routes directly from their own edge costs."""

        def ensure_active() -> None:
            if should_cancel is not None and should_cancel():
                raise routing.RouteSearchCancelled()

        ensure_active()
        self._ensure_fresh_data()
        if self.graph is None:
            raise ValueError("El grafo de rutas aun no esta listo. Intenta nuevamente en unos segundos.")

        origin_snap = self._nearest_road_snap(payload.origin, should_cancel)
        destination_snap = self._nearest_road_snap(payload.destination, should_cancel)
        if origin_snap.distance_km > MAX_POINT_SNAP_KM:
            raise ValueError(
                f"El origen esta a {origin_snap.distance_km:.2f} km de la red vial disponible. "
                "Mueve el punto a una calle con cobertura de datos."
            )
        if destination_snap.distance_km > MAX_POINT_SNAP_KM:
            raise ValueError(
                f"El destino esta a {destination_snap.distance_km:.2f} km de la red vial disponible. "
                "Mueve el punto a una calle con cobertura de datos."
            )

        day_value = _normalize_day(payload.day_of_week)
        hour_bucket = data_loader.hour_bucket(payload.departure_hour)
        analysis_context = {
            "day": day_value,
            "hour_bucket": hour_bucket,
            "include_congestion": True,
            "include_accidents": False,
            "match_filters": True,
        }
        active_congestion_lines = self._active_congestion_lines(payload)
        congestion_scores, congestion_speeds = self._active_congestion_node_metrics(
            active_congestion_lines
        )
        model = RoutingCostModel(
            congestion_scores=congestion_scores,
            congestion_speeds_kmh=congestion_speeds,
            segment_lengths_km=self.segment_lengths_km,
            pm25_factor=self._air_quality_cost_factor(
                payload.departure_hour,
                payload.congestion_date,
            ),
            wellbeing_factor=self._urban_wellbeing_cost_factor(),
            adverse_environment_factor=self._adverse_environment_cost_factor(payload),
        )

        def search(objective: str, max_contextual_time_min: float | None = None) -> List[routing.RouteStep]:
            cost_function = getattr(model, objective)
            source_costs = self._endpoint_objective_costs(
                model,
                origin_snap.source_node_costs,
                objective,
            )
            target_costs = self._endpoint_objective_costs(
                model,
                destination_snap.target_node_costs,
                objective,
            )
            if (
                objective in {"fluent", "environmental"}
                and max_contextual_time_min is not None
                and hasattr(self.graph, "shortest_path_constrained")
            ):
                source_time_costs = self._endpoint_objective_costs(
                    model,
                    origin_snap.source_node_costs,
                    "fastest",
                )
                target_time_costs = self._endpoint_objective_costs(
                    model,
                    destination_snap.target_node_costs,
                    "fastest",
                )
                return self.graph.shortest_path_constrained(
                    source_node_costs=source_costs,
                    target_node_costs=target_costs,
                    source_resource_costs=source_time_costs,
                    target_resource_costs=target_time_costs,
                    edge_filter=_vehicle_edge_allowed,
                    edge_cost=lambda source, target, distance: cost_function(
                        source,
                        target,
                        distance,
                    ).optimization_cost_min,
                    edge_resource_cost=lambda source, target, distance: model.fastest(
                        source,
                        target,
                        distance,
                    ).travel_time_min,
                    max_resource_cost=max_contextual_time_min,
                    should_cancel=should_cancel,
                )
            return self.graph.shortest_path(
                (payload.origin.lat, payload.origin.lon),
                (payload.destination.lat, payload.destination.lon),
                source_node_costs=source_costs,
                target_node_costs=target_costs,
                source_path_distances_km=origin_snap.source_node_costs,
                target_path_distances_km=destination_snap.target_node_costs,
                edge_filter=_vehicle_edge_allowed,
                edge_cost=lambda source, target, distance: cost_function(
                    source,
                    target,
                    distance,
                ).optimization_cost_min,
                apply_penalties=False,
                use_heuristic=False,
                should_cancel=should_cancel,
            )

        logger.info("Generando Llegar antes con costo temporal contextual...")
        fastest_path = search("fastest")
        if not fastest_path:
            raise ValueError(
                "No existe un camino vehicular continuo entre el origen y el destino con los datos disponibles."
            )
        fastest_totals = self._path_cost_totals(
            fastest_path,
            model,
            "fastest",
            origin_snap,
            destination_snap,
        )
        alternative_time_limit_min = fastest_totals.travel_time_min * ALTERNATIVE_MAX_TIME_RATIO

        ensure_active()
        logger.info("Generando Circulacion mas fluida con costo temporal y congestion exacta...")
        fluent_path = search("fluent", alternative_time_limit_min)
        if not fluent_path:
            fluent_path = list(fastest_path)

        ensure_active()
        logger.info("Generando Menor exposicion ambiental con costo ambiental directo...")
        environmental_path = search("environmental", alternative_time_limit_min)
        if not environmental_path:
            environmental_path = list(fastest_path)

        fluent_totals = self._path_cost_totals(
            fluent_path,
            model,
            "fluent",
            origin_snap,
            destination_snap,
        )
        environmental_totals = self._path_cost_totals(
            environmental_path,
            model,
            "environmental",
            origin_snap,
            destination_snap,
        )

        fastest_variant = self._build_response_variant(
            payload,
            fastest_path,
            analysis_context,
            via_factors={},
            variant_name="reference",
            origin_snap=origin_snap,
            destination_snap=destination_snap,
            active_congestion_lines=active_congestion_lines,
            cost_totals=fastest_totals,
            cost_objective="fastest",
        )
        fluent_variant = self._build_response_variant(
            payload,
            fluent_path,
            analysis_context,
            via_factors={},
            variant_name="least_congestion",
            origin_snap=origin_snap,
            destination_snap=destination_snap,
            active_congestion_lines=active_congestion_lines,
            cost_totals=fluent_totals,
            cost_objective="fluent",
        )
        environmental_variant = self._build_response_variant(
            payload,
            environmental_path,
            analysis_context,
            via_factors={},
            variant_name="healthiest",
            origin_snap=origin_snap,
            destination_snap=destination_snap,
            active_congestion_lines=active_congestion_lines,
            cost_totals=environmental_totals,
            cost_objective="environmental",
        )
        comparison = self._build_comparison(
            reference=fastest_variant,
            least_congestion=fluent_variant,
            ubcf=fluent_variant,
            ibcf=fluent_variant,
            healthiest=environmental_variant,
            personalized=fluent_variant,
        )
        self._log_variant_diagnostics(
            {
                "reference": fastest_variant,
                "least_congestion": fluent_variant,
                "healthiest": environmental_variant,
            }
        )
        return RouteResponse(
            reference=fastest_variant,
            least_congestion=fluent_variant,
            ubcf=fluent_variant.model_copy(deep=True),
            ibcf=fluent_variant.model_copy(deep=True),
            healthiest=environmental_variant,
            personalized=fluent_variant.model_copy(deep=True),
            comparison=comparison,
        )

    def _compute_route_legacy(
        self,
        payload: RouteRequest,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RouteResponse:
        def ensure_active() -> None:
            if should_cancel is not None and should_cancel():
                raise routing.RouteSearchCancelled()

        ensure_active()
        self._ensure_fresh_data()
        if self.graph is None:
            raise ValueError("El grafo de rutas aún no está listo. Intenta nuevamente en unos segundos.")
        origin_snap = self._nearest_road_snap(payload.origin, should_cancel)
        destination_snap = self._nearest_road_snap(payload.destination, should_cancel)
        if origin_snap.distance_km > MAX_POINT_SNAP_KM:
            raise ValueError(
                f"El origen esta a {origin_snap.distance_km:.2f} km de la red vial disponible. "
                "Mueve el punto a una calle con cobertura de datos."
            )
        if destination_snap.distance_km > MAX_POINT_SNAP_KM:
            raise ValueError(
                f"El destino esta a {destination_snap.distance_km:.2f} km de la red vial disponible. "
                "Mueve el punto a una calle con cobertura de datos."
            )
        route_endpoint_kwargs = {
            "source_node_costs": origin_snap.source_node_costs,
            "target_node_costs": destination_snap.target_node_costs,
            "edge_filter": _cerro_caracol_edge_allowed,
            "edge_cost_factor": _cerro_caracol_edge_cost_factor,
            "should_cancel": should_cancel,
        }
        day_value = _normalize_day(payload.day_of_week)
        hour_bucket = data_loader.hour_bucket(payload.departure_hour)
        delay_context = {
            "day": day_value,
            "hour_bucket": hour_bucket,
            "include_congestion": True,
            "include_accidents": False,
            "match_filters": True,
        }
        routing_context = None
        needs_context = payload.avoid_congestion
        active_congestion_lines = self._active_congestion_lines(payload)
        if needs_context:
            routing_context = {
                "day": day_value,
                "hour_bucket": hour_bucket,
                "avoid_congestion": payload.avoid_congestion,
                "avoid_accidents": False,
            }
            node_penalties = self._active_congestion_node_penalties(active_congestion_lines)
            if node_penalties:
                routing_context["node_penalties"] = node_penalties

        # Log detallado sobre penalizaciones
        penalty_status = []
        if payload.avoid_congestion:
            penalty_status.append("Congestion historica por severidad y contexto")
        if not penalty_status:
            penalty_status.append("NINGUNA - Rutas solo diferirán por preferencias CF")

        logger.info(
            "Calculando rutas:\n"
            "  Origen: (%.5f, %.5f)\n"
            "  Destino: (%.5f, %.5f)\n"
            "  Perfil de datos: %s\n"
            "  Contexto: %s, hora %.1f (%s)\n"
            "  Penalizaciones activas: %s",
            payload.origin.lat,
            payload.origin.lon,
            payload.destination.lat,
            payload.destination.lon,
            data_loader.get_data_profile(),
            day_value,
            payload.departure_hour,
            hour_bucket,
            ", ".join(penalty_status),
        )

        # -------------------------------
        # Función auxiliar para convertir preferencias en factores de vía
        # ratings altos -> factor < 1 (bonificación),
        # ratings bajos -> factor > 1 (castigo real).
        # -------------------------------
        def compute_via_factors(preferences: List) -> Dict[str, float]:
            """
            Convierte preferencias (ratings 0-1) a factores de costo para Dijkstra.

            Fórmula AMPLIFICADA para forzar diferencias mayores entre rutas:
            - Rating alto (>0.7) → factor muy bajo (0.1-0.5) → PREFERIR fuertemente
            - Rating medio (0.4-0.7) → factor neutro (0.5-1.5)
            - Rating bajo (<0.4) → factor muy alto (1.5-5.0) → EVITAR fuertemente
            """
            factors: Dict[str, float] = {}
            for pref in preferences:
                # pref.weight viene del cliente ya normalizado en [0,1]
                score = max(0.0, min(1.0, float(pref.weight)))

                # Fórmula exponencial para amplificar diferencias
                if score > 0.7:
                    # Rating alto: bonus agresivo
                    # 1.0 -> 0.1, 0.7 -> 0.5
                    factor = 0.1 + (1.0 - score) ** 2 * 1.33
                elif score >= 0.4:
                    # Rating medio: lineal suave
                    # 0.7 -> 0.5, 0.4 -> 1.5
                    factor = 0.5 + (0.7 - score) * 3.33
                else:
                    # Rating bajo: penalización agresiva
                    # 0.4 -> 1.5, 0.0 -> 5.0
                    factor = 1.5 + (0.4 - score) ** 0.5 * 5.53

                # Recorte de seguridad más amplio
                factors[pref.via] = round(max(0.1, min(5.0, factor)), 3)
            return factors

        # Separar las preferencias por estrategia
        ubcf_factors = compute_via_factors(payload.ubcf_preferences)
        ibcf_factors = compute_via_factors(payload.ibcf_preferences)

        # Mantener compatibilidad con el campo 'preferences' legacy
        legacy_factors = compute_via_factors(payload.preferences)
        has_legacy = bool(legacy_factors)

        # Log de diagnóstico: comparar factores UBCF vs IBCF
        logger.info(
            "Factores de preferencia calculados:\n"
            "  UBCF: %d vías con factores\n"
            "  IBCF: %d vías con factores",
            len(ubcf_factors),
            len(ibcf_factors),
        )

        # Verificar si hay vías en común con factores diferentes
        common_vias = set(ubcf_factors.keys()) & set(ibcf_factors.keys())
        if common_vias:
            different_count = sum(
                1 for via in common_vias
                if abs(ubcf_factors[via] - ibcf_factors[via]) > 0.1
            )
            logger.info(
                "  Vías en común: %d, con factores diferentes (Δ>0.1): %d (%.1f%%)",
                len(common_vias),
                different_count,
                100 * different_count / len(common_vias) if common_vias else 0,
            )

            # Mostrar algunos ejemplos de factores diferentes
            examples = []
            for via in list(common_vias)[:5]:
                uf = ubcf_factors[via]
                if_ = ibcf_factors[via]
                if abs(uf - if_) > 0.1:
                    examples.append(f"{via}: UBCF={uf:.2f}, IBCF={if_:.2f}")

            if examples:
                logger.info("  Ejemplos de diferencias:\n    " + "\n    ".join(examples))
        else:
            logger.warning("  ⚠️ Sin vías en común entre UBCF e IBCF")

        default_factor = 1.0

        # -------------------------------
        # GENERAR 3 RUTAS DISTINTAS (CON OPTIMIZACIÓN)
        # -------------------------------

        # 1. RUTA REFERENCE: A* geográfico (sin penalizaciones, sin preferencias)
        logger.info("Generando ruta reference (A* geográfico)...")
        reference_path = self.graph.shortest_path(
            (payload.origin.lat, payload.origin.lon),
            (payload.destination.lat, payload.destination.lon),
            **route_endpoint_kwargs,
            apply_penalties=False,
        )
        if not reference_path:
            raise ValueError(
                "No existe un camino continuo entre el origen y el destino con los datos viales disponibles."
            )
        reference_road_distance_km = self._path_distance_km(reference_path)
        endpoint_snap_distance_km = origin_snap.distance_km + destination_snap.distance_km
        reasonable_route_limit_km = (
            min(
                reference_road_distance_km * ALTERNATIVE_MAX_DISTANCE_RATIO,
                reference_road_distance_km
                + ROUTE_ASSUMED_SPEED_KMH * ALTERNATIVE_MAX_EXTRA_MIN / 60.0,
            )
            + endpoint_snap_distance_km
        )
        environmental_route_limit_km = (
            min(
                reference_road_distance_km * HEALTHY_MAX_DISTANCE_RATIO,
                reference_road_distance_km
                + ROUTE_ASSUMED_SPEED_KMH * HEALTHY_MAX_EXTRA_MIN / 60.0,
            )
            + endpoint_snap_distance_km
        )

        # Optimización: si no hay preferencias de CF, generar solo 1 ruta con penalizaciones
        logger.info("Generando ruta least_congestion (solo penalizacion por congestion historica)...")
        if needs_context:
            least_congestion_path = self.graph.shortest_path(
                (payload.origin.lat, payload.origin.lon),
                (payload.destination.lat, payload.destination.lon),
                **route_endpoint_kwargs,
                incident_ctx=routing_context,
                apply_penalties=True,
                apply_historical_penalties=False,
                geographic_path_limit_km=reasonable_route_limit_km,
            )
            if not least_congestion_path:
                logger.warning("No se pudo construir ruta least_congestion; usando ruta reference como fallback.")
                least_congestion_path = list(reference_path)
        else:
            least_congestion_path = list(reference_path)

        has_ubcf = bool(ubcf_factors)
        has_ibcf = bool(ibcf_factors)
        personalized_path: List[routing.RouteStep]

        if not has_ubcf and not has_ibcf and has_legacy:
            logger.info("Solo hay preferencias legacy; generando ruta personalizada compatible...")
            personalized_path = self.graph.shortest_path(
                (payload.origin.lat, payload.origin.lon),
                (payload.destination.lat, payload.destination.lon),
                **route_endpoint_kwargs,
                via_factors=legacy_factors,
                default_via_factor=default_factor,
                incident_ctx=routing_context,
                apply_penalties=True,
            )
            if not personalized_path:
                logger.warning("No se pudo construir ruta legacy; usando ruta reference como fallback.")
                personalized_path = list(reference_path)
            ubcf_path = list(personalized_path)
            ibcf_path = list(personalized_path)
        elif not has_ubcf and not has_ibcf:
            # Sin preferencias CF: generar solo 1 ruta con penalizaciones y reutilizarla
            logger.info("Sin preferencias CF; reutilizando ruta least_congestion para UBCF e IBCF...")
            ubcf_path = list(least_congestion_path)
            ibcf_path = list(least_congestion_path)
            personalized_path = list(ubcf_path)
        else:
            # 2. Ruta compatible UBCF: solo se usa si llegan preferencias academicas.
            if has_ubcf:
                logger.info("Generando ruta UBCF legacy con congestion historica...")
                ubcf_path = self.graph.shortest_path(
                    (payload.origin.lat, payload.origin.lon),
                    (payload.destination.lat, payload.destination.lon),
                    **route_endpoint_kwargs,
                    via_factors=ubcf_factors,
                    default_via_factor=default_factor,
                    incident_ctx=routing_context,
                    apply_penalties=True,
                )
                if not ubcf_path:
                    logger.warning("No se pudo construir ruta UBCF; usando ruta reference como fallback.")
                    ubcf_path = list(reference_path)
            else:
                # Sin preferencias UBCF: usar ruta con solo penalizaciones
                logger.info("Sin preferencias UBCF; usando ruta con solo penalizaciones...")
                if needs_context:
                    ubcf_path = self.graph.shortest_path(
                        (payload.origin.lat, payload.origin.lon),
                        (payload.destination.lat, payload.destination.lon),
                        **route_endpoint_kwargs,
                        incident_ctx=routing_context,
                        apply_penalties=True,
                    )
                    if not ubcf_path:
                        ubcf_path = list(reference_path)
                else:
                    ubcf_path = list(reference_path)

            # 3. Ruta compatible IBCF: solo se usa si llegan preferencias academicas.
            if has_ibcf:
                logger.info("Generando ruta IBCF legacy con congestion historica...")
                ibcf_path = self.graph.shortest_path(
                    (payload.origin.lat, payload.origin.lon),
                    (payload.destination.lat, payload.destination.lon),
                    **route_endpoint_kwargs,
                    via_factors=ibcf_factors,
                    default_via_factor=default_factor,
                    incident_ctx=routing_context,
                    apply_penalties=True,
                )
                if not ibcf_path:
                    logger.warning("No se pudo construir ruta IBCF; usando ruta reference como fallback.")
                    ibcf_path = list(reference_path)
            else:
                # Sin preferencias IBCF: reutilizar la ruta UBCF si es posible
                if not has_ubcf and needs_context:
                    logger.info("Sin preferencias IBCF; reutilizando ruta con penalizaciones...")
                    ibcf_path = list(ubcf_path)
                elif not has_ubcf:
                    ibcf_path = list(reference_path)
                else:
                    # UBCF tiene preferencias pero IBCF no: generar ruta solo con penalizaciones
                    logger.info("Sin preferencias IBCF; reutilizando ruta least_congestion.")
                    ibcf_path = list(least_congestion_path)

            if has_legacy:
                logger.info("Generando ruta legacy de compatibilidad...")
                personalized_path = self.graph.shortest_path(
                    (payload.origin.lat, payload.origin.lon),
                    (payload.destination.lat, payload.destination.lon),
                    **route_endpoint_kwargs,
                    via_factors=legacy_factors,
                    default_via_factor=default_factor,
                    incident_ctx=routing_context,
                    apply_penalties=True,
                )
                if not personalized_path:
                    logger.warning("No se pudo construir ruta legacy; usando ruta UBCF como fallback.")
                    personalized_path = list(ubcf_path)
            else:
                personalized_path = list(ubcf_path)

        air_quality_factor = self._air_quality_cost_factor(payload.departure_hour)
        wellbeing_factor = self._urban_wellbeing_cost_factor()
        wellbeing_service = get_urban_wellbeing_service()
        logger.info("Generando ruta saludable combinada por PM2.5 y bienestar urbano...")
        healthy_penalty_kwargs = {
            "incident_ctx": routing_context,
            "apply_penalties": bool(needs_context),
        }
        waypoint_candidates = wellbeing_service.candidate_waypoints(
            payload.origin,
            payload.destination,
            limit=HEALTHY_ENVIRONMENT_WAYPOINT_LIMIT,
        )
        healthy_weighted_path = self.graph.shortest_path(
            (payload.origin.lat, payload.origin.lon),
            (payload.destination.lat, payload.destination.lon),
            **route_endpoint_kwargs,
            **healthy_penalty_kwargs,
            apply_historical_penalties=False,
            air_quality_factor=air_quality_factor,
            urban_wellbeing_factor=wellbeing_factor,
            geographic_path_limit_km=environmental_route_limit_km,
        )
        if not healthy_weighted_path:
            healthy_weighted_path = list(reference_path)

        healthy_waypoint_paths: List[List[routing.RouteStep]] = []
        for waypoint in waypoint_candidates:
            waypoint_path = self._path_via_waypoint(
                payload,
                waypoint,
                origin_snap=origin_snap,
                destination_snap=destination_snap,
                incident_ctx=routing_context,
                apply_penalties=bool(needs_context),
                apply_historical_penalties=False,
                air_quality_factor=air_quality_factor,
                urban_wellbeing_factor=wellbeing_factor,
                geographic_path_limit_km=environmental_route_limit_km,
                should_cancel=should_cancel,
            )
            if waypoint_path and not self._same_path(waypoint_path, healthy_weighted_path):
                healthy_waypoint_paths.append(waypoint_path)

        # Construir variantes de respuesta
        ensure_active()
        reference_variant = self._build_response_variant(
            payload,
            reference_path,
            delay_context,
            via_factors={},
            variant_name="reference",
            origin_snap=origin_snap,
            destination_snap=destination_snap,
            active_congestion_lines=active_congestion_lines,
        )
        least_congestion_variant = self._build_response_variant(
            payload,
            least_congestion_path,
            delay_context,
            via_factors={},
            variant_name="least_congestion",
            origin_snap=origin_snap,
            destination_snap=destination_snap,
            active_congestion_lines=active_congestion_lines,
        )
        ubcf_variant = (
            deepcopy(least_congestion_variant)
            if self._same_path(ubcf_path, least_congestion_path) and not ubcf_factors
            else self._build_response_variant(
                payload,
                ubcf_path,
                delay_context,
                via_factors=ubcf_factors,
                variant_name="ubcf",
                origin_snap=origin_snap,
                destination_snap=destination_snap,
                active_congestion_lines=active_congestion_lines,
            )
        )
        ibcf_variant = (
            deepcopy(least_congestion_variant)
            if self._same_path(ibcf_path, least_congestion_path) and not ibcf_factors
            else self._build_response_variant(
                payload,
                ibcf_path,
                delay_context,
                via_factors=ibcf_factors,
                variant_name="ibcf",
                origin_snap=origin_snap,
                destination_snap=destination_snap,
                active_congestion_lines=active_congestion_lines,
            )
        )
        personalized_variant = (
            deepcopy(ubcf_variant)
            if self._same_path(personalized_path, ubcf_path) and not legacy_factors
            else self._build_response_variant(
                payload,
                personalized_path,
                delay_context,
                via_factors=legacy_factors,
                variant_name="personalized",
                origin_snap=origin_snap,
                destination_snap=destination_snap,
                active_congestion_lines=active_congestion_lines,
            )
        )
        healthy_weighted_variant = self._build_response_variant(
            payload,
            healthy_weighted_path,
            delay_context,
            via_factors={},
            variant_name="healthy_weighted_candidate",
            origin_snap=origin_snap,
            destination_snap=destination_snap,
            active_congestion_lines=active_congestion_lines,
        )
        healthy_waypoint_variants = [
            self._build_response_variant(
                payload,
                path,
                delay_context,
                via_factors={},
                variant_name=f"healthy_waypoint_candidate_{index}",
                origin_snap=origin_snap,
                destination_snap=destination_snap,
                active_congestion_lines=active_congestion_lines,
            )
            for index, path in enumerate(healthy_waypoint_paths, start=1)
        ]
        ensure_active()
        healthiest_variant = self._finalize_weighted_environmental_variant(
            reference=reference_variant,
            least_congestion=least_congestion_variant,
            weighted=healthy_weighted_variant,
            waypoint_candidates=healthy_waypoint_variants,
        )
        comparison = self._build_comparison(
            reference=reference_variant,
            least_congestion=least_congestion_variant,
            ubcf=ubcf_variant,
            ibcf=ibcf_variant,
            healthiest=healthiest_variant,
            personalized=personalized_variant,
        )
        self._log_variant_diagnostics(
            {
                "reference": reference_variant,
                "least_congestion": least_congestion_variant,
                "ubcf": ubcf_variant,
                "ibcf": ibcf_variant,
                "healthiest": healthiest_variant,
                "personalized": personalized_variant,
            }
        )

        logger.info(
            "Rutas generadas exitosamente:\n"
            "  - A* (reference): %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total\n"
            "  - Least congestion: %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total\n"
            "  - UBCF: %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total\n"
            "  - IBCF: %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total\n"
            "  - Healthiest: %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total\n"
            "  - Legacy/personalized: %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total",
            reference_variant.distance_km,
            reference_variant.estimated_duration_min,
            reference_variant.extra_delay_min,
            reference_variant.estimated_duration_min + reference_variant.extra_delay_min,
            least_congestion_variant.distance_km,
            least_congestion_variant.estimated_duration_min,
            least_congestion_variant.extra_delay_min,
            least_congestion_variant.estimated_duration_min + least_congestion_variant.extra_delay_min,
            ubcf_variant.distance_km,
            ubcf_variant.estimated_duration_min,
            ubcf_variant.extra_delay_min,
            ubcf_variant.estimated_duration_min + ubcf_variant.extra_delay_min,
            ibcf_variant.distance_km,
            ibcf_variant.estimated_duration_min,
            ibcf_variant.extra_delay_min,
            ibcf_variant.estimated_duration_min + ibcf_variant.extra_delay_min,
            healthiest_variant.distance_km,
            healthiest_variant.estimated_duration_min,
            healthiest_variant.extra_delay_min,
            healthiest_variant.estimated_duration_min + healthiest_variant.extra_delay_min,
            personalized_variant.distance_km,
            personalized_variant.estimated_duration_min,
            personalized_variant.extra_delay_min,
            personalized_variant.estimated_duration_min + personalized_variant.extra_delay_min,
        )

        return RouteResponse(
            reference=reference_variant,
            least_congestion=least_congestion_variant,
            ubcf=ubcf_variant,
            ibcf=ibcf_variant,
            healthiest=healthiest_variant,
            personalized=personalized_variant,
            comparison=comparison,
        )

    @staticmethod
    def _variant_geometry_key(variant: RouteVariant) -> tuple[tuple[float, float], ...]:
        return tuple((round(point.lat, 6), round(point.lon, 6)) for point in variant.geometry)

    @staticmethod
    def _same_path(first: List[routing.RouteStep], second: List[routing.RouteStep]) -> bool:
        return bool(first) and bool(second) and [step.node_id for step in first] == [step.node_id for step in second]

    @staticmethod
    def _path_distance_km(path: List[routing.RouteStep]) -> float:
        return sum(
            routing.haversine_km(previous.lat, previous.lon, current.lat, current.lon)
            for previous, current in zip(path, path[1:])
        )

    def _endpoint_objective_costs(
        self,
        model: RoutingCostModel,
        distances_km: Dict[str, float],
        objective: str,
    ) -> Dict[str, float]:
        if self.graph is None:
            return {}
        cost_function = getattr(model, objective)
        result: Dict[str, float] = {}
        for node_id, distance_km in distances_km.items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            result[node_id] = cost_function(node, node, float(distance_km)).optimization_cost_min
        return result

    def _path_cost_totals(
        self,
        path: List[routing.RouteStep],
        model: RoutingCostModel,
        objective: str,
        origin_snap: RoadSnap,
        destination_snap: RoadSnap,
    ) -> RouteCostTotals:
        if self.graph is None or not path:
            return RouteCostTotals()
        cost_function = getattr(model, objective)
        breakdowns = []
        first_node = self.graph.nodes.get(path[0].node_id)
        last_node = self.graph.nodes.get(path[-1].node_id)
        if first_node is not None:
            source_distance = float(origin_snap.source_node_costs.get(first_node.node_id, 0.0))
            if source_distance > 0:
                breakdowns.append(cost_function(first_node, first_node, source_distance))
        for previous, current in zip(path, path[1:]):
            source = self.graph.nodes.get(previous.node_id)
            target = self.graph.nodes.get(current.node_id)
            if source is None or target is None:
                continue
            distance = routing.haversine_km(source.lat, source.lon, target.lat, target.lon)
            breakdowns.append(cost_function(source, target, distance))
        if last_node is not None:
            target_distance = float(destination_snap.target_node_costs.get(last_node.node_id, 0.0))
            if target_distance > 0:
                breakdowns.append(cost_function(last_node, last_node, target_distance))
        return sum_breakdowns(breakdowns)

    @classmethod
    def _is_reasonable_alternative(
        cls,
        reference: List[routing.RouteStep],
        candidate: List[routing.RouteStep],
    ) -> bool:
        if not candidate or cls._same_path(reference, candidate):
            return False
        reference_distance = cls._path_distance_km(reference)
        candidate_distance = cls._path_distance_km(candidate)
        max_distance = min(
            reference_distance * ALTERNATIVE_MAX_DISTANCE_RATIO,
            reference_distance
            + ROUTE_ASSUMED_SPEED_KMH * ALTERNATIVE_MAX_EXTRA_MIN / 60.0,
        )
        return candidate_distance <= max_distance + 1e-9

    @staticmethod
    def _diversity_edge_cost_factor(
        paths: List[List[routing.RouteStep]],
        *,
        base_factor: Callable[[routing.GraphNode, routing.GraphNode], float] | None = None,
    ) -> Callable[[routing.GraphNode, routing.GraphNode], float]:
        used_edges = {
            (previous.node_id, current.node_id)
            for path in paths
            for previous, current in zip(path, path[1:])
        }

        def factor(previous: routing.GraphNode, current: routing.GraphNode) -> float:
            value = 1.0
            if base_factor is not None:
                value = max(1.0, float(base_factor(previous, current)))
            if (previous.node_id, current.node_id) in used_edges:
                value *= ALTERNATIVE_OVERLAP_PENALTY
            return value

        return factor

    def _log_variant_diagnostics(self, variants: Dict[str, RouteVariant]) -> None:
        rows = []
        for key, variant in variants.items():
            pm25_score = variant.pm25_exposure.average_pm25 if variant.pm25_exposure is not None else None
            rows.append(
                "  - %s: %.2f km, %.1f min total, congestion_score=%.1f, matched=%d, pm25=%s, geometry_points=%d"
                % (
                    key,
                    variant.distance_km,
                    self._variant_total_minutes(variant),
                    variant.risk_score,
                    variant.incident_exposure.matched_incident_segments,
                    f"{pm25_score:.1f}" if pm25_score is not None else "n/a",
                    len(variant.geometry),
                )
            )
        reference = variants.get("reference")
        least_congestion = variants.get("least_congestion")
        if reference is not None and least_congestion is not None:
            same_geometry = self._variant_geometry_key(reference) == self._variant_geometry_key(least_congestion)
            rows.append(f"  - reference_vs_least_congestion_same_geometry={same_geometry}")
        logger.info("Diagnostico de variantes de ruta:\n%s", "\n".join(rows))

    @staticmethod
    def _air_quality_cost_factor(departure_hour: float, snapshot_date: str | None = None):
        service = get_air_quality_service()
        cache: Dict[tuple[float, float], float] = {}
        snapshot = None
        if snapshot_date:
            try:
                snapshot = service.station_snapshot(snapshot_date, int(departure_hour))
            except Exception as exc:
                logger.warning("No se pudo cargar PM2.5 exacto para el costo de ruta: %s", exc)

        def factor(node: routing.GraphNode) -> float:
            coordinate = (round(node.lat, 6), round(node.lon, 6))
            cached = cache.get(coordinate)
            if cached is not None:
                return cached
            if snapshot_date:
                value = (
                    service.route_cost_factor_from_snapshot(snapshot, node.lat, node.lon)
                    if snapshot is not None and snapshot.available
                    else 1.0
                )
            else:
                value = service.route_cost_factor(node.lat, node.lon, departure_hour)
            cache[coordinate] = value
            return value

        return factor

    @staticmethod
    def _urban_wellbeing_cost_factor():
        service = get_urban_wellbeing_service()
        cache: Dict[tuple[float, float], float] = {}

        def factor(node: routing.GraphNode) -> float:
            coordinate = (round(node.lat, 6), round(node.lon, 6))
            cached = cache.get(coordinate)
            if cached is not None:
                return cached
            value = service.route_cost_factor(node.lat, node.lon)
            cache[coordinate] = value
            return value

        return factor

    @staticmethod
    def _adverse_environment_cost_factor(payload: RouteRequest):
        if not payload.congestion_date:
            return None
        try:
            snapshot = get_environmental_impact_service().build_snapshot(
                payload.congestion_date,
                int(payload.departure_hour),
            )
        except Exception as exc:
            logger.warning("No se pudo cargar la capa ambiental exacta para rutas: %s", exc)
            return None
        features = list((snapshot.zones or {}).get("features") or [])
        geometries = []
        penalties = []
        for feature in features:
            properties = feature.get("properties") or {}
            try:
                geometry = shape(feature.get("geometry") or {})
                score = float(properties.get("score_avg") or 0.0)
                recency_weight = float(properties.get("recency_weight") or 0.0)
            except (TypeError, ValueError):
                continue
            if geometry.is_empty:
                continue
            if recency_weight <= 0:
                if int(properties.get("current_focus_count") or 0) > 0:
                    recency_weight = 1.0
                elif int(properties.get("memory_max_lag_hours") or 0) <= 1:
                    recency_weight = 0.25
                else:
                    recency_weight = 0.10
            level = str(properties.get("level") or "low").strip().lower()
            score = max(0.0, min(100.0, score))
            if level == "high":
                low_penalty, high_penalty = ENVIRONMENT_HIGH_ZONE_PENALTY
                progress = max(0.0, min(1.0, (score - 65.0) / 35.0))
            elif level == "medium":
                low_penalty, high_penalty = ENVIRONMENT_MEDIUM_ZONE_PENALTY
                progress = max(0.0, min(1.0, (score - 35.0) / 30.0))
            else:
                low_penalty, high_penalty = ENVIRONMENT_LOW_ZONE_PENALTY
                progress = max(0.0, min(1.0, score / 35.0))
            zone_penalty = (low_penalty + (high_penalty - low_penalty) * progress) * max(
                0.0,
                min(1.0, recency_weight),
            )
            geometries.append(geometry)
            penalties.append(zone_penalty)
        if not geometries:
            return None
        tree = STRtree(geometries)
        cache: Dict[tuple[float, float], float] = {}

        def factor(node: routing.GraphNode) -> float:
            coordinate = (round(node.lat, 6), round(node.lon, 6))
            cached = cache.get(coordinate)
            if cached is not None:
                return cached
            point = Point(float(node.lon), float(node.lat))
            matched = [
                penalties[int(index)]
                for index in tree.query(point)
                if geometries[int(index)].covers(point)
            ]
            value = 1.0 + max(matched, default=0.0)
            cache[coordinate] = value
            return value

        return factor

    def _path_via_waypoint(
        self,
        payload: RouteRequest,
        waypoint: dict,
        *,
        origin_snap: RoadSnap | None = None,
        destination_snap: RoadSnap | None = None,
        incident_ctx: Dict[str, object] | None = None,
        apply_penalties: bool = False,
        apply_historical_penalties: bool = True,
        air_quality_factor: Callable[[routing.GraphNode], float] | None = None,
        urban_wellbeing_factor: Callable[[routing.GraphNode], float] | None = None,
        geographic_path_limit_km: float | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> List[routing.RouteStep]:
        waypoint_lat = float(waypoint["lat"])
        waypoint_lon = float(waypoint["lon"])
        waypoint_point = RoutePoint(lat=waypoint_lat, lon=waypoint_lon)
        waypoint_snap = self._nearest_road_snap(waypoint_point, should_cancel)
        anchor_node_ids = [
            node_id
            for node_id in set(waypoint_snap.source_node_costs) | set(waypoint_snap.target_node_costs)
            if node_id in self.graph.nodes
        ]
        if not anchor_node_ids:
            nearest_nodes = self.graph.nearest_nodes(waypoint_lat, waypoint_lon, limit=1)
            if not nearest_nodes:
                return []
            anchor_node_ids = [nearest_nodes[0][0].node_id]
        anchor_node_id = min(
            anchor_node_ids,
            key=lambda node_id: routing.haversine_km(
                waypoint_lat,
                waypoint_lon,
                self.graph.nodes[node_id].lat,
                self.graph.nodes[node_id].lon,
            ),
        )
        anchor_node_costs = {anchor_node_id: 0.0}
        first_limit = geographic_path_limit_km
        second_limit = geographic_path_limit_km
        if geographic_path_limit_km is not None:
            first_direct = routing.haversine_km(
                payload.origin.lat,
                payload.origin.lon,
                waypoint_lat,
                waypoint_lon,
            )
            second_direct = routing.haversine_km(
                waypoint_lat,
                waypoint_lon,
                payload.destination.lat,
                payload.destination.lon,
            )
            total_direct = first_direct + second_direct
            search_extra = min(
                HEALTHY_WAYPOINT_SEARCH_EXTRA_KM,
                max(0.0, geographic_path_limit_km - total_direct),
            )
            if total_direct > 1e-9:
                first_limit = first_direct + search_extra * first_direct / total_direct
                second_limit = second_direct + search_extra * second_direct / total_direct
        first = self.graph.shortest_path(
            (payload.origin.lat, payload.origin.lon),
            (waypoint_lat, waypoint_lon),
            source_node_costs=origin_snap.source_node_costs if origin_snap else None,
            target_node_costs=anchor_node_costs,
            incident_ctx=incident_ctx,
            apply_penalties=apply_penalties,
            apply_historical_penalties=apply_historical_penalties,
            air_quality_factor=air_quality_factor,
            urban_wellbeing_factor=urban_wellbeing_factor,
            geographic_path_limit_km=first_limit,
            edge_filter=_cerro_caracol_edge_allowed,
            edge_cost_factor=_cerro_caracol_edge_cost_factor,
            should_cancel=should_cancel,
        )
        second = self.graph.shortest_path(
            (waypoint_lat, waypoint_lon),
            (payload.destination.lat, payload.destination.lon),
            source_node_costs=anchor_node_costs,
            target_node_costs=destination_snap.target_node_costs if destination_snap else None,
            incident_ctx=incident_ctx,
            apply_penalties=apply_penalties,
            apply_historical_penalties=apply_historical_penalties,
            air_quality_factor=air_quality_factor,
            urban_wellbeing_factor=urban_wellbeing_factor,
            geographic_path_limit_km=second_limit,
            edge_filter=_cerro_caracol_edge_allowed,
            edge_cost_factor=_cerro_caracol_edge_cost_factor,
            should_cancel=should_cancel,
        )
        if not first or not second:
            return []
        if first[-1].node_id != second[0].node_id:
            logger.warning(
                "Se descarto una ruta ambiental discontinua en el waypoint %s: %s -> %s.",
                waypoint.get("name") or "sin nombre",
                first[-1].node_id,
                second[0].node_id,
            )
            return []
        return [*first, *second[1:]]

    @classmethod
    def _finalize_weighted_environmental_variant(
        cls,
        *,
        reference: RouteVariant,
        least_congestion: RouteVariant,
        weighted: RouteVariant,
        waypoint_candidates: List[RouteVariant],
    ) -> RouteVariant:
        """Compare the direct weighted path with safe environmental detours.

        Dijkstra has already chosen ``weighted`` using pollution, congestion and
        environmental edge costs. Waypoint paths may improve effective contact,
        but they must remain within the time limit and cannot introduce a large
        regression toward the origin.
        """

        fastest_time = min(
            cls._variant_total_minutes(reference),
            cls._variant_total_minutes(least_congestion),
        )
        max_time = fastest_time + HEALTHY_MAX_EXTRA_MIN
        max_distance = reference.distance_km * HEALTHY_MAX_DISTANCE_RATIO

        def within_detour(candidate: RouteVariant) -> bool:
            return (
                cls._variant_total_minutes(candidate) <= max_time + 1e-9
                and candidate.distance_km <= max_distance + 1e-9
            )

        def has_contact(candidate: RouteVariant) -> bool:
            return bool(
                candidate.urban_wellbeing is not None
                and candidate.urban_wellbeing.available
                and candidate.urban_wellbeing.top_features
            )

        def avoids_congestion(candidate: RouteVariant) -> bool:
            return (
                candidate.incident_exposure.matched_incident_segments == 0
                and float(candidate.congestion_coverage.high_pct) <= 1e-9
            )

        weighted_backtracking = cls._route_backtracking_ratio(weighted)
        max_allowed_backtracking = max(
            HEALTHY_MAX_BACKTRACK_RATIO,
            weighted_backtracking + HEALTHY_MAX_BACKTRACK_RATIO / 2.0,
        )
        rejected_backtracking = any(
            cls._route_backtracking_ratio(candidate) > max_allowed_backtracking + 1e-9
            for candidate in waypoint_candidates
        )
        safe_waypoint_candidates = [
            candidate
            for candidate in waypoint_candidates
            if cls._route_backtracking_ratio(candidate) <= max_allowed_backtracking + 1e-9
        ]
        ordered = [*safe_waypoint_candidates, weighted, reference, least_congestion]
        feasible = [candidate for candidate in ordered if within_detour(candidate)]
        chosen = next(
            (
                candidate
                for candidate in feasible
                if has_contact(candidate) and avoids_congestion(candidate)
            ),
            None,
        )
        missing_combined_alternative = chosen is None
        if chosen is None:
            chosen = next(
                (candidate for candidate in feasible if avoids_congestion(candidate)),
                reference,
            )

        result = chosen.model_copy(deep=True) if hasattr(chosen, "model_copy") else deepcopy(chosen)
        result.healthy_route_score = None
        same_as_reference = cls._variant_geometry_key(result) == cls._variant_geometry_key(reference)
        reasons = [
            (
                "La ruta ponderada coincide con la ruta directa para este viaje."
                if same_as_reference
                else "La geometria fue calculada directamente con los pesos ambientales por tramo."
            )
        ]
        if missing_combined_alternative:
            reasons[0] = (
                "No se encontro una alternativa que combine entorno urbano beneficioso "
                "y ausencia de congestion dentro del desvio permitido."
            )
        if has_contact(result):
            feature = result.urban_wellbeing.top_features[0]
            reasons.append(f"El trayecto pasa junto a {feature.name}.")
        else:
            reasons.append("No se encontro un contacto ambiental valido dentro del desvio maximo permitido.")
        if avoids_congestion(result):
            reasons.append("La alternativa evita los segmentos de congestion identificados.")
        else:
            reasons.append("No fue posible evitar toda la congestion identificada dentro del desvio permitido.")
        if rejected_backtracking:
            reasons.append("Se descarto un desvio ambiental porque obligaba a retroceder en el trayecto.")
        if cls._variant_total_minutes(result) > fastest_time + 0.1:
            reasons.append(
                f"La ruta ponderada agrega {cls._variant_total_minutes(result) - fastest_time:.1f} min."
            )
        result.why_changed = reasons[:4]
        return result

    @staticmethod
    def _route_backtracking_ratio(variant: RouteVariant) -> float:
        """Measure cumulative backwards movement along the origin-destination axis."""

        if len(variant.geometry) < 3:
            return 0.0
        origin = variant.geometry[0]
        destination = variant.geometry[-1]
        reference_lat = (float(origin.lat) + float(destination.lat)) / 2.0
        origin_x, origin_y = RoutingService._project_point(
            {"lat": float(origin.lat), "lon": float(origin.lon)},
            reference_lat,
        )
        destination_x, destination_y = RoutingService._project_point(
            {"lat": float(destination.lat), "lon": float(destination.lon)},
            reference_lat,
        )
        axis_x = destination_x - origin_x
        axis_y = destination_y - origin_y
        direct_distance = math.hypot(axis_x, axis_y)
        if direct_distance <= 1e-9:
            return 0.0

        progress_values = []
        for point in variant.geometry:
            point_x, point_y = RoutingService._project_point(
                {"lat": float(point.lat), "lon": float(point.lon)},
                reference_lat,
            )
            progress_values.append(
                ((point_x - origin_x) * axis_x + (point_y - origin_y) * axis_y) / direct_distance
            )
        backwards_distance = sum(
            max(0.0, previous - current)
            for previous, current in zip(progress_values, progress_values[1:])
        )
        return backwards_distance / direct_distance

    @classmethod
    def _select_healthiest_variant(
        cls,
        *,
        reference: RouteVariant,
        candidates: List[RouteVariant],
    ) -> RouteVariant:
        fastest_time = min(cls._variant_total_minutes(candidate) for candidate in candidates)
        max_time = fastest_time + HEALTHY_MAX_EXTRA_MIN
        max_distance = reference.distance_km * HEALTHY_MAX_DISTANCE_RATIO

        unique: Dict[tuple[tuple[float, float], ...], RouteVariant] = {}
        for candidate in candidates:
            key = cls._variant_geometry_key(candidate)
            current = unique.get(key)
            if current is None or cls._variant_total_minutes(candidate) < cls._variant_total_minutes(current):
                unique[key] = candidate
        feasible = [
            candidate
            for candidate in unique.values()
            if cls._variant_total_minutes(candidate) <= max_time + 1e-9
            and candidate.distance_km <= max_distance + 1e-9
        ] or [reference]

        environmental_contacts = [
            candidate
            for candidate in feasible
            if candidate.urban_wellbeing is not None
            and candidate.urban_wellbeing.available
            and bool(candidate.urban_wellbeing.top_features)
        ]
        contact_without_congestion = [
            candidate
            for candidate in environmental_contacts
            if candidate.incident_exposure.matched_incident_segments == 0
            and float(candidate.congestion_coverage.high_pct) <= 1e-9
        ]
        baseline_geometry_keys = {cls._variant_geometry_key(reference)}
        if len(candidates) >= 3:
            baseline_geometry_keys.add(cls._variant_geometry_key(candidates[1]))
        distinct_contact_without_congestion = [
            candidate
            for candidate in contact_without_congestion
            if cls._variant_geometry_key(candidate) not in baseline_geometry_keys
        ]
        if distinct_contact_without_congestion:
            feasible = distinct_contact_without_congestion
        elif contact_without_congestion:
            feasible = contact_without_congestion
        elif environmental_contacts:
            feasible = environmental_contacts

        pm25_values = [
            float(candidate.pm25_exposure.average_pm25)
            for candidate in feasible
            if candidate.pm25_exposure is not None and candidate.pm25_exposure.available
        ]
        congestion_values = [float(candidate.risk_score) for candidate in feasible]
        wellbeing_values = [
            float(candidate.urban_wellbeing.score)
            if candidate.urban_wellbeing is not None and candidate.urban_wellbeing.available
            else 0.0
            for candidate in feasible
        ]
        time_values = [cls._variant_total_minutes(candidate) for candidate in feasible]
        distance_values = [float(candidate.distance_km) for candidate in feasible]

        def normalized(value: float, values: List[float]) -> float:
            if not values:
                return 0.0
            low = min(values)
            high = max(values)
            if high - low <= 1e-9:
                return 0.0
            return (value - low) / (high - low)

        def has_variation(values: List[float]) -> bool:
            return bool(values) and max(values) - min(values) > 1e-9

        best_risk = min(float(candidate.risk_score) for candidate in feasible)
        best_matched_segments = min(candidate.incident_exposure.matched_incident_segments for candidate in feasible)
        best_high_pct = min(float(candidate.congestion_coverage.high_pct) for candidate in feasible)
        congestion_feasible = [
            candidate
            for candidate in feasible
            if float(candidate.risk_score) <= best_risk + HEALTHY_CONGESTION_RISK_TOLERANCE
            and candidate.incident_exposure.matched_incident_segments
            <= best_matched_segments + HEALTHY_CONGESTION_SEGMENT_TOLERANCE
            and float(candidate.congestion_coverage.high_pct)
            <= best_high_pct + HEALTHY_HIGH_CONGESTION_PCT_TOLERANCE
        ]
        scoring_candidates = congestion_feasible or feasible
        reference_pm25 = (
            float(reference.pm25_exposure.average_pm25)
            if reference.pm25_exposure is not None and reference.pm25_exposure.available
            else None
        )
        if reference_pm25 is not None and has_variation(pm25_values):
            lower_pm25_candidates = [
                candidate
                for candidate in scoring_candidates
                if candidate.pm25_exposure is not None
                and candidate.pm25_exposure.available
                and float(candidate.pm25_exposure.average_pm25)
                <= reference_pm25 - HEALTHY_PM25_MEANINGFUL_DELTA
            ]
            if lower_pm25_candidates:
                scoring_candidates = lower_pm25_candidates

        scored: list[tuple[float, RouteVariant]] = []
        for candidate in scoring_candidates:
            pm25 = (
                float(candidate.pm25_exposure.average_pm25)
                if candidate.pm25_exposure is not None and candidate.pm25_exposure.available
                else max(pm25_values, default=0.0)
            )
            wellbeing = (
                float(candidate.urban_wellbeing.score)
                if candidate.urban_wellbeing is not None and candidate.urban_wellbeing.available
                else 0.0
            )
            pm25_cost = normalized(pm25, pm25_values) if has_variation(pm25_values) else 0.0
            congestion_cost = normalized(float(candidate.risk_score), congestion_values) if has_variation(congestion_values) else 0.0
            wellbeing_cost = 1.0 - normalized(wellbeing, wellbeing_values) if has_variation(wellbeing_values) else 0.0
            travel_cost = (
                normalized(cls._variant_total_minutes(candidate), time_values)
                + normalized(float(candidate.distance_km), distance_values)
            ) / 2.0
            cost = (
                HEALTHY_CONGESTION_WEIGHT * congestion_cost
                + HEALTHY_WELLBEING_WEIGHT * wellbeing_cost
                + HEALTHY_PM25_WEIGHT * pm25_cost
                + HEALTHY_TRAVEL_WEIGHT * travel_cost
            )
            scored.append((cost, candidate))

        cost, chosen = min(
            scored,
            key=lambda item: (
                item[0],
                cls._variant_total_minutes(item[1]),
                item[1].distance_km,
            ),
        )
        result = chosen.model_copy(deep=True) if hasattr(chosen, "model_copy") else deepcopy(chosen)
        result.healthy_route_score = round(max(0.0, min(100.0, (1.0 - cost) * 100.0)), 1)
        if cls._variant_geometry_key(result) == cls._variant_geometry_key(reference):
            reasons = [
                "La ruta mas directa ya ofrece la menor exposicion ambiental disponible para este viaje.",
            ]
        else:
            reasons = [
                "Esta variante prioriza una menor exposicion ambiental dentro de un desvio razonable.",
            ]
        if (
            reference_pm25 is not None
            and result.pm25_exposure is not None
            and result.pm25_exposure.available
            and float(result.pm25_exposure.average_pm25) <= reference_pm25 - HEALTHY_PM25_MEANINGFUL_DELTA
        ):
            reasons.append(
                f"Reduce PM2.5 estimado de {reference_pm25:.1f} a {result.pm25_exposure.average_pm25:.1f} ug/m3."
            )
        if congestion_feasible and len(congestion_feasible) < len(feasible):
            reasons.append("Se descartaron alternativas ambientales con congestion claramente mayor.")
        if result.urban_wellbeing is not None and result.urban_wellbeing.top_features:
            feature = result.urban_wellbeing.top_features[0]
            reasons.append(f"El trayecto pasa junto a {feature.name}, considerado por su aporte al entorno del viaje.")
        else:
            reasons.append("No se encontro un contacto ambiental valido dentro del desvio maximo permitido.")
        if (
            result.incident_exposure.matched_incident_segments == 0
            and float(result.congestion_coverage.high_pct) <= 1e-9
        ):
            reasons.append("La alternativa evita los segmentos de congestion identificados.")
        if cls._variant_total_minutes(result) > fastest_time + 0.1:
            reasons.append(
                f"La menor exposicion agrega {cls._variant_total_minutes(result) - fastest_time:.1f} min frente a la ruta mas directa."
            )
        result.why_changed = [*reasons, *result.why_changed[1:]][:4]
        return result

    @staticmethod
    def _variant_total_minutes(variant: RouteVariant) -> float:
        return float(variant.estimated_duration_min + variant.extra_delay_min)

    @staticmethod
    def _match_step_context(step: routing.RouteStep, context: Dict[str, str | bool] | None) -> bool:
        if not context:
            return False
        day_value = str(context.get("day") or "").lower()
        hour_value = context.get("hour_bucket")
        matches_day = not day_value or bool(step.dia_semana and step.dia_semana.lower() == day_value)
        matches_hour = not hour_value or bool(step.franja_horaria and step.franja_horaria == hour_value)
        return matches_day and matches_hour

    def _active_congestion_lines(self, payload: RouteRequest) -> List[dict]:
        if not payload.congestion_date:
            return []
        try:
            snapshot = get_environmental_impact_service().build_snapshot(
                payload.congestion_date,
                int(payload.departure_hour),
            )
        except Exception as exc:
            logger.warning("No se pudo cargar congestion activa para rutas: %s", exc)
            return []
        return list((snapshot.congestion_lines or {}).get("features") or [])

    @staticmethod
    def _congestion_line_points(feature: dict) -> List[Dict[str, float]]:
        coordinates = feature.get("geometry", {}).get("coordinates") or []
        points: List[Dict[str, float]] = []
        for item in coordinates:
            if isinstance(item, list) and len(item) >= 2:
                try:
                    points.append({"lon": float(item[0]), "lat": float(item[1])})
                except (TypeError, ValueError):
                    continue
        return points

    @classmethod
    def _point_to_segment_distance_m(
        cls,
        point: Dict[str, float],
        start: Dict[str, float],
        end: Dict[str, float],
    ) -> float:
        reference_lat = point["lat"]
        px, py = cls._project_point(point, reference_lat)
        ax, ay = cls._project_point(start, reference_lat)
        bx, by = cls._project_point(end, reference_lat)
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    @classmethod
    def _point_to_polyline_distance_m(cls, point: Dict[str, float], line: List[Dict[str, float]]) -> float:
        if not line:
            return float("inf")
        if len(line) == 1:
            return routing.haversine_km(point["lat"], point["lon"], line[0]["lat"], line[0]["lon"]) * 1000
        return min(cls._point_to_segment_distance_m(point, start, end) for start, end in zip(line, line[1:]))

    @classmethod
    def _polyline_distance_m(cls, route: List[Dict[str, float]], line: List[Dict[str, float]]) -> float:
        if not route or not line:
            return float("inf")
        route_to_line = min(cls._point_to_polyline_distance_m(point, line) for point in route)
        line_to_route = min(cls._point_to_polyline_distance_m(point, route) for point in line)
        return min(route_to_line, line_to_route)

    @staticmethod
    def _polyline_length_m(points: List[Dict[str, float]]) -> float:
        if len(points) < 2:
            return 0.0
        return sum(
            routing.haversine_km(start["lat"], start["lon"], end["lat"], end["lon"]) * 1000
            for start, end in zip(points, points[1:])
        )

    @classmethod
    def _route_length_near_polyline_m(
        cls,
        route: List[Dict[str, float]],
        line: List[Dict[str, float]],
        tolerance_m: float,
    ) -> float:
        if len(route) < 2 or len(line) < 2:
            return 0.0
        near_length = 0.0
        for start, end in zip(route, route[1:]):
            midpoint = {
                "lat": (start["lat"] + end["lat"]) / 2,
                "lon": (start["lon"] + end["lon"]) / 2,
            }
            if (
                cls._point_to_polyline_distance_m(start, line) <= tolerance_m
                or cls._point_to_polyline_distance_m(midpoint, line) <= tolerance_m
                or cls._point_to_polyline_distance_m(end, line) <= tolerance_m
            ):
                near_length += routing.haversine_km(start["lat"], start["lon"], end["lat"], end["lon"]) * 1000
        return near_length

    @staticmethod
    def _normalize_road_name(value: str | None) -> str:
        text = unicodedata.normalize("NFKD", str(value or ""))
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"\b(av|avda|avenida|calle|pasaje)\b\.?", " ", text.lower())
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    @classmethod
    def _road_names_compatible(cls, first: str | None, second: str | None) -> bool:
        """Match conservative OSM/source aliases without relying on one-off names."""

        first_name = cls._normalize_road_name(first)
        second_name = cls._normalize_road_name(second)
        invalid_names = {"", "sin nombre", "unknown", "desconocida"}
        if first_name in invalid_names or second_name in invalid_names:
            return False
        if first_name == second_name:
            return True
        first_tokens = set(first_name.split())
        second_tokens = set(second_name.split())
        shorter, longer = (
            (first_tokens, second_tokens)
            if len(first_tokens) <= len(second_tokens)
            else (second_tokens, first_tokens)
        )
        distinctive_tokens = {token for token in shorter if len(token) >= 4 and not token.isdigit()}
        return bool(distinctive_tokens and shorter.issubset(longer))

    @classmethod
    def _route_matches_congestion_feature(
        cls,
        properties: dict,
        route_vias: set[str],
        route_segment_ids: set[str],
    ) -> bool:
        segment_id = str(properties.get("segment_id") or "")
        if segment_id and segment_id in route_segment_ids:
            return True
        via = cls._normalize_road_name(str(properties.get("via") or ""))
        return any(cls._road_names_compatible(via, route_via) for route_via in route_vias)

    @classmethod
    def _node_matches_congestion_feature(cls, node: routing.GraphNode, properties: dict) -> bool:
        segment_id = str(properties.get("segment_id") or "")
        if segment_id and segment_id == node.segment_id:
            return True
        feature_via = cls._normalize_road_name(str(properties.get("via") or ""))
        node_via = cls._normalize_road_name(node.via)
        return cls._road_names_compatible(feature_via, node_via)

    def _active_congestion_node_metrics(
        self,
        active_congestion_lines: List[dict],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        if self.graph is None or not active_congestion_lines:
            return {}, {}
        congestion_coords: list[tuple[float, float]] = []
        congestion_feature_indices: list[int] = []
        congestion_features: list[tuple[dict, List[Dict[str, float]], float, float | None]] = []
        for feature in active_congestion_lines:
            line = self._congestion_line_points(feature)
            if len(line) < 2:
                continue
            properties = feature.get("properties", {})
            score = float(properties.get("score") or 0.0)
            raw_speed = properties.get("speed_kmh")
            try:
                speed_kmh = float(raw_speed) if raw_speed is not None else None
            except (TypeError, ValueError):
                speed_kmh = None
            if speed_kmh is not None and (not math.isfinite(speed_kmh) or speed_kmh < 5.0):
                speed_kmh = None
            feature_index = len(congestion_features)
            congestion_features.append((properties, line, score, speed_kmh))
            for start, end in zip(line, line[1:]):
                segment_length_m = routing.haversine_km(
                    start["lat"], start["lon"], end["lat"], end["lon"]
                ) * 1000
                sample_count = max(1, math.ceil(segment_length_m / ACTIVE_CONGESTION_NODE_TOLERANCE_M))
                for sample_index in range(sample_count):
                    ratio = sample_index / sample_count
                    congestion_coords.append(
                        (
                            start["lat"] + (end["lat"] - start["lat"]) * ratio,
                            start["lon"] + (end["lon"] - start["lon"]) * ratio,
                        )
                    )
                    congestion_feature_indices.append(feature_index)
            congestion_coords.append((line[-1]["lat"], line[-1]["lon"]))
            congestion_feature_indices.append(feature_index)
        if not congestion_coords:
            return {}, {}

        node_ids: list[str] = []
        node_coords: list[tuple[float, float]] = []
        for node_id, node in self.graph.nodes.items():
            if math.isfinite(node.lat) and math.isfinite(node.lon):
                node_ids.append(node_id)
                node_coords.append((node.lat, node.lon))
        if not node_coords:
            return {}, {}

        tree = BallTree(np.radians(np.asarray(congestion_coords, dtype=float)), metric="haversine")
        radius = (ACTIVE_CONGESTION_NODE_TOLERANCE_M * 1.35) / (routing.EARTH_RADIUS_KM * 1000)
        neighbor_indices = tree.query_radius(np.radians(np.asarray(node_coords, dtype=float)), r=radius)
        scores: Dict[str, float] = {}
        speeds: Dict[str, float] = {}
        for node_id, neighbors in zip(node_ids, neighbor_indices):
            node = self.graph.nodes[node_id]
            matching: list[tuple[float, float | None]] = []
            for feature_index in {congestion_feature_indices[int(index)] for index in neighbors}:
                properties, line, score, speed_kmh = congestion_features[feature_index]
                if not self._node_matches_congestion_feature(node, properties):
                    continue
                point = {"lat": node.lat, "lon": node.lon}
                if self._point_to_polyline_distance_m(point, line) <= ACTIVE_CONGESTION_NODE_TOLERANCE_M:
                    matching.append((score, speed_kmh))
            if matching:
                scores[node_id] = max(0.0, min(1.0, max(item[0] for item in matching) / 100.0))
                valid_speeds = [item[1] for item in matching if item[1] is not None]
                if valid_speeds:
                    speeds[node_id] = min(valid_speeds)
        return scores, speeds

    def _active_congestion_node_penalties(self, active_congestion_lines: List[dict]) -> Dict[str, float]:
        scores, _speeds = self._active_congestion_node_metrics(active_congestion_lines)
        return {node_id: 1.0 + score * 80.0 for node_id, score in scores.items()}

    def _active_congestion_segment_impacts(
        self,
        route_geometry: List[Dict[str, float]],
        route_vias: set[str],
        route_segment_ids: set[str],
        active_congestion_lines: List[dict],
        route_path: List[routing.RouteStep] | None = None,
    ) -> tuple[List[SegmentImpact], RouteCongestionCoverage]:
        impacts: List[SegmentImpact] = []
        seen: set[str] = set()
        route_m = self._polyline_length_m(route_geometry)
        high_m = 0.0
        medium_m = 0.0
        low_m = 0.0
        via_lengths: dict[str, float] = {}
        for feature in active_congestion_lines:
            properties = feature.get("properties", {})
            segment_id = str(properties.get("segment_id") or "")
            if not segment_id or segment_id in seen:
                continue
            if not self._route_matches_congestion_feature(properties, route_vias, route_segment_ids):
                continue
            line = self._congestion_line_points(feature)
            matched_route_geometries = [route_geometry]
            if route_path:
                feature_via = self._normalize_road_name(str(properties.get("via") or ""))
                local_pairs: List[List[Dict[str, float]]] = []
                for previous, current in zip(route_path, route_path[1:]):
                    matches_segment = segment_id in {previous.segment_id, current.segment_id}
                    matches_via = bool(
                        feature_via
                        and feature_via not in {"sin nombre", "unknown", "desconocida"}
                        and self._road_names_compatible(feature_via, previous.via)
                        and self._road_names_compatible(feature_via, current.via)
                    )
                    if matches_segment or matches_via:
                        local_pairs.append(
                            [
                                {"lat": previous.lat, "lon": previous.lon},
                                {"lat": current.lat, "lon": current.lon},
                            ]
                        )
                if not local_pairs:
                    continue
                matched_route_geometries = local_pairs
            distance_m = min(
                self._polyline_distance_m(local_geometry, line)
                for local_geometry in matched_route_geometries
            )
            if distance_m > ACTIVE_CONGESTION_MATCH_TOLERANCE_M:
                continue
            seen.add(segment_id)
            score = float(properties.get("score") or 0.0)
            affected_length_m = sum(
                self._route_length_near_polyline_m(
                    local_geometry,
                    line,
                    ACTIVE_CONGESTION_MATCH_TOLERANCE_M,
                )
                for local_geometry in matched_route_geometries
            )
            impact_factor = min(1.0, affected_length_m / ACTIVE_CONGESTION_FULL_IMPACT_M)
            impact_factor = max(ACTIVE_CONGESTION_MIN_IMPACT_FACTOR, impact_factor)
            impact_score = round((score / 12.5) * impact_factor, 2)
            level = str(properties.get("level") or "").strip().lower()
            if level == "high":
                high_m += affected_length_m
            elif level == "medium":
                medium_m += affected_length_m
            else:
                low_m += affected_length_m
            via = str(properties.get("via") or "Tramo congestionado")
            via_lengths[via] = via_lengths.get(via, 0.0) + affected_length_m
            reason = (
                f"Circula por {affected_length_m:.0f} m de congestion "
                f"{properties.get('level', 'detectada')} observada en "
                f"{properties.get('recency', 'la hora seleccionada')}."
            )
            impacts.append(
                SegmentImpact(
                    segment_id=segment_id,
                    via=via,
                    comuna=str(properties.get("comuna") or ""),
                    event_type="Congestion",
                    impact_score=impact_score,
                    reason=reason,
                )
            )
        congested_m = high_m + medium_m + low_m
        primary_via = max(via_lengths, key=via_lengths.get) if via_lengths else None

        def pct(value: float) -> float:
            return round((value / route_m) * 100, 1) if route_m > 0 else 0.0

        return impacts, RouteCongestionCoverage(
            route_m=round(route_m, 1),
            congested_m=round(congested_m, 1),
            high_m=round(high_m, 1),
            medium_m=round(medium_m, 1),
            low_m=round(low_m, 1),
            congested_pct=pct(congested_m),
            high_pct=pct(high_m),
            medium_pct=pct(medium_m),
            low_pct=pct(low_m),
            primary_via=primary_via,
        )

    def _build_variant_analysis(
        self,
        path: List[routing.RouteStep],
        context: Dict[str, str | bool] | None,
        via_factors: Dict[str, float],
        variant_name: str,
        extra_minutes: float,
        route_geometry: List[Dict[str, float]] | None = None,
        active_congestion_lines: List[dict] | None = None,
    ) -> tuple[IncidentExposure, float, List[str], List[SegmentImpact], List[PreferredViaImpact], RouteCongestionCoverage]:
        incident_steps_by_segment = {
            step.segment_id: step
            for step in path
            if self._is_congestion_event(step.tipo_evento)
        }
        matched_steps_by_segment = {
            step.segment_id: step
            for step in incident_steps_by_segment.values()
            if self._match_step_context(step, context)
        }
        incident_steps = list(incident_steps_by_segment.values())
        matched_steps = list(matched_steps_by_segment.values())
        congestion_steps = list(incident_steps)
        accident_steps: List[routing.RouteStep] = []

        scored_segments: List[SegmentImpact] = []
        for step in matched_steps:
            type_weight = 1.3
            base_minutes = max(float(step.duracion_hrs or 0.0) * 60, 5.0)
            impact_score = round(type_weight * base_minutes / 10.0, 2)
            scored_segments.append(
                SegmentImpact(
                    segment_id=step.segment_id,
                    via=step.via,
                    comuna=step.comuna,
                    event_type=step.tipo_evento,
                    impact_score=impact_score,
                    reason=self._incident_reason(step),
                )
            )
        route_vias = {self._normalize_road_name(step.via) for step in path if step.via}
        route_segment_ids = {step.segment_id for step in path if step.segment_id}
        active_impacts, congestion_coverage = self._active_congestion_segment_impacts(
            route_geometry or [],
            route_vias,
            route_segment_ids,
            active_congestion_lines or [],
            route_path=path,
        )
        existing_segment_ids = {segment.segment_id for segment in scored_segments}
        for impact in active_impacts:
            if impact.segment_id not in existing_segment_ids:
                scored_segments.append(impact)
                existing_segment_ids.add(impact.segment_id)

        risk_score = 0.0
        if active_congestion_lines:
            risk_score = round(
                min(
                    100.0,
                    congestion_coverage.high_pct * 2.0
                    + congestion_coverage.medium_pct * 2.0
                    + congestion_coverage.low_pct * 0.5,
                ),
                1,
            )
        elif scored_segments:
            max_impact = max(segment.impact_score for segment in scored_segments)
            avg_impact = sum(segment.impact_score for segment in scored_segments) / len(scored_segments)
            risk_score = round(min(100.0, max(max_impact * 12.5, avg_impact * 10.0)), 1)

        exposure = IncidentExposure(
            total_incident_segments=len({*incident_steps_by_segment.keys(), *(impact.segment_id for impact in active_impacts)}),
            matched_incident_segments=len({*(step.segment_id for step in matched_steps), *(impact.segment_id for impact in active_impacts)}),
            congestion_segments=len({*(step.segment_id for step in congestion_steps), *(impact.segment_id for impact in active_impacts)}),
            accident_segments=len(accident_steps),
            exposure_minutes=round(extra_minutes, 1),
        )
        top_penalized_segments = sorted(
            scored_segments,
            key=lambda item: item.impact_score,
            reverse=True,
        )[:3]
        top_preferred_vias = self._preferred_via_impacts(path, via_factors)
        why_changed = self._default_why_changed(
            variant_name=variant_name,
            exposure=exposure,
            top_penalized_segments=top_penalized_segments,
            top_preferred_vias=top_preferred_vias,
        )
        return exposure, risk_score, why_changed, top_penalized_segments, top_preferred_vias, congestion_coverage

    @staticmethod
    def _incident_reason(step: routing.RouteStep) -> str:
        if RoutingService._is_congestion_event(step.tipo_evento):
            return f"Congestión histórica en {step.franja_horaria or 'franja no definida'}."
        return "Segmento con historial de congestion."

    @staticmethod
    def _is_congestion_event(value: str | None) -> bool:
        return str(value or "").strip().lower().startswith("congesti")

    @staticmethod
    def _preferred_via_impacts(
        path: List[routing.RouteStep],
        via_factors: Dict[str, float],
    ) -> List[PreferredViaImpact]:
        if not via_factors:
            return []
        route_vias = {step.via for step in path if step.via}
        preferred = [
            PreferredViaImpact(
                via=via,
                factor=factor,
                reason="La estrategia colaborativa favorece esta vía en la simulación."
                if factor < 1.0
                else "La estrategia colaborativa penaliza esta vía en la simulación.",
            )
            for via, factor in sorted(via_factors.items(), key=lambda item: item[1])
            if via in route_vias
        ]
        if preferred:
            return preferred[:3]
        return [
            PreferredViaImpact(
                via=via,
                factor=factor,
                reason="Preferencia global del perfil colaborativo."
                if factor < 1.0
                else "Penalización global del perfil colaborativo.",
            )
            for via, factor in sorted(via_factors.items(), key=lambda item: item[1])[:3]
        ]

    @staticmethod
    def _default_why_changed(
        variant_name: str,
        exposure: IncidentExposure,
        top_penalized_segments: List[SegmentImpact],
        top_preferred_vias: List[PreferredViaImpact],
    ) -> List[str]:
        reasons: List[str] = []
        if variant_name == "reference":
            reasons.append("Esta ruta usa Dijkstra y minimiza el tiempo estimado para la fecha y hora seleccionadas.")
        elif variant_name.startswith("healthy_") or variant_name == "healthiest":
            reasons.append("Esta variante prioriza menor exposicion ambiental y mejores condiciones del entorno.")
        else:
            reasons.append("Esta variante evita sectores con mayor congestion para mejorar la fluidez del viaje.")
        if exposure.matched_incident_segments:
            reasons.append(
                f"Se detectaron {exposure.matched_incident_segments} zonas con congestion historica en el contexto seleccionado."
            )
        else:
            reasons.append("No se detecto congestion historica relevante para el dia y horario elegidos.")
        if top_penalized_segments:
            reasons.append(f"El principal punto conflictivo es {top_penalized_segments[0].via}.")
        if top_preferred_vias:
            reasons.append(f"La vía más favorecida por el perfil es {top_preferred_vias[0].via}.")
        return reasons[:4]

    def _build_comparison(
        self,
        *,
        reference: RouteVariant,
        least_congestion: RouteVariant | None,
        ubcf: RouteVariant,
        ibcf: RouteVariant,
        healthiest: RouteVariant,
        personalized: RouteVariant,
    ) -> RouteComparison:
        variants = {"reference": reference}
        if least_congestion is not None:
            variants["least_congestion"] = least_congestion
        variants.update(
            {
                "ubcf": ubcf,
                "ibcf": ibcf,
                "healthiest": healthiest,
                "personalized": personalized,
            }
        )
        fastest_variant = min(variants, key=lambda key: self._variant_total_minutes(variants[key]))
        safest_variant = min(variants, key=lambda key: variants[key].risk_score)
        exposure_candidates = variants

        def pm25_exposure_value(variant: RouteVariant) -> float:
            if variant.pm25_exposure is None or not variant.pm25_exposure.available:
                return float("inf")
            return float(variant.pm25_exposure.average_pm25)

        def wellbeing_value(variant: RouteVariant) -> float:
            if variant.urban_wellbeing is None or not variant.urban_wellbeing.available:
                return 0.0
            return float(variant.urban_wellbeing.score)

        lowest_exposure_variant = min(
            exposure_candidates,
            key=lambda key: (
                pm25_exposure_value(exposure_candidates[key]),
                exposure_candidates[key].incident_exposure.matched_incident_segments,
                exposure_candidates[key].risk_score,
                -wellbeing_value(exposure_candidates[key]),
                self._variant_total_minutes(exposure_candidates[key]),
                exposure_candidates[key].distance_km,
            ),
        )

        def balance_score(variant: RouteVariant) -> float:
            return (
                self._variant_total_minutes(variant) * 0.45
                + variant.risk_score * 0.35
                + variant.incident_exposure.matched_incident_segments * 4.0
                + variant.distance_km * 0.2
            )

        best_balance_variant = min(variants, key=lambda key: balance_score(variants[key]))
        deltas = []
        for key, variant in variants.items():
            deltas.append(
                RouteDelta(
                    variant=key,
                    distance_delta_km=round(variant.distance_km - reference.distance_km, 2),
                    total_duration_delta_min=round(
                        self._variant_total_minutes(variant) - self._variant_total_minutes(reference),
                        1,
                    ),
                    risk_delta=round(variant.risk_score - reference.risk_score, 1),
                    exposure_delta=round(
                        variant.incident_exposure.matched_incident_segments
                        - reference.incident_exposure.matched_incident_segments,
                        1,
                    ),
                )
            )
        return RouteComparison(
            fastest_variant=fastest_variant,
            safest_variant=safest_variant,
            lowest_exposure_variant=lowest_exposure_variant,
            best_balance_variant=best_balance_variant,
            deltas=deltas,
        )

    def _build_response_variant(
        self,
        payload: RouteRequest,
        path: List[routing.RouteStep],
        context: Dict[str, str | bool] | None,
        *,
        via_factors: Dict[str, float],
        variant_name: str,
        origin_snap: RoadSnap | None = None,
        destination_snap: RoadSnap | None = None,
        active_congestion_lines: List[dict] | None = None,
        cost_totals: RouteCostTotals | None = None,
        cost_objective: str | None = None,
    ) -> RouteVariant:
        first_graph = path[0]
        last_graph = path[-1]
        origin_snap = origin_snap or self._nearest_road_snap(payload.origin)
        destination_snap = destination_snap or self._nearest_road_snap(payload.destination)
        origin_road_point = origin_snap.point
        destination_road_point = destination_snap.point
        origin_step = routing.RouteStep(
            node_id="user_origin",
            segment_id=first_graph.segment_id,
            segment_seq=first_graph.segment_seq,
            lat=payload.origin.lat,
            lon=payload.origin.lon,
            via=first_graph.via,
            comuna=first_graph.comuna,
            peso=0.0,
            tipo_evento="Usuario",
            duracion_hrs=0.0,
            dia_semana=_normalize_day(payload.day_of_week),
            franja_horaria=data_loader.hour_bucket(payload.departure_hour),
        )
        dest_step = routing.RouteStep(
            node_id="user_destination",
            segment_id=last_graph.segment_id,
            segment_seq=last_graph.segment_seq,
            lat=payload.destination.lat,
            lon=payload.destination.lon,
            via=last_graph.via,
            comuna=last_graph.comuna,
            peso=path[-1].peso,
            tipo_evento="Usuario",
            duracion_hrs=0.0,
            dia_semana=_normalize_day(payload.day_of_week),
            franja_horaria=data_loader.hour_bucket(payload.departure_hour),
        )
        full_path = [origin_step] + path + [dest_step]

        road_points = self._build_geometry(path)
        if road_points:
            if road_points[0] != origin_road_point:
                road_points = [origin_road_point] + road_points
            if road_points[-1] != destination_road_point:
                road_points = road_points + [destination_road_point]
        else:
            road_points = [origin_road_point, destination_road_point]
        road_points = self._postprocess_geometry(road_points)

        geometry_points = [
            {"lat": payload.origin.lat, "lon": payload.origin.lon},
            *road_points,
            {"lat": payload.destination.lat, "lon": payload.destination.lon},
        ]
        cleaned_geometry_points: List[Dict[str, float]] = []
        for point in geometry_points:
            if not cleaned_geometry_points or cleaned_geometry_points[-1] != point:
                cleaned_geometry_points.append(point)

        distance = 0.0
        for prev, step in zip(cleaned_geometry_points, cleaned_geometry_points[1:]):
            distance += routing.haversine_km(prev["lat"], prev["lon"], step["lat"], step["lon"])

        steps: List[RouteStepResponse] = []
        cumulative_cost = 0.0
        for idx, step in enumerate(full_path):
            if idx > 0:
                prev = full_path[idx - 1]
                cumulative_cost += routing.haversine_km(prev.lat, prev.lon, step.lat, step.lon)
            steps.append(
                RouteStepResponse(
                    node_id=step.node_id,
                    lat=step.lat,
                    lon=step.lon,
                    via=step.via,
                    comuna=step.comuna,
                    cumulative_cost=round(cumulative_cost, 3),
                )
            )
        estimated_minutes = (
            float(cost_totals.base_time_min)
            if cost_totals is not None
            else (distance / 35.0) * 60.0
        )
        extra_minutes = float(cost_totals.congestion_delay_min) if cost_totals is not None else 0.0
        if context and cost_totals is None:
            include_congestion = bool(context.get("include_congestion", True))
            match_filters = bool(context.get("match_filters", True))
            day_value = str(context.get("day") or "").lower() if match_filters else ""
            hour_value = context.get("hour_bucket") if match_filters else None
            buckets: Dict[Tuple[str, str, str], List[float]] = {}
            for step in path:
                if not self._is_congestion_event(step.tipo_evento):
                    continue
                matches_day = True
                if day_value:
                    matches_day = bool(step.dia_semana and step.dia_semana.lower() == day_value)
                matches_hour = True
                if hour_value:
                    matches_hour = bool(step.franja_horaria and step.franja_horaria == hour_value)
                if not (matches_day and matches_hour):
                    continue
                if self._is_congestion_event(step.tipo_evento) and not include_congestion:
                    continue
                key = (step.segment_id, step.tipo_evento, step.franja_horaria or "")
                minutes = max(step.duracion_hrs, 0.1) * 60
                buckets.setdefault(key, []).append(minutes)
            for key, values in buckets.items():
                promedio = sum(values) / len(values)
                extra_minutes += promedio
                if self._is_congestion_event(key[1]):
                    extra_minutes += 5
        exposure, risk_score, why_changed, top_penalized_segments, top_preferred_vias, congestion_coverage = (
            self._build_variant_analysis(
                path=path,
                context=context,
                via_factors=via_factors,
                variant_name=variant_name,
                extra_minutes=extra_minutes,
                route_geometry=road_points,
                active_congestion_lines=active_congestion_lines or [],
            )
        )
        geometry = [RoutePoint(**point) for point in self._postprocess_geometry(cleaned_geometry_points)]
        road_geometry = [RoutePoint(**point) for point in road_points]
        access_geometry = [
            [RoutePoint(lat=payload.origin.lat, lon=payload.origin.lon), RoutePoint(**origin_road_point)],
            [RoutePoint(**destination_road_point), RoutePoint(lat=payload.destination.lat, lon=payload.destination.lon)],
        ]
        try:
            pm25_exposure = get_air_quality_service().estimate_route_exposure(
                geometry=geometry,
                departure_hour=payload.departure_hour,
                snapshot_date=payload.congestion_date,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback for optional layer
            logger.warning("No se pudo estimar exposicion PM2.5 para la ruta: %s", exc)
            pm25_exposure = None
        try:
            urban_wellbeing = get_urban_wellbeing_service().evaluate_route(geometry)
        except Exception as exc:  # pragma: no cover - defensive fallback for optional layer
            logger.warning("No se pudo estimar bienestar urbano para la ruta: %s", exc)
            urban_wellbeing = None

        pm25_data_available = bool(
            pm25_exposure.get("available", False)
            if isinstance(pm25_exposure, dict)
            else getattr(pm25_exposure, "available", False)
        )
        urban_data_available = bool(
            urban_wellbeing.get("available", False)
            if isinstance(urban_wellbeing, dict)
            else getattr(urban_wellbeing, "available", False)
        )
        if variant_name == "healthiest":
            if not pm25_data_available:
                why_changed.append(
                    "No hay datos PM2.5 para la fecha y hora seleccionadas; "
                    "ese componente ambiental se mantuvo neutro."
                )
            if not urban_data_available:
                why_changed.append(
                    "No hay datos de entorno urbano disponibles; "
                    "ese componente ambiental se mantuvo neutro."
                )
            why_changed = why_changed[:4]

        optimization_trace = None
        if cost_totals is not None and cost_objective in {"fastest", "fluent", "environmental"}:
            optimization_trace = RouteOptimizationTrace(
                objective=cost_objective,
                logical_segment_count=len({step.segment_id for step in path}),
                base_time_min=round(cost_totals.base_time_min, 3),
                congestion_delay_min=round(cost_totals.congestion_delay_min, 3),
                congestion_penalty_min=round(cost_totals.congestion_penalty_min, 3),
                stop_penalty_min=round(cost_totals.stop_penalty_min, 3),
                pm25_penalty_min=round(cost_totals.pm25_penalty_min, 3),
                adverse_environment_penalty_min=round(
                    cost_totals.adverse_environment_penalty_min,
                    3,
                ),
                urban_benefit_min=round(cost_totals.urban_benefit_min, 3),
                optimization_cost_min=round(cost_totals.optimization_cost_min, 3),
                pm25_data_available=pm25_data_available,
                urban_data_available=urban_data_available,
            )
        return RouteVariant(
            distance_km=round(distance, 2),
            estimated_duration_min=round(estimated_minutes, 1),
            steps=steps,
            geometry=geometry,
            road_geometry=road_geometry,
            access_geometry=access_geometry,
            extra_delay_min=round(extra_minutes, 1),
            risk_score=risk_score,
            incident_exposure=exposure,
            pm25_exposure=pm25_exposure,
            urban_wellbeing=urban_wellbeing,
            why_changed=why_changed,
            top_penalized_segments=top_penalized_segments,
            top_preferred_vias=top_preferred_vias,
            congestion_coverage=congestion_coverage,
            optimization_trace=optimization_trace,
        )

    def _build_geometry(self, path: List[routing.RouteStep]) -> List[Dict[str, float]]:
        geometry: List[Dict[str, float]] = []
        for idx, step in enumerate(path):
            point = {"lat": step.lat, "lon": step.lon}
            if not geometry or geometry[-1] != point:
                geometry.append(point)
            if idx < len(path) - 1:
                nxt = path[idx + 1]
                if step.segment_id == nxt.segment_id:
                    intermediates = self._segment_between(step.segment_id, step.segment_seq, nxt.segment_seq)
                    for lat, lon in intermediates:
                        candidate = {"lat": lat, "lon": lon}
                        if geometry[-1] != candidate:
                            geometry.append(candidate)
        return self._postprocess_geometry(geometry)

    @staticmethod
    def _project_point(point: Dict[str, float], reference_lat: float) -> Tuple[float, float]:
        lat = math.radians(point["lat"])
        lon = math.radians(point["lon"])
        return (
            routing.EARTH_RADIUS_KM * 1000 * lon * math.cos(math.radians(reference_lat)),
            routing.EARTH_RADIUS_KM * 1000 * lat,
        )

    @classmethod
    def _perpendicular_distance_m(
        cls,
        point: Dict[str, float],
        start: Dict[str, float],
        end: Dict[str, float],
        reference_lat: float,
    ) -> float:
        px, py = cls._project_point(point, reference_lat)
        ax, ay = cls._project_point(start, reference_lat)
        bx, by = cls._project_point(end, reference_lat)
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        nearest_x = ax + t * dx
        nearest_y = ay + t * dy
        return math.hypot(px - nearest_x, py - nearest_y)

    @classmethod
    def _simplify_geometry(
        cls,
        points: List[Dict[str, float]],
        tolerance_m: float = GEOMETRY_SIMPLIFY_TOLERANCE_M,
    ) -> List[Dict[str, float]]:
        if len(points) <= 2:
            return points
        reference_lat = sum(point["lat"] for point in points) / len(points)
        keep = {0, len(points) - 1}

        def simplify_range(start_idx: int, end_idx: int) -> None:
            if end_idx <= start_idx + 1:
                return
            start = points[start_idx]
            end = points[end_idx]
            max_distance = -1.0
            max_idx = start_idx
            for idx in range(start_idx + 1, end_idx):
                distance = cls._perpendicular_distance_m(points[idx], start, end, reference_lat)
                if distance > max_distance:
                    max_distance = distance
                    max_idx = idx
            if max_distance > tolerance_m:
                keep.add(max_idx)
                simplify_range(start_idx, max_idx)
                simplify_range(max_idx, end_idx)

        simplify_range(0, len(points) - 1)
        return [points[idx] for idx in sorted(keep)]

    @staticmethod
    def _densify_geometry(
        points: List[Dict[str, float]],
        max_step_km: float = MAX_GEOMETRY_STEP_KM,
    ) -> List[Dict[str, float]]:
        if len(points) <= 1:
            return points
        densified = [points[0]]
        for start, end in zip(points, points[1:]):
            distance = routing.haversine_km(start["lat"], start["lon"], end["lat"], end["lon"])
            steps = max(1, int(math.ceil(distance / max_step_km)))
            for idx in range(1, steps):
                ratio = idx / steps
                candidate = {
                    "lat": start["lat"] + (end["lat"] - start["lat"]) * ratio,
                    "lon": start["lon"] + (end["lon"] - start["lon"]) * ratio,
                }
                if densified[-1] != candidate:
                    densified.append(candidate)
            if densified[-1] != end:
                densified.append(end)
        return densified

    @classmethod
    def _postprocess_geometry(cls, geometry: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if len(geometry) <= 2:
            return geometry
        cleaned: List[Dict[str, float]] = []
        for point in geometry:
            if not cleaned or cleaned[-1] != point:
                cleaned.append(point)
        return cls._densify_geometry(cleaned)


@lru_cache(maxsize=1)
def get_routing_service() -> RoutingService:
    return RoutingService()
