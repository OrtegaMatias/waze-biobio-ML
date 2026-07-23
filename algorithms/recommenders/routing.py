# -*- coding: utf-8 -*-
"""
Enrutador basado en A* sobre los segmentos del Biobío.
"""

from __future__ import annotations

import heapq
import math
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from . import data_loader

EARTH_RADIUS_KM = 6_371.0
JUNCTION_RADIUS_M = 35
MAX_JUNCTION_NEIGHBORS = 6
BRIDGE_GAP_RADIUS_KM = 2.25
MAX_BRIDGE_GAP_NEIGHBORS = 20_000
SOURCE_CANDIDATES = 8
TARGET_CANDIDATES = 256
MAX_DESTINATION_GAP_KM = 0.35
MIN_CANDIDATE_WINDOW_KM = 0.05
CANDIDATE_DISTANCE_MULTIPLIER = 3.0
GRAPH_MIN_GEOGRAPHIC_WEIGHT_RATIO = 0.25


class RouteSearchCancelled(Exception):
    """Raised when a caller cancels an active graph search."""


@dataclass
class GraphNode:
    node_id: str
    segment_id: str
    segment_seq: int
    lat: float
    lon: float
    tipo_evento: str
    velocidad_kmh: float
    duracion_hrs: float
    via: str
    comuna: str
    penalty_factor: float = 1.0
    dia_semana: str = ""
    franja_horaria: str = ""
    oneway: bool = False


@dataclass
class RouteStep:
    node_id: str
    segment_id: str
    segment_seq: int
    lat: float
    lon: float
    via: str
    comuna: str
    peso: float
    tipo_evento: str = "Referencia"
    duracion_hrs: float = 0.0
    dia_semana: str = ""
    franja_horaria: str = ""


class RouteGraph:
    def __init__(
        self,
        nodes: Dict[str, GraphNode],
        adjacency: Dict[str, List[Tuple[str, float]]],
        spatial_node_ids: Optional[List[str]] = None,
        minimum_geographic_weight_ratio: float = 0.0,
    ):
        self.nodes = nodes
        self.adjacency = adjacency
        self._spatial_node_ids: List[str] = list(spatial_node_ids or [])
        self._minimum_geographic_weight_ratio = max(0.0, float(minimum_geographic_weight_ratio))
        self._ball_tree: BallTree | None = None
        self._rebuild_spatial_index()

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "_minimum_geographic_weight_ratio"):
            # Cached graphs were created by ``from_events`` and obey this same
            # geographic lower bound.
            self._minimum_geographic_weight_ratio = GRAPH_MIN_GEOGRAPHIC_WEIGHT_RATIO
        self._rebuild_spatial_index()

    def _rebuild_spatial_index(self) -> None:
        valid_ids: List[str] = []
        valid_coords: List[Tuple[float, float]] = []
        candidate_ids = self._spatial_node_ids or list(self.nodes.keys())
        for node_id in candidate_ids:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            if math.isfinite(node.lat) and math.isfinite(node.lon):
                valid_ids.append(node_id)
                valid_coords.append((node.lat, node.lon))
        self._spatial_node_ids = valid_ids
        if valid_coords:
            self._ball_tree = BallTree(np.radians(np.asarray(valid_coords, dtype=float)), metric="haversine")
        else:
            self._ball_tree = None

    @staticmethod
    def _add_edge(adjacency: Dict[str, Dict[str, float]], source: str, target: str, weight: float) -> None:
        current = adjacency[source].get(target)
        if current is None or weight < current:
            adjacency[source][target] = weight

    @staticmethod
    def _segment_bounds(nodes: Dict[str, GraphNode]) -> Dict[str, Tuple[int, int, bool]]:
        bounds: Dict[str, Tuple[int, int, bool]] = {}
        for node in nodes.values():
            current = bounds.get(node.segment_id)
            if current is None:
                bounds[node.segment_id] = (node.segment_seq, node.segment_seq, bool(node.oneway))
                continue
            min_seq, max_seq, oneway = current
            bounds[node.segment_id] = (
                min(min_seq, node.segment_seq),
                max(max_seq, node.segment_seq),
                bool(oneway or node.oneway),
            )
        return bounds

    @staticmethod
    def _is_segment_endpoint(node: GraphNode, bounds: Dict[str, Tuple[int, int, bool]]) -> bool:
        segment = bounds.get(node.segment_id)
        if segment is None:
            return True
        min_seq, max_seq, _ = segment
        return node.segment_seq in {min_seq, max_seq}

    @staticmethod
    def _can_enter_segment(node: GraphNode, bounds: Dict[str, Tuple[int, int, bool]]) -> bool:
        segment = bounds.get(node.segment_id)
        if segment is None:
            return True
        min_seq, _, oneway = segment
        return not oneway or node.segment_seq == min_seq

    @staticmethod
    def _can_exit_segment(node: GraphNode, bounds: Dict[str, Tuple[int, int, bool]]) -> bool:
        segment = bounds.get(node.segment_id)
        if segment is None:
            return True
        _, max_seq, oneway = segment
        return not oneway or node.segment_seq == max_seq

    @classmethod
    def _can_transfer_between_segments(
        cls,
        source: GraphNode,
        target: GraphNode,
        bounds: Dict[str, Tuple[int, int, bool]],
    ) -> bool:
        if source.segment_id == target.segment_id:
            return False
        return (
            cls._is_segment_endpoint(source, bounds)
            and cls._is_segment_endpoint(target, bounds)
            and cls._can_exit_segment(source, bounds)
            and cls._can_enter_segment(target, bounds)
        )

    @staticmethod
    def _normalized_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(value or ""))
        ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
        return " ".join(ascii_text.lower().split())

    @classmethod
    def _is_congestion_event(cls, value: str) -> bool:
        return cls._normalized_text(value).startswith("congesti")

    @classmethod
    def _is_bridge_like(cls, node: GraphNode) -> bool:
        via = cls._normalized_text(node.via)
        return "puente" in via

    @staticmethod
    def _component_lookup(
        nodes: Dict[str, GraphNode],
        adjacency: Dict[str, Dict[str, float]],
    ) -> Dict[str, int]:
        component: Dict[str, int] = {}
        undirected: Dict[str, List[str]] = defaultdict(list)
        for source, neighbors in adjacency.items():
            for target in neighbors:
                undirected[source].append(target)
                undirected[target].append(source)
        next_component_id = 0
        for node_id in nodes:
            if node_id in component:
                continue
            component_id = next_component_id
            next_component_id += 1
            queue = deque([node_id])
            component[node_id] = component_id
            while queue:
                current = queue.popleft()
                for neighbor in undirected.get(current, []):
                    if neighbor not in component:
                        component[neighbor] = component_id
                        queue.append(neighbor)
        return component

    @classmethod
    def _connect_bridge_gaps(
        cls,
        nodes: Dict[str, GraphNode],
        adjacency: Dict[str, Dict[str, float]],
        segment_bounds: Dict[str, Tuple[int, int, bool]],
    ) -> None:
        bridge_node_ids = [
            node_id
            for node_id, node in nodes.items()
            if cls._is_bridge_like(node) and math.isfinite(node.lat) and math.isfinite(node.lon)
        ]
        if not bridge_node_ids or len(nodes) < 2:
            return

        node_ids = [
            node_id
            for node_id, node in nodes.items()
            if math.isfinite(node.lat) and math.isfinite(node.lon)
        ]
        if not node_ids:
            return

        component = cls._component_lookup(nodes, adjacency)
        coords = np.asarray([[nodes[node_id].lat, nodes[node_id].lon] for node_id in node_ids], dtype=float)
        tree = BallTree(np.radians(coords), metric="haversine")
        radius = BRIDGE_GAP_RADIUS_KM / EARTH_RADIUS_KM
        best_pairs: Dict[Tuple[int, int], Tuple[float, str, str]] = {}

        for bridge_node_id in bridge_node_ids:
            bridge_node = nodes[bridge_node_id]
            source_component = component.get(bridge_node_id)
            if source_component is None:
                continue
            query = np.radians(np.asarray([[bridge_node.lat, bridge_node.lon]], dtype=float))
            nearby_indices, nearby_distances = tree.query_radius(
                query,
                r=radius,
                return_distance=True,
                sort_results=True,
            )
            checked_for_bridge = 0
            for raw_index, raw_distance in zip(nearby_indices[0], nearby_distances[0]):
                checked_for_bridge += 1
                if checked_for_bridge > MAX_BRIDGE_GAP_NEIGHBORS:
                    break
                target_node_id = node_ids[int(raw_index)]
                if target_node_id == bridge_node_id:
                    continue
                target_component = component.get(target_node_id)
                if target_component is None or target_component == source_component:
                    continue
                target_node = nodes[target_node_id]
                if not cls._can_transfer_between_segments(bridge_node, target_node, segment_bounds):
                    continue
                source_commune = cls._normalized_text(bridge_node.comuna)
                target_commune = cls._normalized_text(target_node.comuna)
                if source_commune == target_commune and not cls._is_bridge_like(target_node):
                    continue
                distance_km = float(raw_distance) * EARTH_RADIUS_KM
                if distance_km <= 0:
                    continue
                key = tuple(sorted((source_component, target_component)))
                current = best_pairs.get(key)
                if current is None or distance_km < current[0]:
                    best_pairs[key] = (distance_km, bridge_node_id, target_node_id)

        for distance_km, source, target in best_pairs.values():
            weight = max(0.05, distance_km)
            source_node = nodes[source]
            target_node = nodes[target]
            if cls._can_transfer_between_segments(source_node, target_node, segment_bounds):
                cls._add_edge(adjacency, source, target, weight)
            if cls._can_transfer_between_segments(target_node, source_node, segment_bounds):
                cls._add_edge(adjacency, target, source, weight)

    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame | None = None,
        progress: Optional[Callable[[str, float], None]] = None,
    ) -> "RouteGraph":
        df = events if events is not None else data_loader.load_raw_events()
        total_rows = len(df)
        processed_rows = 0

        def notify(stage: str, ratio: float) -> None:
            if progress is not None:
                try:
                    progress(stage, float(max(0.0, min(1.0, ratio))))
                except Exception:
                    pass
        nodes: Dict[str, GraphNode] = {}
        adjacency_maps: Dict[str, Dict[str, float]] = defaultdict(dict)
        coord_groups = defaultdict(list)
        spatial_node_ids: List[str] = []
        spatial_coords: List[Tuple[float, float]] = []

        for row in df.itertuples(index=False):
            lat = float(row.lat)
            lon = float(row.lon)
            velocidad = float(row.velocidad_kmh) if math.isfinite(float(row.velocidad_kmh)) else 0.0
            duracion = float(row.duracion_hrs) if math.isfinite(float(row.duracion_hrs)) else 0.0
            node_id = f"{row.segment_id}::{int(row.segment_seq)}"
            nodes[node_id] = GraphNode(
                node_id=node_id,
                segment_id=row.segment_id,
                segment_seq=int(row.segment_seq),
                lat=lat,
                lon=lon,
                tipo_evento=row.tipo_evento,
                velocidad_kmh=velocidad,
                duracion_hrs=duracion,
                via=row.via,
                comuna=row.comuna,
                penalty_factor=float(getattr(row, "penalty_factor", 1.0) or 1.0),
                dia_semana=str(getattr(row, "dia_semana", "") or ""),
                franja_horaria=str(getattr(row, "franja_horaria", "") or ""),
                oneway=bool(getattr(row, "oneway", False)),
            )
            key = (round(lat, 5), round(lon, 5))
            coord_groups[key].append(node_id)
            spatial_node_ids.append(node_id)
            spatial_coords.append((lat, lon))
            processed_rows += 1
            if processed_rows % 50000 == 0 and total_rows:
                notify("nodes", processed_rows / total_rows)

        df_sorted = df.sort_values(["segment_id", "segment_seq"])
        total_groups = max(int(df_sorted["segment_id"].nunique()), 1)
        for gi, (segment_id, group) in enumerate(df_sorted.groupby("segment_id", sort=False)):
            node_ids = [f"{segment_id}::{int(seq)}" for seq in group["segment_seq"]]
            is_oneway = bool(group["oneway"].iloc[0]) if "oneway" in group.columns else False
            for idx in range(len(node_ids) - 1):
                a = node_ids[idx]
                b = node_ids[idx + 1]
                base_dist = cls._base_distance(nodes[a], nodes[b])
                cls._add_edge(adjacency_maps, a, b, base_dist)
                if not is_oneway:
                    cls._add_edge(adjacency_maps, b, a, base_dist)
            if gi % 1000 == 0:
                notify("segments", gi / total_groups)

        segment_bounds = cls._segment_bounds(nodes)

        coord_items = list(coord_groups.values())
        for ci, node_list in enumerate(coord_items):
            if len(node_list) < 2:
                continue
            base_nodes = [nodes[node_id] for node_id in node_list]
            for i in range(len(base_nodes)):
                for j in range(i + 1, len(base_nodes)):
                    a = base_nodes[i]
                    b = base_nodes[j]
                    if not (
                        cls._can_transfer_between_segments(a, b, segment_bounds)
                        or cls._can_transfer_between_segments(b, a, segment_bounds)
                    ):
                        continue
                    base_dist = cls._base_distance(a, b) * 0.25
                    if cls._can_transfer_between_segments(a, b, segment_bounds):
                        cls._add_edge(adjacency_maps, a.node_id, b.node_id, base_dist)
                    if cls._can_transfer_between_segments(b, a, segment_bounds):
                        cls._add_edge(adjacency_maps, b.node_id, a.node_id, base_dist)
            if ci % 5000 == 0 and coord_items:
                notify("junctions", ci / len(coord_items))

        node_ids = list(nodes.keys())
        valid_indices = [
            idx for idx, node_id in enumerate(node_ids)
            if math.isfinite(nodes[node_id].lat) and math.isfinite(nodes[node_id].lon)
        ]
        if valid_indices:
            coords = np.asarray(
                [[nodes[node_ids[idx]].lat, nodes[node_ids[idx]].lon] for idx in valid_indices],
                dtype=float,
            )
            tree = BallTree(np.radians(coords), metric="haversine")
            radius = JUNCTION_RADIUS_M / (EARTH_RADIUS_KM * 1000)
            nearby_indices, nearby_distances = tree.query_radius(
                np.radians(coords),
                r=radius,
                return_distance=True,
                sort_results=True,
            )
            total_valid = len(valid_indices)
            for pos, (neighbors, distances) in enumerate(zip(nearby_indices, nearby_distances)):
                source_node_id = node_ids[valid_indices[pos]]
                source_node = nodes[source_node_id]
                added = 0
                for neighbor_idx, raw_distance in zip(neighbors[1:], distances[1:]):
                    target_node_id = node_ids[valid_indices[int(neighbor_idx)]]
                    target_node = nodes[target_node_id]
                    if not cls._can_transfer_between_segments(source_node, target_node, segment_bounds):
                        continue
                    distance_km = float(raw_distance) * EARTH_RADIUS_KM
                    if distance_km <= 0:
                        continue
                    connection_weight = max(0.01, distance_km * 0.35)
                    cls._add_edge(adjacency_maps, source_node_id, target_node_id, connection_weight)
                    if cls._can_transfer_between_segments(target_node, source_node, segment_bounds):
                        cls._add_edge(adjacency_maps, target_node_id, source_node_id, connection_weight)
                    added += 1
                    if added >= MAX_JUNCTION_NEIGHBORS:
                        break
                if pos % 1000 == 0:
                    notify("junctions", pos / total_valid)

        cls._connect_bridge_gaps(nodes, adjacency_maps, segment_bounds)

        adjacency = {
            node_id: list(neighbors.items())
            for node_id, neighbors in adjacency_maps.items()
        }
        return cls(
            nodes=nodes,
            adjacency=adjacency,
            spatial_node_ids=spatial_node_ids or node_ids,
            minimum_geographic_weight_ratio=GRAPH_MIN_GEOGRAPHIC_WEIGHT_RATIO,
        )

    @staticmethod
    def _base_distance(a: GraphNode, b: GraphNode) -> float:
        distance = haversine_km(a.lat, a.lon, b.lat, b.lon)
        if not math.isfinite(distance) or distance <= 0:
            return 0.05
        return distance

    def nearest_nodes(
        self,
        lat: float,
        lon: float,
        exclude: Optional[Set[str]] = None,
        limit: int = 1,
    ) -> List[Tuple[GraphNode, float]]:
        exclude = exclude or set()
        if not self.nodes:
            raise ValueError("No se encontraron nodos en el grafo.")
        limit = max(1, int(limit))
        if self._ball_tree is None or not self._spatial_node_ids:
            ranked = sorted(
                (
                    (self.nodes[node_id], haversine_km(lat, lon, node.lat, node.lon))
                    for node_id, node in self.nodes.items()
                    if node_id not in exclude
                ),
                key=lambda item: item[1],
            )
            return ranked[:limit]

        query_point = np.radians(np.asarray([[lat, lon]], dtype=float))
        total_nodes = len(self._spatial_node_ids)
        requested = min(total_nodes, max(limit * 4, limit + len(exclude)))
        while True:
            distances, indices = self._ball_tree.query(query_point, k=requested)
            ranked: List[Tuple[GraphNode, float]] = []
            seen: Set[str] = set()
            for raw_distance, raw_index in zip(distances[0], indices[0]):
                node_id = self._spatial_node_ids[int(raw_index)]
                if node_id in exclude or node_id in seen:
                    continue
                seen.add(node_id)
                ranked.append((self.nodes[node_id], float(raw_distance) * EARTH_RADIUS_KM))
                if len(ranked) >= limit:
                    return ranked
            if requested >= total_nodes:
                return ranked
            requested = min(total_nodes, requested * 2)

    def nearest_node(self, lat: float, lon: float, exclude: Optional[Set[str]] = None) -> GraphNode:
        ranked = self.nearest_nodes(lat, lon, exclude=exclude, limit=1)
        if not ranked:
            raise ValueError("No se encontraron nodos en el grafo.")
        return ranked[0][0]

    @staticmethod
    def _filter_snap_candidates(candidates: List[Tuple[GraphNode, float]]) -> List[Tuple[GraphNode, float]]:
        if not candidates:
            return []
        best_distance = candidates[0][1]
        max_distance = max(MIN_CANDIDATE_WINDOW_KM, best_distance * CANDIDATE_DISTANCE_MULTIPLIER)
        filtered = [item for item in candidates if item[1] <= max_distance]
        return filtered or candidates[:1]

    def shortest_path(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        via_factors: Optional[Dict[str, float]] = None,
        default_via_factor: float = 1.0,
        incident_ctx: Optional[Dict[str, str | bool]] = None,
        air_quality_factor: Optional[Callable[[GraphNode], float]] = None,
        urban_wellbeing_factor: Optional[Callable[[GraphNode], float]] = None,
        apply_penalties: bool = True,
        source_node_costs: Optional[Dict[str, float]] = None,
        target_node_costs: Optional[Dict[str, float]] = None,
        edge_filter: Optional[Callable[[GraphNode, GraphNode], bool]] = None,
        edge_cost_factor: Optional[Callable[[GraphNode, GraphNode], float]] = None,
        geographic_path_limit_km: Optional[float] = None,
        use_heuristic: bool = True,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> List[RouteStep]:
        if should_cancel is not None and should_cancel():
            raise RouteSearchCancelled()
        if source_node_costs:
            source_candidates = [
                (self.nodes[node_id], max(0.0, float(cost)))
                for node_id, cost in source_node_costs.items()
                if node_id in self.nodes and math.isfinite(float(cost))
            ]
        else:
            source_candidates = self._filter_snap_candidates(
                self.nearest_nodes(*origin, limit=SOURCE_CANDIDATES)
            )
        source_ids = {node.node_id for node, _ in source_candidates}
        if target_node_costs:
            target_candidates = [
                (self.nodes[node_id], max(0.0, float(cost)))
                for node_id, cost in target_node_costs.items()
                if node_id in self.nodes and math.isfinite(float(cost))
            ]
        else:
            target_candidates = self._filter_snap_candidates(
                self.nearest_nodes(
                    *destination,
                    exclude=source_ids,
                    limit=TARGET_CANDIDATES,
                )
            )
        if not target_candidates and not target_node_costs:
            target_candidates = self._filter_snap_candidates(self.nearest_nodes(*destination, limit=1))
        target_cost_lookup = {node.node_id: max(0.0, float(cost)) for node, cost in target_candidates}
        target_ids = set(target_cost_lookup)
        distances: Dict[str, float] = {}
        previous: Dict[str, Optional[str]] = {}
        queue: List[Tuple[float, float, str]] = []
        best_target = None
        best_target_total = float("inf")

        # A* must never overestimate the remaining adjusted route cost. Base
        # graph edges cost at least 25% of their geodesic distance; the other
        # multipliers below are the exact lower clamps used during expansion.
        preference_floor = float(default_via_factor)
        if via_factors:
            preference_floor = min(
                preference_floor,
                *(float(value) for value in via_factors.values()),
            )
        heuristic_factor = self._minimum_geographic_weight_ratio
        if apply_penalties:
            heuristic_factor *= 0.3
        heuristic_factor *= preference_floor
        if air_quality_factor is not None:
            heuristic_factor *= 0.5
        if urban_wellbeing_factor is not None:
            heuristic_factor *= 0.65
        if not use_heuristic or not math.isfinite(heuristic_factor) or heuristic_factor <= 0:
            heuristic_factor = 0.0
        heuristic_factor = min(1.0, heuristic_factor)
        target_heuristic_correction = max(
            (
                heuristic_factor
                * haversine_km(
                    self.nodes[node_id].lat,
                    self.nodes[node_id].lon,
                    destination[0],
                    destination[1],
                )
                - terminal_cost
                for node_id, terminal_cost in target_cost_lookup.items()
            ),
            default=0.0,
        )
        target_heuristic_correction = max(0.0, target_heuristic_correction)

        def remaining_cost_estimate(node: GraphNode) -> float:
            if heuristic_factor == 0.0:
                return 0.0
            return max(
                0.0,
                heuristic_factor
                * haversine_km(
                    node.lat,
                    node.lon,
                    destination[0],
                    destination[1],
                )
                - target_heuristic_correction,
            )

        for source_node, snap_distance in source_candidates:
            initial_cost = max(0.0, snap_distance)
            if initial_cost < distances.get(source_node.node_id, float("inf")):
                distances[source_node.node_id] = initial_cost
                previous[source_node.node_id] = None
                queue.append(
                    (
                        initial_cost + remaining_cost_estimate(source_node),
                        initial_cost,
                        source_node.node_id,
                    )
                )

        heapq.heapify(queue)
        nodes = self.nodes
        adjacency = self.adjacency
        geographic_eligibility: Dict[str, bool] = {}

        def within_geographic_limit(node: GraphNode) -> bool:
            if geographic_path_limit_km is None:
                return True
            cached = geographic_eligibility.get(node.node_id)
            if cached is not None:
                return cached
            lower_bound = haversine_km(origin[0], origin[1], node.lat, node.lon) + haversine_km(
                node.lat,
                node.lon,
                destination[0],
                destination[1],
            )
            eligible = lower_bound <= max(0.0, float(geographic_path_limit_km)) + 1e-9
            geographic_eligibility[node.node_id] = eligible
            return eligible

        def preference_factor(via: str) -> float:
            if via_factors:
                return float(via_factors.get(via, default_via_factor))
            return default_via_factor

        def pollution_factor(node: GraphNode) -> float:
            if air_quality_factor is None:
                return 1.0
            factor = air_quality_factor(node)
            if not math.isfinite(factor):
                return 1.0
            return max(0.5, min(3.0, float(factor)))

        def wellbeing_factor(node: GraphNode) -> float:
            if urban_wellbeing_factor is None:
                return 1.0
            factor = urban_wellbeing_factor(node)
            if not math.isfinite(factor):
                return 1.0
            return max(0.65, min(1.0, float(factor)))

        def incident_factor(node: GraphNode) -> float:
            """
            Factor adicional de penalización según contexto del viaje.

            - Antes: solo penalizaba si coincidían día Y franja.
            - Ahora: penaliza si coincide el día O la franja O el segmento ya trae
              un penalty_factor > 1.0 (históricamente conflictivo).
            - Además mezcla este factor con penalty_factor para dar más peso a
              segmentos con congestión histórica, incluso si el usuario viaja en
              otro horario.
            """
            if not incident_ctx or not apply_penalties:
                return 1.0

            day = incident_ctx.get("day")
            hour_bucket = incident_ctx.get("hour_bucket")
            avoid_congestion = bool(incident_ctx.get("avoid_congestion"))

            # Si el usuario no pidió evitar nada, no aplicamos incidente extra
            if not avoid_congestion:
                return 1.0

            node_penalties = incident_ctx.get("node_penalties")
            if isinstance(node_penalties, dict):
                node_penalty = node_penalties.get(node.node_id)
                if node_penalty is not None:
                    try:
                        return max(1.0, float(node_penalty))
                    except (TypeError, ValueError):
                        pass

            matches_day = bool(
                day
                and node.dia_semana
                and node.dia_semana.lower() == str(day).lower()
            )
            matches_hour = bool(
                hour_bucket
                and node.franja_horaria
                and node.franja_horaria == hour_bucket
            )
            has_penalty = bool(node.penalty_factor and node.penalty_factor > 1.0)

            # Factor base ligado a la severidad histórica
            # p.ej. penalty_factor=1.5 -> base_incident=1.5
            base_incident = 1.0
            if has_penalty:
                base_incident += (float(node.penalty_factor) - 1.0)

            # --- Congestión ---
            if avoid_congestion and self._is_congestion_event(node.tipo_evento):
                # Match EXACTO día + franja: caso extremo (mantiene el comportamiento previo)
                if matches_day and matches_hour:
                    return max(1.0, base_incident * 400.0)

                # Match parcial (día O franja) o segmento históricamente penalizado:
                if matches_day or matches_hour or has_penalty:
                    # Penalización fuerte pero menor al caso exacto
                    return max(1.0, base_incident * 4.0)

                # Solo factor histórico suave (si lo hubiera)
                return max(1.0, base_incident)

            # Otros tipos de evento: solo factor histórico (si lo hay)
            return max(1.0, base_incident)

        expanded_nodes = 0
        while queue:
            expanded_nodes += 1
            if expanded_nodes % 256 == 0 and should_cancel is not None and should_cancel():
                raise RouteSearchCancelled()
            estimated_total, current_dist, node_id = heapq.heappop(queue)
            if current_dist > distances.get(node_id, float("inf")):
                continue
            if estimated_total >= best_target_total:
                break
            if node_id in target_ids:
                total_with_terminal = current_dist + target_cost_lookup.get(node_id, 0.0)
                if total_with_terminal < best_target_total:
                    best_target_total = total_with_terminal
                    best_target = node_id
            current_node = nodes[node_id]
            for neighbor, base_weight in adjacency.get(node_id, []):
                neighbor_node = nodes[neighbor]
                if not within_geographic_limit(neighbor_node):
                    continue
                if edge_filter is not None and not edge_filter(current_node, neighbor_node):
                    continue
                if apply_penalties:
                    penalty = max((current_node.penalty_factor + neighbor_node.penalty_factor) / 2, 1.0)
                    speed_a = current_node.velocidad_kmh if math.isfinite(current_node.velocidad_kmh) else 0.0
                    speed_b = neighbor_node.velocidad_kmh if math.isfinite(neighbor_node.velocidad_kmh) else 0.0
                    avg_speed = (speed_a + speed_b) / 2 if speed_a > 0 and speed_b > 0 else max(speed_a, speed_b, 0.0)
                    effective_speed = max(avg_speed, 5.0)
                    speed_factor = max(0.3, 40 / effective_speed)
                else:
                    penalty = 1.0
                    speed_factor = 1.0
                preference = preference_factor(neighbor_node.via)
                incident = incident_factor(neighbor_node)
                pollution = pollution_factor(neighbor_node)
                wellbeing = wellbeing_factor(neighbor_node)
                edge_factor = 1.0
                if edge_cost_factor is not None:
                    candidate_factor = edge_cost_factor(current_node, neighbor_node)
                    if math.isfinite(candidate_factor):
                        edge_factor = max(1.0, float(candidate_factor))
                adjusted_weight = (
                    base_weight
                    * penalty
                    * speed_factor
                    * preference
                    * incident
                    * pollution
                    * wellbeing
                    * edge_factor
                )
                new_dist = current_dist + adjusted_weight
                if new_dist < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_dist
                    previous[neighbor] = node_id
                    heapq.heappush(
                        queue,
                        (
                            new_dist + remaining_cost_estimate(neighbor_node),
                            new_dist,
                            neighbor,
                        ),
                    )

        if best_target is None and not target_node_costs:
            reachable_target = None
            reachable_gap = float("inf")
            for node_id, total_cost in distances.items():
                if not math.isfinite(total_cost):
                    continue
                node = self.nodes[node_id]
                gap = haversine_km(destination[0], destination[1], node.lat, node.lon)
                if gap < reachable_gap:
                    reachable_gap = gap
                    reachable_target = node_id
            if reachable_target is not None and reachable_gap <= MAX_DESTINATION_GAP_KM:
                best_target = reachable_target

        if best_target is None or not math.isfinite(distances.get(best_target, float("inf"))):
            return []

        path: List[RouteStep] = []
        current = best_target
        while current is not None:
            node = nodes[current]
            peso = distances[current]
            node_lat = node.lat
            node_lon = node.lon
            if not math.isfinite(node_lat) or not math.isfinite(node_lon):
                prev_id = previous.get(current)
                if prev_id is not None:
                    prev_node = nodes[prev_id]
                    node_lat = prev_node.lat
                    node_lon = prev_node.lon
                else:
                    node_lat, node_lon = origin
            path.append(
                RouteStep(
                    node_id=node.node_id,
                    segment_id=node.segment_id,
                    segment_seq=node.segment_seq,
                    lat=node_lat,
                    lon=node_lon,
                    via=node.via,
                    comuna=node.comuna,
                    peso=peso,
                    tipo_evento=node.tipo_evento,
                    duracion_hrs=node.duracion_hrs,
                    dia_semana=node.dia_semana,
                    franja_horaria=node.franja_horaria,
                )
            )
            current = previous.get(current)
        path.reverse()
        return path


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c
