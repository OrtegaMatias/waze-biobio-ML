# -*- coding: utf-8 -*-
"""
Enrutador basado en Dijkstra sobre los segmentos del Biobío.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from . import data_loader

EARTH_RADIUS_KM = 6_371.0
JUNCTION_RADIUS_M = 35
MAX_JUNCTION_NEIGHBORS = 6
SOURCE_CANDIDATES = 8
TARGET_CANDIDATES = 256
MAX_DESTINATION_GAP_KM = 0.35
MIN_CANDIDATE_WINDOW_KM = 0.05
CANDIDATE_DISTANCE_MULTIPLIER = 3.0


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
    ):
        self.nodes = nodes
        self.adjacency = adjacency
        self._spatial_node_ids: List[str] = list(spatial_node_ids or [])
        self._ball_tree: BallTree | None = None
        self._rebuild_spatial_index()

    def __setstate__(self, state):
        self.__dict__.update(state)
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

        coord_items = list(coord_groups.values())
        for ci, node_list in enumerate(coord_items):
            if len(node_list) < 2:
                continue
            base_nodes = [nodes[node_id] for node_id in node_list]
            for i in range(len(base_nodes)):
                for j in range(i + 1, len(base_nodes)):
                    a = base_nodes[i]
                    b = base_nodes[j]
                    base_dist = cls._base_distance(a, b) * 0.25
                    cls._add_edge(adjacency_maps, a.node_id, b.node_id, base_dist)
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
                    if source_node.segment_id == target_node.segment_id:
                        continue
                    distance_km = float(raw_distance) * EARTH_RADIUS_KM
                    if distance_km <= 0:
                        continue
                    connection_weight = max(0.01, distance_km * 0.35)
                    cls._add_edge(adjacency_maps, source_node_id, target_node_id, connection_weight)
                    cls._add_edge(adjacency_maps, target_node_id, source_node_id, connection_weight)
                    added += 1
                    if added >= MAX_JUNCTION_NEIGHBORS:
                        break
                if pos % 1000 == 0:
                    notify("junctions", pos / total_valid)

        adjacency = {
            node_id: list(neighbors.items())
            for node_id, neighbors in adjacency_maps.items()
        }
        return cls(nodes=nodes, adjacency=adjacency, spatial_node_ids=spatial_node_ids or node_ids)

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
        apply_penalties: bool = True,
    ) -> List[RouteStep]:
        source_candidates = self._filter_snap_candidates(
            self.nearest_nodes(*origin, limit=SOURCE_CANDIDATES)
        )
        source_ids = {node.node_id for node, _ in source_candidates}
        target_candidates = self._filter_snap_candidates(
            self.nearest_nodes(
                *destination,
                exclude=source_ids,
                limit=TARGET_CANDIDATES,
            )
        )
        if not target_candidates:
            target_candidates = self._filter_snap_candidates(self.nearest_nodes(*destination, limit=1))
        target_ids = {node.node_id for node, _ in target_candidates}
        distances = {node_id: float("inf") for node_id in self.nodes}
        previous: Dict[str, Optional[str]] = {node_id: None for node_id in self.nodes}
        queue: List[Tuple[float, str]] = []
        best_target = None

        for source_node, snap_distance in source_candidates:
            initial_cost = max(0.0, snap_distance)
            if initial_cost < distances[source_node.node_id]:
                distances[source_node.node_id] = initial_cost
                queue.append((initial_cost, source_node.node_id))

        heapq.heapify(queue)
        nodes = self.nodes
        adjacency = self.adjacency

        def preference_factor(via: str) -> float:
            if via_factors:
                return float(via_factors.get(via, default_via_factor))
            return default_via_factor

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
            avoid_accidents = bool(incident_ctx.get("avoid_accidents"))

            # Si el usuario no pidió evitar nada, no aplicamos incidente extra
            if not (avoid_congestion or avoid_accidents):
                return 1.0

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
            if avoid_congestion and node.tipo_evento == "Congestión":
                # Match EXACTO día + franja: caso extremo (mantiene el comportamiento previo)
                if matches_day and matches_hour:
                    return max(1.0, base_incident * 400.0)

                # Match parcial (día O franja) o segmento históricamente penalizado:
                if matches_day or matches_hour or has_penalty:
                    # Penalización fuerte pero menor al caso exacto
                    return max(1.0, base_incident * 4.0)

                # Solo factor histórico suave (si lo hubiera)
                return max(1.0, base_incident)

            # --- Accidentes ---
            if avoid_accidents and node.tipo_evento == "Accidente":
                if matches_day and matches_hour:
                    return max(1.0, base_incident * 200.0)

                if matches_day or matches_hour or has_penalty:
                    return max(1.0, base_incident * 2.0)

                return max(1.0, base_incident)

            # Otros tipos de evento: solo factor histórico (si lo hay)
            return max(1.0, base_incident)

        while queue:
            current_dist, node_id = heapq.heappop(queue)
            if current_dist > distances[node_id]:
                continue
            if node_id in target_ids:
                best_target = node_id
                break
            current_node = nodes[node_id]
            for neighbor, base_weight in adjacency.get(node_id, []):
                neighbor_node = nodes[neighbor]
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
                adjusted_weight = base_weight * penalty * speed_factor * preference * incident
                new_dist = current_dist + adjusted_weight
                if new_dist < distances.get(neighbor, float("inf")):
                    distances[neighbor] = new_dist
                    previous[neighbor] = node_id
                    heapq.heappush(queue, (new_dist, neighbor))

        if best_target is None:
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
