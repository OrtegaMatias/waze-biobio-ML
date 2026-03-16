# -*- coding: utf-8 -*-
"""
Enrutador basado en Dijkstra sobre los segmentos del Biobío.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from . import data_loader


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
        spatial_coords: Optional[np.ndarray] = None,
        spatial_tree: Optional[BallTree] = None,
    ):
        self.nodes = nodes
        self.adjacency = adjacency
        self._spatial_node_ids = list(spatial_node_ids or [])
        self._spatial_coords = spatial_coords
        self._spatial_tree = spatial_tree
        self._ensure_spatial_index()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ensure_spatial_index()

    def _ensure_spatial_index(self) -> None:
        if self._spatial_coords is None:
            if not self._spatial_node_ids:
                self._spatial_node_ids = list(self.nodes.keys())
            coords = [
                (self.nodes[node_id].lat, self.nodes[node_id].lon)
                for node_id in self._spatial_node_ids
                if node_id in self.nodes
            ]
            if coords:
                self._spatial_coords = np.radians(np.asarray(coords, dtype=float))
            else:
                self._spatial_coords = np.empty((0, 2), dtype=float)
        if self._spatial_tree is None and len(self._spatial_coords):
            self._spatial_tree = BallTree(self._spatial_coords, metric="haversine")

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
        adjacency: Dict[str, List[Tuple[str, float]]] = {}
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
                adjacency.setdefault(a, []).append((b, base_dist))
                if not is_oneway:
                    adjacency.setdefault(b, []).append((a, base_dist))
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
                    adjacency.setdefault(a.node_id, []).append((b.node_id, base_dist))
                    adjacency.setdefault(b.node_id, []).append((a.node_id, base_dist))
            if ci % 5000 == 0 and coord_items:
                notify("junctions", ci / len(coord_items))
        coords_array = (
            np.radians(np.asarray(spatial_coords, dtype=float))
            if spatial_coords
            else np.empty((0, 2), dtype=float)
        )
        spatial_tree = BallTree(coords_array, metric="haversine") if len(coords_array) else None
        return cls(
            nodes=nodes,
            adjacency=adjacency,
            spatial_node_ids=spatial_node_ids,
            spatial_coords=coords_array,
            spatial_tree=spatial_tree,
        )

    @staticmethod
    def _base_distance(a: GraphNode, b: GraphNode) -> float:
        distance = haversine_km(a.lat, a.lon, b.lat, b.lon)
        if not math.isfinite(distance) or distance <= 0:
            return 0.05
        return distance

    def nearest_node(self, lat: float, lon: float, exclude: Optional[Set[str]] = None) -> GraphNode:
        exclude = exclude or set()
        if self._spatial_tree is None or not self._spatial_node_ids:
            best_node = None
            best_dist = float("inf")
            for node_id, node in self.nodes.items():
                if node_id in exclude:
                    continue
                d = haversine_km(lat, lon, node.lat, node.lon)
                if d < best_dist:
                    best_dist = d
                    best_node = node
            if best_node is None:
                raise ValueError("No se encontraron nodos en el grafo.")
            return best_node

        point = np.radians(np.asarray([[lat, lon]], dtype=float))
        total_candidates = len(self._spatial_node_ids)
        k = min(max(len(exclude) + 1, 4), total_candidates)
        while k <= total_candidates:
            _, indices = self._spatial_tree.query(point, k=k)
            for idx in indices[0]:
                node_id = self._spatial_node_ids[int(idx)]
                if node_id not in exclude:
                    return self.nodes[node_id]
            if k == total_candidates:
                break
            k = min(total_candidates, k * 2)
        raise ValueError("No se encontraron nodos en el grafo.")

    def shortest_path(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        via_factors: Optional[Dict[str, float]] = None,
        default_via_factor: float = 1.0,
        incident_ctx: Optional[Dict[str, str | bool]] = None,
        apply_penalties: bool = True,
    ) -> List[RouteStep]:
        source_node = self.nearest_node(*origin)
        try:
            target_node = self.nearest_node(*destination, exclude={source_node.node_id})
        except ValueError:
            neighbors = sorted(self.adjacency.get(source_node.node_id, []), key=lambda x: x[1])
            if neighbors:
                target_node = self.nodes[neighbors[0][0]]
            else:
                target_node = source_node

        source = source_node.node_id
        target = target_node.node_id
        distances: Dict[str, float] = {source: 0.0}
        previous: Dict[str, Optional[str]] = {source: None}
        visited: Set[str] = set()
        queue: List[Tuple[float, str]] = [(0.0, source)]
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
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id == target:
                break
            current_node = nodes[node_id]
            for neighbor, base_weight in adjacency.get(node_id, []):
                if neighbor in visited:
                    continue
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
        if not math.isfinite(distances.get(target, float("inf"))):
            return []

        path: List[RouteStep] = []
        current = target
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
