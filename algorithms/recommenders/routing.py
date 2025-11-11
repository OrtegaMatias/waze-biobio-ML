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


class RouteGraph:
    def __init__(self, nodes: Dict[str, GraphNode], adjacency: Dict[str, List[Tuple[str, float]]]):
        self.nodes = nodes
        self.adjacency = adjacency

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

        for _, row in df.iterrows():
            node_id = f"{row['segment_id']}::{row['segment_seq']}"
            nodes[node_id] = GraphNode(
                node_id=node_id,
                segment_id=row["segment_id"],
                segment_seq=int(row["segment_seq"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                tipo_evento=row["tipo_evento"],
                velocidad_kmh=float(row["velocidad_kmh"]) if not math.isnan(row["velocidad_kmh"]) else 0.0,
                duracion_hrs=float(row["duracion_hrs"]) if not math.isnan(row["duracion_hrs"]) else 0.0,
                via=row["via"],
                comuna=row["comuna"],
                penalty_factor=float(row.get("penalty_factor", 1.0) or 1.0),
            )
            key = (round(float(row["lat"]), 5), round(float(row["lon"]), 5))
            coord_groups[key].append(node_id)
            processed_rows += 1
            if processed_rows % 50000 == 0 and total_rows:
                notify("nodes", processed_rows / total_rows)

        df_sorted = df.sort_values(["segment_id", "segment_seq"])
        groups = list(df_sorted.groupby("segment_id"))
        total_groups = len(groups) if groups else 1
        for gi, (segment_id, group) in enumerate(groups):
            group = group.sort_values("segment_seq")
            node_ids = [f"{segment_id}::{int(seq)}" for seq in group["segment_seq"]]
            is_oneway = bool(group["oneway"].iloc[0]) if "oneway" in group.columns else False
            for idx in range(len(node_ids) - 1):
                a = node_ids[idx]
                b = node_ids[idx + 1]
                weight = cls._edge_weight(nodes[a], nodes[b])
                adjacency.setdefault(a, []).append((b, weight))
                if not is_oneway:
                    adjacency.setdefault(b, []).append((a, weight))
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
                    weight = cls._edge_weight(a, b) * 0.25
                    adjacency.setdefault(a.node_id, []).append((b.node_id, weight))
                    adjacency.setdefault(b.node_id, []).append((a.node_id, weight))
            if ci % 5000 == 0 and coord_items:
                notify("junctions", ci / len(coord_items))
        return cls(nodes=nodes, adjacency=adjacency)

    @staticmethod
    def _edge_weight(a: GraphNode, b: GraphNode) -> float:
        distance = haversine_km(a.lat, a.lon, b.lat, b.lon)
        if distance == 0:
            distance = 0.05
        penalty = max((a.penalty_factor + b.penalty_factor) / 2, 1.0)
        avg_speed = (a.velocidad_kmh + b.velocidad_kmh) / 2 or 10
        speed_factor = max(0.3, 40 / max(avg_speed, 5))
        return distance * penalty * speed_factor

    def nearest_node(self, lat: float, lon: float, exclude: Optional[Set[str]] = None) -> GraphNode:
        exclude = exclude or set()
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

    def shortest_path(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> List[RouteStep]:
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
        distances = {node_id: float("inf") for node_id in self.nodes}
        previous: Dict[str, Optional[str]] = {node_id: None for node_id in self.nodes}
        distances[source] = 0.0
        queue: List[Tuple[float, str]] = [(0.0, source)]

        while queue:
            current_dist, node_id = heapq.heappop(queue)
            if node_id == target:
                break
            if current_dist > distances[node_id]:
                continue
            for neighbor, weight in self.adjacency.get(node_id, []):
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = node_id
                    heapq.heappush(queue, (new_dist, neighbor))

        path: List[RouteStep] = []
        current = target
        while current is not None:
            node = self.nodes[current]
            peso = distances[current]
            node_lat = node.lat
            node_lon = node.lon
            if not math.isfinite(node_lat) or not math.isfinite(node_lon):
                if previous[current] is not None:
                    prev_node = self.nodes[previous[current]]
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
                )
            )
            current = previous[current]
        path.reverse()
        return path


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c
