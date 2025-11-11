# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from functools import lru_cache
from threading import Lock
from typing import Dict, List, Tuple

import math
import pandas as pd

from algorithms.recommenders import data_loader, routing

from ..schemas.routes import RouteRequest, RouteResponse, RouteStepResponse

logger = logging.getLogger(__name__)

class RoutingService:
    def __init__(self, progress=None) -> None:
        self._data_version = None
        self.events = None
        self.graph = None
        self.segment_lookup = {}
        self._progress = progress
        self._build_lock = Lock()
        logger.info("Inicializando RoutingService (lazy build)")

    def _build_structures(self, progress=None) -> None:
        if self._build_lock.locked():
            # Otra hebra ya está construyendo; espera a que termine
            with self._build_lock:
                return
        with self._build_lock:
            logger.info("Construyendo estructuras de ruta...")
            if progress:
                progress("Cargando eventos", 0.2)
            self.events = data_loader.load_raw_events()
            if progress:
                progress("Eventos cargados", 0.4)

            def rg_progress(stage: str, ratio: float) -> None:
                base = {"nodes": 0.4, "segments": 0.8, "junctions": 0.95}.get(stage, 0.4)
                span = 0.4 if stage == "nodes" else (0.15 if stage == "segments" else 0.05)
                percent = min(0.99, base + span * ratio)
                if progress:
                    progress(f"Construyendo grafo ({stage})", percent)

            self.graph = routing.RouteGraph.from_events(self.events, progress=rg_progress)
            self.segment_lookup = self._build_segment_lookup(self.events)
            logger.info(
                "Grafo cargado: %d nodos, %d segmentos",
                len(self.graph.nodes),
                len(self.segment_lookup),
            )
            if progress:
                progress("Grafo listo", 1.0)

    def _ensure_fresh_data(self) -> None:
        current_version = data_loader.data_version()
        if self._data_version is None:
            self._data_version = current_version
            self._build_structures(self._progress)
            return
        if current_version != self._data_version:
            self._data_version = current_version
            logger.info("Detectado cambio en datos (%s), reconstruyendo grafo...", current_version)
            self._build_structures(self._progress)

    def build(self, progress=None) -> None:
        self._data_version = data_loader.data_version()
        self._build_structures(progress)

    @staticmethod
    def _build_segment_lookup(events) -> Dict[str, Dict[int, Tuple[float, float]]]:
        lookup: Dict[str, Dict[int, Tuple[float, float]]] = {}
        df = events.copy()
        df["segment_seq"] = pd.to_numeric(df["segment_seq"], errors="coerce").fillna(0).astype(int)
        for segment_id, group in df.groupby("segment_id"):
            seq_map = {
                int(row.segment_seq): (float(row.lat), float(row.lon))
                for row in group.sort_values("segment_seq").itertuples()
            }
            lookup[segment_id] = seq_map
        return lookup

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

    def compute_route(self, payload: RouteRequest) -> RouteResponse:
        self._ensure_fresh_data()
        if self.graph is None:
            raise ValueError("El grafo de rutas aún no está listo. Intenta nuevamente en unos segundos.")
        logger.info(
            "Calculando ruta origen=(%.5f, %.5f) destino=(%.5f, %.5f)",
            payload.origin.lat,
            payload.origin.lon,
            payload.destination.lat,
            payload.destination.lon,
        )
        path = self.graph.shortest_path(
            (payload.origin.lat, payload.origin.lon),
            (payload.destination.lat, payload.destination.lon),
        )
        if not path:
            raise ValueError("No se pudo construir una ruta con los datos disponibles.")
        first_graph = path[0]
        last_graph = path[-1]
        origin_step = routing.RouteStep(
            node_id="user_origin",
            segment_id=first_graph.segment_id,
            segment_seq=first_graph.segment_seq,
            lat=payload.origin.lat,
            lon=payload.origin.lon,
            via=first_graph.via,
            comuna=first_graph.comuna,
            peso=0.0,
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
        )
        full_path = [origin_step] + path + [dest_step]

        distance = 0.0
        steps: List[RouteStepResponse] = []
        cumulative_cost = 0.0
        for idx, step in enumerate(full_path):
            if idx > 0:
                prev = full_path[idx - 1]
                distance += routing.haversine_km(prev.lat, prev.lon, step.lat, step.lon)
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
        avg_speed = 35
        estimated_minutes = (distance / max(avg_speed, 5)) * 60
        return RouteResponse(
            distance_km=round(distance, 2),
            estimated_duration_min=round(estimated_minutes, 1),
            steps=steps,
            geometry=self._build_geometry(full_path),
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
        return geometry


@lru_cache(maxsize=1)
def get_routing_service() -> RoutingService:
    return RoutingService()
