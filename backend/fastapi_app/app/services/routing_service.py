# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import math
import pickle
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

import pandas as pd

from algorithms.recommenders import data_loader, routing

from ..schemas.routes import RouteRequest, RouteResponse, RouteStepResponse, RouteVariant

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "cache"
GRAPH_CACHE = CACHE_DIR / "route_graph.pkl"
GRAPH_META = CACHE_DIR / "route_graph.meta.json"
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


def _load_graph_cache(signature):
    if not GRAPH_CACHE.exists() or not GRAPH_META.exists():
        return None
    try:
        meta = json.loads(GRAPH_META.read_text())
    except Exception:
        return None
    if meta.get("signature") != list(signature):
        return None
    try:
        with GRAPH_CACHE.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _store_graph_cache(signature, bundle):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with GRAPH_CACHE.open("wb") as fh:
        pickle.dump(bundle, fh)
    GRAPH_META.write_text(json.dumps({"signature": list(signature)}))


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
            if progress:
                progress("Cargando eventos", 0.2)
            self.events = data_loader.load_raw_events()
            if progress:
                progress("Eventos cargados", 0.4)
            cached = _load_graph_cache(signature)
            if cached:
                self.graph = cached["graph"]
                self.segment_lookup = cached["segment_lookup"]
                logger.info(
                    "Grafo cargado desde cache: %d nodos, %d segmentos",
                    len(self.graph.nodes),
                    len(self.segment_lookup),
                )
                if progress:
                    progress("Grafo listo (cache)", 1.0)
                return

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
            _store_graph_cache(signature, {"graph": self.graph, "segment_lookup": self.segment_lookup})
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
        day_value = _normalize_day(payload.day_of_week)
        hour_bucket = data_loader.hour_bucket(payload.departure_hour)
        delay_context = {
            "day": day_value,
            "hour_bucket": hour_bucket,
            "include_congestion": True,
            "include_accidents": True,
            "match_filters": True,
        }
        routing_context = None
        needs_context = payload.avoid_congestion or payload.avoid_accidents
        if needs_context:
            routing_context = {
                "day": day_value,
                "hour_bucket": hour_bucket,
                "avoid_congestion": payload.avoid_congestion,
                "avoid_accidents": payload.avoid_accidents,
            }
        logger.info(
            "Calculando ruta origen=(%.5f, %.5f) destino=(%.5f, %.5f)",
            payload.origin.lat,
            payload.origin.lon,
            payload.destination.lat,
            payload.destination.lon,
        )

        # -------------------------------
        # Factores por vía desde CF (UBCF/IBCF):
        # ratings altos -> factor < 1 (bonificación),
        # ratings bajos -> factor > 1 (castigo real).
        # -------------------------------
        via_factors: Dict[str, float] = {}
        for pref in payload.preferences:
            # pref.weight viene de Streamlit ya normalizado en [0,1]
            score = max(0.0, min(1.0, float(pref.weight)))

            # Zona neutra: 0.4–0.6 ~ sin efecto
            if 0.4 <= score <= 0.6:
                factor = 1.0
            # Muy bien valorada: premio fuerte (baja el costo)
            elif score > 0.6:
                # 0.6 -> 1.0 ; 1.0 -> ~0.3
                factor = 1.0 - (score - 0.6) * 1.75
            # Mal valorada: castigo fuerte (sube el costo)
            else:  # score < 0.4
                # 0.4 -> 1.0 ; 0.0 -> ~3.0
                factor = 1.0 + (0.4 - score) * 5.0

            # Recorte de seguridad
            via_factors[pref.via] = round(max(0.2, min(3.0, factor)), 3)

        default_factor = 1.0

        reference_path = self.graph.shortest_path(
            (payload.origin.lat, payload.origin.lon),
            (payload.destination.lat, payload.destination.lon),
            apply_penalties=False,
        )
        if not reference_path:
            raise ValueError("No se pudo construir una ruta con los datos disponibles.")

        need_personalized = bool(via_factors) or needs_context
        if need_personalized:
            personalized_path = self.graph.shortest_path(
                (payload.origin.lat, payload.origin.lon),
                (payload.destination.lat, payload.destination.lon),
                via_factors=via_factors if via_factors else None,
                default_via_factor=default_factor,
                incident_ctx=routing_context,
                apply_penalties=True,
            )
        else:
            personalized_path = list(reference_path)

        if need_personalized and not personalized_path:
            logger.warning("No se pudo construir una ruta personalizada; se usará la referencia.")
            personalized_path = list(reference_path)

        reference_variant = self._build_response_variant(payload, reference_path, delay_context)
        personalized_variant = self._build_response_variant(payload, personalized_path, delay_context)
        return RouteResponse(reference=reference_variant, personalized=personalized_variant)

    def _build_response_variant(
        self,
        payload: RouteRequest,
        path: List[routing.RouteStep],
        context: Dict[str, str | bool] | None,
    ):
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
        extra_minutes = 0.0
        if context:
            include_congestion = bool(context.get("include_congestion", True))
            include_accidents = bool(context.get("include_accidents", True))
            match_filters = bool(context.get("match_filters", True))
            day_value = str(context.get("day") or "").lower() if match_filters else ""
            hour_value = context.get("hour_bucket") if match_filters else None
            buckets: Dict[Tuple[str, str, str], List[float]] = {}
            for step in path:
                if step.tipo_evento not in {"Congestión", "Accidente"}:
                    continue
                matches_day = True
                if day_value:
                    matches_day = bool(step.dia_semana and step.dia_semana.lower() == day_value)
                matches_hour = True
                if hour_value:
                    matches_hour = bool(step.franja_horaria and step.franja_horaria == hour_value)
                if not (matches_day and matches_hour):
                    continue
                if step.tipo_evento == "Congestión" and not include_congestion:
                    continue
                if step.tipo_evento == "Accidente" and not include_accidents:
                    continue
                key = (step.segment_id, step.tipo_evento, step.franja_horaria or "")
                minutes = max(step.duracion_hrs, 0.1) * 60
                buckets.setdefault(key, []).append(minutes)
            for key, values in buckets.items():
                promedio = sum(values) / len(values)
                extra_minutes += promedio
                if key[1] == "Congestión":
                    extra_minutes += 5
        return RouteVariant(
            distance_km=round(distance, 2),
            estimated_duration_min=round(estimated_minutes, 1),
            steps=steps,
            geometry=self._build_geometry(full_path),
            extra_delay_min=round(extra_minutes, 1),
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
