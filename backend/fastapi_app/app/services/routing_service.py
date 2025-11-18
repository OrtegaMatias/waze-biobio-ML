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

        # Log detallado sobre penalizaciones
        penalty_status = []
        if payload.avoid_congestion:
            penalty_status.append("Congestiones (4x-400x)")
        if payload.avoid_accidents:
            penalty_status.append("Accidentes (2x-200x)")
        if not penalty_status:
            penalty_status.append("NINGUNA - Rutas solo diferirán por preferencias CF")

        logger.info(
            "Calculando rutas:\n"
            "  Origen: (%.5f, %.5f)\n"
            "  Destino: (%.5f, %.5f)\n"
            "  Contexto: %s, hora %.1f (%s)\n"
            "  Penalizaciones activas: %s",
            payload.origin.lat,
            payload.origin.lon,
            payload.destination.lat,
            payload.destination.lon,
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
                # pref.weight viene de Streamlit ya normalizado en [0,1]
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

        # 1. RUTA REFERENCE: Dijkstra puro (sin penalizaciones, sin preferencias)
        logger.info("Generando ruta reference (Dijkstra puro)...")
        reference_path = self.graph.shortest_path(
            (payload.origin.lat, payload.origin.lon),
            (payload.destination.lat, payload.destination.lon),
            apply_penalties=False,
        )
        if not reference_path:
            raise ValueError("No se pudo construir una ruta con los datos disponibles.")

        # Optimización: si no hay preferencias de CF, generar solo 1 ruta con penalizaciones
        has_ubcf = bool(ubcf_factors)
        has_ibcf = bool(ibcf_factors)

        if not has_ubcf and not has_ibcf:
            # Sin preferencias CF: generar solo 1 ruta con penalizaciones y reutilizarla
            logger.info("Sin preferencias CF; generando 1 ruta con penalizaciones para UBCF e IBCF...")
            if needs_context:
                penalty_path = self.graph.shortest_path(
                    (payload.origin.lat, payload.origin.lon),
                    (payload.destination.lat, payload.destination.lon),
                    incident_ctx=routing_context,
                    apply_penalties=True,
                )
                ubcf_path = penalty_path if penalty_path else list(reference_path)
                ibcf_path = list(ubcf_path)
            else:
                # Sin penalizaciones ni preferencias: todas las rutas son iguales a reference
                ubcf_path = list(reference_path)
                ibcf_path = list(reference_path)
        else:
            # 2. RUTA UBCF: Con preferencias UBCF + penalizaciones de incidentes
            if has_ubcf:
                logger.info("Generando ruta UBCF (evita incidentes según usuarios similares)...")
                ubcf_path = self.graph.shortest_path(
                    (payload.origin.lat, payload.origin.lon),
                    (payload.destination.lat, payload.destination.lon),
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
                        incident_ctx=routing_context,
                        apply_penalties=True,
                    )
                    if not ubcf_path:
                        ubcf_path = list(reference_path)
                else:
                    ubcf_path = list(reference_path)

            # 3. RUTA IBCF: Con preferencias IBCF + penalizaciones de incidentes
            if has_ibcf:
                logger.info("Generando ruta IBCF (evita incidentes según vías similares)...")
                ibcf_path = self.graph.shortest_path(
                    (payload.origin.lat, payload.origin.lon),
                    (payload.destination.lat, payload.destination.lon),
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
                    logger.info("Sin preferencias IBCF; generando ruta con solo penalizaciones...")
                    if needs_context:
                        ibcf_path = self.graph.shortest_path(
                            (payload.origin.lat, payload.origin.lon),
                            (payload.destination.lat, payload.destination.lon),
                            incident_ctx=routing_context,
                            apply_penalties=True,
                        )
                        if not ibcf_path:
                            ibcf_path = list(reference_path)
                    else:
                        ibcf_path = list(reference_path)

        # Construir las 3 variantes de respuesta
        reference_variant = self._build_response_variant(payload, reference_path, delay_context)
        ubcf_variant = self._build_response_variant(payload, ubcf_path, delay_context)
        ibcf_variant = self._build_response_variant(payload, ibcf_path, delay_context)

        logger.info(
            "Rutas generadas exitosamente:\n"
            "  - Dijkstra (reference): %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total\n"
            "  - UBCF: %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total\n"
            "  - IBCF: %.2f km, %.1f min base, +%.1f min retrasos = %.1f min total",
            reference_variant.distance_km,
            reference_variant.estimated_duration_min,
            reference_variant.extra_delay_min,
            reference_variant.estimated_duration_min + reference_variant.extra_delay_min,
            ubcf_variant.distance_km,
            ubcf_variant.estimated_duration_min,
            ubcf_variant.extra_delay_min,
            ubcf_variant.estimated_duration_min + ubcf_variant.extra_delay_min,
            ibcf_variant.distance_km,
            ibcf_variant.estimated_duration_min,
            ibcf_variant.extra_delay_min,
            ibcf_variant.estimated_duration_min + ibcf_variant.extra_delay_min,
        )

        return RouteResponse(
            reference=reference_variant,
            ubcf=ubcf_variant,
            ibcf=ibcf_variant,
            personalized=None,  # Ya no se genera para reducir tiempo de cómputo
        )

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
