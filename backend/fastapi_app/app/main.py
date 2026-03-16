# -*- coding: utf-8 -*-
"""
FastAPI principal para exponer la demo de ruta segura explicable.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from algorithms.recommenders import data_loader

from .core import dataset
from .core.demo_scenarios import DEMO_SCENARIOS
from .schemas.recommendations import (
    CollaborativeRequest,
    CollaborativeResponse,
    PlaygroundRequest,
    PlaygroundResponse,
)
from .schemas.routes import HotspotResponse, MetadataResponse, RouteRequest, RouteResponse
from .schemas.system import (
    BootstrapStatus,
    DataQualitySummary,
    DatasetChangeRequest,
    DatasetInfo,
    DatasetStatus,
    DemoScenarioList,
    ReadinessStatus,
)
from .services import data_quality_service
from .services.recommendation_service import RecommendationService, get_recommendation_service
from .services.routing_service import RoutingService, get_routing_service

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
AUTO_BOOTSTRAP_ENABLED = os.getenv("AUTO_BOOTSTRAP_ENABLED", "1") != "0"


def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("uvicorn.error")
    logger.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logger.info("Logging configurado en nivel INFO")
    return logger


logger = configure_logging()

bootstrap_lock = threading.Lock()
bootstrap_thread: threading.Thread | None = None


def _default_bootstrap_state(message: str = "Esperando warm-up") -> dict:
    return {
        "status": "idle",
        "message": message,
        "percent": 0,
        "routing_nodes": 0,
        "routing_segments": 0,
        "duration_ms": 0.0,
        "dataset_profile": dataset.get_profile(),
        "quality": None,
    }


bootstrap_state = _default_bootstrap_state()


def _snapshot_bootstrap_state() -> dict:
    with bootstrap_lock:
        return deepcopy(bootstrap_state)


def _update_bootstrap_state(**kwargs) -> dict:
    with bootstrap_lock:
        bootstrap_state.update(**kwargs)
        return deepcopy(bootstrap_state)


def _quality_summary_dict() -> dict:
    summary = DataQualitySummary(**data_quality_service.inspect_data_quality())
    return summary.model_dump() if hasattr(summary, "model_dump") else summary.dict()


def _build_dataset_status() -> DatasetStatus:
    available = [DatasetInfo(key=key, label=label) for key, label in dataset.available_profiles()]
    return DatasetStatus(
        current=dataset.get_profile(),
        current_label=dataset.get_profile_label(),
        available=available,
    )


def _reset_runtime_state(message: str) -> None:
    get_recommendation_service.cache_clear()
    get_routing_service.cache_clear()
    with _hotspot_cache_lock:
        _hotspot_cache["signature"] = None
        _hotspot_cache["points"] = []
    with bootstrap_lock:
        bootstrap_state.clear()
        bootstrap_state.update(_default_bootstrap_state(message=message))


def _run_bootstrap() -> None:
    start = time.perf_counter()
    try:
        _update_bootstrap_state(
            status="running",
            message="Validando calidad de datos...",
            percent=5,
            dataset_profile=dataset.get_profile(),
        )
        quality = _quality_summary_dict()
        _update_bootstrap_state(quality=quality, message="Preparando metadatos del perfil activo...", percent=12)

        rec_service = get_recommendation_service()
        rec_service.available_options()

        _update_bootstrap_state(message="Construyendo grafo de rutas...", percent=20)
        routing_service = get_routing_service()

        def progress(msg: str, frac: float) -> None:
            _update_bootstrap_state(message=msg, percent=int(max(0.0, min(1.0, frac)) * 100))

        routing_service.build(progress=progress)
        nodes = len(routing_service.graph.nodes) if routing_service.graph else 0
        segments = len(routing_service.segment_lookup)
        duration = (time.perf_counter() - start) * 1000
        _update_bootstrap_state(
            status="completed",
            message="Infraestructura lista para la demo",
            percent=100,
            routing_nodes=nodes,
            routing_segments=segments,
            duration_ms=round(duration, 1),
            dataset_profile=dataset.get_profile(),
            quality=quality,
        )
        logger.info(
            "Bootstrap completado para perfil=%s: %d nodos, %d segmentos (%.1f ms)",
            dataset.get_profile(),
            nodes,
            segments,
            duration,
        )
    except Exception as exc:  # pragma: no cover
        _update_bootstrap_state(
            status="error",
            message=str(exc),
            percent=0,
            dataset_profile=dataset.get_profile(),
            quality=_snapshot_bootstrap_state().get("quality"),
        )
        logger.exception("Falló el bootstrap")


def _start_bootstrap_thread(force: bool = False) -> dict:
    global bootstrap_thread
    snapshot = _snapshot_bootstrap_state()
    if snapshot.get("status") == "running" and bootstrap_thread and bootstrap_thread.is_alive():
        return snapshot
    if not force and snapshot.get("status") == "completed":
        return snapshot
    _update_bootstrap_state(
        status="running",
        message="Iniciando warm-up del backend...",
        percent=1,
        dataset_profile=dataset.get_profile(),
    )
    bootstrap_thread = threading.Thread(target=_run_bootstrap, daemon=True)
    bootstrap_thread.start()
    return _snapshot_bootstrap_state()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if AUTO_BOOTSTRAP_ENABLED:
        _start_bootstrap_thread(force=True)
    yield


app = FastAPI(title="Biobío ML API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "dataset_profile": dataset.get_profile()}


@app.get("/readyz", response_model=ReadinessStatus, tags=["meta"])
def readyz() -> ReadinessStatus:
    snapshot = _snapshot_bootstrap_state()
    service = get_routing_service()
    ready = snapshot.get("status") == "completed" and service.graph is not None
    status = "ready" if ready else ("error" if snapshot.get("status") == "error" else "warming")
    message = (
        "Backend listo para generar la demo."
        if ready
        else snapshot.get("message") or "El backend sigue calentando el perfil activo."
    )
    return ReadinessStatus(
        status=status,
        ready=ready,
        message=message,
        dataset_profile=dataset.get_profile(),
        bootstrap=BootstrapStatus(**snapshot),
    )


@app.get("/metadata/options", response_model=MetadataResponse, tags=["meta"])
def metadata(service: RecommendationService = Depends(get_recommendation_service)) -> MetadataResponse:
    start = time.perf_counter()
    options = service.available_options()
    duration = (time.perf_counter() - start) * 1000
    logger.info("GET /metadata/options -> %d eventos (%.1f ms)", options.get("total_events", 0), duration)
    return MetadataResponse(**options)


@app.get("/system/dataset", response_model=DatasetStatus, tags=["meta"])
def dataset_status() -> DatasetStatus:
    return _build_dataset_status()


@app.post("/system/dataset", response_model=DatasetStatus, tags=["meta"])
def dataset_switch(payload: DatasetChangeRequest) -> DatasetStatus:
    snapshot = _snapshot_bootstrap_state()
    if snapshot.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="No se puede cambiar el perfil mientras el backend está en warm-up.",
        )
    dataset.set_profile(payload.profile)
    _reset_runtime_state("Perfil cambiado. Preparando nuevo warm-up.")
    _start_bootstrap_thread(force=True)
    logger.info("Perfil de datos actualizado a %s", payload.profile)
    return _build_dataset_status()


@app.get("/system/demo-scenarios", response_model=DemoScenarioList, tags=["meta"])
def demo_scenarios() -> DemoScenarioList:
    return DemoScenarioList(scenarios=DEMO_SCENARIOS)


@app.post(
    "/recommendations/collaborative",
    response_model=CollaborativeResponse,
    tags=["recommendations"],
)
def collaborative_recommendations(
    payload: CollaborativeRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> CollaborativeResponse:
    start = time.perf_counter()
    recs = service.collaborative_recommendations(payload)
    if not recs:
        raise HTTPException(status_code=404, detail="No se encontraron recomendaciones para el perfil simulado.")
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /recommendations/collaborative -> %d recs (user=%s,strategy=%s) in %.1f ms",
        len(recs),
        payload.user_id,
        payload.strategy,
        duration,
    )
    return CollaborativeResponse(recommendations=recs)


@app.post(
    "/recommendations/playground",
    response_model=PlaygroundResponse,
    tags=["recommendations"],
)
def collaborative_playground(
    payload: PlaygroundRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> PlaygroundResponse:
    start = time.perf_counter()
    recs = service.playground_recommendations(payload)
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /recommendations/playground -> ubcf=%d ibcf=%d (user=%s) in %.1f ms",
        len(recs.get("ubcf", [])),
        len(recs.get("ibcf", [])),
        payload.user_id,
        duration,
    )
    return PlaygroundResponse(
        ubcf=recs.get("ubcf", []),
        ibcf=recs.get("ibcf", []),
    )


@app.post("/routes/optimal", response_model=RouteResponse, tags=["routes"])
def optimal_route(
    payload: RouteRequest,
    service: RoutingService = Depends(get_routing_service),
) -> RouteResponse:
    snapshot = _snapshot_bootstrap_state()
    if snapshot.get("status") != "completed" and service.graph is None:
        raise HTTPException(
            status_code=503,
            detail="El backend aún está calentando el grafo del perfil activo. Consulta /readyz o /system/bootstrap/status.",
        )
    start = time.perf_counter()
    try:
        route = service.compute_route(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /routes/optimal -> base=%.2f km, mejor balance=%s en %.1f ms",
        route.reference.distance_km if route.reference else 0.0,
        route.comparison.best_balance_variant,
        duration,
    )
    return route


@app.post("/system/bootstrap", response_model=BootstrapStatus, tags=["meta"])
def bootstrap() -> BootstrapStatus:
    return BootstrapStatus(**_start_bootstrap_thread(force=True))


@app.get("/system/bootstrap/status", response_model=BootstrapStatus, tags=["meta"])
def bootstrap_status() -> BootstrapStatus:
    return BootstrapStatus(**_snapshot_bootstrap_state())


_hotspot_cache = {"signature": None, "points": []}
_hotspot_cache_lock = threading.Lock()


def _build_hotspot_points() -> List[dict]:
    events = data_loader.load_raw_events()
    congestions = events[events["tipo_evento"] == "Congestión"].dropna(subset=["lat", "lon"])
    if congestions.empty:
        return []
    bucketed = []
    for _, row in congestions.iterrows():
        try:
            hora_inicio = pd.to_datetime(row.get("hora_inicio"), format="%H:%M", errors="coerce")
            hora_fin = pd.to_datetime(row.get("hora_fin"), format="%H:%M", errors="coerce")
        except Exception:
            hora_inicio = hora_fin = None
        if pd.isna(hora_inicio):
            hora_inicio = None
        if pd.isna(hora_fin):
            hora_fin = None
        start_float = float(hora_inicio.hour + hora_inicio.minute / 60) if hora_inicio is not None else None
        end_float = float(hora_fin.hour + hora_fin.minute / 60) if hora_fin is not None else None
        speed = row.get("velocidad_kmh")
        try:
            speed_value = float(speed) if speed is not None else None
        except Exception:
            speed_value = None
        weight = 0.5
        if speed_value is not None and speed_value > 0:
            weight = min(2.0, max(0.1, 1 / max(speed_value, 5)))
        bucketed.append(
            {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "weight": float(weight),
                "day": str(row.get("dia_semana") or ""),
                "bucket": str(row.get("franja_horaria") or ""),
                "segment_id": str(row.get("segment_id") or ""),
                "hora_inicio_float": start_float,
                "hora_fin_float": end_float,
            }
        )
    return bucketed


def _cached_hotspots(limit: int) -> List[dict]:
    limit = max(0, limit)
    signature = data_loader.data_version()
    with _hotspot_cache_lock:
        if _hotspot_cache["signature"] != signature:
            _hotspot_cache["points"] = _build_hotspot_points()
            _hotspot_cache["signature"] = signature
        return list(_hotspot_cache["points"][:limit])


@app.get("/metadata/hotspots", response_model=HotspotResponse, tags=["meta"])
def metadata_hotspots(limit: int = 2000) -> HotspotResponse:
    limit = max(200, min(limit, 10000))
    points = _cached_hotspots(limit)
    logger.info("Hotspots solicitados -> %d puntos", len(points))
    return HotspotResponse(points=points)
