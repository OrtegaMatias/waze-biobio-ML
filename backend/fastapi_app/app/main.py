# -*- coding: utf-8 -*-
"""
FastAPI principal para exponer las recomendaciones y rutas.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas.recommendations import (
    AssociationRequest,
    AssociationResponse,
    CollaborativeRequest,
    CollaborativeResponse,
)
from .schemas.routes import HotspotResponse, MetadataResponse, RouteRequest, RouteResponse
from .services.recommendation_service import RecommendationService, get_recommendation_service
from .services.routing_service import RoutingService, get_routing_service
from algorithms.recommenders import data_loader

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger("uvicorn.error")
    logger.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logger.info("Logging configurado en nivel INFO")
    return logger


logger = configure_logging()

app = FastAPI(title="Biobío ML API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/metadata/options", response_model=MetadataResponse, tags=["meta"])
def metadata(service: RecommendationService = Depends(get_recommendation_service)) -> MetadataResponse:
    start = time.perf_counter()
    options = service.available_options()
    duration = (time.perf_counter() - start) * 1000
    logger.info("GET /metadata/options -> %d eventos (%.1f ms)", options.get("total_events", 0), duration)
    return MetadataResponse(**options)


@app.post(
    "/recommendations/association",
    response_model=AssociationResponse,
    tags=["recommendations"],
)
def association_recommendations(
    payload: AssociationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> AssociationResponse:
    start = time.perf_counter()
    recs = service.association_recommendations(payload)
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /recommendations/association -> %d recs (events=%s) in %.1f ms",
        len(recs),
        ",".join(payload.event_types),
        duration,
    )
    return AssociationResponse(recommendations=recs)


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
        raise HTTPException(status_code=404, detail="No se encontraron recomendaciones personalizadas.")
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /recommendations/collaborative -> %d recs (user=%s,strategy=%s) in %.1f ms",
        len(recs),
        payload.user_id,
        payload.strategy,
        duration,
    )
    return CollaborativeResponse(recommendations=recs)


@app.post("/routes/optimal", response_model=RouteResponse, tags=["routes"])
def optimal_route(
    payload: RouteRequest,
    service: RoutingService = Depends(get_routing_service),
) -> RouteResponse:
    start = time.perf_counter()
    path = service.compute_route(payload)
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /routes/optimal -> %.2f km (%d steps) in %.1f ms",
        path.distance_km,
        len(path.steps),
        duration,
    )
    return path
bootstrap_lock = threading.Lock()
bootstrap_state = {
    "status": "idle",
    "message": "Esperando ejecución",
    "percent": 0,
    "routing_nodes": 0,
    "routing_segments": 0,
    "duration_ms": 0.0,
}


def _run_bootstrap() -> None:
    start = time.perf_counter()
    try:
        with bootstrap_lock:
            bootstrap_state.update(status="running", message="Preparando recomendaciones...", percent=5)
        rec_service = get_recommendation_service()
        rec_service.available_options()
        with bootstrap_lock:
            bootstrap_state.update(message="Construyendo grafo de rutas...", percent=20)
        routing_service = get_routing_service()
        def progress(msg: str, frac: float) -> None:
            with bootstrap_lock:
                bootstrap_state.update(message=msg, percent=int(frac * 100))
        routing_service.build(progress=progress)
        nodes = len(routing_service.graph.nodes) if routing_service.graph else 0
        segments = len(routing_service.segment_lookup)
        duration = (time.perf_counter() - start) * 1000
        with bootstrap_lock:
            bootstrap_state.update(
                status="completed",
                message="Infraestructura lista",
                percent=100,
                routing_nodes=nodes,
                routing_segments=segments,
                duration_ms=round(duration, 1),
            )
        logger.info("Bootstrap completado: %d nodos, %d segmentos (%.1f ms)", nodes, segments, duration)
    except Exception as exc:  # pragma: no cover
        with bootstrap_lock:
            bootstrap_state.update(status="error", message=str(exc), percent=0)
        logger.exception("Falló el bootstrap")


@app.post("/system/bootstrap", tags=["meta"])
def bootstrap() -> dict:
    with bootstrap_lock:
        status = bootstrap_state.get("status")
        if status == "running":
            return bootstrap_state
        if status in {"idle", "completed", "error"}:
            bootstrap_state.update(status="running", message="Iniciando bootstrap...", percent=1)
            thread = threading.Thread(target=_run_bootstrap, daemon=True)
            thread.start()
            return bootstrap_state
    return bootstrap_state


@app.get("/system/bootstrap/status", tags=["meta"])
def bootstrap_status() -> dict:
    return bootstrap_state


@lru_cache(maxsize=8)
def _cached_hotspots(limit: int) -> List[tuple]:
    events = data_loader.load_raw_events()
    congestions = events[events["tipo_evento"] == "Congestión"].dropna(subset=["lat", "lon"])
    if congestions.empty:
        return []
    if len(congestions) > limit:
        congestions = congestions.sample(n=limit, random_state=42)
    speeds = congestions["velocidad_kmh"].clip(lower=5, upper=60).fillna(20)
    weights = (1 / speeds).values
    return list(zip(congestions["lat"].values, congestions["lon"].values, weights))


@app.get("/metadata/hotspots", response_model=HotspotResponse, tags=["meta"])
def metadata_hotspots(limit: int = 2000) -> HotspotResponse:
    limit = max(200, min(limit, 10000))
    points = _cached_hotspots(limit)
    logger.info("Hotspots solicitados -> %d puntos", len(points))
    return HotspotResponse(
        points=[
            {"lat": float(lat), "lon": float(lon), "weight": float(weight)}
            for lat, lon, weight in points
        ]
    )
