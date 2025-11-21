# -*- coding: utf-8 -*-
"""
FastAPI principal para exponer las recomendaciones y rutas.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from typing import List, Tuple

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas.recommendations import (
    CollaborativeRequest,
    CollaborativeResponse,
    PlaygroundRequest,
    PlaygroundResponse,
)
from .schemas.routes import HotspotResponse, MetadataResponse, RouteRequest, RouteResponse
from .schemas.system import DatasetChangeRequest, DatasetStatus, DatasetInfo
from .services.recommendation_service import RecommendationService, get_recommendation_service
from .services.routing_service import RoutingService, get_routing_service
from .core import dataset
from .core.exceptions import (
    WazeBiobioException,
    NoRouteFoundException,
    InvalidCoordinatesException,
    UserNotFoundException,
    InvalidStrategyException,
    DataNotLoadedException,
)
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


# Exception handler for custom exceptions
@app.exception_handler(WazeBiobioException)
async def waze_biobio_exception_handler(request: Request, exc: WazeBiobioException):
    """Handle custom Waze Biobío exceptions."""
    logger.error(f"WazeBiobioException: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__},
    )


# Exception handler for validation errors
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc), "error_type": "ValueError"},
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


def _build_dataset_status() -> DatasetStatus:
    available = [DatasetInfo(key=key, label=label) for key, label in dataset.available_profiles()]
    return DatasetStatus(current=dataset.get_profile(), available=available)


@app.get("/system/dataset", response_model=DatasetStatus, tags=["meta"])
def dataset_status() -> DatasetStatus:
    return _build_dataset_status()


@app.post("/system/dataset", response_model=DatasetStatus, tags=["meta"])
def dataset_switch(payload: DatasetChangeRequest) -> DatasetStatus:
    dataset.set_profile(payload.profile)
    get_recommendation_service.cache_clear()
    logger.info("Perfil de datos actualizado a %s", payload.profile)
    return _build_dataset_status()


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
    """
    Calculate optimal route between two points.

    Validates coordinates and handles routing errors gracefully.
    """
    start = time.perf_counter()

    # Additional validation beyond Pydantic
    origin = payload.origin
    destination = payload.destination

    # Check if coordinates are within reasonable bounds for Biobío region
    # Approximate bounds: lat [-38, -36], lon [-73.5, -71.5]
    if not (-39 < origin.lat < -35):
        raise InvalidCoordinatesException(
            origin.lat, origin.lon,
            "El origen está fuera de la región del Biobío"
        )
    if not (-39 < destination.lat < -35):
        raise InvalidCoordinatesException(
            destination.lat, destination.lon,
            "El destino está fuera de la región del Biobío"
        )
    if not (-74 < origin.lon < -71):
        raise InvalidCoordinatesException(
            origin.lat, origin.lon,
            "El origen está fuera de la región del Biobío"
        )
    if not (-74 < destination.lon < -71):
        raise InvalidCoordinatesException(
            destination.lat, destination.lon,
            "El destino está fuera de la región del Biobío"
        )

    # Check if service is properly initialized
    if not service.graph or not service.segment_lookup:
        raise DataNotLoadedException("grafo de rutas")

    try:
        route = service.compute_route(payload)
    except Exception as exc:
        logger.exception("Error calculando ruta")
        # If it's not a custom exception, wrap it
        if not isinstance(exc, WazeBiobioException):
            raise NoRouteFoundException(
                (origin.lat, origin.lon),
                (destination.lat, destination.lon)
            )
        raise

    duration = (time.perf_counter() - start) * 1000
    reference = route.reference
    personalized = route.personalized
    distance_base = reference.distance_km if reference else 0.0
    distance_personalized = personalized.distance_km if personalized else distance_base
    logger.info(
        "POST /routes/optimal -> base=%.2f km personalizada=%.2f km en %.1f ms",
        distance_base,
        distance_personalized,
        duration,
    )
    return route

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
