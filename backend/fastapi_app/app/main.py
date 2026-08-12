# -*- coding: utf-8 -*-
"""
FastAPI principal para exponer la demo academica y la superficie producto.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from algorithms.recommenders import data_loader, routing

from .core import dataset
from .core.demo_scenarios import DEMO_SCENARIOS
from .schemas.recommendations import (
    CollaborativeRequest,
    CollaborativeResponse,
    PlaygroundRequest,
    PlaygroundResponse,
)
from .schemas.routes import (
    CyclewayCoverage,
    ActiveMobilityEstimate,
    CongestionDateCoverageResponse,
    CongestionHourAvailabilityResponse,
    ContextualMobilityMessage,
    CyclewayResponse,
    EnvironmentalImpactResponse,
    HotspotPoint,
    HotspotResponse,
    MetadataResponse,
    Pm25SnapshotResponse,
    PlaceReverseResponse,
    PlaceSearchResponse,
    PlanRouteRequest,
    PlanRouteResponse,
    RegionBounds,
    RouteBadge,
    RoutePoint,
    RouteRequest,
    RouteResponse,
    RouteVariant,
    UserRouteAlert,
    UserRouteCard,
    UserRouteSummary,
    UrbanWellbeingResponse,
)
from .schemas.system import (
    BootstrapStatus,
    DataQualitySummary,
    DatasetInfo,
    DatasetStatus,
    DemoScenarioList,
    ReadinessStatus,
)
from .services import active_mobility_messages, cycleway_service, urban_wellbeing_service
from .services import data_quality_service
from .services.air_quality_service import get_air_quality_service
from .services.environmental_impact_service import get_environmental_impact_service
from .services.geocoding_service import (
    GeocodingConfigError,
    GeocodingLookupError,
    GeocodingService,
    get_geocoding_service,
)
from .services.plan_execution_service import PlanExecutionCoordinator
from .services.plan_cache_service import PlanResultCache
from .services.recommendation_service import RecommendationService, get_recommendation_service
from .services.routing_service import RoutingService, get_routing_service

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
AUTO_BOOTSTRAP_ENABLED = os.getenv("AUTO_BOOTSTRAP_ENABLED", "1") != "0"
TRAVEL_STYLE_TO_PROFILE = {
    "safe": "safety_focused",
    "balanced": "usuario_demo",
    "fast": "risk_taker",
}
BADGE_LABELS = {
    "base": "Mas corta",
    "recommended": "Recomendada",
    "fastest": "Llegar antes",
    "least_exposure": "Menor congestion",
    "least_congestion": "Circulación más fluida",
    "healthiest": "Menor exposición ambiental",
}
USER_ROUTE_LABELS = {
    "base": "Ruta mas corta",
    "recommended": "Ruta recomendada",
    "fastest": "Llegar antes",
    "least_exposure": "Menor congestion historica",
    "least_congested": "Circulación más fluida",
    "healthiest": "Menor exposición ambiental",
}
BICYCLE_SUGGESTION_TEXT = (
    "Esta ruta tiene buena cobertura de ciclovia y buena calidad del aire. "
    "Podrias considerar hacerla en bicicleta."
)
plan_execution_coordinator = PlanExecutionCoordinator[RouteResponse]()
plan_result_cache = PlanResultCache[PlanRouteResponse](max_entries=32, ttl_seconds=900)


def _plan_request_key(payload: PlanRouteRequest) -> tuple:
    return (
        data_loader.data_version(),
        payload.origin.lat,
        payload.origin.lon,
        payload.destination.lat,
        payload.destination.lon,
        payload.congestion_date,
        payload.day_of_week,
        payload.departure_hour,
        payload.avoid_congestion,
        payload.avoid_accidents,
    )


def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    uvicorn_logger.info("Logging configurado en nivel INFO")
    return uvicorn_logger


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
    get_geocoding_service.cache_clear()
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
            message="Infraestructura lista para planificar viajes",
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
        logger.exception("Fallo el bootstrap")


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


def _ensure_routing_ready(service: RoutingService) -> None:
    snapshot = _snapshot_bootstrap_state()
    if snapshot.get("status") != "completed" and service.graph is None:
        raise HTTPException(
            status_code=503,
            detail="El backend aun esta calentando el grafo del perfil activo. Consulta /readyz o /system/bootstrap/status.",
        )


def _build_preference_payload(items: list) -> list[dict]:
    return [
        {
            "via": item.via,
            "weight": round(float(item.estimated_rating) / 5.0, 3),
        }
        for item in items[:6]
    ]


def _build_route_request_from_plan(
    payload: PlanRouteRequest,
    recommendation_service: RecommendationService,
) -> RouteRequest:
    return RouteRequest(
        origin=payload.origin,
        destination=payload.destination,
        congestion_date=payload.congestion_date,
        preferences=[],
        ubcf_preferences=[],
        ibcf_preferences=[],
        day_of_week=payload.day_of_week,
        departure_hour=payload.departure_hour,
        avoid_congestion=payload.avoid_congestion,
        avoid_accidents=payload.avoid_accidents,
    )


def _risk_level(score: float) -> str:
    if score <= 10:
        return "low"
    if score <= 25:
        return "medium"
    return "high"


def _route_total_minutes(variant: RouteVariant) -> float:
    return round(float(variant.estimated_duration_min + variant.extra_delay_min), 1)


def _variant_alerts(variant: RouteVariant) -> list[UserRouteAlert]:
    alerts = [
        UserRouteAlert(
            title=f"{segment.event_type} en {segment.via}",
            detail=segment.reason,
            severity="medium",
        )
        for segment in variant.top_penalized_segments[:3]
    ]
    if alerts:
        return alerts
    if variant.incident_exposure.matched_incident_segments:
        return [
            UserRouteAlert(
                title="Congestion historica detectada",
                detail=(
                    f"La ruta cruza {variant.incident_exposure.matched_incident_segments} zonas con congestion historica "
                    "en el contexto elegido."
                ),
                severity="low",
            )
        ]
    return []


def _variant_summary_text(variant: RouteVariant) -> str:
    primary_reason = variant.why_changed[0] if variant.why_changed else "Trayecto calculado con el mejor ajuste disponible."
    return (
        f"{_route_total_minutes(variant):.1f} min en total, "
        f"{variant.incident_exposure.matched_incident_segments} zonas historicas en ruta. {primary_reason}"
    )


def _has_good_air_quality(variant: RouteVariant) -> bool:
    exposure = variant.pm25_exposure
    if exposure is None or not exposure.available:
        return False
    return exposure.category in {"Baja", "Media"} and exposure.average_pm25 < active_mobility_messages.HIGH_PM25_UG_M3


def _cycleway_coverage_for_variant(variant: RouteVariant) -> CyclewayCoverage:
    return CyclewayCoverage(**cycleway_service.estimate_route_coverage(variant.geometry))


def _bicycle_suggestion_for_variant(variant: RouteVariant, coverage: CyclewayCoverage) -> str | None:
    low_congestion = (
        variant.risk_score <= 10
        and variant.incident_exposure.matched_incident_segments <= 1
        and variant.incident_exposure.exposure_minutes <= 3.0
        and variant.extra_delay_min <= 3.0
    )
    if (
        coverage.available
        and coverage.has_high_coverage
        and _has_good_air_quality(variant)
        and low_congestion
        and variant.distance_km <= active_mobility_messages.BIKE_MAX_KM
    ):
        return BICYCLE_SUGGESTION_TEXT
    return None


def _contextual_messages_for_card(card: UserRouteCard) -> list[ContextualMobilityMessage]:
    return active_mobility_messages.build_route_messages(
        route_key=card.key,
        distance_km=card.distance_km,
        delay_min=card.delay_min,
        risk_level=card.risk_level,
        incident_exposure=card.incident_exposure,
        pm25_exposure=card.pm25_exposure,
        cycleway_coverage=card.cycleway_coverage,
        active_mobility_estimate=card.active_mobility_estimate,
    )


def _active_mobility_estimate_for_card(card: UserRouteCard) -> ActiveMobilityEstimate:
    return active_mobility_messages.estimate_active_travel_times(
        distance_km=card.distance_km,
        auto_min=card.duration_min,
        cycleway_coverage=card.cycleway_coverage,
    )


def _bounds_from_points(points: list[RoutePoint]) -> RegionBounds:
    if not points:
        return RegionBounds(lat_min=0.0, lat_max=0.0, lon_min=0.0, lon_max=0.0)
    lats = [point.lat for point in points]
    lons = [point.lon for point in points]
    return RegionBounds(
        lat_min=min(lats),
        lat_max=max(lats),
        lon_min=min(lons),
        lon_max=max(lons),
    )


def _expand_bounds(bounds: RegionBounds, pad: float = 0.003) -> tuple[float, float, float, float]:
    return (
        bounds.lon_min - pad,
        bounds.lat_min - pad,
        bounds.lon_max + pad,
        bounds.lat_max + pad,
    )


def _variant_lookup(route: RouteResponse) -> dict[str, RouteVariant]:
    variants = {
        "reference": route.reference,
        "ubcf": route.ubcf,
        "ibcf": route.ibcf,
    }
    if route.least_congestion is not None:
        variants["least_congestion"] = route.least_congestion
    if route.healthiest is not None:
        variants["healthiest"] = route.healthiest
    if route.personalized is not None:
        variants["personalized"] = route.personalized
    return variants


def _healthiest_variant_key(variants: dict[str, RouteVariant], fallback_key: str) -> str:
    available = [
        (key, variant.pm25_exposure.average_pm25)
        for key, variant in variants.items()
        if variant.pm25_exposure is not None
    ]
    if not available:
        return fallback_key
    return min(available, key=lambda item: item[1])[0]


def _filter_hotspots(
    *,
    limit: int,
    bbox: tuple[float, float, float, float] | None = None,
    day_of_week: str | None = None,
    departure_hour: float | None = None,
) -> list[dict]:
    points = _cached_hotspots(10000)
    normalized_day = day_of_week.strip().lower() if day_of_week else None
    target_bucket = data_loader.hour_bucket(departure_hour) if departure_hour is not None else None
    filtered: list[dict] = []
    for point in points:
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            if point["lon"] < min_lon or point["lon"] > max_lon or point["lat"] < min_lat or point["lat"] > max_lat:
                continue
        if normalized_day and str(point.get("day") or "").strip().lower() not in {"", normalized_day}:
            continue
        if target_bucket and str(point.get("bucket") or "") not in {"", target_bucket}:
            continue
        filtered.append(point)
        if len(filtered) >= limit:
            break
    return filtered


def _parse_bbox_param(raw_bbox: str | None) -> tuple[float, float, float, float] | None:
    if not raw_bbox:
        return None
    try:
        min_lon, min_lat, max_lon, max_lat = [float(part.strip()) for part in raw_bbox.split(",")]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="bbox debe venir como minLon,minLat,maxLon,maxLat.") from exc
    if min_lon > max_lon or min_lat > max_lat:
        raise HTTPException(status_code=400, detail="bbox invalido: revisa el orden de coordenadas.")
    return (min_lon, min_lat, max_lon, max_lat)


def _build_plan_response(route: RouteResponse, payload: PlanRouteRequest) -> PlanRouteResponse:
    variants = _variant_lookup(route)
    fastest_variant = route.comparison.fastest_variant if route.comparison.fastest_variant in variants else "reference"
    least_congestion_variant = "least_congestion" if "least_congestion" in variants else "reference"
    healthiest_variant = (
        "healthiest"
        if "healthiest" in variants
        else _healthiest_variant_key(variants, fallback_key=least_congestion_variant)
    )
    semantic_targets = [
        ("fastest", "fastest", fastest_variant),
        ("least_congested", "least_congestion", least_congestion_variant),
        ("healthiest", "healthiest", healthiest_variant),
    ]
    routes: list[UserRouteCard] = []
    visible_geometry_labels: dict[tuple[tuple[float, float], ...], str] = {}
    for route_key, badge_key, variant_key in semantic_targets:
        variant = variants[variant_key]
        badge = RouteBadge(key=badge_key, label=BADGE_LABELS[badge_key])
        cycleway_coverage = _cycleway_coverage_for_variant(variant)
        geometry_key = tuple((round(point.lat, 6), round(point.lon, 6)) for point in variant.geometry)
        why_changed = list(variant.why_changed)
        matching_label = visible_geometry_labels.get(geometry_key)
        if matching_label is not None:
            explanation = (
                f"Coincide con {matching_label}: el mismo trayecto obtuvo el mejor resultado "
                f"para ambos criterios y no se encontro una alternativa valida que lo mejorara."
            )
            if explanation not in why_changed:
                why_changed.insert(0, explanation)
        else:
            visible_geometry_labels[geometry_key] = USER_ROUTE_LABELS[route_key]
        card = UserRouteCard(
            key=route_key,
            label=USER_ROUTE_LABELS[route_key],
            badges=[badge],
            duration_min=_route_total_minutes(variant),
            distance_km=round(variant.distance_km, 2),
            delay_min=round(variant.extra_delay_min, 1),
            congestion_score=round(variant.risk_score, 1),
            risk_level=_risk_level(variant.risk_score),
            summary=_variant_summary_text(variant),
            geometry=variant.geometry,
            road_geometry=variant.road_geometry,
            access_geometry=variant.access_geometry,
            top_alerts=_variant_alerts(variant),
            why_changed=why_changed,
            top_penalized_segments=variant.top_penalized_segments,
            top_preferred_vias=variant.top_preferred_vias,
            congestion_coverage=variant.congestion_coverage,
            incident_exposure=variant.incident_exposure,
            pm25_exposure=variant.pm25_exposure,
            urban_wellbeing=variant.urban_wellbeing,
            healthy_route_score=variant.healthy_route_score,
            optimization_trace=variant.optimization_trace,
            cycleway_coverage=cycleway_coverage,
            bicycle_suggestion=_bicycle_suggestion_for_variant(variant, cycleway_coverage),
        )
        card.active_mobility_estimate = _active_mobility_estimate_for_card(card)
        card.contextual_messages = _contextual_messages_for_card(card)
        routes.append(card)
    selected = next((item for item in routes if item.key == "least_congested"), routes[0])
    all_points = [point for item in routes for point in item.geometry]
    bounds = _bounds_from_points(all_points)
    hotspot_bbox = _expand_bounds(bounds)
    hotspots = [
        HotspotPoint(**point)
        for point in _filter_hotspots(
            limit=80,
            bbox=hotspot_bbox,
            day_of_week=payload.day_of_week,
            departure_hour=payload.departure_hour,
        )
    ]
    summary = UserRouteSummary(
        eta_total_min=selected.duration_min,
        distance_km=selected.distance_km,
        delay_min=selected.delay_min,
        alerts_on_route=selected.incident_exposure.matched_incident_segments,
        main_reason=selected.why_changed[0] if selected.why_changed else selected.summary,
    )
    contextual_messages = active_mobility_messages.select_plan_messages(routes, selected.key)
    return PlanRouteResponse(
        selected_route_key=selected.key,
        routes=routes,
        routes_by_type={item.key: item for item in routes},
        summary=summary,
        alerts=selected.top_alerts,
        contextual_messages=contextual_messages,
        hotspots=hotspots,
        map_bounds=bounds,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if AUTO_BOOTSTRAP_ENABLED:
        _start_bootstrap_thread(force=True)
    yield


app = FastAPI(title="Biobio ML API", version="0.3.0", lifespan=lifespan)

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
        "Backend listo para planificar viajes."
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
    _ensure_routing_ready(service)
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


@app.post("/routes/plan", response_model=PlanRouteResponse, tags=["routes"])
async def plan_route(
    payload: PlanRouteRequest,
    request: Request,
    routing_service: RoutingService = Depends(get_routing_service),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> PlanRouteResponse:
    _ensure_routing_ready(routing_service)
    start = time.perf_counter()
    lease = None
    client_cancelled = False
    request_key = _plan_request_key(payload)
    cached_response = plan_result_cache.get(request_key)
    if cached_response is not None:
        logger.info("POST /routes/plan -> resultado servido desde cache")
        return cached_response
    try:
        internal_payload = _build_route_request_from_plan(payload, recommendation_service)
        lease = await plan_execution_coordinator.acquire(
            request_key,
            lambda should_cancel: routing_service.compute_route(internal_payload, should_cancel),
        )
        route_task = lease.task
        while not route_task.done():
            await asyncio.wait({route_task}, timeout=0.1)
            if not route_task.done() and await request.is_disconnected():
                client_cancelled = True
                logger.info("POST /routes/plan cancelado por desconexión del cliente")
                raise HTTPException(status_code=499, detail="Planificación cancelada.")
        route = route_task.result()
    except asyncio.CancelledError:
        client_cancelled = True
        raise
    except routing.RouteSearchCancelled as exc:
        raise HTTPException(status_code=499, detail="Planificación cancelada.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if lease is not None:
            await lease.release(cancelled=client_cancelled)
    response = _build_plan_response(route, payload)
    plan_result_cache.set(request_key, response)
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /routes/plan -> rutas=%d en %.1f ms",
        len(response.routes),
        duration,
    )
    return response


@app.post("/system/bootstrap", response_model=BootstrapStatus, tags=["meta"])
def bootstrap() -> BootstrapStatus:
    plan_result_cache.clear()
    return BootstrapStatus(**_start_bootstrap_thread(force=True))


@app.get("/system/bootstrap/status", response_model=BootstrapStatus, tags=["meta"])
def bootstrap_status() -> BootstrapStatus:
    return BootstrapStatus(**_snapshot_bootstrap_state())


_hotspot_cache = {"signature": None, "points": []}
_hotspot_cache_lock = threading.Lock()
CONGESTION_COVERAGE_FILES = {
    "gran_concepcion": data_loader.ROOT_DIR / "data_analysis" / "congestion_aggregated_gran_concepcion_core.csv",
    "regional": data_loader.ROOT_DIR / "data_analysis" / "congestion_aggregated.csv",
}
RAIN_DAILY_PATH = data_loader.ROOT_DIR / "data_processed" / "gran_concepcion_rain_daily.csv"


def _build_hotspot_points(limit: int = 10000) -> List[dict]:
    events = data_loader.load_congestion_events()
    event_type_series = (
        events["tipo_evento"]
        .fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.strip()
        .str.lower()
    )
    congestions = events[event_type_series == "congestion"].dropna(subset=["lat", "lon"]).head(limit)
    if congestions.empty:
        return []
    empty = pd.Series(index=congestions.index, dtype=object)
    start = pd.to_datetime(congestions.get("hora_inicio", empty), format="%H:%M", errors="coerce")
    end = pd.to_datetime(congestions.get("hora_fin", empty), format="%H:%M", errors="coerce")
    speed = pd.to_numeric(congestions.get("velocidad_kmh", empty), errors="coerce")
    valid_speed = speed.notna() & (speed > 0)
    weights = pd.Series(0.5, index=congestions.index, dtype=float)
    weights.loc[valid_speed] = (1.0 / speed.loc[valid_speed].clip(lower=5.0)).clip(lower=0.1, upper=2.0)

    def text_column(name: str) -> pd.Series:
        return congestions.get(name, empty).fillna("").astype(str)

    result = pd.DataFrame(
        {
            "lat": pd.to_numeric(congestions["lat"], errors="coerce"),
            "lon": pd.to_numeric(congestions["lon"], errors="coerce"),
            "weight": weights,
            "day": text_column("dia_semana"),
            "bucket": text_column("franja_horaria"),
            "segment_id": text_column("segment_id"),
            "hora_inicio_float": (start.dt.hour + start.dt.minute / 60.0).astype(object).where(start.notna(), None),
            "hora_fin_float": (end.dt.hour + end.dt.minute / 60.0).astype(object).where(end.notna(), None),
        }
    )
    return result.to_dict(orient="records")


def _cached_hotspots(limit: int) -> List[dict]:
    limit = max(0, limit)
    signature = data_loader.data_version()
    with _hotspot_cache_lock:
        if _hotspot_cache["signature"] != signature:
            _hotspot_cache["points"] = _build_hotspot_points(limit)
            _hotspot_cache["signature"] = signature
        return list(_hotspot_cache["points"][:limit])


@lru_cache(maxsize=4)
def _load_coverage_date_rows(path: str, modified_ns: int, size: int) -> tuple[pd.Series, str]:
    del modified_ns, size
    coverage_path = data_loader.ROOT_DIR / path
    columns = pd.read_csv(coverage_path, nrows=0).columns
    date_column = "fecha_dia_dt" if "fecha_dia_dt" in columns else "fecha"
    df = pd.read_csv(coverage_path, usecols=[date_column])
    return df[date_column].drop_duplicates().reset_index(drop=True), coverage_path.name


def _load_congestion_date_rows() -> tuple[pd.Series, str]:
    profile = data_loader.get_data_profile()
    coverage_path = CONGESTION_COVERAGE_FILES.get(profile)
    if coverage_path is not None and coverage_path.exists():
        stat = coverage_path.stat()
        relative_path = str(coverage_path.relative_to(data_loader.ROOT_DIR))
        return _load_coverage_date_rows(relative_path, stat.st_mtime_ns, stat.st_size)

    events = data_loader.load_congestion_events()
    if events.empty or "fecha" not in events.columns:
        return pd.Series(dtype=str), data_loader.CONGESTION_PATH.name
    return events["fecha"], data_loader.CONGESTION_PATH.name


@lru_cache(maxsize=4)
def _load_coverage_hour_rows(path: str, modified_ns: int, size: int) -> tuple[pd.DataFrame, str]:
    del modified_ns, size
    coverage_path = data_loader.ROOT_DIR / path
    columns = pd.read_csv(coverage_path, nrows=0).columns
    date_column = "fecha_dia_dt" if "fecha_dia_dt" in columns else "fecha"
    hour_column = "hora" if "hora" in columns else "periodo_hora"
    if date_column not in columns or hour_column not in columns:
        return pd.DataFrame(columns=["date", "hour"]), coverage_path.name
    rows = pd.read_csv(coverage_path, usecols=[date_column, hour_column])
    parsed_hours = (
        pd.to_datetime(rows[hour_column], format="%H:%M", errors="coerce")
        if hour_column == "hora"
        else pd.to_datetime(rows[hour_column], errors="coerce")
    )
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(rows[date_column], errors="coerce").dt.strftime("%Y-%m-%d"),
            "hour": parsed_hours.dt.hour,
        }
    ).dropna()
    return result, coverage_path.name


def _load_congestion_hour_rows() -> tuple[pd.DataFrame, str]:
    profile = data_loader.get_data_profile()
    coverage_path = CONGESTION_COVERAGE_FILES.get(profile)
    if coverage_path is not None and coverage_path.exists():
        stat = coverage_path.stat()
        relative_path = str(coverage_path.relative_to(data_loader.ROOT_DIR))
        return _load_coverage_hour_rows(relative_path, stat.st_mtime_ns, stat.st_size)

    events = data_loader.load_congestion_events()
    if events.empty or "fecha" not in events.columns or "hora_inicio" not in events.columns:
        return pd.DataFrame(columns=["date", "hour"]), data_loader.CONGESTION_PATH.name
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(events["fecha"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "hour": pd.to_datetime(events["hora_inicio"], format="%H:%M", errors="coerce").dt.hour,
        }
    ).dropna()
    return result, data_loader.CONGESTION_PATH.name


@lru_cache(maxsize=4)
def _load_rain_dates(path: str, modified_ns: int, size: int) -> list[str]:
    del modified_ns, size
    rain_path = data_loader.ROOT_DIR / path
    if not rain_path.exists():
        return []
    columns = pd.read_csv(rain_path, nrows=0).columns
    if "date" not in columns:
        return []
    usecols = ["date"]
    if "rain_mm" in columns:
        usecols.append("rain_mm")
    if "wet_hours" in columns:
        usecols.append("wet_hours")
    daily = pd.read_csv(rain_path, usecols=usecols)
    if daily.empty:
        return []
    rain_signal = pd.Series(False, index=daily.index)
    if "rain_mm" in daily.columns:
        rain_signal = rain_signal | (pd.to_numeric(daily["rain_mm"], errors="coerce").fillna(0.0) > 0.0)
    if "wet_hours" in daily.columns:
        rain_signal = rain_signal | (pd.to_numeric(daily["wet_hours"], errors="coerce").fillna(0.0) > 0)
    dates = pd.to_datetime(daily.loc[rain_signal, "date"], errors="coerce").dropna().dt.normalize()
    return sorted({value.date().isoformat() for value in dates})


def _rain_dates() -> list[str]:
    if not RAIN_DAILY_PATH.exists():
        return []
    stat = RAIN_DAILY_PATH.stat()
    relative_path = str(RAIN_DAILY_PATH.relative_to(data_loader.ROOT_DIR))
    return _load_rain_dates(relative_path, stat.st_mtime_ns, stat.st_size)


@app.get("/metadata/hotspots", response_model=HotspotResponse, tags=["meta"])
def metadata_hotspots(
    limit: int = 2000,
    bbox: str | None = Query(default=None),
    day_of_week: str | None = Query(default=None),
    departure_hour: float | None = Query(default=None, ge=0.0, le=24.0),
) -> HotspotResponse:
    limit = max(200, min(limit, 10000))
    parsed_bbox = _parse_bbox_param(bbox)
    points = _filter_hotspots(
        limit=limit,
        bbox=parsed_bbox,
        day_of_week=day_of_week,
        departure_hour=departure_hour,
    )
    logger.info("Hotspots solicitados -> %d puntos", len(points))
    return HotspotResponse(points=[HotspotPoint(**point) for point in points])


@app.get("/metadata/congestion/dates", response_model=CongestionDateCoverageResponse, tags=["meta"])
def metadata_congestion_dates() -> CongestionDateCoverageResponse:
    raw_dates, data_source = _load_congestion_date_rows()
    if raw_dates.empty:
        return CongestionDateCoverageResponse(
            available_dates=[],
            missing_dates=[],
            rain_dates=_rain_dates(),
            available_days=0,
            calendar_days=0,
            data_source=data_source,
        )

    dates = pd.to_datetime(raw_dates, errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return CongestionDateCoverageResponse(
            available_dates=[],
            missing_dates=[],
            rain_dates=_rain_dates(),
            available_days=0,
            calendar_days=0,
            data_source=data_source,
        )

    available_dates = sorted({value.date().isoformat() for value in dates})
    start = available_dates[0]
    end = available_dates[-1]
    calendar_range = pd.date_range(start=start, end=end, freq="D")
    available_set = set(available_dates)
    missing_dates = [value.date().isoformat() for value in calendar_range if value.date().isoformat() not in available_set]
    return CongestionDateCoverageResponse(
        start=start,
        end=end,
        available_dates=available_dates,
        missing_dates=missing_dates,
        rain_dates=_rain_dates(),
        available_days=len(available_dates),
        calendar_days=len(calendar_range),
        data_source=data_source,
    )


@app.get("/metadata/congestion/hours", response_model=CongestionHourAvailabilityResponse, tags=["meta"])
def metadata_congestion_hours(
    date: str = Query(..., description="Fecha historica local en formato YYYY-MM-DD"),
) -> CongestionHourAvailabilityResponse:
    parsed_date = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed_date) or parsed_date.strftime("%Y-%m-%d") != date:
        raise HTTPException(status_code=422, detail="date debe usar el formato YYYY-MM-DD.")

    rows, data_source = _load_congestion_hour_rows()
    matching = rows.loc[rows["date"] == date, "hour"] if not rows.empty else pd.Series(dtype=float)
    available_hours = sorted(
        {
            int(hour)
            for hour in pd.to_numeric(matching, errors="coerce").dropna()
            if 0 <= int(hour) <= 23
        }
    )
    return CongestionHourAvailabilityResponse(
        date=date,
        available_hours=available_hours,
        count=len(available_hours),
        data_source=data_source,
    )


@app.get("/metadata/cycleways", response_model=CyclewayResponse, tags=["meta"])
def metadata_cycleways() -> CyclewayResponse:
    payload = cycleway_service.load_cycleways()
    logger.info("Ciclovias solicitadas -> %d segmentos", len(payload.get("features", [])))
    return CyclewayResponse(**payload)


@app.get("/metadata/urban-wellbeing", response_model=UrbanWellbeingResponse, tags=["meta"])
def metadata_urban_wellbeing() -> UrbanWellbeingResponse:
    payload = urban_wellbeing_service.load_wellbeing_features()
    logger.info("Bienestar urbano solicitado -> %d elementos", len(payload.get("features", [])))
    return UrbanWellbeingResponse(**payload)


@app.get("/metadata/pm25/snapshot", response_model=Pm25SnapshotResponse, tags=["meta"])
def metadata_pm25_snapshot(
    date: str = Query(..., description="Fecha historica 2025 en formato YYYY-MM-DD"),
    hour: int = Query(..., ge=0, le=23, description="Hora local completa, 0-23"),
) -> Pm25SnapshotResponse:
    try:
        snapshot = get_air_quality_service().station_snapshot(date, hour)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Snapshot PM2.5 solicitado -> %s estaciones=%d", snapshot.requested_at, len(snapshot.stations))
    return snapshot


@app.get("/metadata/environmental-impact", response_model=EnvironmentalImpactResponse, tags=["meta"])
def metadata_environmental_impact(
    date: str = Query(..., description="Fecha historica 2025 en formato YYYY-MM-DD"),
    hour: int = Query(..., ge=0, le=23, description="Hora local completa, 0-23"),
) -> EnvironmentalImpactResponse:
    try:
        snapshot = get_environmental_impact_service().build_snapshot(date, hour)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "Impacto ambiental solicitado -> %s puntos=%d",
        snapshot.summary.requested_at,
        len(snapshot.points),
    )
    return snapshot


@app.get("/places/search", response_model=PlaceSearchResponse, tags=["places"])
def places_search(
    q: str,
    limit: int = Query(default=5, ge=1, le=10),
    service: GeocodingService = Depends(get_geocoding_service),
) -> PlaceSearchResponse:
    query = q.strip()
    if len(query) < 2:
        raise HTTPException(status_code=400, detail="q debe tener al menos 2 caracteres.")
    try:
        results = service.search(query, limit=limit)
    except GeocodingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeocodingLookupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PlaceSearchResponse(results=results)


@app.get("/places/reverse", response_model=PlaceReverseResponse, tags=["places"])
def places_reverse(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    service: GeocodingService = Depends(get_geocoding_service),
) -> PlaceReverseResponse:
    try:
        result = service.reverse(lat=lat, lon=lon)
    except GeocodingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeocodingLookupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return PlaceReverseResponse(result=result)


def _frontend_dist_dir() -> Path | None:
    candidates = [
        Path(os.getenv("FRONTEND_DIST", "/app/frontend_dist")),
        Path(__file__).resolve().parents[3] / "frontend" / "react_app" / "dist",
    ]
    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate
    return None


frontend_dist = _frontend_dist_dir()
if frontend_dist is not None:
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    def serve_frontend_index() -> FileResponse:
        return FileResponse(frontend_dist / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str) -> FileResponse:
        requested_file = frontend_dist / full_path
        if requested_file.is_file():
            return FileResponse(requested_file)
        return FileResponse(frontend_dist / "index.html")
