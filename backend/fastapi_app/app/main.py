# -*- coding: utf-8 -*-
"""
FastAPI principal para exponer la demo academica y la superficie producto.
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
from fastapi import Depends, FastAPI, HTTPException, Query
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
from .schemas.routes import (
    HotspotPoint,
    HotspotResponse,
    MetadataResponse,
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
)
from .schemas.system import (
    BootstrapStatus,
    DataQualitySummary,
    DatasetInfo,
    DatasetStatus,
    DemoScenarioList,
    ReadinessStatus,
)
from .services import data_quality_service
from .services.geocoding_service import (
    GeocodingConfigError,
    GeocodingLookupError,
    GeocodingService,
    get_geocoding_service,
)
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
    "base": "Base",
    "recommended": "Recomendada",
    "fastest": "Mas rapida",
    "least_exposure": "Menor congestion",
    "least_congestion": "Menor congestion",
    "healthiest": "Mas saludable",
}
USER_ROUTE_LABELS = {
    "base": "Ruta base",
    "recommended": "Ruta recomendada",
    "fastest": "Ruta mas rapida",
    "least_exposure": "Menor congestion historica",
    "least_congestion": "Menor congestion",
    "healthiest": "Mas saludable",
}


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
        preferences=[],
        ubcf_preferences=[],
        ibcf_preferences=[],
        day_of_week=payload.day_of_week,
        departure_hour=payload.departure_hour,
        avoid_congestion=payload.avoid_congestion,
        avoid_accidents=False,
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


def _route_signature(variant: RouteVariant) -> tuple:
    geometry = tuple((round(point.lat, 5), round(point.lon, 5)) for point in variant.geometry)
    return (
        geometry,
        round(variant.distance_km, 2),
        round(_route_total_minutes(variant), 1),
        round(variant.risk_score, 1),
    )


def _variant_lookup(route: RouteResponse) -> dict[str, RouteVariant]:
    variants = {
        "reference": route.reference,
        "ubcf": route.ubcf,
        "ibcf": route.ibcf,
    }
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
    least_congestion_variant = route.comparison.lowest_exposure_variant
    healthiest_variant = _healthiest_variant_key(variants, fallback_key=least_congestion_variant)
    semantic_targets = [
        ("base", "reference"),
        ("least_congestion", least_congestion_variant),
        ("healthiest", healthiest_variant),
    ]
    grouped_routes: dict[object, UserRouteCard] = {}
    cards_by_signature: dict[tuple, UserRouteCard] = {}
    route_order: list[object] = []
    for badge_key, variant_key in semantic_targets:
        variant = variants[variant_key]
        signature = _route_signature(variant)
        badge = RouteBadge(key=badge_key, label=BADGE_LABELS[badge_key])
        keep_separate = badge_key in {"base", "least_congestion", "healthiest"}
        route_key = (badge_key, signature) if keep_separate else signature
        if not keep_separate and signature in cards_by_signature:
            cards_by_signature[signature].badges.append(badge)
            continue
        card = UserRouteCard(
            key=badge_key,
            label=USER_ROUTE_LABELS[badge_key],
            badges=[badge],
            duration_min=_route_total_minutes(variant),
            distance_km=round(variant.distance_km, 2),
            delay_min=round(variant.extra_delay_min, 1),
            risk_level=_risk_level(variant.risk_score),
            summary=_variant_summary_text(variant),
            geometry=variant.geometry,
            top_alerts=_variant_alerts(variant),
            why_changed=variant.why_changed,
            top_penalized_segments=variant.top_penalized_segments,
            top_preferred_vias=variant.top_preferred_vias,
            incident_exposure=variant.incident_exposure,
            pm25_exposure=variant.pm25_exposure,
        )
        grouped_routes[route_key] = card
        cards_by_signature.setdefault(signature, card)
        route_order.append(route_key)

    routes = [grouped_routes[signature] for signature in route_order]
    selected = next((item for item in routes if any(badge.key == "least_congestion" for badge in item.badges)), routes[0])
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
    return PlanRouteResponse(
        selected_route_key=selected.key,
        routes=routes,
        summary=summary,
        alerts=selected.top_alerts,
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
def plan_route(
    payload: PlanRouteRequest,
    routing_service: RoutingService = Depends(get_routing_service),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
) -> PlanRouteResponse:
    _ensure_routing_ready(routing_service)
    start = time.perf_counter()
    try:
        internal_payload = _build_route_request_from_plan(payload, recommendation_service)
        route = routing_service.compute_route(internal_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    response = _build_plan_response(route, payload)
    duration = (time.perf_counter() - start) * 1000
    logger.info(
        "POST /routes/plan -> rutas=%d estilo=%s en %.1f ms",
        len(response.routes),
        payload.travel_style,
        duration,
    )
    return response


@app.post("/system/bootstrap", response_model=BootstrapStatus, tags=["meta"])
def bootstrap() -> BootstrapStatus:
    return BootstrapStatus(**_start_bootstrap_thread(force=True))


@app.get("/system/bootstrap/status", response_model=BootstrapStatus, tags=["meta"])
def bootstrap_status() -> BootstrapStatus:
    return BootstrapStatus(**_snapshot_bootstrap_state())


_hotspot_cache = {"signature": None, "points": []}
_hotspot_cache_lock = threading.Lock()


def _build_hotspot_points() -> List[dict]:
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
    congestions = events[event_type_series == "congestion"].dropna(subset=["lat", "lon"])
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
