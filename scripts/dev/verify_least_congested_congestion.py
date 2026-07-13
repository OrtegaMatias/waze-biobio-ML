# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import LineString
from sklearn.neighbors import BallTree

PROJECT_ROOT = Path(os.environ.get("APP_ROOT", ""))
if not str(PROJECT_ROOT) or not PROJECT_ROOT.exists():
    candidate = Path("/app")
    PROJECT_ROOT = candidate if candidate.exists() else Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.fastapi_app.app import main  # noqa: E402
from backend.fastapi_app.app.schemas.routes import RoutePoint, RouteRequest  # noqa: E402
from backend.fastapi_app.app.services.environmental_impact_service import (  # noqa: E402
    get_environmental_impact_service,
)
from backend.fastapi_app.app.services import routing_service as routing_service_module  # noqa: E402
from backend.fastapi_app.app.services.routing_service import (  # noqa: E402
    ACTIVE_CONGESTION_MATCH_TOLERANCE_M,
    get_routing_service,
)

METERS_PER_DEGREE_LAT = 111_320.0


@dataclass(frozen=True)
class RoutePair:
    name: str
    origin: tuple[float, float]
    destination: tuple[float, float]


@dataclass
class ProblemCase:
    kind: str
    date: str
    hour: int
    pair: str
    segment_id: str | None
    via: str | None
    distance_m: float | None
    crossing_type: str | None
    congestion_score: float
    matched_incident_segments: int
    top_penalized_segments: list[dict]
    detail: str


@dataclass
class RouteCheckResult:
    geometry: list[RoutePoint]
    road_geometry: list[RoutePoint]
    risk_score: float
    incident_exposure: object
    top_penalized_segments: list


DEFAULT_ROUTE_PAIRS = [
    RoutePair(
        name="colo_colo_regression",
        origin=(-36.8328, -73.0523),
        destination=(-36.8245, -73.0474),
    ),
    RoutePair(
        name="centro_concepcion_sw_ne",
        origin=(-36.8335, -73.0610),
        destination=(-36.8233, -73.0442),
    ),
    RoutePair(
        name="centro_concepcion_w_e",
        origin=(-36.8287, -73.0637),
        destination=(-36.8257, -73.0398),
    ),
    RoutePair(
        name="udec_plaza_peru_a_plaza_independencia",
        origin=(-36.8273, -73.0389),
        destination=(-36.8271, -73.0503),
    ),
    RoutePair(
        name="costanera_a_centro",
        origin=(-36.8188, -73.0642),
        destination=(-36.8295, -73.0460),
    ),
    RoutePair(
        name="hualpen_centro_concepcion",
        origin=(-36.7912, -73.0972),
        destination=(-36.8274, -73.0499),
    ),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verifica que la ruta least_congested de /routes/plan no cruce lineas rojas "
            "actuales de congestion_lines a 45 m o menos."
        )
    )
    parser.add_argument("--date", dest="dates", action="append", help="Fecha YYYY-MM-DD. Repetible.")
    parser.add_argument("--hour", dest="hours", type=int, action="append", help="Hora 0-23. Repetible.")
    parser.add_argument("--max-dates", type=int, default=None, help="Limita fechas para corridas rapidas.")
    parser.add_argument("--only-pair", action="append", choices=[pair.name for pair in DEFAULT_ROUTE_PAIRS])
    parser.add_argument("--origin", help="Par manual lat,lon. Ejemplo: --origin -36.8321,-73.0521")
    parser.add_argument("--destination", help="Par manual lat,lon. Ejemplo: --destination -36.8243,-73.0497")
    parser.add_argument(
        "--use-full-context",
        action="store_true",
        help=(
            "Usa PM2.5, bienestar urbano y ciclovias reales. Por defecto se neutralizan porque no "
            "cambian la ruta least_congested y hacen el barrido exhaustivo mucho mas lento."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "cache" / "least_congested_congestion_verification.json",
        help="Ruta del reporte JSON.",
    )
    return parser.parse_args()


def _parse_lat_lon(value: str, *, label: str) -> tuple[float, float]:
    try:
        lat_text, lon_text = value.split(",", maxsplit=1)
        return (float(lat_text.strip()), float(lon_text.strip()))
    except ValueError as exc:
        raise SystemExit(f"{label} debe venir como lat,lon.") from exc


class _NeutralAirQualityService:
    def route_cost_factor(self, *_args, **_kwargs) -> float:
        return 1.0

    def estimate_route_exposure(self, **_kwargs):
        return None


class _NeutralUrbanWellbeingService:
    def route_cost_factor(self, *_args, **_kwargs) -> float:
        return 1.0

    def candidate_waypoints(self, *_args, **_kwargs) -> list:
        return []

    def evaluate_route(self, *_args, **_kwargs):
        return None


def _neutralize_non_congestion_context() -> None:
    routing_service_module.get_air_quality_service = lambda: _NeutralAirQualityService()
    routing_service_module.get_urban_wellbeing_service = lambda: _NeutralUrbanWellbeingService()
    main.cycleway_service.estimate_route_coverage = lambda _geometry: {
        "available": False,
        "coverage_ratio": 0.0,
        "nearby_cycleway_km": 0.0,
        "route_km": 0.0,
        "nearby_buffer_m": 80.0,
        "has_high_coverage": False,
        "data_source": "neutralized_for_congestion_verification",
    }
    main._filter_hotspots = lambda **_kwargs: []


def _available_dates() -> list[str]:
    coverage = main.metadata_congestion_dates()
    return list(coverage.available_dates)


def _weekday(iso_date: str) -> str:
    return date.fromisoformat(iso_date).strftime("%A")


def _high_actual_lines(snapshot) -> list[dict]:
    features = (snapshot.congestion_lines or {}).get("features") or []
    selected = []
    for feature in features:
        props = feature.get("properties", {})
        if props.get("level") == "high" and props.get("recency") == "actual":
            selected.append(feature)
    return selected


def _route_points(route_card) -> list[tuple[float, float]]:
    raw_points = route_card.road_geometry or route_card.geometry
    return [(float(point.lat), float(point.lon)) for point in raw_points]


def _line_points(feature: dict) -> list[tuple[float, float]]:
    coords = feature.get("geometry", {}).get("coordinates") or []
    points: list[tuple[float, float]] = []
    for item in coords:
        if isinstance(item, list) and len(item) >= 2:
            points.append((float(item[1]), float(item[0])))
    return points


def _project_lines(
    route_points: list[tuple[float, float]],
    congestion_points: list[tuple[float, float]],
) -> tuple[LineString | None, LineString | None]:
    if len(route_points) < 2 or len(congestion_points) < 2:
        return None, None
    all_points = route_points + congestion_points
    origin_lat = sum(point[0] for point in all_points) / len(all_points)
    origin_lon = sum(point[1] for point in all_points) / len(all_points)
    meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(origin_lat))

    def project(point: tuple[float, float]) -> tuple[float, float]:
        lat, lon = point
        return (
            (lon - origin_lon) * meters_per_degree_lon,
            (lat - origin_lat) * METERS_PER_DEGREE_LAT,
        )

    return LineString([project(point) for point in route_points]), LineString(
        [project(point) for point in congestion_points]
    )


def _distance_m(
    route_points: list[tuple[float, float]],
    congestion_points: list[tuple[float, float]],
) -> tuple[float, str]:
    route_line, congestion_line = _project_lines(route_points, congestion_points)
    if route_line is None or congestion_line is None:
        return float("inf"), "invalid_geometry"
    distance = float(route_line.distance(congestion_line))
    if distance <= 1.0 or route_line.intersects(congestion_line):
        return distance, "real_line_crossing"
    return distance, "nearby_within_45m"


def _segment_ids(items: Iterable[dict]) -> set[str]:
    return {str(item.get("segment_id") or "") for item in items if item.get("segment_id")}


def _build_node_index(service):
    node_ids: list[str] = []
    node_coords: list[tuple[float, float]] = []
    for node_id, node in service.graph.nodes.items():
        if math.isfinite(node.lat) and math.isfinite(node.lon):
            node_ids.append(node_id)
            node_coords.append((node.lat, node.lon))
    if not node_coords:
        raise ValueError("El grafo no tiene nodos georreferenciados.")
    return node_ids, BallTree(np.radians(np.asarray(node_coords, dtype=float)), metric="haversine")


def _active_congestion_node_penalties_fast(
    node_index,
    active_congestion_lines: list[dict],
) -> dict[str, float]:
    node_ids, node_tree = node_index
    radius = (
        routing_service_module.ACTIVE_CONGESTION_NODE_TOLERANCE_M * 1.35
    ) / (routing_service_module.routing.EARTH_RADIUS_KM * 1000)
    penalties: dict[str, float] = {}
    for feature in active_congestion_lines:
        properties = feature.get("properties", {})
        try:
            score = float(properties.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        line = _line_points(feature)
        if not line:
            continue
        query_coords = np.radians(np.asarray(line, dtype=float))
        neighbor_indices = node_tree.query_radius(query_coords, r=radius)
        penalty = 1.0 + (score / 100.0) * 80.0
        for neighbors in neighbor_indices:
            for raw_index in neighbors:
                node_id = node_ids[int(raw_index)]
                penalties[node_id] = max(penalties.get(node_id, 1.0), penalty)
    return penalties


def _plan_least_congested(
    date_value: str,
    hour: int,
    pair: RoutePair,
    *,
    active_congestion_lines: list[dict],
    node_penalties: dict[str, float],
):
    service = get_routing_service()
    origin = RoutePoint(lat=pair.origin[0], lon=pair.origin[1])
    destination = RoutePoint(lat=pair.destination[0], lon=pair.destination[1])
    payload = RouteRequest(
        origin=origin,
        destination=destination,
        congestion_date=date_value,
        day_of_week=_weekday(date_value),
        departure_hour=float(hour),
        avoid_congestion=True,
        avoid_accidents=False,
    )
    if service.graph is None:
        raise ValueError("El grafo de rutas no esta listo.")

    day_value = routing_service_module._normalize_day(payload.day_of_week)
    hour_bucket = routing_service_module.data_loader.hour_bucket(payload.departure_hour)
    active_congestion_lines = service._active_congestion_lines(payload)
    routing_context = {
        "day": day_value,
        "hour_bucket": hour_bucket,
        "avoid_congestion": True,
        "avoid_accidents": False,
    }
    if node_penalties:
        routing_context["node_penalties"] = node_penalties

    least_congestion_path = service.graph.shortest_path(
        (payload.origin.lat, payload.origin.lon),
        (payload.destination.lat, payload.destination.lon),
        incident_ctx=routing_context,
        apply_penalties=True,
    )
    if not least_congestion_path:
        raise ValueError("No existe ruta least_congestion para el par indicado.")

    delay_context = {
        "day": day_value,
        "hour_bucket": hour_bucket,
        "include_congestion": True,
        "include_accidents": False,
        "match_filters": True,
    }
    road_points = service._build_geometry(least_congestion_path)
    exposure, risk_score, _why_changed, top_penalized_segments, _top_preferred_vias, _congestion_coverage = (
        service._build_variant_analysis(
            path=least_congestion_path,
            context=delay_context,
            via_factors={},
            variant_name="least_congestion",
            extra_minutes=0.0,
            route_geometry=road_points,
            active_congestion_lines=active_congestion_lines,
        )
    )
    road_geometry = [RoutePoint(**point) for point in road_points]
    return RouteCheckResult(
        geometry=road_geometry,
        road_geometry=road_geometry,
        risk_score=risk_score,
        incident_exposure=exposure,
        top_penalized_segments=top_penalized_segments,
    )


def main_cli() -> int:
    args = _parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    if not args.use_full_context:
        _neutralize_non_congestion_context()
    dates = args.dates or _available_dates()
    if args.max_dates is not None:
        dates = dates[: args.max_dates]
    hours = args.hours or list(range(24))
    if bool(args.origin) != bool(args.destination):
        raise SystemExit("--origin y --destination deben usarse juntos.")
    if args.origin and args.destination:
        pairs = [
            RoutePair(
                name="manual",
                origin=_parse_lat_lon(args.origin, label="--origin"),
                destination=_parse_lat_lon(args.destination, label="--destination"),
            )
        ]
    else:
        pairs = [pair for pair in DEFAULT_ROUTE_PAIRS if not args.only_pair or pair.name in args.only_pair]

    output = {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tolerance_m": ACTIVE_CONGESTION_MATCH_TOLERANCE_M,
        "dates": dates,
        "hours": hours,
        "pairs": [asdict(pair) for pair in pairs],
        "summary": {},
        "problem_cases": [],
        "colo_colo_regression": None,
    }
    problems: list[ProblemCase] = []
    clean_hours = 0
    routed_cases = 0
    skipped_without_high_actual = 0
    route_errors = 0

    env_service = get_environmental_impact_service()
    routing_service = get_routing_service()
    routing_service._ensure_fresh_data()
    if routing_service.graph is None:
        raise ValueError("El grafo de rutas no esta listo.")
    node_index = _build_node_index(routing_service)

    for date_index, date_value in enumerate(dates, start=1):
        print(f"[{date_index}/{len(dates)}] {date_value}: start", flush=True)
        for hour in hours:
            snapshot = env_service.build_snapshot(date_value, hour)
            active_congestion_lines = list((snapshot.congestion_lines or {}).get("features") or [])
            high_lines = _high_actual_lines(snapshot)
            if not high_lines:
                clean_hours += 1
                skipped_without_high_actual += len(pairs)
                continue
            node_penalties = _active_congestion_node_penalties_fast(node_index, active_congestion_lines)
            for pair in pairs:
                try:
                    least_route = _plan_least_congested(
                        date_value,
                        hour,
                        pair,
                        active_congestion_lines=active_congestion_lines,
                        node_penalties=node_penalties,
                    )
                except Exception as exc:
                    route_errors += 1
                    problems.append(
                        ProblemCase(
                            kind="route_error",
                            date=date_value,
                            hour=hour,
                            pair=pair.name,
                            segment_id=None,
                            via=None,
                            distance_m=None,
                            crossing_type=None,
                            congestion_score=0.0,
                            matched_incident_segments=0,
                            top_penalized_segments=[],
                            detail=str(exc),
                        )
                    )
                    continue
                routed_cases += 1
                route_points = _route_points(least_route)
                congestion_score = float(
                    getattr(least_route, "congestion_score", getattr(least_route, "risk_score", 0.0))
                )
                penalized_segment_ids = _segment_ids(
                    [segment.model_dump() for segment in least_route.top_penalized_segments]
                )
                for feature in high_lines:
                    props = feature.get("properties", {})
                    distance_m, crossing_type = _distance_m(route_points, _line_points(feature))
                    if distance_m > ACTIVE_CONGESTION_MATCH_TOLERANCE_M:
                        continue
                    segment_id = str(props.get("segment_id") or "")
                    reflected = (
                        congestion_score > 0
                        and least_route.incident_exposure.matched_incident_segments > 0
                        and segment_id in penalized_segment_ids
                    )
                    kind = "reflected_crossing" if reflected else "unreflected_crossing"
                    problems.append(
                        ProblemCase(
                            kind=kind,
                            date=date_value,
                            hour=hour,
                            pair=pair.name,
                            segment_id=segment_id,
                            via=props.get("via"),
                            distance_m=round(distance_m, 2),
                            crossing_type=crossing_type,
                            congestion_score=congestion_score,
                            matched_incident_segments=int(
                                least_route.incident_exposure.matched_incident_segments
                            ),
                            top_penalized_segments=[
                                segment.model_dump() for segment in least_route.top_penalized_segments
                            ],
                            detail=(
                                "least_congested queda a 45 m o menos de una linea high/actual; "
                                "se marca como reflejado solo si score, exposure y top_penalized_segments "
                                "incluyen el segmento."
                            ),
                        )
                    )
        print(
            f"[{date_index}/{len(dates)}] {date_value}: "
            f"routed={routed_cases} problems={len(problems)}",
            flush=True,
        )

    colo = [
        problem
        for problem in problems
        if problem.date == "2025-03-13"
        and problem.hour == 8
        and problem.pair == "colo_colo_regression"
    ]
    output["colo_colo_regression"] = {
        "date": "2025-03-13",
        "hour": 8,
        "origin": DEFAULT_ROUTE_PAIRS[0].origin,
        "destination": DEFAULT_ROUTE_PAIRS[0].destination,
        "avoids_high_actual_within_45m": not colo,
        "problem_cases": [asdict(problem) for problem in colo],
    }
    output["problem_cases"] = [asdict(problem) for problem in problems]
    output["summary"] = {
        "date_count": len(dates),
        "hour_count": len(hours),
        "pair_count": len(pairs),
        "clean_hours_without_high_actual": clean_hours,
        "skipped_pair_hours_without_high_actual": skipped_without_high_actual,
        "routed_cases_with_high_actual_lines": routed_cases,
        "route_errors": route_errors,
        "problem_count": len(problems),
        "unreflected_crossing_count": sum(1 for problem in problems if problem.kind == "unreflected_crossing"),
        "reflected_crossing_count": sum(1 for problem in problems if problem.kind == "reflected_crossing"),
        "real_line_crossing_count": sum(1 for problem in problems if problem.crossing_type == "real_line_crossing"),
        "nearby_within_45m_count": sum(1 for problem in problems if problem.crossing_type == "nearby_within_45m"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"Reporte: {args.output}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
