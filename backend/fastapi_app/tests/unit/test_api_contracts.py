# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.fastapi_app.app import main
from backend.fastapi_app.app.schemas.recommendations import CollaborativeRecommendation
from backend.fastapi_app.app.schemas.routes import (
    IncidentExposure,
    PlaceResult,
    PreferredViaImpact,
    RouteComparison,
    RouteDelta,
    RoutePoint,
    RouteRequest,
    RouteResponse,
    RouteStepResponse,
    RouteVariant,
    SegmentImpact,
)


def build_variant(
    *,
    via: str,
    duration: float,
    delay: float,
    risk: float,
    offset: float,
    exposure: int,
) -> RouteVariant:
    return RouteVariant(
        distance_km=3.2 + offset,
        estimated_duration_min=duration,
        steps=[
            RouteStepResponse(
                node_id=f"{via}-a",
                lat=-36.82,
                lon=-73.04,
                via=via,
                comuna="Concepcion",
                cumulative_cost=0.0,
            )
        ],
        geometry=[
            RoutePoint(lat=-36.82, lon=-73.04),
            RoutePoint(lat=-36.815 + offset / 100.0, lon=-73.045 - offset / 100.0),
            RoutePoint(lat=-36.81, lon=-73.05),
        ],
        extra_delay_min=delay,
        risk_score=risk,
        incident_exposure=IncidentExposure(
            total_incident_segments=max(1, exposure),
            matched_incident_segments=exposure,
            congestion_segments=exposure,
            accident_segments=0,
            exposure_minutes=delay,
        ),
        why_changed=[f"Ruta ajustada por {via}."],
        top_penalized_segments=[
            SegmentImpact(
                segment_id=f"seg-{via}",
                via=via,
                comuna="Concepcion",
                event_type="Congestion",
                impact_score=4.2,
                reason=f"Congestion historica en {via}.",
            )
        ],
        top_preferred_vias=[
            PreferredViaImpact(
                via=via,
                factor=0.6,
                reason="La estrategia colaborativa favorece esta via en la simulacion.",
            )
        ],
    )


class DummyRoutingService:
    graph = object()

    def compute_route(self, _payload: RouteRequest) -> RouteResponse:
        reference = build_variant(via="Barros Arana", duration=8.5, delay=2.0, risk=12.5, offset=0.0, exposure=1)
        ubcf = build_variant(via="O'Higgins", duration=7.5, delay=0.0, risk=5.0, offset=0.2, exposure=0)
        ibcf = build_variant(via="Paicavi", duration=7.8, delay=0.5, risk=7.0, offset=0.4, exposure=0)
        return RouteResponse(
            reference=reference,
            ubcf=ubcf,
            ibcf=ibcf,
            personalized=ubcf,
            comparison=RouteComparison(
                fastest_variant="ubcf",
                safest_variant="ubcf",
                lowest_exposure_variant="ibcf",
                best_balance_variant="ubcf",
                deltas=[
                    RouteDelta(
                        variant="ubcf",
                        distance_delta_km=0.2,
                        total_duration_delta_min=-3.0,
                        risk_delta=-7.5,
                        exposure_delta=-1.0,
                    )
                ],
            ),
        )


class DummyRecommendationService:
    def available_options(self) -> dict:
        return {"total_events": 1}

    def collaborative_recommendations(self, _payload):
        return []

    def playground_recommendations(self, _payload):
        return {
            "ubcf": [CollaborativeRecommendation(via="O'Higgins", estimated_rating=4.7, strategy="ubcf")],
            "ibcf": [CollaborativeRecommendation(via="Paicavi", estimated_rating=4.4, strategy="ibcf")],
        }


class DummyGeocodingService:
    def search(self, query: str, limit: int = 5):
        return [
            PlaceResult(
                id="place-1",
                label=f"{query} Centro",
                lat=-36.82,
                lon=-73.04,
                bbox=[-73.06, -36.84, -73.02, -36.80],
            )
        ][:limit]

    def reverse(self, lat: float, lon: float):
        return PlaceResult(id="reverse-1", label="Punto manual", lat=lat, lon=lon, bbox=None)


def test_readyz_reports_ready(monkeypatch):
    monkeypatch.setattr(main, "AUTO_BOOTSTRAP_ENABLED", False)
    main.bootstrap_state.clear()
    main.bootstrap_state.update(
        {
            "status": "completed",
            "message": "Infraestructura lista para planificar viajes",
            "percent": 100,
            "routing_nodes": 10,
            "routing_segments": 5,
            "duration_ms": 15.0,
            "dataset_profile": "gran_concepcion",
            "quality": {
                "status": "warning",
                "dataset_profile": "gran_concepcion",
                "duplicate_incident_sources": True,
                "date_range": {"start": "2025-07-01", "end": "2025-07-31", "days": 31},
                "missing_via_ratio": 0.02,
                "anomalous_communes": [],
                "raw_counts": {"accidents": 1, "congestions": 1, "combined": 2},
                "warnings": ["Fuentes identicas."],
                "notes": [],
            },
        }
    )
    monkeypatch.setattr(main, "get_routing_service", lambda: DummyRoutingService())

    with TestClient(main.app) as client:
        response = client.get("/readyz")

    payload = response.json()
    assert response.status_code == 200
    assert payload["ready"] is True
    assert payload["bootstrap"]["quality"]["duplicate_incident_sources"] is True


def test_demo_scenarios_endpoint_returns_curated_cases(monkeypatch):
    monkeypatch.setattr(main, "AUTO_BOOTSTRAP_ENABLED", False)

    with TestClient(main.app) as client:
        response = client.get("/system/demo-scenarios")

    payload = response.json()
    assert response.status_code == 200
    assert len(payload["scenarios"]) >= 3
    assert payload["scenarios"][0]["profile"]


def test_routes_optimal_returns_explanatory_contract(monkeypatch):
    monkeypatch.setattr(main, "AUTO_BOOTSTRAP_ENABLED", False)
    main.bootstrap_state.clear()
    main.bootstrap_state.update(
        {
            "status": "completed",
            "message": "Infraestructura lista para planificar viajes",
            "percent": 100,
            "routing_nodes": 10,
            "routing_segments": 5,
            "duration_ms": 15.0,
            "dataset_profile": "gran_concepcion",
            "quality": None,
        }
    )
    main.app.dependency_overrides[main.get_routing_service] = lambda: DummyRoutingService()

    payload = {
        "origin": {"lat": -36.82, "lon": -73.04},
        "destination": {"lat": -36.81, "lon": -73.05},
        "preferences": [],
        "ubcf_preferences": [],
        "ibcf_preferences": [],
        "day_of_week": "Wednesday",
        "departure_hour": 8.0,
        "avoid_congestion": True,
        "avoid_accidents": False,
    }

    with TestClient(main.app) as client:
        response = client.post("/routes/optimal", json=payload)

    main.app.dependency_overrides.clear()
    body = response.json()
    assert response.status_code == 200
    assert body["ubcf"]["risk_score"] == 5.0
    assert body["ubcf"]["incident_exposure"]["matched_incident_segments"] == 0
    assert body["ubcf"]["top_penalized_segments"][0]["segment_id"] == "seg-O'Higgins"
    assert body["comparison"]["best_balance_variant"] == "ubcf"


def test_routes_plan_returns_user_facing_contract(monkeypatch):
    monkeypatch.setattr(main, "AUTO_BOOTSTRAP_ENABLED", False)
    main.bootstrap_state.clear()
    main.bootstrap_state.update(
        {
            "status": "completed",
            "message": "Infraestructura lista para planificar viajes",
            "percent": 100,
            "routing_nodes": 10,
            "routing_segments": 5,
            "duration_ms": 15.0,
            "dataset_profile": "gran_concepcion",
            "quality": None,
        }
    )
    main.app.dependency_overrides[main.get_routing_service] = lambda: DummyRoutingService()
    main.app.dependency_overrides[main.get_recommendation_service] = lambda: DummyRecommendationService()
    monkeypatch.setattr(
        main,
        "_cached_hotspots",
        lambda _limit: [
            {
                "lat": -36.819,
                "lon": -73.044,
                "weight": 0.5,
                "day": "Wednesday",
                "bucket": "Punta AM (06-09h)",
                "segment_id": "seg-1",
                "hora_inicio_float": 7.0,
                "hora_fin_float": 8.0,
            }
        ],
    )

    payload = {
        "origin": {"lat": -36.82, "lon": -73.04},
        "destination": {"lat": -36.81, "lon": -73.05},
        "day_of_week": "Wednesday",
        "departure_hour": 8.0,
        "travel_style": "balanced",
        "avoid_congestion": True,
        "avoid_accidents": False,
    }

    with TestClient(main.app) as client:
        response = client.post("/routes/plan", json=payload)

    main.app.dependency_overrides.clear()
    body = response.json()
    assert response.status_code == 200
    assert body["selected_route_key"] == "least_congestion"
    assert len(body["routes"]) == 3
    assert body["routes"][0]["badges"][0]["key"] == "base"
    assert body["routes"][1]["badges"][0]["key"] == "least_congestion"
    assert body["routes"][2]["badges"][0]["key"] == "healthiest"
    assert body["summary"]["eta_total_min"] == 8.3
    assert body["hotspots"][0]["segment_id"] == "seg-1"


def test_places_endpoints_return_normalized_results(monkeypatch):
    monkeypatch.setattr(main, "AUTO_BOOTSTRAP_ENABLED", False)
    main.app.dependency_overrides[main.get_geocoding_service] = lambda: DummyGeocodingService()

    with TestClient(main.app) as client:
        search_response = client.get("/places/search", params={"q": "UdeC", "limit": 5})
        reverse_response = client.get("/places/reverse", params={"lat": -36.82, "lon": -73.04})

    main.app.dependency_overrides.clear()
    assert search_response.status_code == 200
    assert reverse_response.status_code == 200
    assert search_response.json()["results"][0]["label"] == "UdeC Centro"
    assert reverse_response.json()["result"]["label"] == "Punto manual"
