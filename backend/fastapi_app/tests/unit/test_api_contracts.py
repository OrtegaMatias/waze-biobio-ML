# -*- coding: utf-8 -*-
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.fastapi_app.app import main
from backend.fastapi_app.app.schemas.routes import (
    IncidentExposure,
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


class DummyRoutingService:
    graph = object()

    def compute_route(self, _payload: RouteRequest) -> RouteResponse:
        variant = RouteVariant(
            distance_km=3.2,
            estimated_duration_min=8.5,
            steps=[
                RouteStepResponse(
                    node_id="a",
                    lat=-36.82,
                    lon=-73.04,
                    via="Barros Arana",
                    comuna="Concepción",
                    cumulative_cost=0.0,
                )
            ],
            geometry=[RoutePoint(lat=-36.82, lon=-73.04)],
            extra_delay_min=2.0,
            risk_score=12.5,
            incident_exposure=IncidentExposure(
                total_incident_segments=2,
                matched_incident_segments=1,
                congestion_segments=1,
                accident_segments=0,
                exposure_minutes=2.0,
            ),
            why_changed=["La ruta evita una franja con exposición histórica."],
            top_penalized_segments=[
                SegmentImpact(
                    segment_id="seg-1",
                    via="Barros Arana",
                    comuna="Concepción",
                    event_type="Congestión",
                    impact_score=4.2,
                    reason="Congestión histórica en Punta AM (06-09h).",
                )
            ],
            top_preferred_vias=[
                PreferredViaImpact(
                    via="Barros Arana",
                    factor=0.6,
                    reason="La estrategia colaborativa favorece esta vía en la simulación.",
                )
            ],
        )
        return RouteResponse(
            reference=variant,
            ubcf=variant,
            ibcf=variant,
            personalized=variant,
            comparison=RouteComparison(
                fastest_variant="reference",
                safest_variant="ubcf",
                lowest_exposure_variant="ubcf",
                best_balance_variant="ubcf",
                deltas=[RouteDelta(variant="ubcf", distance_delta_km=0.2, total_duration_delta_min=-1.5, risk_delta=-3.0, exposure_delta=-1.0)],
            ),
        )


def test_readyz_reports_ready(monkeypatch):
    monkeypatch.setattr(main, "AUTO_BOOTSTRAP_ENABLED", False)
    main.bootstrap_state.clear()
    main.bootstrap_state.update(
        {
            "status": "completed",
            "message": "Infraestructura lista para la demo",
            "percent": 100,
            "routing_nodes": 10,
            "routing_segments": 5,
            "duration_ms": 15.0,
            "dataset_profile": "concepcion",
            "quality": {
                "status": "warning",
                "dataset_profile": "concepcion",
                "duplicate_incident_sources": True,
                "date_range": {"start": "2025-07-01", "end": "2025-07-31", "days": 31},
                "missing_via_ratio": 0.02,
                "anomalous_communes": [],
                "raw_counts": {"accidents": 1, "congestions": 1, "combined": 2},
                "warnings": ["Fuentes idénticas."],
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
            "message": "Infraestructura lista para la demo",
            "percent": 100,
            "routing_nodes": 10,
            "routing_segments": 5,
            "duration_ms": 15.0,
            "dataset_profile": "concepcion",
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
    assert body["ubcf"]["risk_score"] == 12.5
    assert body["ubcf"]["incident_exposure"]["matched_incident_segments"] == 1
    assert body["ubcf"]["top_penalized_segments"][0]["segment_id"] == "seg-1"
    assert body["comparison"]["best_balance_variant"] == "ubcf"
