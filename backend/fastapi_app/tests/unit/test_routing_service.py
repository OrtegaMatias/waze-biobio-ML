# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pytest

from backend.fastapi_app.app.schemas.routes import (
    IncidentExposure,
    Pm25Exposure,
    RouteCongestionCoverage,
    RoutePoint,
    RouteRequest,
    RouteVariant,
    UrbanWellbeingAnalysis,
    ViaPreference,
    WellbeingFeatureImpact,
)
from backend.fastapi_app.app.services import routing_service
from algorithms.recommenders import routing


def _event_dataframe():
    return pd.DataFrame(
        [
            {
                "segment_id": "segA",
                "segment_seq": 0,
                "lat": -36.0,
                "lon": -73.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.1,
                "via": "Ruta Azul",
                "comuna": "Test",
                "oneway": False,
            }
        ]
    )


def test_routing_service_rebuilds_on_new_signature(monkeypatch):
    service = routing_service.RoutingService()
    events = _event_dataframe()

    signature_values = [("a", 1, 1), ("b", 2, 2)]
    call_index = {"value": 0}

    def fake_data_version():
        idx = min(call_index["value"], len(signature_values) - 1)
        call_index["value"] += 1
        return signature_values[idx]

    class DummyGraph:
        def __init__(self):
            self.nodes = {"node": None}

    def fake_from_events(cls, *_args, **_kwargs):
        return DummyGraph()

    seen_signatures = []

    monkeypatch.setattr(routing_service.data_loader, "load_route_network", lambda: events)
    monkeypatch.setattr(routing_service.data_loader, "data_version", fake_data_version)
    monkeypatch.setattr(routing_service, "_load_graph_cache", lambda signature: seen_signatures.append(signature))
    monkeypatch.setattr(routing_service, "_store_graph_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(routing_service.routing.RouteGraph, "from_events", classmethod(fake_from_events))
    monkeypatch.setattr(routing_service.RoutingService, "_build_segment_lookup", staticmethod(lambda _events: {}))

    service._build_structures()
    service._build_structures()

    assert seen_signatures == signature_values


def test_routing_service_uses_cache_without_loading_events(monkeypatch):
    service = routing_service.RoutingService()

    class DummyGraph:
        def __init__(self):
            self.nodes = {"node": None}

    load_calls = {"count": 0}
    signature = ("sig", 1, 1)

    def fake_load_route_network():
        load_calls["count"] += 1
        return _event_dataframe()

    monkeypatch.setattr(routing_service.data_loader, "data_version", lambda: signature)
    monkeypatch.setattr(routing_service.data_loader, "load_route_network", fake_load_route_network)
    monkeypatch.setattr(
        routing_service,
        "_load_graph_cache",
        lambda current_signature: {
            "graph": DummyGraph(),
            "segment_lookup": {"segA": {0: (-36.0, -73.0)}},
        } if current_signature == signature else None,
    )

    service._build_structures()

    assert load_calls["count"] == 0
    assert service.events is None
    assert service.graph is not None
    assert service.segment_lookup == {"segA": {0: (-36.0, -73.0)}}


def test_default_factor_remains_neutral_when_preferences_present(monkeypatch):
    service = routing_service.RoutingService()

    class DummyGraph:
        def __init__(self):
            self.nodes = {"node": None}
            self.calls = []

        def shortest_path(self, _origin, _destination, **kwargs):
            self.calls.append(kwargs)
            return [
                routing.RouteStep(
                    node_id="a",
                    segment_id="seg1",
                    segment_seq=0,
                    lat=0.0,
                    lon=0.0,
                    via="Ruta Azul",
                    comuna="Test",
                    peso=0.0,
                ),
                routing.RouteStep(
                    node_id="b",
                    segment_id="seg1",
                    segment_seq=1,
                    lat=0.1,
                    lon=0.1,
                    via="Ruta Azul",
                    comuna="Test",
                    peso=1.0,
                ),
            ]

    dummy_graph = DummyGraph()
    service.graph = dummy_graph
    service.segment_lookup = {"seg1": {0: (0.0, 0.0), 1: (0.1, 0.1)}}
    monkeypatch.setattr(service, "_ensure_fresh_data", lambda: None)

    payload = RouteRequest(
        origin=RoutePoint(lat=0.0, lon=0.0),
        destination=RoutePoint(lat=0.1, lon=0.1),
        preferences=[ViaPreference(via="Ruta Azul", weight=1.0)],
        avoid_congestion=False,
        avoid_accidents=False,
    )

    service.compute_route(payload)

    assert len(dummy_graph.calls) == 5
    personalized_call = next(call for call in dummy_graph.calls if call.get("via_factors"))
    assert personalized_call["default_via_factor"] == 1.0
    healthy_calls = [call for call in dummy_graph.calls if "air_quality_factor" in call]
    assert len(healthy_calls) == 2
    assert all("urban_wellbeing_factor" in call for call in healthy_calls)


def test_diversity_factor_penalizes_only_edges_from_existing_paths():
    def graph_node(node_id: str) -> routing.GraphNode:
        return routing.GraphNode(
            node_id=node_id,
            segment_id=node_id,
            segment_seq=0,
            lat=0.0,
            lon=0.0,
            tipo_evento="Referencia",
            velocidad_kmh=30.0,
            duracion_hrs=0.1,
            via="Test",
            comuna="Test",
        )

    a = graph_node("a")
    b = graph_node("b")
    c = graph_node("c")
    path = [
        routing.RouteStep(
            node_id="a", segment_id="a", segment_seq=0, lat=0.0, lon=0.0, via="Test", comuna="Test", peso=0.0
        ),
        routing.RouteStep(
            node_id="b", segment_id="b", segment_seq=0, lat=0.0, lon=0.01, via="Test", comuna="Test", peso=1.0
        ),
    ]

    factor = routing_service.RoutingService._diversity_edge_cost_factor(
        [path],
        base_factor=lambda _previous, _current: 2.0,
    )

    assert factor(a, b) == pytest.approx(2.0 * routing_service.ALTERNATIVE_OVERLAP_PENALTY)
    assert factor(a, c) == pytest.approx(2.0)


def test_reasonable_alternative_rejects_identical_and_excessive_detours():
    def step(node_id: str, lon: float) -> routing.RouteStep:
        return routing.RouteStep(
            node_id=node_id,
            segment_id=node_id,
            segment_seq=0,
            lat=0.0,
            lon=lon,
            via="Test",
            comuna="Test",
            peso=0.0,
        )

    reference = [step("a", 0.0), step("b", 0.01)]
    reasonable = [step("a", 0.0), step("c", 0.005), step("b", 0.01)]
    excessive = [step("a", 0.0), step("far", 0.02), step("b", 0.01)]

    assert not routing_service.RoutingService._is_reasonable_alternative(reference, list(reference))
    assert routing_service.RoutingService._is_reasonable_alternative(reference, reasonable)
    assert not routing_service.RoutingService._is_reasonable_alternative(reference, excessive)


def test_reference_variant_includes_congestion_delay(monkeypatch):
    service = routing_service.RoutingService()

    class DummyGraph:
        def __init__(self):
            self.nodes = {"node": None}

        def shortest_path(self, _origin, _destination, **_kwargs):
            return [
                routing.RouteStep(
                    node_id="a",
                    segment_id="segDelay",
                    segment_seq=0,
                    lat=-36.0,
                    lon=-73.0,
                    via="Via Base",
                    comuna="Test",
                    peso=0.0,
                    tipo_evento="Referencia",
                    duracion_hrs=0.1,
                    dia_semana="Wednesday",
                    franja_horaria="Punta AM (06-09h)",
                ),
                routing.RouteStep(
                    node_id="b",
                    segment_id="segDelay",
                    segment_seq=1,
                    lat=-36.01,
                    lon=-73.01,
                    via="Via Cong",
                    comuna="Test",
                    peso=1.0,
                    tipo_evento="Congestión",
                    duracion_hrs=0.5,
                    dia_semana="Wednesday",
                    franja_horaria="Punta AM (06-09h)",
                ),
            ]

    service.graph = DummyGraph()
    service.segment_lookup = {"segDelay": {0: (-36.0, -73.0), 1: (-36.01, -73.01)}}
    monkeypatch.setattr(service, "_ensure_fresh_data", lambda: None)

    payload = RouteRequest(
        origin=RoutePoint(lat=-36.0, lon=-73.0),
        destination=RoutePoint(lat=-36.01, lon=-73.01),
        preferences=[],
        day_of_week="Wednesday",
        departure_hour=8.0,
        avoid_congestion=False,
        avoid_accidents=False,
    )

    route = service.compute_route(payload)

    assert route.reference.extra_delay_min == pytest.approx(35.0)
    assert route.personalized.extra_delay_min == pytest.approx(35.0)
    assert route.reference.estimated_duration_min < 10
    assert route.personalized.estimated_duration_min < 10


def test_least_congestion_variant_is_dedicated_penalized_route(monkeypatch):
    service = routing_service.RoutingService()

    direct_path = [
        routing.RouteStep(
            node_id="direct-a",
            segment_id="segDirect",
            segment_seq=0,
            lat=-36.0,
            lon=-73.0,
            via="Via Directa",
            comuna="Test",
            peso=0.0,
            tipo_evento="Referencia",
            duracion_hrs=0.1,
            dia_semana="Wednesday",
            franja_horaria="Punta AM (06-09h)",
        ),
        routing.RouteStep(
            node_id="direct-b",
            segment_id="segDirect",
            segment_seq=1,
            lat=-36.01,
            lon=-73.01,
            via="Via Directa",
            comuna="Test",
            peso=1.0,
            tipo_evento="CongestiÃ³n",
            duracion_hrs=0.4,
            dia_semana="Wednesday",
            franja_horaria="Punta AM (06-09h)",
        ),
    ]
    detour_path = [
        routing.RouteStep(
            node_id="detour-a",
            segment_id="segDetour",
            segment_seq=0,
            lat=-36.0,
            lon=-73.0,
            via="Via Alternativa",
            comuna="Test",
            peso=0.0,
            tipo_evento="Referencia",
            duracion_hrs=0.1,
            dia_semana="Wednesday",
            franja_horaria="Punta AM (06-09h)",
        ),
        routing.RouteStep(
            node_id="detour-b",
            segment_id="segDetour",
            segment_seq=1,
            lat=-36.0,
            lon=-73.02,
            via="Via Alternativa",
            comuna="Test",
            peso=1.0,
            tipo_evento="Referencia",
            duracion_hrs=0.1,
            dia_semana="Wednesday",
            franja_horaria="Punta AM (06-09h)",
        ),
        routing.RouteStep(
            node_id="detour-c",
            segment_id="segDetour",
            segment_seq=2,
            lat=-36.01,
            lon=-73.02,
            via="Via Alternativa",
            comuna="Test",
            peso=2.0,
            tipo_evento="Referencia",
            duracion_hrs=0.1,
            dia_semana="Wednesday",
            franja_horaria="Punta AM (06-09h)",
        ),
    ]

    class DummyGraph:
        nodes = {"node": None}

        def __init__(self):
            self.calls = []

        def shortest_path(self, _origin, _destination, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("apply_penalties") is False:
                return direct_path
            return detour_path

    class DummyAirQualityService:
        def route_cost_factor(self, _lat, _lon, _departure_hour):
            return 1.0

        def estimate_route_exposure(self, **_kwargs):
            return None

    service.graph = DummyGraph()
    service.segment_lookup = {
        "segDirect": {0: (-36.0, -73.0), 1: (-36.01, -73.01)},
        "segDetour": {0: (-36.0, -73.0), 1: (-36.0, -73.02), 2: (-36.01, -73.02)},
    }
    monkeypatch.setattr(service, "_ensure_fresh_data", lambda: None)
    monkeypatch.setattr(routing_service, "get_air_quality_service", lambda: DummyAirQualityService())

    payload = RouteRequest(
        origin=RoutePoint(lat=-36.0, lon=-73.0),
        destination=RoutePoint(lat=-36.01, lon=-73.02),
        preferences=[],
        day_of_week="Wednesday",
        departure_hour=8.0,
        avoid_congestion=True,
        avoid_accidents=False,
    )

    route = service.compute_route(payload)

    assert route.least_congestion is not None
    assert route.reference.geometry != route.least_congestion.geometry
    assert route.reference.incident_exposure.matched_incident_segments == 1
    assert route.least_congestion.incident_exposure.matched_incident_segments == 0
    assert route.least_congestion.distance_km > route.reference.distance_km
    assert route.comparison.lowest_exposure_variant == "least_congestion"
    assert "air_quality_factor" not in service.graph.calls[1]
    assert "urban_wellbeing_factor" not in service.graph.calls[1]


def test_cerro_caracol_blocks_internal_edges_but_allows_victor_lamas():
    def node(node_id: str, lat: float, lon: float, via: str) -> routing.GraphNode:
        return routing.GraphNode(
            node_id=node_id,
            segment_id=node_id,
            segment_seq=0,
            lat=lat,
            lon=lon,
            tipo_evento="Referencia",
            velocidad_kmh=30.0,
            duracion_hrs=0.1,
            via=via,
            comuna="Concepción",
        )

    outside = node("outside", -36.8280, -73.0470, "Cochrane")
    inside = node("inside", -36.8360, -73.0470, "Veteranos del 79")
    victor_lamas = node("victor", -36.8360, -73.0470, "Víctor Lamas")

    assert routing_service._cerro_caracol_edge_allowed(outside, outside)
    assert not routing_service._cerro_caracol_edge_allowed(inside, inside)
    assert routing_service._cerro_caracol_edge_allowed(outside, victor_lamas)
    assert routing_service._cerro_caracol_edge_cost_factor(inside, inside) > 1.0
    assert routing_service._cerro_caracol_edge_cost_factor(outside, victor_lamas) == 1.0


def _healthy_candidate(
    *,
    lat: float,
    distance_km: float = 1.0,
    minutes: float = 10.0,
    risk_score: float = 0.0,
    matched_segments: int = 0,
    high_pct: float = 0.0,
    wellbeing_score: float = 0.0,
    pm25: float = 20.0,
    environmental_contact: bool = True,
) -> RouteVariant:
    return RouteVariant(
        distance_km=distance_km,
        estimated_duration_min=minutes,
        steps=[],
        geometry=[RoutePoint(lat=lat, lon=-73.0), RoutePoint(lat=lat + 0.001, lon=-73.0)],
        risk_score=risk_score,
        incident_exposure=IncidentExposure(matched_incident_segments=matched_segments),
        pm25_exposure=Pm25Exposure(
            available=True,
            average_pm25=pm25,
            category="Baja" if pm25 < 20 else ("Media" if pm25 < 50 else "Alta"),
            method="test",
            data_source="test",
        ),
        urban_wellbeing=UrbanWellbeingAnalysis(
            available=True,
            score=wellbeing_score,
            nearby_feature_count=1 if environmental_contact else 0,
            top_features=(
                [
                    WellbeingFeatureImpact(
                        feature_id="test-park",
                        name="Parque de prueba",
                        category="green_space",
                        subtype="park",
                        source="test",
                    )
                ]
                if environmental_contact
                else []
            ),
            method="test",
            data_source="test",
        ),
        congestion_coverage=RouteCongestionCoverage(high_pct=high_pct),
    )


def test_healthiest_selection_rejects_candidate_with_much_worse_congestion():
    reference = _healthy_candidate(lat=-36.0, risk_score=0.0, high_pct=0.0, wellbeing_score=5.0)
    congested_green = _healthy_candidate(
        lat=-36.01,
        risk_score=90.0,
        matched_segments=4,
        high_pct=75.0,
        wellbeing_score=100.0,
    )

    selected = routing_service.RoutingService._select_healthiest_variant(
        reference=reference,
        candidates=[reference, congested_green],
    )

    assert selected.geometry == reference.geometry
    assert selected.risk_score == 0.0


def test_healthiest_selection_uses_wellbeing_when_congestion_is_equivalent():
    reference = _healthy_candidate(lat=-36.0, distance_km=1.0, minutes=10.0, wellbeing_score=10.0)
    greener = _healthy_candidate(lat=-36.01, distance_km=1.04, minutes=10.4, wellbeing_score=90.0)

    selected = routing_service.RoutingService._select_healthiest_variant(
        reference=reference,
        candidates=[reference, greener],
    )

    assert selected.geometry == greener.geometry
    assert selected.urban_wellbeing is not None
    assert selected.urban_wellbeing.score == 90.0


def test_healthiest_selection_requires_effective_environmental_contact():
    reference = _healthy_candidate(
        lat=-36.0,
        wellbeing_score=95.0,
        environmental_contact=False,
    )
    route_beside_park = _healthy_candidate(
        lat=-36.01,
        distance_km=1.04,
        minutes=10.4,
        wellbeing_score=20.0,
        environmental_contact=True,
    )

    selected = routing_service.RoutingService._select_healthiest_variant(
        reference=reference,
        candidates=[reference, route_beside_park],
    )

    assert selected.geometry == route_beside_park.geometry
    assert any("pasa junto a" in reason for reason in selected.why_changed)


def test_healthiest_selection_prefers_contact_route_without_congestion():
    congested_park = _healthy_candidate(
        lat=-36.0,
        wellbeing_score=95.0,
        risk_score=5.0,
        matched_segments=1,
        high_pct=2.0,
    )
    clear_park = _healthy_candidate(
        lat=-36.01,
        distance_km=1.04,
        minutes=10.4,
        wellbeing_score=20.0,
    )

    selected = routing_service.RoutingService._select_healthiest_variant(
        reference=congested_park,
        candidates=[congested_park, clear_park],
    )

    assert selected.geometry == clear_park.geometry
    assert selected.incident_exposure.matched_incident_segments == 0


def test_healthiest_selection_prefers_distinct_environmental_geometry_when_valid():
    reference = _healthy_candidate(lat=-36.0, wellbeing_score=40.0)
    least_congestion = _healthy_candidate(lat=-36.005, wellbeing_score=45.0)
    distinct_environmental = _healthy_candidate(
        lat=-36.01,
        distance_km=1.04,
        minutes=10.4,
        wellbeing_score=50.0,
    )

    selected = routing_service.RoutingService._select_healthiest_variant(
        reference=reference,
        candidates=[reference, least_congestion, distinct_environmental],
    )

    assert selected.geometry == distinct_environmental.geometry


def test_weighted_environmental_route_is_not_reordered_by_a_final_score():
    reference = _healthy_candidate(lat=-36.0, wellbeing_score=90.0, pm25=10.0)
    least_congestion = _healthy_candidate(lat=-36.005, wellbeing_score=90.0, pm25=10.0)
    weighted = _healthy_candidate(lat=-36.01, wellbeing_score=100.0, pm25=5.0)
    waypoint = _healthy_candidate(
        lat=-36.015,
        distance_km=1.04,
        minutes=10.4,
        wellbeing_score=1.0,
        pm25=80.0,
    )

    selected = routing_service.RoutingService._finalize_weighted_environmental_variant(
        reference=reference,
        least_congestion=least_congestion,
        weighted=weighted,
        waypoint_candidates=[waypoint],
    )

    assert selected.geometry == waypoint.geometry
    assert selected.healthy_route_score is None
    assert selected.why_changed[0].startswith("La geometria fue calculada directamente")


def test_weighted_environmental_route_uses_hard_constraints_not_ranking():
    reference = _healthy_candidate(lat=-36.0)
    least_congestion = _healthy_candidate(lat=-36.005)
    weighted = _healthy_candidate(lat=-36.01, wellbeing_score=10.0, pm25=40.0)
    congested_waypoint = _healthy_candidate(
        lat=-36.015,
        distance_km=1.04,
        minutes=10.4,
        risk_score=1.0,
        matched_segments=1,
        high_pct=1.0,
        wellbeing_score=100.0,
        pm25=5.0,
    )

    selected = routing_service.RoutingService._finalize_weighted_environmental_variant(
        reference=reference,
        least_congestion=least_congestion,
        weighted=weighted,
        waypoint_candidates=[congested_waypoint],
    )

    assert selected.geometry == weighted.geometry
    assert selected.incident_exposure.matched_incident_segments == 0


def test_weighted_environmental_route_allows_ten_extra_minutes_from_ten_minute_trip():
    reference = _healthy_candidate(lat=-36.0, minutes=10.0)
    least_congestion = _healthy_candidate(lat=-36.005, minutes=10.0)
    weighted = _healthy_candidate(
        lat=-36.01,
        distance_km=1.10,
        minutes=20.0,
    )

    selected = routing_service.RoutingService._finalize_weighted_environmental_variant(
        reference=reference,
        least_congestion=least_congestion,
        weighted=weighted,
        waypoint_candidates=[],
    )

    assert selected.geometry == weighted.geometry


def test_weighted_environmental_route_rejects_more_than_ten_extra_minutes():
    reference = _healthy_candidate(lat=-36.0, minutes=10.0)
    least_congestion = _healthy_candidate(lat=-36.005, minutes=10.0)
    weighted = _healthy_candidate(
        lat=-36.01,
        distance_km=1.10,
        minutes=20.1,
    )

    selected = routing_service.RoutingService._finalize_weighted_environmental_variant(
        reference=reference,
        least_congestion=least_congestion,
        weighted=weighted,
        waypoint_candidates=[],
    )

    assert selected.geometry == reference.geometry


def test_healthiest_selection_prefers_meaningfully_lower_pm25_with_reasonable_detour():
    reference = _healthy_candidate(lat=-36.0, distance_km=1.0, minutes=10.0, pm25=56.0, wellbeing_score=90.0)
    cleaner = _healthy_candidate(lat=-36.01, distance_km=1.08, minutes=11.0, pm25=22.0, wellbeing_score=0.0)

    selected = routing_service.RoutingService._select_healthiest_variant(
        reference=reference,
        candidates=[reference, cleaner],
    )

    assert selected.geometry == cleaner.geometry
    assert selected.pm25_exposure is not None
    assert selected.pm25_exposure.average_pm25 == 22.0
    assert any("Reduce PM2.5" in reason for reason in selected.why_changed)


def test_lowest_exposure_comparison_uses_pm25_before_fastest_time():
    service = routing_service.RoutingService()
    reference = _healthy_candidate(lat=-36.0, distance_km=1.0, minutes=10.0, pm25=56.0)
    healthier = _healthy_candidate(lat=-36.01, distance_km=1.08, minutes=11.0, pm25=22.0)

    comparison = service._build_comparison(
        reference=reference,
        least_congestion=reference,
        ubcf=reference,
        ibcf=reference,
        healthiest=healthier,
        personalized=reference,
    )

    assert comparison.fastest_variant == "reference"
    assert comparison.lowest_exposure_variant == "healthiest"


def test_reference_variant_ignores_incidents_if_day_differs(monkeypatch):
    service = routing_service.RoutingService()

    class DummyGraph:
        def __init__(self):
            self.nodes = {"node": None}

        def shortest_path(self, _origin, _destination, **_kwargs):
            return [
                routing.RouteStep(
                    node_id="a",
                    segment_id="segDelay",
                    segment_seq=0,
                    lat=-36.0,
                    lon=-73.0,
                    via="Via Base",
                    comuna="Test",
                    peso=0.0,
                    tipo_evento="Referencia",
                    duracion_hrs=0.1,
                    dia_semana="Wednesday",
                    franja_horaria="Punta AM (06-09h)",
                ),
                routing.RouteStep(
                    node_id="b",
                    segment_id="segDelay",
                    segment_seq=1,
                    lat=-36.01,
                    lon=-73.01,
                    via="Via Cong",
                    comuna="Test",
                    peso=1.0,
                    tipo_evento="Congestión",
                    duracion_hrs=0.5,
                    dia_semana="Wednesday",
                    franja_horaria="Punta AM (06-09h)",
                ),
            ]

    service.graph = DummyGraph()
    service.segment_lookup = {"segDelay": {0: (-36.0, -73.0), 1: (-36.01, -73.01)}}
    monkeypatch.setattr(service, "_ensure_fresh_data", lambda: None)

    payload = RouteRequest(
        origin=RoutePoint(lat=-36.0, lon=-73.0),
        destination=RoutePoint(lat=-36.01, lon=-73.01),
        preferences=[],
        day_of_week="Monday",
        departure_hour=8.0,
        avoid_congestion=False,
        avoid_accidents=False,
    )

    route = service.compute_route(payload)

    assert route.reference.extra_delay_min == pytest.approx(0.0)
    assert route.personalized.extra_delay_min == pytest.approx(0.0)
    assert route.reference.estimated_duration_min < 10
    assert route.personalized.estimated_duration_min < 10


def test_variant_analysis_counts_unique_congested_segments_and_keeps_severity():
    service = routing_service.RoutingService()
    path = [
        routing.RouteStep(
            node_id=f"clean-{index}",
            segment_id=f"clean-{index}",
            segment_seq=index,
            lat=0.0,
            lon=index * 0.001,
            via="Via Limpia",
            comuna="Test",
            peso=float(index),
            tipo_evento="Referencia",
            duracion_hrs=0.1,
            dia_semana="Wednesday",
            franja_horaria="Punta AM (06-09h)",
        )
        for index in range(12)
    ]
    path.extend(
        [
            routing.RouteStep(
                node_id="cong-a",
                segment_id="red-segment",
                segment_seq=0,
                lat=0.0,
                lon=0.02,
                via="Via Roja",
                comuna="Test",
                peso=12.0,
                tipo_evento="CONGESTION",
                duracion_hrs=0.5,
                dia_semana="Wednesday",
                franja_horaria="Punta AM (06-09h)",
            ),
            routing.RouteStep(
                node_id="cong-b",
                segment_id="red-segment",
                segment_seq=1,
                lat=0.0,
                lon=0.021,
                via="Via Roja",
                comuna="Test",
                peso=13.0,
                tipo_evento="CONGESTION",
                duracion_hrs=0.5,
                dia_semana="Wednesday",
                franja_horaria="Punta AM (06-09h)",
            ),
        ]
    )

    exposure, risk_score, *_ = service._build_variant_analysis(
        path=path,
        context={"day": "Wednesday", "hour_bucket": "Punta AM (06-09h)"},
        via_factors={},
        variant_name="least_congestion",
        extra_minutes=35.0,
    )

    assert exposure.total_incident_segments == 1
    assert exposure.matched_incident_segments == 1
    assert exposure.congestion_segments == 1
    assert risk_score > 25


def test_geometry_postprocess_preserves_street_vertices_and_densifies_long_steps():
    geometry = [
        {"lat": 0.0, "lon": 0.0},
        {"lat": 0.00003, "lon": 0.0005},
        {"lat": -0.00003, "lon": 0.0010},
        {"lat": 0.0, "lon": 0.0025},
    ]

    processed = routing_service.RoutingService._postprocess_geometry(geometry)

    assert processed[0] == geometry[0]
    assert processed[-1] == geometry[-1]
    cursor = 0
    for point in processed:
        if point == geometry[cursor]:
            cursor += 1
            if cursor == len(geometry):
                break
    assert cursor == len(geometry)
    assert all(
        routing.haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) <= routing_service.MAX_GEOMETRY_STEP_KM + 0.001
        for a, b in zip(processed, processed[1:])
    )


def test_active_congestion_nearby_touch_gets_reduced_impact():
    service = routing_service.RoutingService()
    route_geometry = [
        {"lat": -36.83220, "lon": -73.05210},
        {"lat": -36.83210, "lon": -73.05210},
    ]
    active_line = {
        "type": "Feature",
        "properties": {
            "segment_id": "red-colo-colo",
            "via": "Colo Colo",
            "comuna": "Concepcion",
            "level": "high",
            "recency": "actual",
            "score": 70.0,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-73.05211, -36.83220],
                [-73.05211, -36.82480],
            ],
        },
    }

    impacts, coverage = service._active_congestion_segment_impacts(
        route_geometry,
        {service._normalize_road_name("Colo Colo")},
        {"route-segment"},
        [active_line],
    )

    assert len(impacts) == 1
    assert impacts[0].via == "Colo Colo"
    assert impacts[0].impact_score < 1.0
    assert coverage.high_pct > 0
    assert "Circula por" in impacts[0].reason


def test_active_congestion_parallel_street_does_not_count_as_route_congestion():
    service = routing_service.RoutingService()
    route_geometry = [
        {"lat": -36.83220, "lon": -73.05210},
        {"lat": -36.83210, "lon": -73.05210},
    ]
    active_line = {
        "type": "Feature",
        "properties": {
            "segment_id": "red-colo-colo",
            "via": "Colo Colo",
            "comuna": "Concepcion",
            "level": "high",
            "recency": "actual",
            "score": 70.0,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-73.05211, -36.83220],
                [-73.05211, -36.82480],
            ],
        },
    }

    impacts, coverage = service._active_congestion_segment_impacts(
        route_geometry,
        {service._normalize_road_name("Lincoyan")},
        {"route-segment"},
        [active_line],
    )

    assert impacts == []
    assert coverage.high_pct == 0.0


def test_active_congestion_perpendicular_street_does_not_penalize_crossing_nodes():
    service = routing_service.RoutingService()
    service.graph = routing.RouteGraph(
        nodes={
            "crossing": routing.GraphNode(
                "crossing", "route-lincoyan", 0, -36.83215, -73.05210,
                "Referencia", 30.0, 0.1, "Lincoyan", "Concepcion",
            ),
            "congested": routing.GraphNode(
                "congested", "red-colo-colo", 0, -36.83215, -73.05211,
                "Referencia", 30.0, 0.1, "Colo Colo", "Concepcion",
            ),
        },
        adjacency={},
    )
    perpendicular_line = {
        "type": "Feature",
        "properties": {
            "segment_id": "red-colo-colo",
            "via": "Colo Colo",
            "score": 70.0,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-73.05211, -36.83300],
                [-73.05211, -36.83100],
            ],
        },
    }

    penalties = service._active_congestion_node_penalties([perpendicular_line])

    assert "crossing" not in penalties
    assert penalties["congested"] > 1.0


def test_active_congestion_perpendicular_street_does_not_count_in_route_coverage():
    service = routing_service.RoutingService()
    route_geometry = [
        {"lat": -36.83215, "lon": -73.05300},
        {"lat": -36.83215, "lon": -73.05100},
    ]
    perpendicular_line = {
        "type": "Feature",
        "properties": {
            "segment_id": "red-colo-colo",
            "via": "Colo Colo",
            "comuna": "Concepcion",
            "level": "high",
            "recency": "actual",
            "score": 70.0,
        },
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [-73.05210, -36.83300],
                [-73.05210, -36.83100],
            ],
        },
    }

    impacts, coverage = service._active_congestion_segment_impacts(
        route_geometry,
        {service._normalize_road_name("Lincoyan")},
        {"route-lincoyan"},
        [perpendicular_line],
    )

    assert impacts == []
    assert coverage.congested_pct == 0.0


def test_destination_snap_uses_mid_block_perpendicular_and_oneway_target(monkeypatch):
    service = routing_service.RoutingService()
    events = pd.DataFrame(
        [
            {
                "segment_id": "segOneWay",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.1,
                "via": "Las Heras",
                "comuna": "Test",
                "oneway": True,
            },
            {
                "segment_id": "segOneWay",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.01,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.1,
                "via": "Las Heras",
                "comuna": "Test",
                "oneway": True,
            },
        ]
    )

    class DummyAirQualityService:
        def route_cost_factor(self, *_args):
            return 1.0

        def estimate_route_exposure(self, **_kwargs):
            return None

    class DummyUrbanWellbeingService:
        def route_cost_factor(self, *_args):
            return 1.0

        def candidate_waypoints(self, *_args, **_kwargs):
            return []

        def evaluate_route(self, *_args, **_kwargs):
            return None

    service.graph = routing.RouteGraph.from_events(events)
    service.segment_lookup = service._build_segment_lookup(events)
    monkeypatch.setattr(service, "_ensure_fresh_data", lambda: None)
    monkeypatch.setattr(routing_service, "get_air_quality_service", lambda: DummyAirQualityService())
    monkeypatch.setattr(routing_service, "get_urban_wellbeing_service", lambda: DummyUrbanWellbeingService())

    destination = RoutePoint(lat=0.001, lon=0.005)
    snap = service._nearest_road_snap(destination)

    assert snap.point["lat"] == pytest.approx(0.0, abs=1e-6)
    assert snap.point["lon"] == pytest.approx(0.005, abs=1e-6)
    assert set(snap.target_node_costs) == {"segOneWay::0"}
    assert snap.target_node_costs["segOneWay::0"] == pytest.approx(
        routing.haversine_km(0.0, 0.0, 0.0, 0.005)
    )

    payload = RouteRequest(
        origin=RoutePoint(lat=0.0, lon=0.0),
        destination=destination,
        preferences=[],
        avoid_congestion=False,
        avoid_accidents=False,
    )

    route = service.compute_route(payload)

    assert route.reference.steps[-2].node_id == "segOneWay::0"
    assert route.reference.steps[-1].node_id == "user_destination"
    assert route.reference.road_geometry[-1].lat == pytest.approx(0.0, abs=1e-6)
    assert route.reference.road_geometry[-1].lon == pytest.approx(0.005, abs=1e-6)
    assert route.reference.access_geometry[-1][0].lat == pytest.approx(0.0, abs=1e-6)
    assert route.reference.access_geometry[-1][0].lon == pytest.approx(0.005, abs=1e-6)


def test_road_snap_stops_when_plan_is_cancelled():
    service = routing_service.RoutingService()
    service.graph = object()
    service.segment_lookup = {"segment": {0: (0.0, 0.0), 1: (0.0, 0.01)}}

    with pytest.raises(routing.RouteSearchCancelled):
        service._nearest_road_snap(RoutePoint(lat=0.0, lon=0.005), should_cancel=lambda: True)
