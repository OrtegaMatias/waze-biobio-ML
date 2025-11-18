# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
import pytest

from backend.fastapi_app.app.schemas.routes import RoutePoint, RouteRequest, ViaPreference
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

    monkeypatch.setattr(routing_service.data_loader, "load_raw_events", lambda: events)
    monkeypatch.setattr(routing_service.data_loader, "data_version", fake_data_version)
    monkeypatch.setattr(routing_service, "_load_graph_cache", lambda signature: seen_signatures.append(signature))
    monkeypatch.setattr(routing_service, "_store_graph_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(routing_service.routing.RouteGraph, "from_events", classmethod(fake_from_events))
    monkeypatch.setattr(routing_service.RoutingService, "_build_segment_lookup", staticmethod(lambda _events: {}))

    service._build_structures()
    service._build_structures()

    assert seen_signatures == signature_values


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

    assert len(dummy_graph.calls) == 2
    assert dummy_graph.calls[1]["default_via_factor"] == 1.0


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
