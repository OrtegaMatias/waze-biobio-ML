from __future__ import annotations

import pandas as pd

from algorithms.recommenders import routing
from backend.fastapi_app.app.services.routing_costs import (
    ENVIRONMENT_MIN_EDGE_FACTOR,
    RoutingCostModel,
    congestion_avoidance_penalty_factor,
    free_flow_speed_kmh,
)


def _node(node_id: str, lon: float, *, segment: str = "block-a", speed: float = 40.0) -> routing.GraphNode:
    return routing.GraphNode(
        node_id=node_id,
        segment_id=segment,
        segment_seq=0,
        lat=0.0,
        lon=lon,
        tipo_evento="Referencia",
        velocidad_kmh=speed,
        duracion_hrs=0.0,
        via="Calle de prueba",
        comuna="Concepcion",
    )


def test_free_flow_speed_uses_road_class_when_specific_speed_is_invalid():
    node = _node("a", 0.0, speed=0.0)
    node.road_class = "residential"

    assert free_flow_speed_kmh(node) == 30.0


def test_fastest_cost_uses_contextual_observed_speed():
    source = _node("a", 0.0)
    target = _node("b", 0.01)
    model = RoutingCostModel(
        congestion_scores={"b": 1.0},
        congestion_speeds_kmh={"b": 10.0},
    )

    cost = model.fastest(source, target, 1.0)

    assert cost.base_time_min == 1.5
    assert cost.travel_time_min == 6.0
    assert cost.congestion_delay_min == 4.5


def test_fluent_cost_equals_fastest_when_the_block_has_no_congestion():
    source = _node("a", 0.0)
    target = _node("b", 0.01)
    model = RoutingCostModel(segment_lengths_km={"block-a": 1.0})

    fastest = model.fastest(source, target, 1.0)
    fluent = model.fluent(source, target, 1.0)

    assert fluent.optimization_cost_min == fastest.optimization_cost_min


def test_congestion_penalty_prioritizes_avoiding_orange_and_red():
    green = congestion_avoidance_penalty_factor(0.25)
    orange = congestion_avoidance_penalty_factor(0.50)
    red = congestion_avoidance_penalty_factor(0.80)

    assert 0.0 < green < orange < red
    assert orange >= green * 5.0
    assert red >= orange * 2.0


def test_fluent_and_environmental_costs_strongly_penalize_red_congestion():
    source = _node("a", 0.0)
    target = _node("b", 0.01)
    model = RoutingCostModel(
        congestion_scores={"b": 0.8},
        congestion_speeds_kmh={"b": 10.0},
        segment_lengths_km={"block-a": 1.0},
    )

    fastest = model.fastest(source, target, 1.0)
    fluent = model.fluent(source, target, 1.0)
    environmental = model.environmental(source, target, 1.0)

    assert fluent.optimization_cost_min > fastest.optimization_cost_min * 10.0
    assert environmental.optimization_cost_min > fastest.optimization_cost_min * 7.0


def test_environmental_benefits_never_make_an_edge_free_or_negative():
    source = _node("a", 0.0)
    target = _node("b", 0.01)
    model = RoutingCostModel(wellbeing_factor=lambda _node: 0.0)

    cost = model.environmental(source, target, 1.0)

    assert cost.optimization_cost_min >= cost.travel_time_min * ENVIRONMENT_MIN_EDGE_FACTOR
    assert cost.optimization_cost_min > 0


def test_custom_edge_cost_keeps_objective_cost_separate_from_distance_limit():
    rows = []
    for segment_id, coords in {
        "direct": [(0.0, 0.0), (0.0, 0.01)],
        "detour-a": [(0.0, 0.0), (0.01, 0.0)],
        "detour-b": [(0.01, 0.0), (0.0, 0.01)],
    }.items():
        for seq, (lat, lon) in enumerate(coords):
            rows.append(
                {
                    "segment_id": segment_id,
                    "segment_seq": seq,
                    "lat": lat,
                    "lon": lon,
                    "tipo_evento": "Referencia",
                    "velocidad_kmh": 40.0,
                    "duracion_hrs": 0.0,
                    "via": segment_id,
                    "comuna": "Concepcion",
                    "penalty_factor": 1.0,
                    "dia_semana": "",
                    "franja_horaria": "",
                    "oneway": False,
                }
            )
    graph = routing.RouteGraph.from_events(pd.DataFrame(rows))

    path = graph.shortest_path(
        (0.0, 0.0),
        (0.0, 0.01),
        edge_cost=lambda _source, _target, distance: distance * 10.0,
        use_heuristic=False,
        geographic_path_limit_km=1.5,
    )

    assert path
    assert {step.segment_id for step in path} == {"direct"}
