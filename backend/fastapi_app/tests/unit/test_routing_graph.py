import pandas as pd

from algorithms.recommenders import routing


def _base_segment(oneway: bool):
    return pd.DataFrame(
        [
            {
                "segment_id": "segA",
                "segment_seq": 0,
                "lat": -36.82,
                "lon": -73.05,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta Azul",
                "comuna": "Test",
                "oneway": oneway,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segA",
                "segment_seq": 1,
                "lat": -36.821,
                "lon": -73.049,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta Azul",
                "comuna": "Test",
                "oneway": oneway,
                "penalty_factor": 1.0,
            },
        ]
    )


def test_route_graph_respects_oneway_edges():
    df = _base_segment(oneway=True)
    graph = routing.RouteGraph.from_events(df)

    outgoing = dict(graph.adjacency["segA::0"])
    assert "segA::1" in outgoing

    incoming = [neighbor for neighbor, _ in graph.adjacency.get("segA::1", [])]
    assert "segA::0" not in incoming


def test_route_graph_two_way_connects_both_sides():
    df = _base_segment(oneway=False)
    graph = routing.RouteGraph.from_events(df)
    outgoing = dict(graph.adjacency["segA::0"])
    incoming = dict(graph.adjacency["segA::1"])

    assert "segA::1" in outgoing
    assert "segA::0" in incoming


def test_edge_weight_increases_with_penalty():
    df = _base_segment(oneway=False)
    graph = routing.RouteGraph.from_events(df)
    baseline_path = graph.shortest_path(
        (-36.82, -73.05),
        (-36.821, -73.049),
        apply_penalties=True,
    )
    baseline_cost = baseline_path[-1].peso

    df_high = df.copy()
    df_high["penalty_factor"] = 2.0
    graph_penalized = routing.RouteGraph.from_events(df_high)
    penalized_path = graph_penalized.shortest_path(
        (-36.82, -73.05),
        (-36.821, -73.049),
        apply_penalties=True,
    )
    penalized_cost = penalized_path[-1].peso

    assert penalized_cost > baseline_cost


def test_shortest_path_applies_preferences():
    nodes = {
        "start": routing.GraphNode("start", "seg_start", 0, 0.0, 0.0, "Referencia", 30.0, 0.1, "Inicio", "Test"),
        "viaA": routing.GraphNode("viaA", "segA", 1, 0.1, 0.1, "Referencia", 30.0, 0.1, "Ruta Azul", "Test"),
        "viaB": routing.GraphNode("viaB", "segB", 1, 0.1, -0.1, "Referencia", 30.0, 0.1, "Ruta Roja", "Test"),
        "end": routing.GraphNode("end", "seg_end", 2, 0.2, 0.0, "Referencia", 30.0, 0.1, "Fin", "Test"),
    }
    adjacency = {
        "start": [("viaA", 1.0), ("viaB", 1.0)],
        "viaA": [("end", 1.0)],
        "viaB": [("end", 1.0)],
    }
    graph = routing.RouteGraph(nodes=nodes, adjacency=adjacency)
    pref_path = graph.shortest_path(
        (0.0, 0.0),
        (0.2, 0.0),
        via_factors={"Ruta Roja": 0.5},
        default_via_factor=1.3,
    )
    vias = [step.via for step in pref_path]
    assert "Ruta Roja" in vias


def test_congestion_penalty_forces_alternate_path():
    nodes = {
        "start": routing.GraphNode("start", "seg_start", 0, 0.0, 0.0, "Referencia", 30.0, 0.1, "Inicio", "Test"),
        "cong": routing.GraphNode(
            "cong",
            "segCong",
            1,
            0.1,
            0.0,
            "Congestión",
            10.0,
            0.5,
            "Via Congestionada",
            "Test",
            dia_semana="Monday",
            franja_horaria="Punta AM (06-09h)",
        ),
        "alt": routing.GraphNode(
            "alt",
            "segAlt",
            1,
            0.0,
            0.1,
            "Referencia",
            30.0,
            0.1,
            "Via Alterna",
            "Test",
        ),
        "end": routing.GraphNode("end", "seg_end", 2, 0.2, 0.1, "Referencia", 30.0, 0.1, "Fin", "Test"),
    }
    adjacency = {
        "start": [("cong", 1.0), ("alt", 1.0)],
        "cong": [("end", 1.0)],
        "alt": [("end", 1.0)],
    }
    graph = routing.RouteGraph(nodes=nodes, adjacency=adjacency)
    path = graph.shortest_path(
        (0.0, 0.0),
        (0.2, 0.1),
        via_factors=None,
        default_via_factor=1.0,
        incident_ctx={"day": "Monday", "hour_bucket": "Punta AM (06-09h)", "avoid_congestion": True, "avoid_accidents": False},
    )
    vias = [step.via for step in path]
    assert "Via Congestionada" not in vias
