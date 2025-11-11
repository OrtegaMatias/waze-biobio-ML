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
    baseline = dict(graph.adjacency["segA::0"])["segA::1"]

    df_high = df.copy()
    df_high["penalty_factor"] = 2.0
    graph_penalized = routing.RouteGraph.from_events(df_high)
    penalized = dict(graph_penalized.adjacency["segA::0"])["segA::1"]

    assert penalized > baseline
