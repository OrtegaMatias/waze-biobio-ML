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


def test_nearby_junction_does_not_enter_oneway_from_exit_endpoint():
    df = pd.DataFrame(
        [
            {
                "segment_id": "oneway",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Un Sentido",
                "comuna": "Test",
                "oneway": True,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "oneway",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.0002,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Un Sentido",
                "comuna": "Test",
                "oneway": True,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "cross",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.00021,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Cruce",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "cross",
                "segment_seq": 1,
                "lat": 0.0002,
                "lon": 0.00021,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Cruce",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
            },
        ]
    )
    graph = routing.RouteGraph.from_events(df)

    cross_outgoing = dict(graph.adjacency["cross::0"])
    oneway_exit_outgoing = dict(graph.adjacency["oneway::1"])

    assert "oneway::1" not in cross_outgoing
    assert "cross::0" in oneway_exit_outgoing


def test_route_graph_two_way_connects_both_sides():
    df = _base_segment(oneway=False)
    graph = routing.RouteGraph.from_events(df)
    outgoing = dict(graph.adjacency["segA::0"])
    incoming = dict(graph.adjacency["segA::1"])

    assert "segA::1" in outgoing
    assert "segA::0" in incoming


def test_route_graph_connects_nearby_segments():
    df = pd.DataFrame(
        [
            {
                "segment_id": "segA",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta A",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segA",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.00020,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta A",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segB",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.00022,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta B",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segB",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.00042,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta B",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
            },
        ]
    )
    graph = routing.RouteGraph.from_events(df)

    path = graph.shortest_path((0.0, 0.0), (0.0, 0.00042), apply_penalties=False)

    assert path
    assert path[-1].segment_id == "segB"


def test_shortest_path_respects_edge_filter():
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
            comuna="Test",
        )

    nodes = {
        "source": node("source", 0.0, 0.0, "Inicio"),
        "blocked": node("blocked", 0.0, 0.001, "Interior protegido"),
        "detour": node("detour", 0.001, 0.001, "Corredor permitido"),
        "target": node("target", 0.0, 0.002, "Destino"),
    }
    graph = routing.RouteGraph(
        nodes,
        {
            "source": [("blocked", 1.0), ("detour", 2.0)],
            "blocked": [("target", 1.0)],
            "detour": [("target", 2.0)],
            "target": [],
        },
    )

    path = graph.shortest_path(
        (0.0, 0.0),
        (0.0, 0.002),
        apply_penalties=False,
        source_node_costs={"source": 0.0},
        target_node_costs={"target": 0.0},
        edge_filter=lambda _source, target: target.node_id != "blocked",
    )

    assert [step.node_id for step in path] == ["source", "detour", "target"]


def test_route_graph_penalizes_normalized_congestion_event_types():
    df = pd.DataFrame(
        [
            {
                "segment_id": "direct",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.1,
                "via": "Directa",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
                "dia_semana": "Wednesday",
                "franja_horaria": "Punta AM (06-09h)",
            },
            {
                "segment_id": "direct",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.003,
                "tipo_evento": "CONGESTION",
                "velocidad_kmh": 8.0,
                "duracion_hrs": 0.5,
                "via": "Directa",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
                "dia_semana": "Wednesday",
                "franja_horaria": "Punta AM (06-09h)",
            },
            {
                "segment_id": "detour",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.1,
                "via": "Alternativa",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
                "dia_semana": "Wednesday",
                "franja_horaria": "Punta AM (06-09h)",
            },
            {
                "segment_id": "detour",
                "segment_seq": 1,
                "lat": 0.001,
                "lon": 0.0015,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.1,
                "via": "Alternativa",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
                "dia_semana": "Wednesday",
                "franja_horaria": "Punta AM (06-09h)",
            },
            {
                "segment_id": "detour",
                "segment_seq": 2,
                "lat": 0.0,
                "lon": 0.003,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.1,
                "via": "Alternativa",
                "comuna": "Test",
                "oneway": False,
                "penalty_factor": 1.0,
                "dia_semana": "Wednesday",
                "franja_horaria": "Punta AM (06-09h)",
            },
        ]
    )
    graph = routing.RouteGraph.from_events(df)

    reference = graph.shortest_path((0.0, 0.0), (0.0, 0.003), apply_penalties=False)
    penalized = graph.shortest_path(
        (0.0, 0.0),
        (0.0, 0.003),
        incident_ctx={
            "day": "Wednesday",
            "hour_bucket": "Punta AM (06-09h)",
            "avoid_congestion": True,
        },
        apply_penalties=True,
    )

    assert reference[-1].segment_id == "direct"
    assert penalized[-1].segment_id == "detour"


def test_route_graph_connects_bridge_gaps_between_components():
    df = pd.DataFrame(
        [
            {
                "segment_id": "segBridge",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Puente Llacolen",
                "comuna": "Concepcion",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segBridge",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.001,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Puente Llacolen",
                "comuna": "Concepcion",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segApproach",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.018,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Acceso San Pedro",
                "comuna": "San Pedro De La Paz",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segApproach",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.019,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Acceso San Pedro",
                "comuna": "San Pedro De La Paz",
                "oneway": False,
                "penalty_factor": 1.0,
            },
        ]
    )
    graph = routing.RouteGraph.from_events(df)

    path = graph.shortest_path((0.0, 0.0), (0.0, 0.019), apply_penalties=False)

    assert path
    assert path[-1].segment_id == "segApproach"


def test_route_graph_does_not_connect_long_non_bridge_gaps():
    df = pd.DataFrame(
        [
            {
                "segment_id": "segA",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.0,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta A",
                "comuna": "Concepcion",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segA",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.001,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta A",
                "comuna": "Concepcion",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segB",
                "segment_seq": 0,
                "lat": 0.0,
                "lon": 0.018,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta B",
                "comuna": "San Pedro De La Paz",
                "oneway": False,
                "penalty_factor": 1.0,
            },
            {
                "segment_id": "segB",
                "segment_seq": 1,
                "lat": 0.0,
                "lon": 0.019,
                "tipo_evento": "Referencia",
                "velocidad_kmh": 30.0,
                "duracion_hrs": 0.2,
                "via": "Ruta B",
                "comuna": "San Pedro De La Paz",
                "oneway": False,
                "penalty_factor": 1.0,
            },
        ]
    )
    graph = routing.RouteGraph.from_events(df)

    path = graph.shortest_path((0.0, 0.0), (0.0, 0.019), apply_penalties=False)

    assert path == []


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


def test_shortest_path_applies_air_quality_factor():
    nodes = {
        "start": routing.GraphNode("start", "seg_start", 0, 0.0, 0.0, "Referencia", 30.0, 0.1, "Inicio", "Test"),
        "dirty": routing.GraphNode("dirty", "seg_dirty", 1, 0.1, 0.0, "Referencia", 30.0, 0.1, "Ruta Corta", "Test"),
        "clean": routing.GraphNode("clean", "seg_clean", 1, 0.0, 0.1, "Referencia", 30.0, 0.1, "Ruta Limpia", "Test"),
        "end": routing.GraphNode("end", "seg_end", 2, 0.2, 0.0, "Referencia", 30.0, 0.1, "Fin", "Test"),
    }
    adjacency = {
        "start": [("dirty", 1.0), ("clean", 1.3)],
        "dirty": [("end", 1.0)],
        "clean": [("end", 1.3)],
    }
    graph = routing.RouteGraph(nodes=nodes, adjacency=adjacency)

    path = graph.shortest_path(
        (0.0, 0.0),
        (0.2, 0.0),
        air_quality_factor=lambda node: 2.0 if node.via == "Ruta Corta" else 1.0,
        apply_penalties=False,
    )

    vias = [step.via for step in path]
    assert "Ruta Limpia" in vias
    assert "Ruta Corta" not in vias


def test_shortest_path_applies_urban_wellbeing_factor():
    nodes = {
        "start": routing.GraphNode("start", "seg_start", 0, 0.0, 0.0, "Referencia", 30.0, 0.1, "Inicio", "Test"),
        "plain": routing.GraphNode("plain", "seg_plain", 1, 0.1, 0.0, "Referencia", 30.0, 0.1, "Ruta Corta", "Test"),
        "pleasant": routing.GraphNode("pleasant", "seg_pleasant", 1, 0.0, 0.1, "Referencia", 30.0, 0.1, "Ruta Parque", "Test"),
        "end": routing.GraphNode("end", "seg_end", 2, 0.2, 0.0, "Referencia", 30.0, 0.1, "Fin", "Test"),
    }
    adjacency = {
        "start": [("plain", 1.0), ("pleasant", 1.15)],
        "plain": [("end", 1.0)],
        "pleasant": [("end", 1.15)],
    }
    graph = routing.RouteGraph(nodes=nodes, adjacency=adjacency)

    path = graph.shortest_path(
        (0.0, 0.0),
        (0.2, 0.0),
        urban_wellbeing_factor=lambda node: 0.7 if node.via == "Ruta Parque" else 1.0,
        apply_penalties=False,
    )

    vias = [step.via for step in path]
    assert "Ruta Parque" in vias
    assert "Ruta Corta" not in vias


def test_shortest_path_uses_reachable_destination_candidate():
    nodes = {
        "start": routing.GraphNode("start", "seg_start", 0, 0.0, 0.0, "Referencia", 30.0, 0.1, "Inicio", "Test"),
        "mid": routing.GraphNode("mid", "seg_mid", 1, 0.0, 0.001, "Referencia", 30.0, 0.1, "Via Media", "Test"),
        "reachable_end": routing.GraphNode(
            "reachable_end", "seg_end", 2, 0.0, 0.002, "Referencia", 30.0, 0.1, "Fin Alcanzable", "Test"
        ),
        "blocked_end": routing.GraphNode(
            "blocked_end", "seg_blocked", 2, 0.0, 0.00205, "Referencia", 30.0, 0.1, "Fin Aislado", "Test"
        ),
    }
    adjacency = {
        "start": [("mid", 1.0)],
        "mid": [("reachable_end", 1.0)],
    }
    graph = routing.RouteGraph(nodes=nodes, adjacency=adjacency)

    path = graph.shortest_path((0.0, 0.0), (0.0, 0.00205), apply_penalties=False)

    assert path
    assert path[-1].node_id == "reachable_end"


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


def test_accident_penalty_only_applies_when_requested():
    nodes = {
        "start": routing.GraphNode("start", "seg_start", 0, 0.0, 0.0, "Referencia", 30.0, 0.1, "Inicio", "Test"),
        "accident": routing.GraphNode(
            "accident",
            "segAccident",
            1,
            0.1,
            0.0,
            "Accidente",
            30.0,
            0.1,
            "Via Accidente",
            "Test",
            penalty_factor=1.75,
            accident_penalty_factor=1.75,
            dia_semana="Monday",
            franja_horaria="Punta AM (06-09h)",
        ),
        "alt": routing.GraphNode(
            "alt", "segAlt", 1, 0.0, 0.1, "Referencia", 30.0, 0.1, "Via Alterna", "Test"
        ),
        "end": routing.GraphNode("end", "seg_end", 2, 0.2, 0.1, "Referencia", 30.0, 0.1, "Fin", "Test"),
    }
    adjacency = {
        "start": [("accident", 1.0), ("alt", 1.0)],
        "accident": [("end", 1.0)],
        "alt": [("end", 1.0)],
    }
    graph = routing.RouteGraph(nodes=nodes, adjacency=adjacency)

    unfiltered = graph.shortest_path(
        (0.0, 0.0),
        (0.2, 0.1),
        incident_ctx={"day": "Monday", "hour_bucket": "Punta AM (06-09h)", "avoid_congestion": False, "avoid_accidents": False},
    )
    filtered = graph.shortest_path(
        (0.0, 0.0),
        (0.2, 0.1),
        incident_ctx={"day": "Monday", "hour_bucket": "Punta AM (06-09h)", "avoid_congestion": False, "avoid_accidents": True},
    )

    assert "Via Accidente" in [step.via for step in unfiltered]
    assert "Via Accidente" not in [step.via for step in filtered]


def test_nearest_node_uses_spatial_index_and_exclude():
    df = _base_segment(oneway=False)
    graph = routing.RouteGraph.from_events(df)

    source = graph.nearest_node(-36.82, -73.05)
    target = graph.nearest_node(-36.82, -73.05, exclude={source.node_id})

    assert source.node_id == "segA::0"
    assert target.node_id == "segA::1"
