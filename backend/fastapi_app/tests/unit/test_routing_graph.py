import pandas as pd
import pytest

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


def test_shortest_path_prunes_nodes_outside_geographic_path_limit():
    def node(node_id: str, lat: float, lon: float) -> routing.GraphNode:
        return routing.GraphNode(
            node_id=node_id,
            segment_id=node_id,
            segment_seq=0,
            lat=lat,
            lon=lon,
            tipo_evento="Referencia",
            velocidad_kmh=30.0,
            duracion_hrs=0.1,
            via="Test",
            comuna="Test",
        )

    nodes = {
        "source": node("source", 0.0, 0.0),
        "direct": node("direct", 0.0, 0.001),
        "far": node("far", 0.02, 0.001),
        "target": node("target", 0.0, 0.002),
    }
    graph = routing.RouteGraph(
        nodes,
        {
            "source": [("direct", 5.0), ("far", 1.0)],
            "direct": [("target", 5.0)],
            "far": [("target", 1.0)],
            "target": [],
        },
    )

    path = graph.shortest_path(
        (0.0, 0.0),
        (0.0, 0.002),
        apply_penalties=False,
        source_node_costs={"source": 0.0},
        target_node_costs={"target": 0.0},
        geographic_path_limit_km=0.3,
    )

    assert [step.node_id for step in path] == ["source", "direct", "target"]


def test_shortest_path_enforces_real_accumulated_distance_limit():
    def node(node_id: str, lat: float, lon: float) -> routing.GraphNode:
        return routing.GraphNode(
            node_id, node_id, 0, lat, lon, "Referencia", 30.0, 0.1, node_id, "Test"
        )

    nodes = {
        "source": node("source", 0.0, 0.0),
        "direct": node("direct", 0.0, 0.001),
        "detour-a": node("detour-a", 0.001, 0.0005),
        "detour-b": node("detour-b", -0.001, 0.001),
        "detour-c": node("detour-c", 0.001, 0.0015),
        "target": node("target", 0.0, 0.002),
    }
    graph = routing.RouteGraph(
        nodes,
        {
            "source": [("direct", 5.0), ("detour-a", 1.0)],
            "direct": [("target", 5.0)],
            "detour-a": [("detour-b", 1.0)],
            "detour-b": [("detour-c", 1.0)],
            "detour-c": [("target", 1.0)],
            "target": [],
        },
    )

    path = graph.shortest_path(
        (0.0, 0.0),
        (0.0, 0.002),
        apply_penalties=False,
        source_node_costs={"source": 0.0},
        target_node_costs={"target": 0.0},
        geographic_path_limit_km=0.3,
    )

    assert [step.node_id for step in path] == ["source", "direct", "target"]


def test_shortest_path_can_ignore_global_historical_penalties():
    def node(node_id: str, lat: float, lon: float, penalty: float = 1.0) -> routing.GraphNode:
        return routing.GraphNode(
            node_id,
            node_id,
            0,
            lat,
            lon,
            "Referencia",
            30.0,
            0.1,
            node_id,
            "Test",
            penalty_factor=penalty,
        )

    nodes = {
        "source": node("source", 0.0, 0.0),
        "historical": node("historical", 0.0, 0.001, penalty=2.75),
        "clear-detour": node("clear-detour", 0.001, 0.001),
        "target": node("target", 0.0, 0.002),
    }
    graph = routing.RouteGraph(
        nodes,
        {
            "source": [("historical", 0.11), ("clear-detour", 0.16)],
            "historical": [("target", 0.11)],
            "clear-detour": [("target", 0.16)],
            "target": [],
        },
    )
    common = {
        "source_node_costs": {"source": 0.0},
        "target_node_costs": {"target": 0.0},
        "incident_ctx": {"avoid_congestion": True},
        "apply_penalties": True,
    }

    historical = graph.shortest_path((0.0, 0.0), (0.0, 0.002), **common)
    contextual = graph.shortest_path(
        (0.0, 0.0),
        (0.0, 0.002),
        **common,
        apply_historical_penalties=False,
    )

    assert "clear-detour" in [step.node_id for step in historical]
    assert "historical" in [step.node_id for step in contextual]


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


def test_nearest_node_uses_spatial_index_and_exclude():
    df = _base_segment(oneway=False)
    graph = routing.RouteGraph.from_events(df)

    source = graph.nearest_node(-36.82, -73.05)
    target = graph.nearest_node(-36.82, -73.05, exclude={source.node_id})

    assert source.node_id == "segA::0"
    assert target.node_id == "segA::1"


def test_astar_matches_dijkstra_with_all_cost_factors():
    def node(node_id: str, lat: float, lon: float, via: str) -> routing.GraphNode:
        return routing.GraphNode(
            node_id, node_id, 0, lat, lon, "Referencia", 30.0, 0.1, via, "Test"
        )

    nodes = {
        "start": node("start", 0.0, 0.0, "Inicio"),
        "direct": node("direct", 0.0, 0.001, "Ruta directa"),
        "preferred": node("preferred", 0.001, 0.001, "Ruta preferida"),
        "target": node("target", 0.0, 0.002, "Destino"),
    }
    graph = routing.RouteGraph(
        nodes=nodes,
        adjacency={
            "start": [("direct", 0.12), ("preferred", 0.16)],
            "direct": [("target", 0.12)],
            "preferred": [("target", 0.16)],
            "target": [],
        },
        minimum_geographic_weight_ratio=0.7,
    )
    kwargs = {
        "source_node_costs": {"start": 0.0},
        "target_node_costs": {"target": 0.0},
        "via_factors": {"Ruta preferida": 0.5},
        "air_quality_factor": lambda item: 0.8 if item.node_id == "preferred" else 1.0,
        "urban_wellbeing_factor": lambda item: 0.7 if item.node_id == "preferred" else 1.0,
        "apply_penalties": False,
    }

    astar = graph.shortest_path((0.0, 0.0), (0.0, 0.002), **kwargs)
    dijkstra = graph.shortest_path((0.0, 0.0), (0.0, 0.002), **kwargs, use_heuristic=False)

    assert [step.node_id for step in astar] == [step.node_id for step in dijkstra]
    assert abs(astar[-1].peso - dijkstra[-1].peso) < 1e-12


def test_astar_explores_fewer_edges_than_dijkstra():
    def node(node_id: str, lat: float, lon: float) -> routing.GraphNode:
        return routing.GraphNode(
            node_id, node_id, 0, lat, lon, "Referencia", 30.0, 0.1, node_id, "Test"
        )

    nodes = {
        f"main-{index}": node(f"main-{index}", 0.0, index * 0.001)
        for index in range(11)
    }
    adjacency = {node_id: [] for node_id in nodes}
    for index in range(10):
        source = f"main-{index}"
        target = f"main-{index + 1}"
        adjacency[source].append(
            (target, routing.haversine_km(nodes[source].lat, nodes[source].lon, nodes[target].lat, nodes[target].lon))
        )
    for index in range(20):
        branch_id = f"branch-{index}"
        nodes[branch_id] = node(branch_id, 0.003, 0.0)
        branch_cost = routing.haversine_km(0.0, 0.0, nodes[branch_id].lat, nodes[branch_id].lon)
        adjacency["main-0"].append((branch_id, branch_cost))
        adjacency[branch_id] = [(branch_id, 0.01)]

    graph = routing.RouteGraph(
        nodes=nodes,
        adjacency=adjacency,
        minimum_geographic_weight_ratio=1.0,
    )
    astar_edges = 0
    dijkstra_edges = 0

    def count_astar(_source, _target):
        nonlocal astar_edges
        astar_edges += 1
        return True

    def count_dijkstra(_source, _target):
        nonlocal dijkstra_edges
        dijkstra_edges += 1
        return True

    common = {
        "source_node_costs": {"main-0": 0.0},
        "target_node_costs": {"main-10": 0.0},
        "apply_penalties": False,
    }
    astar = graph.shortest_path((0.0, 0.0), (0.0, 0.01), **common, edge_filter=count_astar)
    dijkstra = graph.shortest_path(
        (0.0, 0.0),
        (0.0, 0.01),
        **common,
        edge_filter=count_dijkstra,
        use_heuristic=False,
    )

    assert [step.node_id for step in astar] == [step.node_id for step in dijkstra]
    assert astar_edges < dijkstra_edges


def test_astar_stops_when_the_caller_cancels():
    nodes = {
        f"node-{index}": routing.GraphNode(
            f"node-{index}",
            "segment",
            index,
            0.0,
            index * 0.0001,
            "Referencia",
            30.0,
            0.1,
            "Ruta",
            "Test",
        )
        for index in range(400)
    }
    adjacency = {node_id: [] for node_id in nodes}
    for index in range(399):
        source = f"node-{index}"
        target = f"node-{index + 1}"
        adjacency[source].append((target, 0.02))
    graph = routing.RouteGraph(nodes=nodes, adjacency=adjacency)
    cancellation_checks = 0

    def should_cancel():
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks > 1

    with pytest.raises(routing.RouteSearchCancelled):
        graph.shortest_path(
            (0.0, 0.0),
            (0.0, 0.0399),
            source_node_costs={"node-0": 0.0},
            target_node_costs={"node-399": 0.0},
            should_cancel=should_cancel,
        )
