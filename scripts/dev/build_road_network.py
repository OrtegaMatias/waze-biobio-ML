# -*- coding: utf-8 -*-
"""
Descarga la red vial desde OpenStreetMap y la serializa como CSV para reforzar el grafo.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, MultiLineString

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "road_network.csv"
DEFAULT_PLACE = "Región del Biobío, Chile"


ROAD_CLASS_DEFAULT_SPEED_KMH = {
    "motorway": 50.0,
    "motorway_link": 50.0,
    "trunk": 50.0,
    "trunk_link": 50.0,
    "primary": 50.0,
    "primary_link": 45.0,
    "secondary": 45.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 35.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "unclassified": 30.0,
}


def parse_maxspeed(value, highway: str = "") -> tuple[float, str]:
    fallback = min(50.0, ROAD_CLASS_DEFAULT_SPEED_KMH.get(str(highway or "").lower(), 40.0))
    if value is None:
        return fallback, "road_class"
    if isinstance(value, (list, tuple, set)):
        value = next(iter(value), fallback)
    if isinstance(value, str):
        value = value.replace("km/h", "").replace("mph", "").strip()
    try:
        parsed = float(value)
        if parsed <= 0 or parsed > 130:
            return fallback, "road_class"
        return parsed, "osm_maxspeed"
    except (TypeError, ValueError):
        return fallback, "road_class"


def flatten_coords(geometry) -> List[Tuple[float, float]]:
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
    elif isinstance(geometry, MultiLineString):
        coords: List[Tuple[float, float]] = []
        for line in geometry.geoms:
            coords.extend(line.coords)
    else:
        coords = []
    return [(float(lon), float(lat)) for lon, lat in coords]


def infer_comuna(edge_attrs: dict, node_attrs: Sequence[dict], default_place: str) -> str:
    candidates = [
        edge_attrs.get("addr:city"),
        edge_attrs.get("name:es"),
        edge_attrs.get("ref"),
    ]
    for attrs in node_attrs:
        candidates.extend(
            [
                attrs.get("addr:city"),
                attrs.get("city"),
                attrs.get("name"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().title()
    return default_place


def normalize_oneway(value) -> Tuple[bool, str]:
    if value is None:
        return False, "both"
    value_str = str(value).strip().lower()
    if value_str in {"false", "no", "0"}:
        return False, "both"
    if value_str == "-1":
        return True, "reverse"
    if value_str in {"1", "true", "yes"}:
        return True, "forward"
    return bool(value_str), "both"


def iter_edge_rows(G, place_label: str) -> Iterable[dict]:
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        coords = flatten_coords(geom)
        if not coords:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            coords = [(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])]
        oneway_flag, direction = normalize_oneway(data.get("oneway"))
        if direction == "reverse":
            coords = list(reversed(coords))
        name = data.get("name") or data.get("name:es") or "Sin nombre"
        length_m = float(data.get("length") or 0.0)
        length_km = length_m / 1000
        highway = str(data.get("highway") or "")
        maxspeed, maxspeed_source = parse_maxspeed(data.get("maxspeed"), highway)
        duration_hours = length_km / max(maxspeed, 5.0) if length_km > 0 else 0.0
        if isinstance(data.get("osmid"), (list, tuple, set)):
            osmid = next(iter(data["osmid"]))
        else:
            osmid = data.get("osmid") or f"{u}-{v}"
        segment_id = f"osm::{osmid}::{u}->{v}"
        comuna = infer_comuna(data, [G.nodes[u], G.nodes[v]], default_place=place_label)
        for idx, (lon, lat) in enumerate(coords):
            yield {
                "row_idx": idx,
                "archivo_origen": f"osm::{place_label}",
                "duracion_hrs": duration_hours,
                "distancia_km": length_km,
                "velocidad_kmh": maxspeed,
                "hora_inicio": "00:00",
                "hora_fin": "00:00",
                "comuna": comuna,
                "via": name,
                "indice_coord": idx,
                "lon": lon,
                "lat": lat,
                "alt": 0.0,
                "fecha": "2025-01-01",
                "segment_id": segment_id,
                "oneway": oneway_flag,
                "highway": highway,
                "motor_vehicle": str(data.get("motor_vehicle") or data.get("access") or ""),
                "maxspeed_source": maxspeed_source,
            }


def build_dataframe(place: str, dist: int | None, lat: float | None, lon: float | None, network_type: str) -> pd.DataFrame:
    if place:
        graph = ox.graph_from_place(place, network_type=network_type, simplify=True)
        label = place
    else:
        if lat is None or lon is None or dist is None:
            raise ValueError("Debe proporcionar --place o bien los parámetros --lat, --lon y --dist.")
        graph = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type, simplify=True)
        label = f"point::{lat},{lon}"
    rows = list(iter_edge_rows(graph, label))
    if not rows:
        raise RuntimeError("No se obtuvieron segmentos desde OSM.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera data/processed/road_network.csv desde OSM.")
    parser.add_argument("--place", default=os.getenv("ROAD_NETWORK_PLACE", DEFAULT_PLACE))
    parser.add_argument("--network-type", default=os.getenv("ROAD_NETWORK_TYPE", "drive"))
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--dist", type=int, default=None, help="Radio en metros si usa lat/lon.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    df = build_dataframe(args.place, args.dist, args.lat, args.lon, args.network_type)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Se escribieron {len(df)} filas en {output_path}")


if __name__ == "__main__":
    main()
