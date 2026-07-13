# -*- coding: utf-8 -*-
"""
Build a lightweight GeoJSON layer with cycling infrastructure for Gran Concepcion.

The script can fetch fresh data from Overpass or process existing Overpass JSON
files from cache/. It intentionally writes an independent layer and does not
modify the routed road network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_processed" / "gran_concepcion_cycleways.geojson"
DEFAULT_MINVU_OUTPUT = PROJECT_ROOT / "data_processed" / "gran_concepcion_cycleways_minvu.geojson"
DEFAULT_OVERPASS_JSON = PROJECT_ROOT / "data_processed" / "gran_concepcion_cycleways_overpass.json"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
MINVU_CYCLEWAYS_URL = "https://geoide.minvu.cl/server/rest/services/Planes_Programas/Ciclov%C3%ADas_Minvu/FeatureServer/1/query"
MINVU_CYCLEWAYS_SOURCE_URL = "https://geoide.minvu.cl/server/rest/services/Planes_Programas/Ciclov%C3%ADas_Minvu/FeatureServer/1"

# south, west, north, east
GRAN_CONCEPCION_BBOX = (-37.05, -73.25, -36.68, -72.85)
CYCLEWAY_VALUES = {"lane", "track", "opposite", "opposite_lane", "opposite_track", "shared_lane", "share_busway"}
BICYCLE_VALUES = {"yes", "designated"}
TAG_KEYS = (
    "highway",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "bicycle",
    "name",
    "surface",
    "oneway",
)


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _minvu_category(stage: str | None) -> str:
    stage_key = (stage or "").lower()
    if "dise" in stage_key or "proyect" in stage_key:
        return "minvu_planned_cycleway"
    if "ejec" in stage_key or "obra" in stage_key:
        return "minvu_in_execution_cycleway"
    return "minvu_existing_cycleway"


def _should_include_minvu_stage(stage: str | None, include_planned: bool) -> bool:
    return include_planned or _minvu_category(stage) != "minvu_planned_cycleway"


def overpass_query(bbox: tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    bounds = f"{south},{west},{north},{east}"
    return f"""
    [out:json][timeout:180];
    (
      way["highway"="cycleway"]({bounds});
      way["cycleway"]({bounds});
      way["cycleway:left"]({bounds});
      way["cycleway:right"]({bounds});
      way["cycleway:both"]({bounds});
      way["bicycle"~"^(yes|designated)$"]({bounds});
      way["cycleway"="shared_lane"]({bounds});
      way["cycleway:left"="shared_lane"]({bounds});
      way["cycleway:right"="shared_lane"]({bounds});
    );
    (._;>;);
    out body;
    """


def fetch_overpass(output_json: Path, bbox: tuple[float, float, float, float]) -> dict:
    response = requests.post(OVERPASS_URL, data={"data": overpass_query(bbox)}, timeout=240)
    response.raise_for_status()
    payload = response.json()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def fetch_minvu(output_json: Path, *, include_planned: bool) -> dict:
    params = {
        "f": "json",
        "where": "REGION='08'",
        "outFields": (
            "OBJECTID,EJE_1,PROYECTO_1,Capa,CUT_2010_2011,REGION,PROVINCIA,COMUNA,IDI_1,"
            "KILOMETROS_1,ESTADO_DE_AVANCE,CODIGO_CEHU,Linea_incluyendo_2016,INICIO_OBRA,TERMINO_OBRA,Sistema"
        ),
        "returnGeometry": "true",
        "outSR": "4326",
        "resultRecordCount": "2000",
    }
    response = requests.get(MINVU_CYCLEWAYS_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    geojson = build_minvu_geojson(payload, include_planned=include_planned)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    return geojson


def iter_payloads(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Saltando {path}: {exc}")
            continue
        if isinstance(payload, dict):
            yield payload
        elif isinstance(payload, list):
            yield {"elements": payload}


def has_cycleway_tags(tags: dict) -> bool:
    if tags.get("highway") == "cycleway":
        return True
    if str(tags.get("cycleway", "")).lower() in CYCLEWAY_VALUES:
        return True
    if str(tags.get("cycleway:left", "")).lower() in CYCLEWAY_VALUES:
        return True
    if str(tags.get("cycleway:right", "")).lower() in CYCLEWAY_VALUES:
        return True
    if str(tags.get("cycleway:both", "")).lower() in CYCLEWAY_VALUES:
        return True
    if str(tags.get("bicycle", "")).lower() in BICYCLE_VALUES:
        return True
    return False


def classify(tags: dict) -> str:
    if tags.get("highway") == "cycleway":
        return "segregated_cycleway"
    values = {
        str(tags.get("cycleway", "")).lower(),
        str(tags.get("cycleway:left", "")).lower(),
        str(tags.get("cycleway:right", "")).lower(),
        str(tags.get("cycleway:both", "")).lower(),
    }
    if "track" in values or "opposite_track" in values:
        return "cycle_track"
    if "lane" in values or "opposite_lane" in values:
        return "cycle_lane"
    if "shared_lane" in values:
        return "shared_lane"
    if str(tags.get("bicycle", "")).lower() in BICYCLE_VALUES:
        return "bicycle_access"
    return "cycling_infrastructure"


def in_bbox(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def build_geojson(payloads: Iterable[dict], bbox: tuple[float, float, float, float]) -> dict:
    nodes: dict[int, tuple[float, float]] = {}
    candidate_ways: dict[int, dict] = {}
    for payload in payloads:
        for element in payload.get("elements", []):
            if element.get("type") == "node" and "lat" in element and "lon" in element:
                nodes[int(element["id"])] = (float(element["lon"]), float(element["lat"]))
            elif element.get("type") == "way":
                tags = element.get("tags") or {}
                if has_cycleway_tags(tags):
                    candidate_ways[int(element["id"])] = element

    features = []
    for way_id, way in sorted(candidate_ways.items()):
        coords = [nodes[node_id] for node_id in way.get("nodes", []) if node_id in nodes]
        coords = [(lon, lat) for lon, lat in coords if in_bbox(lon, lat, bbox)]
        if len(coords) < 2:
            continue
        tags = way.get("tags") or {}
        properties = {key.replace(":", "_"): tags.get(key) for key in TAG_KEYS if tags.get(key) is not None}
        properties.update(
            {
                "osm_id": way_id,
                "name": tags.get("name") or "Sin nombre",
                "category": classify(tags),
                "source": "OpenStreetMap/Overpass",
            }
        )
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )

    return {
        "type": "FeatureCollection",
        "name": "gran_concepcion_cycleways",
        "features": features,
    }


def build_minvu_geojson(payload: dict, *, include_planned: bool) -> dict:
    features = []
    for raw_feature in payload.get("features", []):
        attributes = raw_feature.get("attributes") or {}
        geometry = raw_feature.get("geometry") or {}
        stage = _clean_text(attributes.get("ETAPA"))
        if not _should_include_minvu_stage(stage, include_planned):
            continue

        for path_index, path in enumerate(geometry.get("paths") or []):
            coords = []
            for point in path:
                if not isinstance(point, list) or len(point) < 2:
                    continue
                lon, lat = float(point[0]), float(point[1])
                if in_bbox(lon, lat, GRAN_CONCEPCION_BBOX):
                    coords.append((lon, lat))
            if len(coords) < 2:
                continue

            object_id = attributes.get("OBJECTID")
            stage = _clean_text(attributes.get("ESTADO_DE_AVANCE")) or stage
            eje = _clean_text(attributes.get("EJE_1")) or "Sin nombre"
            properties = {
                "minvu_id": object_id,
                "minvu_path_index": path_index,
                "name": eje,
                "category": _minvu_category(stage),
                "source": "MINVU GeoIDE",
                "source_detail": "Ciclovias medida presidencial FeatureServer/1",
                "source_url": MINVU_CYCLEWAYS_SOURCE_URL,
                "comuna": _clean_text(attributes.get("COMUNA")),
                "project": _clean_text(attributes.get("PROYECTO_1")),
                "km": attributes.get("KILOMETROS_1"),
                "type": _clean_text(attributes.get("Capa")),
                "project_code": _clean_text(attributes.get("IDI_1")),
                "cehu_code": _clean_text(attributes.get("CODIGO_CEHU")),
                "system": _clean_text(attributes.get("Sistema")),
                "stage": stage,
                "stage_detail": _clean_text(attributes.get("Linea_incluyendo_2016")),
            }
            features.append(
                {
                    "type": "Feature",
                    "properties": {key: value for key, value in properties.items() if value is not None},
                    "geometry": {"type": "LineString", "coordinates": coords},
                }
            )

    return {
        "type": "FeatureCollection",
        "name": "gran_concepcion_cycleways_minvu",
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera GeoJSON de ciclovias desde OSM/Overpass y MINVU.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--minvu-output", default=str(DEFAULT_MINVU_OUTPUT))
    parser.add_argument("--overpass-json", default=str(DEFAULT_OVERPASS_JSON))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--from-cache", action="store_true", help="Procesa JSON existentes en cache/ sin red.")
    parser.add_argument("--input-json", action="append", default=[], help="JSON Overpass local; se puede repetir.")
    parser.add_argument("--bbox", default=",".join(str(v) for v in GRAN_CONCEPCION_BBOX))
    parser.add_argument("--skip-overpass", action="store_true", help="No actualiza la capa OSM/Overpass.")
    parser.add_argument("--include-minvu", action="store_true", help="Descarga y escribe la capa oficial MINVU.")
    parser.add_argument(
        "--include-minvu-planned",
        action="store_true",
        help="Incluye ciclovias MINVU en etapa de diseno/proyectadas.",
    )
    args = parser.parse_args()

    bbox_parts = tuple(float(part.strip()) for part in args.bbox.split(","))
    if len(bbox_parts) != 4:
        raise ValueError("--bbox debe venir como south,west,north,east")
    bbox = bbox_parts  # type: ignore[assignment]

    input_paths = [Path(path) for path in args.input_json]
    if args.from_cache:
        input_paths.extend(sorted(Path(args.cache_dir).glob("*.json")))

    if not args.skip_overpass:
        if input_paths:
            payloads = iter_payloads(input_paths)
        else:
            payloads = [fetch_overpass(Path(args.overpass_json), bbox)]

        geojson = build_geojson(payloads, bbox)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Se escribieron {len(geojson['features'])} ciclovias OSM/Overpass en {output_path}")

    if args.include_minvu:
        minvu_path = Path(args.minvu_output)
        minvu_geojson = fetch_minvu(minvu_path, include_planned=args.include_minvu_planned)
        print(f"Se escribieron {len(minvu_geojson['features'])} ciclovias MINVU en {minvu_path}")


if __name__ == "__main__":
    main()
