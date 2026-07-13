# -*- coding: utf-8 -*-
"""Build the urban wellbeing GeoJSON layer from OpenStreetMap/Overpass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from shapely.geometry import mapping, shape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data_processed" / "gran_concepcion_urban_wellbeing.geojson"
DEFAULT_RAW = PROJECT_ROOT / "data_processed" / "gran_concepcion_urban_wellbeing_overpass.json"
OVERPASS_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
GRAN_CONCEPCION_BBOX = (-37.05, -73.25, -36.68, -72.85)


def overpass_query(bbox: tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    bounds = f"{south},{west},{north},{east}"
    return f"""
    [out:json][timeout:240];
    (
      node["leisure"~"^(park|garden|nature_reserve|community_garden)$"]({bounds});
      way["leisure"~"^(park|garden|nature_reserve|community_garden)$"]({bounds});
      way["landuse"~"^(forest|recreation_ground|village_green)$"]({bounds});
      way["natural"~"^(wood|wetland|water|tree_row)$"]({bounds});
      way["waterway"~"^(river|stream|canal)$"]({bounds});
      node["place"="square"]({bounds});
      way["place"="square"]({bounds});
      node["amenity"="recycling"]({bounds});
      way["amenity"="recycling"]({bounds});
    );
    out geom;
    """


def classify(tags: dict) -> tuple[str, str, float] | None:
    leisure = str(tags.get("leisure") or "")
    landuse = str(tags.get("landuse") or "")
    natural = str(tags.get("natural") or "")
    if leisure in {"park", "garden", "nature_reserve", "community_garden"}:
        return "green_space", leisure, 1.0 if leisure in {"park", "nature_reserve"} else 0.8
    if landuse in {"forest", "recreation_ground", "village_green"}:
        return "green_space", landuse, 0.85
    if natural in {"wood", "wetland"}:
        return "green_space", natural, 0.9
    if natural == "water" or tags.get("waterway"):
        return "blue_space", str(tags.get("water") or tags.get("waterway") or "water"), 1.0
    if natural == "tree_row":
        return "tree_cover", "tree_row", 1.0
    if tags.get("place") == "square":
        return "public_space", "square", 0.8
    if tags.get("amenity") == "recycling":
        return "sustainability", "recycling", 0.7
    return None


def geometry_for(element: dict, category: str) -> dict | None:
    if element.get("type") == "node" and "lon" in element and "lat" in element:
        return {"type": "Point", "coordinates": [float(element["lon"]), float(element["lat"])]}
    points = element.get("geometry") or []
    coordinates = [
        [float(point["lon"]), float(point["lat"])]
        for point in points
        if isinstance(point, dict) and "lon" in point and "lat" in point
    ]
    if len(coordinates) < 2:
        return None
    if coordinates[0] == coordinates[-1] and len(coordinates) >= 4 and category not in {"tree_cover"}:
        return {"type": "Polygon", "coordinates": [coordinates]}
    return {"type": "LineString", "coordinates": coordinates}


def simplify_geometry(geometry: dict) -> dict:
    simplified = shape(geometry).simplify(0.00004, preserve_topology=True)
    result = mapping(simplified)

    def rounded(value):
        if isinstance(value, (list, tuple)):
            return [rounded(item) for item in value]
        if isinstance(value, float):
            return round(value, 6)
        return value

    return {"type": result["type"], "coordinates": rounded(result["coordinates"])}


def build_geojson(payload: dict) -> dict:
    features = []
    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        classification = classify(tags)
        if classification is None:
            continue
        category, subtype, base_weight = classification
        geometry = geometry_for(element, category)
        if geometry is None:
            continue
        geometry = simplify_geometry(geometry)
        osm_type = str(element.get("type") or "element")
        osm_id = element.get("id")
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "feature_id": f"osm-{osm_type}-{osm_id}",
                    "osm_id": osm_id,
                    "osm_type": osm_type,
                    "name": tags.get("name") or subtype.replace("_", " ").title(),
                    "category": category,
                    "subtype": subtype,
                    "base_weight": base_weight,
                    "access": tags.get("access") or "unknown",
                    "source": "OpenStreetMap/Overpass",
                },
                "geometry": geometry,
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "gran_concepcion_urban_wellbeing",
        "features": features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera la capa de bienestar urbano desde OSM/Overpass.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--raw-json", default=str(DEFAULT_RAW))
    parser.add_argument("--from-raw", action="store_true")
    args = parser.parse_args()

    raw_path = Path(args.raw_json)
    if args.from_raw:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        last_error: Exception | None = None
        payload = None
        for url in OVERPASS_URLS:
            try:
                response = requests.post(
                    url,
                    data={"data": overpass_query(GRAN_CONCEPCION_BBOX)},
                    headers={"User-Agent": "waze-biobio-ml-thesis/1.0"},
                    timeout=300,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except Exception as exc:
                last_error = exc
        if payload is None:
            raise RuntimeError(f"No se pudo descargar la capa desde Overpass: {last_error}")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = build_geojson(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Se escribieron {len(result['features'])} elementos de bienestar urbano en {output}")


if __name__ == "__main__":
    main()
