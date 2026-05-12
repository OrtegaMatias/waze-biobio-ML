from __future__ import annotations

import argparse
import csv
import html
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}
FIELD_PATTERN = re.compile(r"<b>([^<]+):</b>\s*(.*?)\s*(?:<br>|$)", re.IGNORECASE)
JUMP_THRESHOLD_KM = 0.4
OUTPUT_COLUMNS = [
    "row_idx",
    "archivo_origen",
    "duracion_hrs",
    "distancia_km",
    "velocidad_kmh",
    "hora_inicio",
    "hora_fin",
    "comuna",
    "via",
    "indice_coord",
    "lon",
    "lat",
    "alt",
    "fecha",
    "segment_id",
    "fecha_dia",
    "reset_flag",
    "lat_prev",
    "lon_prev",
    "dist_from_prev_km",
    "__jump_thr__",
    "jump_flag",
    "new_segment_flag",
    "segment_seq",
]


@dataclass
class PointRow:
    archivo_origen: str
    duracion_hrs: float
    distancia_km: float
    velocidad_kmh: float
    hora_inicio: str
    hora_fin: str
    comuna: str
    via: str
    indice_coord: int
    lon: float
    lat: float
    alt: float
    fecha: str
    reset_flag: bool = False
    lat_prev: float | None = None
    lon_prev: float | None = None
    dist_from_prev_km: float | None = None
    jump_flag: bool = False
    new_segment_flag: bool = False
    segment_seq: int = 0
    segment_id: str = ""


def _fix_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    if not value:
        return ""
    if "Ã" in value or "Â" in value:
        try:
            value = value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return value.strip()


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value.strip("'\" ")


def _parse_description(raw_description: str | None) -> dict[str, str]:
    description = html.unescape(raw_description or "")
    fields: dict[str, str] = {}
    for key, value in FIELD_PATTERN.findall(description):
        clean_key = _fix_text(key)
        clean_value = _fix_text(re.sub(r"<[^>]+>", "", value))
        fields[clean_key] = clean_value
    return fields


def _parse_float(value: str | None) -> float:
    if not value:
        return math.nan
    clean = (
        value.replace("km/h", "")
        .replace("hrs", "")
        .replace("km", "")
        .replace(",", ".")
        .strip()
    )
    try:
        return float(clean)
    except ValueError:
        return math.nan


def _parse_date(value: str | None) -> str:
    if not value:
        return ""
    clean = _fix_text(value)
    for fmt in ("%d/%m/%Y", "%d_%m_%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return clean


def _parse_coordinates(raw_coordinates: str | None) -> list[tuple[float, float, float]]:
    coordinates: list[tuple[float, float, float]] = []
    if not raw_coordinates:
        return coordinates
    for item in raw_coordinates.split():
        parts = item.split(",")
        if len(parts) < 2:
            continue
        lon = float(parts[0])
        lat = float(parts[1])
        alt = float(parts[2]) if len(parts) > 2 and parts[2] else math.nan
        coordinates.append((lon, lat, alt))
    return coordinates


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _iter_rows(kml_paths: Iterable[Path]) -> list[PointRow]:
    rows: list[PointRow] = []
    previous_row: PointRow | None = None

    for kml_path in sorted(kml_paths, key=lambda path: path.name):
        current_segment_seq = 0
        tree = ET.parse(kml_path)
        document = tree.getroot()
        for placemark in document.findall(".//kml:Placemark", KML_NS):
            fields = _parse_description(placemark.findtext("kml:description", default="", namespaces=KML_NS))
            coords = _parse_coordinates(
                placemark.findtext(".//kml:LineString/kml:coordinates", default="", namespaces=KML_NS)
            )
            if not coords:
                continue

            fecha = _parse_date(fields.get("Fecha"))
            base = {
                "archivo_origen": kml_path.name,
                "duracion_hrs": _parse_float(fields.get("Duración")),
                "distancia_km": round(_parse_float(fields.get("Largo")), 2),
                "velocidad_kmh": _parse_float(fields.get("Velocidad")),
                "hora_inicio": _fix_text(fields.get("Hora Inicio")),
                "hora_fin": _fix_text(fields.get("Hora Fin")),
                "comuna": _fix_text(fields.get("Comuna")),
                "via": _strip_quotes(_fix_text(fields.get("Calle"))),
                "fecha": fecha,
            }

            for index, (lon, lat, alt) in enumerate(coords):
                row = PointRow(
                    indice_coord=index,
                    lon=lon,
                    lat=lat,
                    alt=alt,
                    **base,
                )

                if previous_row is not None:
                    same_file_as_previous = previous_row.archivo_origen == row.archivo_origen
                    if same_file_as_previous:
                        row.lat_prev = previous_row.lat
                        row.lon_prev = previous_row.lon
                        row.dist_from_prev_km = _haversine_km(
                            previous_row.lat,
                            previous_row.lon,
                            row.lat,
                            row.lon,
                        )
                    row.reset_flag = bool(index == 0 and same_file_as_previous)
                    row.jump_flag = bool(
                        same_file_as_previous
                        and not row.reset_flag
                        and row.dist_from_prev_km is not None
                        and row.dist_from_prev_km > JUMP_THRESHOLD_KM
                    )
                    row.new_segment_flag = bool(row.reset_flag or row.jump_flag)
                    if row.new_segment_flag:
                        current_segment_seq += 1
                row.segment_seq = current_segment_seq
                row.segment_id = f"{row.archivo_origen}::{row.segment_seq}"

                rows.append(row)
                previous_row = row

    return rows


def _stringify(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_csv(rows: list[PointRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row_idx, row in enumerate(rows):
            record = {
                "row_idx": row_idx,
                "archivo_origen": row.archivo_origen,
                "duracion_hrs": row.duracion_hrs,
                "distancia_km": row.distancia_km,
                "velocidad_kmh": row.velocidad_kmh,
                "hora_inicio": row.hora_inicio,
                "hora_fin": row.hora_fin,
                "comuna": row.comuna,
                "via": row.via,
                "indice_coord": row.indice_coord,
                "lon": row.lon,
                "lat": row.lat,
                "alt": row.alt,
                "fecha": row.fecha,
                "segment_id": row.segment_id,
                "fecha_dia": row.fecha,
                "reset_flag": row.reset_flag,
                "lat_prev": row.lat_prev,
                "lon_prev": row.lon_prev,
                "dist_from_prev_km": row.dist_from_prev_km,
                "__jump_thr__": JUMP_THRESHOLD_KM,
                "jump_flag": row.jump_flag,
                "new_segment_flag": row.new_segment_flag,
                "segment_seq": row.segment_seq,
            }
            writer.writerow({key: _stringify(value) for key, value in record.items()})


def _collect_inputs(input_paths: list[str]) -> list[Path]:
    results: list[Path] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        if path.is_dir():
            results.extend(sorted(path.glob("*.kml")))
        elif "*" in raw_path or "?" in raw_path:
            results.extend(sorted(path.parent.glob(path.name)))
        else:
            results.append(path)
    return [path for path in results if path.suffix.lower() == ".kml"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte KMLs de congestión al mismo esquema tabular de CONGESTIONES.csv."
    )
    parser.add_argument("inputs", nargs="+", help="Archivos KML, carpetas o globs a convertir.")
    parser.add_argument("--output", required=True, help="Ruta CSV de salida.")
    args = parser.parse_args()

    kml_paths = _collect_inputs(args.inputs)
    if not kml_paths:
        raise SystemExit("No se encontraron archivos KML para convertir.")

    rows = _iter_rows(kml_paths)
    write_csv(rows, Path(args.output))


if __name__ == "__main__":
    main()
