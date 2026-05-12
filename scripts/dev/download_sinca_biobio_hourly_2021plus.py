from __future__ import annotations

import csv
import html
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


BASE_URL = "https://sinca.mma.gob.cl"
REGION_URL = f"{BASE_URL}/index.php/region/index/id/VIII"
START_YYMMDDHH = "21010100"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; thesis-prototype-downloader/1.0)",
}

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "air_quality" / "sinca_biobio_hourly_2021plus"
RAW_DIR = OUTPUT_DIR / "raw_txt"
CSV_DIR = OUTPUT_DIR / "series_csv"

POLLUTANT_LABELS = {
    "0001": "SO2",
    "0002": "NO",
    "0003": "NO2",
    "0004": "CO",
    "0008": "O3",
    "0NOX": "NOX",
    "PM10": "PM10",
    "PM25": "PM25",
}
MET_LABELS = {
    "GLOB": "global_radiation",
    "PRES": "pressure",
    "RAIN": "rain",
    "RHUM": "relative_humidity",
    "TEMP": "temperature",
    "WDIR": "wind_direction",
    "WSPD": "wind_speed",
}


@dataclass
class Station:
    station_id: str
    station_name: str
    station_url: str


@dataclass
class SeriesLink:
    station_id: str
    station_name: str
    category: str
    parameter_code: str
    parameter_label: str
    macro: str
    macropath: str
    from_raw: str
    to_raw: str
    export_url: str
    source_url: str


def fetch_text(url: str, timeout: int = 60) -> str:
    request = Request(url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read()
    return body.decode(charset, errors="replace")


def sanitize_name(value: str) -> str:
    normalized = re.sub(r"\s+", "_", value.strip().lower())
    normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
    return normalized.strip("_") or "unknown"


def parse_region_stations(region_html: str) -> list[Station]:
    pattern = re.compile(
        r'href="(?P<href>/index\.php/estacion/index/id/(?P<id>\d+))"[^>]*>(?P<name>[^<]+)</a>',
        re.IGNORECASE,
    )
    stations: list[Station] = []
    seen: set[str] = set()
    for match in pattern.finditer(region_html):
        station_id = match.group("id")
        if station_id in seen:
            continue
        seen.add(station_id)
        stations.append(
            Station(
                station_id=station_id,
                station_name=html.unescape(match.group("name")).strip(),
                station_url=urljoin(BASE_URL, match.group("href")),
            )
        )
    return stations


def parse_station_metadata(station: Station, station_html: str) -> dict[str, str]:
    text = html.unescape(station_html)
    compact = re.sub(r"\s+", " ", text)

    def capture_table_value(label: str) -> str:
        pattern = re.compile(
            rf"<th[^>]*>\s*{re.escape(label)}\s*</th>\s*<td[^>]*>\s*(.*?)\s*</td>",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(station_html)
        if not match:
            return ""
        value = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        return re.sub(r"\s+", " ", value).strip()

    latlon_match = re.search(
        r"new google\.maps\.LatLng\(\s*([\-0-9.]+)\s*,\s*([\-0-9.]+)\s*\)",
        station_html,
        re.IGNORECASE,
    )
    latitude = latlon_match.group(1) if latlon_match else ""
    longitude = latlon_match.group(2) if latlon_match else ""

    metadata = {
        "station_id": station.station_id,
        "station_name": station.station_name,
        "station_url": station.station_url,
        "owner": capture_table_value("Propietario"),
        "operator": capture_table_value("Operador"),
        "region": capture_table_value("Región"),
        "province": capture_table_value("Provincia"),
        "commune": capture_table_value("Comuna"),
        "utm_coordinates": capture_table_value("Coordenadas UTM"),
        "timezone_huso": capture_table_value("Huso horario"),
        "online_reception": capture_table_value("Recepción de datos"),
        "reported_operation_start": capture_table_value("Inicio de operación reportada"),
        "latitude": latitude,
        "longitude": longitude,
    }
    return metadata


def infer_parameter(macropath: str) -> tuple[str, str, str]:
    cleaned = macropath.replace("\\", "/")
    parts = [part for part in cleaned.split("/") if part and part != "."]
    category = "unknown"
    parameter_code = parts[-1] if parts else "unknown"
    if "Cal" in parts:
        category = "pollutant"
        return category, parameter_code, POLLUTANT_LABELS.get(parameter_code, parameter_code)
    if "Met" in parts:
        category = "meteorology"
        suffix = ""
        return category, parameter_code, MET_LABELS.get(parameter_code, parameter_code) + suffix
    return category, parameter_code, parameter_code


def parse_station_series(station: Station, station_html: str) -> list[SeriesLink]:
    href_pattern = re.compile(
        r'href=["\'](?P<href>[^"\']*apub\.htmlindico2\.cgi\?page=pageFrame[^"\']+)["\']',
        re.IGNORECASE,
    )
    series: list[SeriesLink] = []
    seen: set[tuple[str, str]] = set()
    for match in href_pattern.finditer(station_html):
        href = html.unescape(match.group("href"))
        if href.startswith("//"):
            href = "https:" + href
        elif href.startswith("/"):
            href = urljoin(BASE_URL, href)
        parsed = urlparse(href)
        query = {key: values[0] for key, values in parse_qs(parsed.query).items() if values}
        macro = query.get("macro", "")
        if "horario" not in macro.lower():
            continue
        macropath = query.get("macropath", "")
        category, parameter_code, parameter_label = infer_parameter(macropath)
        key = (macropath, macro)
        if key in seen:
            continue
        seen.add(key)
        export_query = {
            "outtype": "txt",
            "macro": f"{macropath}//{macro}.ic",
            "from": max(START_YYMMDDHH, query.get("from", "") + "00"),
            "to": query.get("to", "") + "23",
            "path": "/usr/airviro/data/CONAMA/",
            "lang": "esp",
            "rsrc": "",
            "macropath": "",
        }
        series.append(
            SeriesLink(
                station_id=station.station_id,
                station_name=station.station_name,
                category=category,
                parameter_code=parameter_code,
                parameter_label=parameter_label,
                macro=macro,
                macropath=macropath,
                from_raw=query.get("from", ""),
                to_raw=query.get("to", ""),
                export_url=f"{BASE_URL}/cgi-bin/APUB-MMA/apub.tsindico2.cgi?{urlencode(export_query)}",
                source_url=href,
            )
        )
    return series


def parse_txt_series(content: str) -> list[dict[str, str | float | None]]:
    rows: list[dict[str, str | float | None]] = []
    in_data = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not in_data:
            if line == "#DATA":
                in_data = True
            continue
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        date_part = parts[0]
        time_part = parts[1]
        if not re.fullmatch(r"\d{6}", date_part) or not re.fullmatch(r"\d{4}", time_part):
            continue
        year_prefix = "19" if int(date_part[:2]) >= 90 else "20"
        validated_value = parse_numeric(parts[2]) if len(parts) >= 3 else None
        preliminary_value = parse_numeric(parts[3]) if len(parts) >= 4 else None
        unvalidated_value = parse_numeric(parts[4]) if len(parts) >= 5 else None
        rows.append(
            {
                "date_code": date_part,
                "time_code": time_part,
                "datetime_local": f"{year_prefix}{date_part[:2]}-{date_part[2:4]}-{date_part[4:6]} {time_part[:2]}:00:00",
                "validated_value": validated_value,
                "preliminary_value": preliminary_value,
                "unvalidated_value": unvalidated_value,
            }
        )
    return rows


def parse_numeric(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned == "@":
        return None
    cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def choose_preferred_value(row: dict[str, str | float | None]) -> float | None:
    for key in ("validated_value", "preliminary_value", "unvalidated_value"):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def write_series_csv(series_link: SeriesLink, rows: Iterable[dict[str, str | float | None]]) -> tuple[Path, int]:
    station_slug = sanitize_name(series_link.station_name)
    parameter_slug = sanitize_name(series_link.parameter_label)
    macro_slug = sanitize_name(series_link.macro)
    csv_path = (
        CSV_DIR
        / f"{series_link.station_id}__{station_slug}__{series_link.category}__{parameter_slug}__{macro_slug}.csv"
    )
    count = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "station_id",
                "station_name",
                "category",
                "parameter_code",
                "parameter_label",
                "macro",
                "datetime_local",
                "validated_value",
                "preliminary_value",
                "unvalidated_value",
                "preferred_value",
                "source_url",
                "export_url",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "station_id": series_link.station_id,
                    "station_name": series_link.station_name,
                    "category": series_link.category,
                    "parameter_code": series_link.parameter_code,
                    "parameter_label": series_link.parameter_label,
                    "macro": series_link.macro,
                    "datetime_local": row["datetime_local"],
                    "validated_value": row["validated_value"],
                    "preliminary_value": row["preliminary_value"],
                    "unvalidated_value": row["unvalidated_value"],
                    "preferred_value": choose_preferred_value(row),
                    "source_url": series_link.source_url,
                    "export_url": series_link.export_url,
                }
            )
            count += 1
    return csv_path, count


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching region page: {REGION_URL}")
    region_html = fetch_text(REGION_URL)
    stations = parse_region_stations(region_html)
    print(f"Found {len(stations)} stations in Biobio.")

    manifest_rows: list[dict[str, str | int]] = []
    station_metadata_rows: list[dict[str, str]] = []

    for index, station in enumerate(stations, start=1):
        print(f"[{index}/{len(stations)}] Station {station.station_name} ({station.station_id})")
        try:
            station_html = fetch_text(station.station_url)
            station_metadata_rows.append(parse_station_metadata(station, station_html))
            series_links = parse_station_series(station, station_html)
        except Exception as exc:  # pragma: no cover
            print(f"  ERROR reading station page: {exc}", file=sys.stderr)
            manifest_rows.append(
                {
                    "station_id": station.station_id,
                    "station_name": station.station_name,
                    "category": "",
                    "parameter_code": "",
                    "parameter_label": "",
                    "rows": 0,
                    "status": f"station_page_error: {exc}",
                    "raw_path": "",
                    "csv_path": "",
                }
            )
            continue

        print(f"  Found {len(series_links)} hourly series.")
        for series_link in series_links:
            station_slug = sanitize_name(series_link.station_name)
            parameter_slug = sanitize_name(series_link.parameter_label)
            macro_slug = sanitize_name(series_link.macro)
            raw_path = (
                RAW_DIR
                / f"{series_link.station_id}__{station_slug}__{series_link.category}__{parameter_slug}__{macro_slug}.txt"
            )
            try:
                content = fetch_text(series_link.export_url, timeout=120)
                raw_path.write_text(content, encoding="utf-8")
                rows = parse_txt_series(content)
                csv_path, row_count = write_series_csv(series_link, rows)
                status = "ok"
            except Exception as exc:  # pragma: no cover
                print(
                    f"  ERROR downloading {series_link.parameter_label} for {series_link.station_name}: {exc}",
                    file=sys.stderr,
                )
                csv_path = Path("")
                row_count = 0
                status = f"download_error: {exc}"
            manifest_rows.append(
                {
                    "station_id": series_link.station_id,
                    "station_name": series_link.station_name,
                    "category": series_link.category,
                    "parameter_code": series_link.parameter_code,
                    "parameter_label": series_link.parameter_label,
                    "rows": row_count,
                    "status": status,
                    "raw_path": str(raw_path.relative_to(ROOT_DIR)) if raw_path.exists() else "",
                    "csv_path": str(csv_path.relative_to(ROOT_DIR)) if csv_path else "",
                }
            )
            time.sleep(0.25)

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "station_id",
                "station_name",
                "category",
                "parameter_code",
                "parameter_label",
                "rows",
                "status",
                "raw_path",
                "csv_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    station_metadata_path = OUTPUT_DIR / "stations_metadata.csv"
    with station_metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "station_id",
                "station_name",
                "station_url",
                "owner",
                "operator",
                "region",
                "province",
                "commune",
                "utm_coordinates",
                "timezone_huso",
                "online_reception",
                "reported_operation_start",
                "latitude",
                "longitude",
            ],
        )
        writer.writeheader()
        writer.writerows(station_metadata_rows)

    ok_count = sum(1 for row in manifest_rows if row["status"] == "ok")
    print(f"Finished. {ok_count} series downloaded.")
    print(f"Manifest: {manifest_path}")
    print(f"Station metadata: {station_metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
