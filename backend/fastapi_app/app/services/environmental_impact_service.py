# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
from shapely.geometry import LineString, Point, mapping
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from ..schemas.routes import (
    EnvironmentalImpactPoint,
    EnvironmentalImpactResponse,
    EnvironmentalImpactSummary,
    EnvironmentalWeatherSummary,
)
from .air_quality_service import get_air_quality_service

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[4]
CONGESTION_CORE_PATH = ROOT_DIR / "data_analysis" / "congestion_clean_gran_concepcion_core.csv"
CONGESTION_REGIONAL_PATH = ROOT_DIR / "data_analysis" / "congestion_clean.csv"
RAIN_NETWORK_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_rain_network_hourly.csv"
WIND_NETWORK_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_wind_network_hourly.csv"
SINCA_MANIFEST_PATH = ROOT_DIR / "data" / "air_quality" / "sinca_biobio_hourly_2021plus" / "manifest.csv"
ENVIRONMENTAL_NORMALIZATION_PATH = ROOT_DIR / "data_analysis" / "environmental_normalization_v1.json"
ENVIRONMENTAL_YEAR = 2025
MAX_POINTS = 280
CONGESTION_CACHE_DIR = ROOT_DIR / "data" / "cache"
CONGESTION_CACHE_SCHEMA_VERSION = 2
ZONE_BUFFER_METERS = {
    "low": 80.0,
    "medium": 120.0,
    "high": 170.0,
}
CONGESTION_WEIGHT = 0.55
PM25_WEIGHT = 0.45
LAYER_PM25_WEIGHT = 0.45
LAYER_CONGESTION_WEIGHT = 0.30
LAYER_LOW_WIND_WEIGHT = 0.25
WIND_KMH_PER_MPS = 3.6
MEMORY_PREVIOUS_HOUR_WEIGHT = 0.25
MEMORY_PREVIOUS_HOUR_ADVERSE_WEATHER_WEIGHT = 0.10
MEMORY_TWO_HOURS_WEIGHT = 0.10
MEMORY_LOW_WIND_MPS = 1.30
INFLUENCE_BAND_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
BUILTIN_NORMALIZATION_REFERENCE = {
    "version": "environmental-normalization-v1",
    "reference_period": {
        "start": "2021-01-01T00:00:00",
        "end_exclusive": "2025-01-01T00:00:00",
    },
    "congestion_reference_period": {
        "start": "2025-03-13T00:00:00",
        "end_exclusive": "2025-08-23T00:00:00",
    },
    "variables": {
        "pm25": {"p10": 3.0, "p50": 10.0, "p90": 36.0, "sample_size": 274986},
        "wind_speed": {"p10": 0.79712, "p50": 1.821, "p90": 3.45762, "sample_size": 35063},
        "congestion_speed_kmh": {
            "p10": 10.54,
            "p50": 18.48,
            "p90": 23.66,
            "sample_size": 29529,
        },
        "congestion_duration_min": {
            "p10": 15.0,
            "p50": 30.0,
            "p90": 109.8,
            "sample_size": 29529,
        },
    },
}


@dataclass(frozen=True)
class WeatherSnapshot:
    rain_mm: float | None
    wind_speed: float | None
    wet_station_count: int | None
    global_radiation: float | None


@dataclass(frozen=True)
class EnvironmentalImpactCandidate:
    lat: float
    lon: float
    raw_score: float
    congestion_score: float
    pm25: float | None
    segment_id: str
    via: str | None
    comuna: str | None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _level(score: float) -> str:
    if score < 35:
        return "low"
    if score < 65:
        return "medium"
    return "high"


def _local_scores(raw_scores: list[float]) -> list[float]:
    if not raw_scores:
        return []
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    span = max_score - min_score
    if span <= 1e-9:
        return [round(_clamp(score / 100.0) * 100, 1) for score in raw_scores]
    return [round(_clamp((score - min_score) / span) * 100, 1) for score in raw_scores]


def _local_component(value: float | None, value_range: tuple[float, float] | None, *, invert: bool = False) -> float:
    if value is None or value_range is None:
        return 0.5
    low, high = value_range
    if high <= low:
        return 0.5
    normalized = _clamp((value - low) / (high - low))
    return 1.0 - normalized if invert else normalized


def _wind_kmh(wind_speed: float | None) -> float | None:
    return wind_speed * WIND_KMH_PER_MPS if wind_speed is not None else None


def _dominant_level(points: list[EnvironmentalImpactPoint]) -> str:
    if not points:
        return "none"
    counts = {"high": 0, "medium": 0, "low": 0}
    for point in points:
        counts[point.level] += 1
    return max(counts, key=lambda key: (counts[key], {"high": 2, "medium": 1, "low": 0}[key]))


def _empty_zone_collection() -> dict:
    return {"type": "FeatureCollection", "features": []}


def _canonical_line_key(feature: dict) -> tuple:
    coordinates = feature.get("geometry", {}).get("coordinates", [])
    forward = tuple(tuple(float(value) for value in point[:2]) for point in coordinates)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def _deduplicate_congestion_line_features(features: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for feature in features:
        grouped.setdefault(_canonical_line_key(feature), []).append(feature)

    consolidated: list[dict] = []
    for matching_features in grouped.values():
        selected = max(
            matching_features,
            key=lambda feature: (
                feature["properties"].get("recency") == "actual",
                float(feature["properties"].get("score") or 0),
                float(feature["properties"].get("environmental_score") or 0),
            ),
        )
        selected = {
            **selected,
            "properties": {
                **selected["properties"],
                "observation_count": len(matching_features),
            },
        }
        consolidated.append(selected)
    return consolidated


def _project_lon_lat(lon: float, lat: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(origin_lat))
    return (
        (lon - origin_lon) * meters_per_degree_lon,
        (lat - origin_lat) * meters_per_degree_lat,
    )


def _unproject_xy(x: float, y: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(origin_lat))
    return (
        origin_lon + x / meters_per_degree_lon,
        origin_lat + y / meters_per_degree_lat,
    )


def _unproject_coordinates(coordinates, origin_lon: float, origin_lat: float):
    if not coordinates:
        return coordinates
    first = coordinates[0]
    if isinstance(first, (float, int)):
        lon, lat = _unproject_xy(float(coordinates[0]), float(coordinates[1]), origin_lon, origin_lat)
        return [round(lon, 6), round(lat, 6)]
    return [_unproject_coordinates(item, origin_lon, origin_lat) for item in coordinates]


def _unproject_geometry(geometry, origin_lon: float, origin_lat: float) -> dict:
    mapped = mapping(geometry)
    return {
        "type": mapped["type"],
        "coordinates": _unproject_coordinates(mapped["coordinates"], origin_lon, origin_lat),
    }


class EnvironmentalImpactService:
    def __init__(
        self,
        congestion_path: Path = CONGESTION_CORE_PATH,
        rain_path: Path = RAIN_NETWORK_PATH,
        wind_path: Path = WIND_NETWORK_PATH,
        radiation_manifest_path: Path = SINCA_MANIFEST_PATH,
        normalization_path: Path = ENVIRONMENTAL_NORMALIZATION_PATH,
        year: int = ENVIRONMENTAL_YEAR,
        include_radiation: bool = False,
    ) -> None:
        self.congestion_path = congestion_path if congestion_path.exists() else CONGESTION_REGIONAL_PATH
        self.rain_path = rain_path
        self.wind_path = wind_path
        self.radiation_manifest_path = radiation_manifest_path
        self.normalization_path = normalization_path
        self.year = year
        self.include_radiation = include_radiation
        self._congestion: pd.DataFrame | None = None
        self._rain: pd.DataFrame | None = None
        self._wind: pd.DataFrame | None = None
        self._radiation: pd.DataFrame | None = None
        self._congestion_hours: set[datetime] | None = None
        self._normalization_reference: dict | None = None
        self._normalization_source: str | None = None

    @staticmethod
    def _validate_normalization_reference(payload: dict) -> None:
        if payload.get("version") != "environmental-normalization-v1":
            raise ValueError("Version de referencia ambiental no soportada.")
        variables = payload.get("variables") or {}
        for variable in ("pm25", "wind_speed", "congestion_speed_kmh", "congestion_duration_min"):
            reference = variables.get(variable) or {}
            p10 = float(reference["p10"])
            p50 = float(reference["p50"])
            p90 = float(reference["p90"])
            sample_size = int(reference["sample_size"])
            if not all(math.isfinite(value) for value in (p10, p50, p90)):
                raise ValueError(f"Referencia no finita para {variable}.")
            if not p10 <= p50 <= p90 or p90 <= p10 or sample_size <= 0:
                raise ValueError(f"Rango historico insuficiente para {variable}.")

    def _load_normalization_reference(self) -> tuple[dict, str]:
        if self._normalization_reference is not None and self._normalization_source is not None:
            return self._normalization_reference, self._normalization_source
        try:
            payload = json.loads(self.normalization_path.read_text(encoding="utf-8"))
            self._validate_normalization_reference(payload)
            source = self.normalization_path.name
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Referencia ambiental no disponible o invalida (%s); se usa fallback fijo incorporado.",
                exc,
            )
            payload = BUILTIN_NORMALIZATION_REFERENCE
            self._validate_normalization_reference(payload)
            source = "builtin_environmental_normalization_v1_fallback"
        self._normalization_reference = payload
        self._normalization_source = source
        return payload, source

    def _normalization_variable(self, variable: str) -> dict:
        payload, _source = self._load_normalization_reference()
        return payload["variables"][variable]

    def _normalization_range(self, variable: str) -> tuple[float, float]:
        reference = self._normalization_variable(variable)
        return float(reference["p10"]), float(reference["p90"])

    @staticmethod
    def _parse_timestamp(snapshot_date: str, hour: int, year: int) -> datetime:
        try:
            parsed_date = date.fromisoformat(snapshot_date)
        except ValueError as exc:
            raise ValueError("date debe venir en formato YYYY-MM-DD.") from exc
        if parsed_date.year != year:
            raise ValueError(f"La capa ambiental historica solo acepta fechas del anio {year}.")
        if hour < 0 or hour > 23:
            raise ValueError("hour debe estar entre 0 y 23.")
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, hour)

    def _load_congestion(self) -> pd.DataFrame:
        if self._congestion is not None:
            return self._congestion
        if not self.congestion_path.exists():
            logger.info("Dataset de congestion no disponible: %s", self.congestion_path)
            self._congestion = pd.DataFrame()
            return self._congestion

        cache_path: Path | None = None
        if self.congestion_path in {CONGESTION_CORE_PATH, CONGESTION_REGIONAL_PATH}:
            stat = self.congestion_path.stat()
            cache_path = CONGESTION_CACHE_DIR / (
                f"environmental-congestion-v{CONGESTION_CACHE_SCHEMA_VERSION}-"
                f"{self.congestion_path.stem}-{stat.st_mtime_ns}-{stat.st_size}.pkl"
            )
            if cache_path.exists():
                try:
                    self._congestion = pd.read_pickle(cache_path)
                    return self._congestion
                except Exception:
                    logger.warning("No se pudo leer cache ambiental: %s", cache_path)

        available_columns = set(pd.read_csv(self.congestion_path, nrows=0).columns)
        required_columns = {
            "segment_id",
            "lat",
            "lon",
            "velocidad_kmh",
            "duracion_min",
            "duracion_hrs",
            "via",
            "comuna",
            "datetime_inicio",
            "datetime_fin",
            "fecha",
            "hora_inicio",
            "hora_fin",
            "segment_seq",
            "indice_coord",
        }
        df = pd.read_csv(self.congestion_path, usecols=sorted(available_columns & required_columns))
        if "datetime_inicio" in df.columns:
            df["_start_ts"] = pd.to_datetime(df["datetime_inicio"], errors="coerce")
        else:
            df["_start_ts"] = pd.to_datetime(
                df.get("fecha", "").astype(str) + " " + df.get("hora_inicio", "").astype(str),
                errors="coerce",
            )
        if "datetime_fin" in df.columns:
            df["_end_ts"] = pd.to_datetime(df["datetime_fin"], errors="coerce")
        else:
            df["_end_ts"] = pd.to_datetime(
                df.get("fecha", "").astype(str) + " " + df.get("hora_fin", "").astype(str),
                errors="coerce",
            )
        if "duracion_min" not in df.columns:
            df["duracion_min"] = pd.to_numeric(df.get("duracion_hrs", 0), errors="coerce").fillna(0) * 60
        df["duracion_min"] = pd.to_numeric(df["duracion_min"], errors="coerce")
        df["velocidad_kmh"] = pd.to_numeric(df.get("velocidad_kmh"), errors="coerce")
        df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
        df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
        df = df.dropna(subset=["_start_ts", "lat", "lon"])
        if "_end_ts" not in df.columns:
            df["_end_ts"] = pd.NaT
        missing_end = df["_end_ts"].isna()
        df.loc[missing_end, "_end_ts"] = df.loc[missing_end, "_start_ts"] + pd.to_timedelta(
            df.loc[missing_end, "duracion_min"].fillna(30).clip(lower=15),
            unit="m",
        )
        self._congestion = df[df["_start_ts"].dt.year == self.year].copy()
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self._congestion.to_pickle(cache_path)
            except Exception:
                logger.warning("No se pudo escribir cache ambiental: %s", cache_path)
        return self._congestion

    def _congestion_hour_index(self, hour: int | None = None) -> set[datetime]:
        if self._congestion_hours is not None:
            return (
                self._congestion_hours
                if hour is None
                else {timestamp for timestamp in self._congestion_hours if timestamp.hour == hour}
            )
        df = self._load_congestion()
        hours: set[datetime] = set()
        if df.empty:
            self._congestion_hours = hours
            return hours
        start_hours = df["_start_ts"].dt.floor("h")
        end_hours = (df["_end_ts"] - pd.Timedelta(microseconds=1)).dt.floor("h")
        valid = start_hours.notna() & end_hours.notna() & (end_hours >= start_hours)
        nanoseconds_per_hour = 3_600_000_000_000
        start_ids = start_hours[valid].astype("int64") // nanoseconds_per_hour
        end_ids = end_hours[valid].astype("int64") // nanoseconds_per_hour
        hour_ids: set[int] = set()
        for start_id, end_id in zip(start_ids, end_ids):
            hour_ids.update(range(int(start_id), int(end_id) + 1))
        epoch = datetime(1970, 1, 1)
        hours = {
            timestamp
            for hour_id in hour_ids
            if (timestamp := epoch + timedelta(hours=hour_id)).year == self.year
        }
        self._congestion_hours = hours
        if hour is None:
            return hours
        return {timestamp for timestamp in hours if timestamp.hour == hour}

    @staticmethod
    def _load_hourly_network(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.dropna(subset=["timestamp"])

    def _load_radiation_network(self) -> pd.DataFrame:
        if self._radiation is not None:
            return self._radiation
        if not self.radiation_manifest_path.exists():
            self._radiation = pd.DataFrame()
            return self._radiation

        manifest = pd.read_csv(self.radiation_manifest_path)
        required = {"parameter_label", "rows", "csv_path"}
        if not required.issubset(manifest.columns):
            self._radiation = pd.DataFrame()
            return self._radiation

        radiation_rows = manifest[
            (manifest["parameter_label"] == "global_radiation")
            & (pd.to_numeric(manifest["rows"], errors="coerce").fillna(0) > 0)
        ]
        frames: list[pd.DataFrame] = []
        for row in radiation_rows.itertuples():
            csv_path = ROOT_DIR / str(row.csv_path).replace("\\", "/")
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, usecols=["datetime_local", "preferred_value"])
            except (OSError, ValueError):
                continue
            df["timestamp"] = pd.to_datetime(df["datetime_local"], errors="coerce")
            df["global_radiation"] = pd.to_numeric(df["preferred_value"], errors="coerce")
            df = df.dropna(subset=["timestamp", "global_radiation"])
            if not df.empty:
                frames.append(df[["timestamp", "global_radiation"]])

        if not frames:
            self._radiation = pd.DataFrame()
            return self._radiation

        combined = pd.concat(frames, ignore_index=True)
        self._radiation = (
            combined.groupby("timestamp", as_index=False)["global_radiation"]
            .median()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        return self._radiation

    def _weather_snapshot(self, timestamp: datetime) -> WeatherSnapshot:
        if self._rain is None:
            self._rain = self._load_hourly_network(self.rain_path)
        if self._wind is None:
            self._wind = self._load_hourly_network(self.wind_path)
        if self.include_radiation and self._radiation is None:
            self._radiation = self._load_radiation_network()
        elif self._radiation is None:
            self._radiation = pd.DataFrame()

        rain_mm: float | None = None
        wet_count: int | None = None
        wind_speed: float | None = None
        global_radiation: float | None = None

        if not self._rain.empty:
            rain_row = self._rain[self._rain["timestamp"] == timestamp]
            if not rain_row.empty:
                raw_rain = pd.to_numeric(rain_row.iloc[0].get("rain_mm_mean"), errors="coerce")
                raw_wet = pd.to_numeric(rain_row.iloc[0].get("wet_station_count"), errors="coerce")
                rain_mm = None if pd.isna(raw_rain) else float(raw_rain)
                wet_count = None if pd.isna(raw_wet) else int(raw_wet)

        if not self._wind.empty:
            wind_row = self._wind[self._wind["timestamp"] == timestamp]
            if not wind_row.empty:
                raw_wind = pd.to_numeric(wind_row.iloc[0].get("wind_speed_mean"), errors="coerce")
                wind_speed = None if pd.isna(raw_wind) else float(raw_wind)

        if not self._radiation.empty:
            radiation_row = self._radiation[self._radiation["timestamp"] == timestamp]
            if not radiation_row.empty:
                raw_radiation = pd.to_numeric(radiation_row.iloc[0].get("global_radiation"), errors="coerce")
                global_radiation = None if pd.isna(raw_radiation) else float(raw_radiation)

        return WeatherSnapshot(
            rain_mm=rain_mm,
            wind_speed=wind_speed,
            wet_station_count=wet_count,
            global_radiation=global_radiation,
        )

    def _wind_range(self, hour: int | None = None) -> tuple[float, float] | None:
        if self._wind is None:
            self._wind = self._load_hourly_network(self.wind_path)
        if self._wind.empty or "wind_speed_mean" not in self._wind.columns:
            return None
        congestion_hours = self._congestion_hour_index(hour)
        wind_rows = self._wind[self._wind["timestamp"].isin(congestion_hours)] if congestion_hours else self._wind
        if wind_rows.empty:
            wind_rows = self._wind
        values = pd.to_numeric(wind_rows["wind_speed_mean"], errors="coerce").dropna()
        values = values[values.map(math.isfinite)]
        if values.empty:
            return None
        return float(values.min()), float(values.max())

    def _pm25_range(self, hour: int | None = None) -> tuple[float, float] | None:
        try:
            return get_air_quality_service().local_pm25_range(self._congestion_hour_index(hour))
        except TypeError:
            return get_air_quality_service().local_pm25_range()
        except Exception:
            return None

    @staticmethod
    def _snapshot_pm25_average(snapshot) -> float | None:
        if snapshot is None:
            return None
        raw_average = getattr(snapshot, "average_pm25", None)
        if raw_average is not None:
            try:
                return float(raw_average)
            except (TypeError, ValueError):
                return None
        stations = getattr(snapshot, "stations", []) or []
        values: list[float] = []
        for station in stations:
            try:
                value = float(station.pm25)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if not values:
            return None
        return sum(values) / len(values)

    def _weather_summary(
        self,
        weather: WeatherSnapshot,
        pm25_snapshot=None,
        pm25_range: tuple[float, float] | None = None,
        wind_range: tuple[float, float] | None = None,
    ) -> EnvironmentalWeatherSummary:
        if weather.rain_mm is None:
            rain_label = "Sin dato"
        elif weather.rain_mm > 10.0:
            rain_label = "Lluvia fuerte"
        elif weather.rain_mm >= 2.0:
            rain_label = "Lluvia"
        elif weather.rain_mm >= 0.1:
            rain_label = "Llovizna"
        else:
            rain_label = "Sin lluvia"

        wind_kmh = _wind_kmh(weather.wind_speed)
        wind_range = wind_range if wind_range is not None else self._wind_range()
        pm25_range = pm25_range if pm25_range is not None else self._pm25_range()
        pm25_average = self._snapshot_pm25_average(pm25_snapshot)

        if weather.wind_speed is None:
            wind_label = "Sin dato"
        elif wind_kmh is not None and wind_kmh >= 39.0:
            wind_label = "Viento fuerte"
        elif wind_kmh is not None and wind_kmh >= 20.0:
            wind_label = "Viento moderado"
        else:
            wind_label = "Viento suave"

        if weather.global_radiation is None:
            sky_label = "Sin dato"
        elif weather.global_radiation < 20.0:
            sky_label = "Oscuro"
        elif weather.global_radiation >= 350.0:
            sky_label = "Despejado"
        elif weather.global_radiation >= 150.0:
            sky_label = "Parcial"
        else:
            sky_label = "Nublado"

        return EnvironmentalWeatherSummary(
            pm25=round(pm25_average, 1) if pm25_average is not None else None,
            pm25_min=round(pm25_range[0], 1) if pm25_range is not None else None,
            pm25_max=round(pm25_range[1], 1) if pm25_range is not None else None,
            rain_mm=round(weather.rain_mm, 2) if weather.rain_mm is not None else None,
            has_rain=weather.rain_mm is not None and weather.rain_mm >= 0.1,
            rain_label=rain_label,
            wind_speed=round(weather.wind_speed, 2) if weather.wind_speed is not None else None,
            wind_speed_kmh=round(wind_kmh, 1) if wind_kmh is not None else None,
            wind_speed_min=round(wind_range[0], 2) if wind_range is not None else None,
            wind_speed_max=round(wind_range[1], 2) if wind_range is not None else None,
            wind_speed_min_kmh=round(_wind_kmh(wind_range[0]), 1) if wind_range is not None else None,
            wind_speed_max_kmh=round(_wind_kmh(wind_range[1]), 1) if wind_range is not None else None,
            wind_label=wind_label,
            global_radiation=round(weather.global_radiation, 1) if weather.global_radiation is not None else None,
            sky_label=sky_label,
        )

    @staticmethod
    def _nearest_pm25(snapshot, lat: float, lon: float) -> float | None:
        stations = getattr(snapshot, "stations", []) or []
        if not stations:
            return None
        nearest = min(stations, key=lambda station: _haversine_km(lat, lon, station.lat, station.lon))
        return float(nearest.pm25)

    def _congestion_score(self, speed_kmh: float | None, duration_min: float | None) -> float:
        components: list[float] = []
        if speed_kmh is not None and math.isfinite(speed_kmh):
            components.append(
                _local_component(
                    speed_kmh,
                    self._normalization_range("congestion_speed_kmh"),
                    invert=True,
                )
            )
        if duration_min is not None and math.isfinite(duration_min):
            components.append(
                _local_component(
                    duration_min,
                    self._normalization_range("congestion_duration_min"),
                )
            )
        if not components:
            return 0.0
        # Severity (low speed) and temporal extent (duration) are separate
        # congestion dimensions. Equal contribution is explicit and provisional:
        # it ensures that a slow and long event scores above an event that is only
        # extreme in one dimension, without allowing either signal to hide the other.
        return _clamp(sum(components) / len(components))

    @staticmethod
    def _score(
        *,
        congestion_score: float,
        pm25: float | None,
        rain_mm: float | None,
        wind_speed: float | None,
    ) -> float:
        pm25_score = _clamp(((pm25 or 0.0) - 20.0) / 60.0) if pm25 is not None else 0.0
        pm25_factor = _clamp(
            EnvironmentalImpactService._rain_pm25_factor(rain_mm)
            * EnvironmentalImpactService._wind_pm25_factor(wind_speed),
            0.80,
            1.06,
        )
        congestion_factor = _clamp(
            EnvironmentalImpactService._rain_congestion_factor(rain_mm)
            * EnvironmentalImpactService._wind_congestion_factor(wind_speed),
            1.00,
            1.20,
        )
        pm25_adjusted = _clamp(pm25_score * pm25_factor)
        congestion_adjusted = _clamp(congestion_score * congestion_factor)
        raw = CONGESTION_WEIGHT * congestion_adjusted + PM25_WEIGHT * pm25_adjusted
        return round(_clamp(raw) * 100, 1)

    @staticmethod
    def _layer_condition_score(
        *,
        congestion_score: float,
        pm25: float | None,
        pm25_range: tuple[float, float] | None,
        rain_mm: float | None,
        wind_speed: float | None,
        wind_range: tuple[float, float] | None,
    ) -> float:
        pm25_component = _local_component(pm25, pm25_range)
        low_wind_component = _local_component(wind_speed, wind_range, invert=True)
        rain_relief = 0.0
        if rain_mm is not None:
            if rain_mm > 10.0:
                rain_relief = 0.12
            elif rain_mm >= 2.0:
                rain_relief = 0.08
            elif rain_mm >= 0.1:
                rain_relief = 0.04
        pressure = (
            LAYER_PM25_WEIGHT * pm25_component
            + LAYER_CONGESTION_WEIGHT * _clamp(congestion_score)
            + LAYER_LOW_WIND_WEIGHT * low_wind_component
            - rain_relief
        )
        return round(_clamp(pressure) * 100, 1)

    @staticmethod
    def _rain_pm25_factor(rain_mm: float | None) -> float:
        if rain_mm is None or rain_mm < 0.1:
            return 1.00
        if rain_mm < 2.0:
            return 0.99
        if rain_mm <= 10.0:
            return 0.94
        return 0.88

    @staticmethod
    def _wind_pm25_factor(wind_speed: float | None) -> float:
        wind_kmh = _wind_kmh(wind_speed)
        if wind_kmh is None:
            return 1.00
        if wind_kmh < 20.0:
            return 1.05
        if wind_kmh < 39.0:
            return 0.98
        if wind_kmh < 62.0:
            return 0.93
        return 0.92

    @staticmethod
    def _rain_congestion_factor(rain_mm: float | None) -> float:
        if rain_mm is None or rain_mm < 0.1:
            return 1.00
        if rain_mm < 2.0:
            return 1.03
        if rain_mm <= 10.0:
            return 1.08
        return 1.15

    @staticmethod
    def _wind_congestion_factor(wind_speed: float | None) -> float:
        wind_kmh = _wind_kmh(wind_speed)
        if wind_kmh is None or wind_kmh < 50.0:
            return 1.00
        if wind_kmh < 75.0:
            return 1.03
        return 1.06

    @staticmethod
    def _point_message(
        score: float,
        raw_score: float,
        pm25: float | None,
        rain_mm: float | None,
        wind_speed: float | None,
    ) -> str:
        if rain_mm is not None and rain_mm >= 2.0 and score >= 65:
            return "La lluvia reduce parcialmente PM2.5, pero aumenta la dificultad vial."
        if rain_mm is not None and rain_mm >= 0.1:
            return "La lluvia reduce suavemente PM2.5 sin eliminar la senal de congestion."
        if pm25 is not None and pm25 >= 80:
            return "Exposicion PM2.5 muy elevada como referencia historica."
        if pm25 is not None and pm25 >= 50:
            return "Exposicion PM2.5 elevada para esta referencia horaria."
        if raw_score >= 65:
            return "Alta presion ambiental por congestion y baja ventilacion."
        if score >= 65:
            return "Mayor presion ambiental local dentro de la hora seleccionada."
        if wind_speed is not None and wind_speed < 1.0:
            return "Baja ventilacion: posible acumulacion local."
        if score < 35:
            return "Condiciones favorables para caminar."
        return "Condiciones ambientales intermedias para movilizarse."

    @staticmethod
    def _summary_messages(points: list[EnvironmentalImpactPoint], weather: WeatherSnapshot) -> list[str]:
        if not points:
            return ["No hay congestion historica para esa fecha y hora; no se dibuja capa ambiental."]
        messages: list[str] = []
        max_score = max(point.score for point in points)
        if max_score >= 70:
            messages.append("Se detectan focos de mayor impacto local en la hora seleccionada.")
            messages.append("Prioriza rutas que eviten las zonas marcadas como altas.")
        elif max_score < 35:
            messages.append("Condiciones favorables para caminar.")
        else:
            messages.append("Condiciones intermedias: revisa PM2.5, viento y lluvia antes de salir.")
        if weather.rain_mm is not None and weather.rain_mm >= 0.1:
            messages.append("La lluvia ajusta PM2.5 y congestion por separado con factores suaves.")
        if weather.wind_speed is not None and weather.wind_speed < 1.0:
            messages.append("Baja ventilacion: posible acumulacion local.")
        return list(dict.fromkeys(messages))[:4]

    def _matching_congestions(self, timestamp: datetime) -> pd.DataFrame:
        df = self._load_congestion()
        if df.empty:
            return df
        end_ts = timestamp + pd.Timedelta(hours=1)
        mask = (df["_start_ts"] < end_ts) & (df["_end_ts"] > timestamp)
        return df.loc[mask].copy()

    @staticmethod
    def _memory_weights(weather: WeatherSnapshot) -> dict[int, float]:
        weights = {0: 1.00}
        has_rain = weather.rain_mm is not None and weather.rain_mm >= 0.1
        has_strong_ventilation = weather.wind_speed is not None and weather.wind_speed > MEMORY_LOW_WIND_MPS
        weights[1] = (
            MEMORY_PREVIOUS_HOUR_ADVERSE_WEATHER_WEIGHT
            if has_rain or has_strong_ventilation
            else MEMORY_PREVIOUS_HOUR_WEIGHT
        )
        if not has_rain and not has_strong_ventilation:
            weights[2] = MEMORY_TWO_HOURS_WEIGHT
        return weights

    def _matching_congestions_with_memory(
        self,
        timestamp: datetime,
        weather: WeatherSnapshot,
    ) -> pd.DataFrame:
        df = self._load_congestion()
        if df.empty:
            return df

        frames: list[pd.DataFrame] = []
        for lag_hours, memory_weight in self._memory_weights(weather).items():
            lag_start = timestamp - pd.Timedelta(hours=lag_hours)
            lag_end = lag_start + pd.Timedelta(hours=1)
            mask = (df["_start_ts"] < lag_end) & (df["_end_ts"] > lag_start)
            lag_rows = df.loc[mask].copy()
            if lag_rows.empty:
                continue
            lag_rows["_memory_lag_hours"] = lag_hours
            lag_rows["_memory_weight"] = memory_weight
            frames.append(lag_rows)

        if not frames:
            empty = df.iloc[0:0].copy()
            empty["_memory_lag_hours"] = pd.Series(dtype="int")
            empty["_memory_weight"] = pd.Series(dtype="float")
            return empty
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _segment_geometry(rows: pd.DataFrame, origin_lon: float, origin_lat: float):
        sort_columns = [column for column in ["segment_seq", "indice_coord"] if column in rows.columns]
        ordered = rows.sort_values(sort_columns) if sort_columns else rows
        coords: list[tuple[float, float]] = []
        for row in ordered.itertuples():
            lon = float(row.lon)
            lat = float(row.lat)
            projected = _project_lon_lat(lon, lat, origin_lon, origin_lat)
            if not coords or coords[-1] != projected:
                coords.append(projected)
        if len(coords) >= 2:
            return LineString(coords)
        if coords:
            return Point(coords[0])
        return None

    def _build_zones(self, rows: pd.DataFrame, points: list[EnvironmentalImpactPoint]) -> dict:
        if rows.empty or not points:
            return _empty_zone_collection()

        origin_lon = float(rows["lon"].mean())
        origin_lat = float(rows["lat"].mean())
        score_by_segment = {point.segment_id: point for point in points}
        sources: list[dict] = []

        for segment_id, segment_rows in rows.groupby("segment_id", dropna=False):
            point = score_by_segment.get(str(segment_id))
            if point is None:
                continue
            lag_hours = (
                int(segment_rows["_memory_lag_hours"].min())
                if "_memory_lag_hours" in segment_rows.columns
                else 0
            )
            recency_weight = (
                float(segment_rows["_memory_weight"].max())
                if "_memory_weight" in segment_rows.columns
                else 1.0
            )
            base_geometry = self._segment_geometry(segment_rows, origin_lon, origin_lat)
            if base_geometry is None or base_geometry.is_empty:
                continue
            buffer_m = ZONE_BUFFER_METERS[point.level] + point.score * 0.3
            buffered = base_geometry.buffer(buffer_m, cap_style=1, join_style=1)
            if not buffered.is_empty:
                sources.append(
                    {
                        "base_geometry": base_geometry,
                        "geometry": buffered,
                        "buffer_m": buffer_m,
                        "score": point.score,
                        "background_score": max(
                            0.0,
                            point.score - LAYER_CONGESTION_WEIGHT * point.congestion_score * 100.0,
                        ),
                        "congestion_contribution": LAYER_CONGESTION_WEIGHT * point.congestion_score * 100.0,
                        "segment_id": point.segment_id,
                        "via": point.via,
                        "lag_hours": lag_hours,
                        "recency_weight": recency_weight,
                    }
                )

        if not sources:
            return _empty_zone_collection()

        # Split all buffers at their shared boundaries. Each resulting cell receives
        # one combined score, so lower-level polygons never remain visible inside a
        # higher-level cloud. Disconnected buffers still remain separate geometries.
        influence_boundaries = []
        for source in sources:
            for fraction in INFLUENCE_BAND_FRACTIONS:
                influence_boundaries.append(
                    source["base_geometry"].buffer(
                        float(source["buffer_m"]) * fraction,
                        cap_style=1,
                        join_style=1,
                    ).boundary
                )
        boundaries = unary_union(influence_boundaries)
        source_tree = STRtree([source["geometry"] for source in sources])
        records_by_level: dict[str, list[dict]] = {"low": [], "medium": [], "high": []}

        for cell in polygonize(boundaries):
            if cell.is_empty or cell.area <= 1e-6:
                continue
            sample = cell.representative_point()
            contributor_indexes = source_tree.query(sample, predicate="within")
            contributors = [sources[int(index)] for index in contributor_indexes]
            if not contributors:
                continue
            # PM2.5, wind and rain describe the shared local background and must
            # be counted once. Only each congestion's incremental contribution
            # is additive where influence areas overlap. Congestion influence
            # decreases progressively with distance from its road segment.
            background_score = max(float(source["background_score"]) for source in contributors)
            congestion_score = sum(
                float(source["congestion_contribution"])
                * _clamp(
                    1.0
                    - source["base_geometry"].distance(sample)
                    / max(float(source["buffer_m"]), 1e-9)
                )
                for source in contributors
            )
            combined_score = min(100.0, background_score + congestion_score)
            level = _level(combined_score)
            records_by_level[level].append(
                {
                    "geometry": cell,
                    "score": combined_score,
                    "area": cell.area,
                    "segment_ids": {str(source["segment_id"]) for source in contributors},
                    "vias": {str(source["via"]) for source in contributors if source["via"]},
                    "current_segment_ids": {
                        str(source["segment_id"]) for source in contributors if int(source["lag_hours"]) == 0
                    },
                    "memory_segment_ids": {
                        str(source["segment_id"]) for source in contributors if int(source["lag_hours"]) > 0
                    },
                    "memory_max_lag_hours": max(int(source["lag_hours"]) for source in contributors),
                    "recency_weight": max(float(source["recency_weight"]) for source in contributors),
                    "contributor_count": len(contributors),
                }
            )

        features: list[dict] = []
        level_order = {"low": 1, "medium": 2, "high": 3}
        for level in ["low", "medium", "high"]:
            records = records_by_level[level]
            if not records:
                continue
            merged = unary_union([record["geometry"] for record in records]).buffer(0)
            if merged.is_empty:
                continue
            components = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]
            components.sort(key=lambda geometry: (round(geometry.centroid.y, 3), round(geometry.centroid.x, 3)))
            for component_index, component in enumerate(components, start=1):
                local_records = [
                    record
                    for record in records
                    if component.covers(record["geometry"].representative_point())
                ]
                if not local_records:
                    continue
                total_area = sum(float(record["area"]) for record in local_records)
                score_avg = (
                    sum(float(record["score"]) * float(record["area"]) for record in local_records) / total_area
                    if total_area > 0
                    else max(float(record["score"]) for record in local_records)
                )
                recency_weight_avg = (
                    sum(float(record["recency_weight"]) * float(record["area"]) for record in local_records)
                    / total_area
                    if total_area > 0
                    else max(float(record["recency_weight"]) for record in local_records)
                )
                segment_ids = sorted(
                    set().union(*(record["segment_ids"] for record in local_records))
                )
                vias = sorted(set().union(*(record["vias"] for record in local_records)))
                current_segment_ids = sorted(
                    set().union(*(record["current_segment_ids"] for record in local_records))
                )
                memory_segment_ids = sorted(
                    set().union(*(record["memory_segment_ids"] for record in local_records))
                )
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "zone_id": f"environment-{level}-{component_index}",
                            "level": level,
                            "score_avg": round(score_avg, 1),
                            "score_max": round(max(float(record["score"]) for record in local_records), 1),
                            "segment_count": len(segment_ids),
                            "segment_ids": segment_ids,
                            "vias": vias,
                            "current_focus_count": len(current_segment_ids),
                            "memory_focus_count": len(memory_segment_ids),
                            "memory_max_lag_hours": max(
                                int(record["memory_max_lag_hours"]) for record in local_records
                            ),
                            "recency_weight": round(recency_weight_avg, 3),
                            "overlap_count_max": max(int(record["contributor_count"]) for record in local_records),
                            "composition": "background_plus_distance_weighted_congestion",
                            "z_index": level_order[level],
                        },
                        "geometry": _unproject_geometry(component, origin_lon, origin_lat),
                    }
                )

        features.sort(key=lambda item: item["properties"]["z_index"])
        return {"type": "FeatureCollection", "features": features}

    def _build_congestion_lines(
        self,
        rows: pd.DataFrame,
        points: list[EnvironmentalImpactPoint],
        segment_metrics: pd.DataFrame | None = None,
    ) -> dict:
        if rows.empty or not points:
            return _empty_zone_collection()

        origin_lon = float(rows["lon"].mean())
        origin_lat = float(rows["lat"].mean())
        point_by_segment = {point.segment_id: point for point in points}
        metrics_by_segment = (
            {str(row["segment_id"]): row for _, row in segment_metrics.iterrows()}
            if segment_metrics is not None and not segment_metrics.empty
            else {}
        )
        features: list[dict] = []

        for segment_id, segment_rows in rows.groupby("segment_id", dropna=False):
            point = point_by_segment.get(str(segment_id))
            if point is None:
                continue
            metrics = metrics_by_segment.get(str(segment_id))
            geometry = self._segment_geometry(segment_rows, origin_lon, origin_lat)
            if geometry is None or geometry.is_empty or geometry.geom_type != "LineString":
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "layer_kind": "congestion",
                        "segment_id": point.segment_id,
                        "level": point.congestion_level,
                        "score": round(point.congestion_score * 100, 1),
                        "congestion_score": point.congestion_score,
                        "environmental_level": point.level,
                        "environmental_score": point.score,
                        "pm25": point.pm25,
                        "via": point.via,
                        "comuna": point.comuna,
                        "speed_kmh": round(float(metrics["velocidad_kmh"]), 1)
                        if metrics is not None and pd.notna(metrics.get("velocidad_kmh"))
                        else None,
                        "duration_min": round(float(metrics["duracion_min"]), 1)
                        if metrics is not None and pd.notna(metrics.get("duracion_min"))
                        else None,
                        "lag_hours": int(metrics["_memory_lag_hours"])
                        if metrics is not None and "_memory_lag_hours" in metrics
                        else 0,
                        "memory_weight": round(float(metrics["_memory_weight"]), 2)
                        if metrics is not None and "_memory_weight" in metrics
                        else 1.0,
                        "recency": "actual"
                        if metrics is None
                        or "_memory_lag_hours" not in metrics
                        or int(metrics["_memory_lag_hours"]) == 0
                        else "reciente",
                    },
                    "geometry": _unproject_geometry(geometry, origin_lon, origin_lat),
                }
            )

        return {
            "type": "FeatureCollection",
            "features": _deduplicate_congestion_line_features(features),
        }

    @lru_cache(maxsize=96)
    def build_snapshot(self, snapshot_date: str, hour: int) -> EnvironmentalImpactResponse:
        timestamp = self._parse_timestamp(snapshot_date, hour, self.year)
        requested_at = timestamp.strftime("%Y-%m-%d %H:00:00")
        weather = self._weather_snapshot(timestamp)
        rows = self._matching_congestions_with_memory(timestamp, weather)
        normalization_payload, normalization_source = self._load_normalization_reference()
        pm25_reference = normalization_payload["variables"]["pm25"]
        wind_reference = normalization_payload["variables"]["wind_speed"]
        try:
            air_quality_service = get_air_quality_service()
            pm25_snapshot = air_quality_service.station_snapshot(snapshot_date, hour)
        except Exception:
            pm25_snapshot = None
        pm25_range = self._normalization_range("pm25")
        wind_range = self._normalization_range("wind_speed")
        fallback_labels: list[str] = []
        if normalization_source.startswith("builtin_"):
            fallback_labels.append("normalization_reference=builtin_fixed_v1")
        if not getattr(pm25_snapshot, "stations", None):
            fallback_labels.append("pm25=historical_p50")
        if weather.wind_speed is None:
            fallback_labels.append("wind=historical_p50")
        if weather.rain_mm is None:
            fallback_labels.append("rain=no_relief")
        data_source = ", ".join(
            [
                self.congestion_path.name,
                self.rain_path.name if self.rain_path.exists() else "rain_missing",
                self.wind_path.name if self.wind_path.exists() else "wind_missing",
                normalization_source,
                *(fallback_labels or ["fallbacks=none"]),
            ]
        )

        if rows.empty:
            summary = EnvironmentalImpactSummary(
                available=False,
                requested_at=requested_at,
                point_count=0,
                dominant_level="none",
                weather=self._weather_summary(weather, pm25_snapshot, pm25_range, wind_range),
                messages=self._summary_messages([], weather),
                method=(
                    "No se generan manchas si no existen registros de congestion historica "
                    "activos durante la hora consultada."
                ),
                data_source=data_source,
            )
            return EnvironmentalImpactResponse(
                summary=summary,
                points=[],
                zones=_empty_zone_collection(),
                congestion_lines=_empty_zone_collection(),
            )

        lagged_segments = (
            rows.groupby(["segment_id", "_memory_lag_hours", "_memory_weight"], dropna=False)
            .agg(
                lat=("lat", "mean"),
                lon=("lon", "mean"),
                velocidad_kmh=("velocidad_kmh", "mean"),
                duracion_min=("duracion_min", "max"),
                via=("via", "first"),
                comuna=("comuna", "first"),
            )
            .reset_index()
        )
        if lagged_segments.empty:
            grouped = lagged_segments
        else:
            lagged_segments["base_congestion_score"] = lagged_segments.apply(
                lambda item: self._congestion_score(
                    None if pd.isna(item["velocidad_kmh"]) else float(item["velocidad_kmh"]),
                    None if pd.isna(item["duracion_min"]) else float(item["duracion_min"]),
                ),
                axis=1,
            )
            lagged_segments["weighted_congestion_score"] = (
                lagged_segments["base_congestion_score"] * lagged_segments["_memory_weight"]
            ).clip(upper=1.0)
            grouped = (
                lagged_segments.sort_values(
                    ["segment_id", "weighted_congestion_score", "_memory_lag_hours"],
                    ascending=[True, False, True],
                )
                .drop_duplicates(subset=["segment_id"], keep="first")
                .reset_index(drop=True)
            )
        grouped = grouped.head(MAX_POINTS)

        candidates: list[EnvironmentalImpactCandidate] = []
        for row in grouped.itertuples():
            lat = float(row.lat)
            lon = float(row.lon)
            pm25 = self._nearest_pm25(pm25_snapshot, lat, lon) if pm25_snapshot is not None else None
            pm25_for_score = float(pm25_reference["p50"]) if pm25 is None else pm25
            wind_for_score = (
                float(wind_reference["p50"])
                if weather.wind_speed is None
                else weather.wind_speed
            )
            congestion_score = (
                float(row.weighted_congestion_score)
                if hasattr(row, "weighted_congestion_score")
                else self._congestion_score(
                    None if pd.isna(row.velocidad_kmh) else float(row.velocidad_kmh),
                    None if pd.isna(row.duracion_min) else float(row.duracion_min),
                )
            )
            layer_score = self._layer_condition_score(
                congestion_score=congestion_score,
                pm25=pm25_for_score,
                pm25_range=pm25_range,
                rain_mm=weather.rain_mm,
                wind_speed=wind_for_score,
                wind_range=wind_range,
            )
            candidates.append(
                EnvironmentalImpactCandidate(
                    lat=lat,
                    lon=lon,
                    raw_score=layer_score,
                    congestion_score=congestion_score,
                    pm25=pm25,
                    segment_id=str(row.segment_id),
                    via=None if pd.isna(row.via) else str(row.via),
                    comuna=None if pd.isna(row.comuna) else str(row.comuna),
                )
            )

        points: list[EnvironmentalImpactPoint] = []
        for candidate in candidates:
            points.append(
                EnvironmentalImpactPoint(
                    lat=round(candidate.lat, 6),
                    lon=round(candidate.lon, 6),
                    score=candidate.raw_score,
                    level=_level(candidate.raw_score),
                    congestion_score=round(candidate.congestion_score, 3),
                    congestion_level=_level(candidate.congestion_score * 100),
                    pm25=round(candidate.pm25, 1) if candidate.pm25 is not None else None,
                    rain_mm=round(weather.rain_mm, 2) if weather.rain_mm is not None else None,
                    wind_speed=round(weather.wind_speed, 2) if weather.wind_speed is not None else None,
                    segment_id=candidate.segment_id,
                    via=candidate.via,
                    comuna=candidate.comuna,
                    message=self._point_message(
                        candidate.raw_score,
                        candidate.raw_score,
                        candidate.pm25,
                        weather.rain_mm,
                        weather.wind_speed,
                    ),
                )
            )

        zones = self._build_zones(rows, points)
        current_rows = rows[rows["_memory_lag_hours"] == 0].copy()
        current_grouped = grouped[grouped["_memory_lag_hours"] == 0].copy()
        congestion_lines = self._build_congestion_lines(current_rows, points, current_grouped)
        summary = EnvironmentalImpactSummary(
            available=bool(points),
            requested_at=requested_at,
            point_count=len(points),
            dominant_level=_dominant_level(points),
            weather=self._weather_summary(weather, pm25_snapshot, pm25_range, wind_range),
            messages=self._summary_messages(points, weather),
            method=(
                "La nube representa impacto ambiental potencial asociado al trafico: PM2.5 medido "
                "por estaciones, viento y lluvia "
                "se reportan como variables directas y se normalizan con una referencia historica fija "
                "y versionada 2021-2024 usando P10 y P90, independiente del dia consultado. La nube incorpora "
                "memoria temporal calibrada con los datos locales: 25% para la hora previa, reducida "
                "a 10% con lluvia o mayor ventilacion; dos horas previas aportan 10% solo sin lluvia "
                "y con viento bajo. Las lineas muestran exclusivamente congestiones activas durante "
                "la hora consultada. Las zonas son buffers urbanos de esos segmentos, "
                "sin interpolacion espacial. Cuando se superponen, PM2.5, viento y lluvia se "
                "cuentan una sola vez y solo se suman los aportes de congestion, cuya influencia "
                "disminuye gradualmente al alejarse de cada calle; cada "
                "ubicacion recibe un unico nivel ambiental final. El color de la nube compara presion ambiental "
                "entre dias: PM2.5 y baja ventilacion usan la misma referencia historica fija, junto con "
                "la congestion historica del segmento. El color de las lineas representa "
                "congestion segun velocidad baja y duracion, normalizadas con P10 y P90 historicos "
                "y combinadas con igual aporte provisional, mas su persistencia temporal."
            ),
            data_source=data_source,
        )
        return EnvironmentalImpactResponse(
            summary=summary,
            points=points,
            zones=zones,
            congestion_lines=congestion_lines,
        )


@lru_cache(maxsize=1)
def get_environmental_impact_service() -> EnvironmentalImpactService:
    return EnvironmentalImpactService()
