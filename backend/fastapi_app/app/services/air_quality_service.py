# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Iterable

from ..schemas.routes import Pm25Exposure, Pm25SnapshotResponse, Pm25StationCondition, Pm25StationExposure, RoutePoint

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[4]
PM25_HOURLY_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_pm25_core_hourly_clean.csv"
PM25_STATION_SUMMARY_PATH = ROOT_DIR / "data_analysis" / "station_summary_gran_concepcion_core.csv"
PM25_SNAPSHOT_YEAR = 2025
MAX_ROUTE_SAMPLE_POINTS = 80
PM25_CLEAN_BASELINE = 15.0
PM25_HIGH_REFERENCE = 35.0
PM25_ROUTE_PENALTY_STRENGTH = 1.75
PM25_IDW_STATION_COUNT = 3
PM25_IDW_POWER = 2.0
PM25_IDW_EPSILON_KM = 0.25
PM25_ABSOLUTE_SCORE_WEIGHT = 0.70
PM25_RELATIVE_SCORE_WEIGHT = 0.30


@dataclass(frozen=True)
class StationProfile:
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    hourly_pm25: dict[int, float]
    overall_pm25: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _category(pm25: float) -> str:
    if pm25 < 20:
        return "Baja"
    if pm25 < 50:
        return "Media"
    return "Alta"


def _sample_points(points: list[RoutePoint]) -> list[RoutePoint]:
    if len(points) <= MAX_ROUTE_SAMPLE_POINTS:
        return points
    step = max(1, math.ceil(len(points) / MAX_ROUTE_SAMPLE_POINTS))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class AirQualityService:
    def __init__(
        self,
        hourly_path: Path = PM25_HOURLY_PATH,
        station_summary_path: Path = PM25_STATION_SUMMARY_PATH,
        snapshot_year: int = PM25_SNAPSHOT_YEAR,
    ) -> None:
        self.hourly_path = hourly_path
        self.station_summary_path = station_summary_path
        self.snapshot_year = snapshot_year
        self._profiles: list[StationProfile] | None = None
        self._snapshot_index: dict[datetime, dict[str, float]] | None = None
        self._snapshot_date_range: tuple[str | None, str | None] = (None, None)
        self._load_lock = Lock()

    def _load_station_coords(self) -> dict[str, tuple[str, float, float]]:
        if not self.station_summary_path.exists():
            logger.info("PM2.5 station summary no disponible: %s", self.station_summary_path)
            return {}
        coords: dict[str, tuple[str, float, float]] = {}
        with self.station_summary_path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            required = {"station_id", "station_name", "latitude", "longitude"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                logger.warning("PM2.5 station summary sin columnas requeridas: %s", sorted(missing))
                return {}
            for row in reader:
                try:
                    station_id = str(row["station_id"]).strip()
                    coords[station_id] = (
                        str(row["station_name"]).strip(),
                        float(row["latitude"]),
                        float(row["longitude"]),
                    )
                except (TypeError, ValueError):
                    continue
        return coords

    def _load_profiles(self) -> list[StationProfile]:
        coords = self._load_station_coords()
        if not coords:
            return []
        if not self.hourly_path.exists():
            logger.info("PM2.5 hourly dataset no disponible: %s", self.hourly_path)
            return []

        hourly_values: dict[str, dict[int, list[float]]] = {}
        overall_values: dict[str, list[float]] = {}
        with self.hourly_path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            required = {"timestamp", "station_id", "station_name", "PM25"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                logger.warning("PM2.5 hourly dataset sin columnas requeridas: %s", sorted(missing))
                return []
            for row in reader:
                station_id = str(row.get("station_id") or "").strip()
                if station_id not in coords:
                    continue
                raw_pm25 = row.get("PM25")
                raw_timestamp = row.get("timestamp")
                try:
                    pm25 = float(raw_pm25)
                    hour = datetime.fromisoformat(str(raw_timestamp)).hour
                except (TypeError, ValueError):
                    continue
                hourly_values.setdefault(station_id, {}).setdefault(hour, []).append(pm25)
                overall_values.setdefault(station_id, []).append(pm25)

        profiles: list[StationProfile] = []
        for station_id, values in overall_values.items():
            if not values:
                continue
            station_name, latitude, longitude = coords[station_id]
            hourly_pm25 = {
                hour: round(sum(hour_values) / len(hour_values), 2)
                for hour, hour_values in hourly_values.get(station_id, {}).items()
                if hour_values
            }
            profiles.append(
                StationProfile(
                    station_id=station_id,
                    station_name=station_name,
                    latitude=latitude,
                    longitude=longitude,
                    hourly_pm25=hourly_pm25,
                    overall_pm25=round(sum(values) / len(values), 2),
                )
            )
        logger.info(
            "PM2.5 cargado: %d estaciones desde %s",
            len(profiles),
            self.hourly_path.relative_to(ROOT_DIR) if self.hourly_path.is_relative_to(ROOT_DIR) else self.hourly_path,
        )
        return profiles

    def _get_profiles(self) -> list[StationProfile]:
        if self._profiles is not None:
            return self._profiles
        with self._load_lock:
            if self._profiles is None:
                self._profiles = self._load_profiles()
        return self._profiles

    def _load_snapshot_index(self) -> dict[datetime, dict[str, float]]:
        if not self.hourly_path.exists():
            logger.info("PM2.5 hourly dataset no disponible: %s", self.hourly_path)
            self._snapshot_date_range = (None, None)
            return {}

        snapshots: dict[datetime, dict[str, list[float]]] = {}
        timestamps: list[datetime] = []
        with self.hourly_path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            required = {"timestamp", "station_id", "PM25"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                logger.warning("PM2.5 hourly dataset sin columnas requeridas: %s", sorted(missing))
                self._snapshot_date_range = (None, None)
                return {}
            for row in reader:
                station_id = str(row.get("station_id") or "").strip()
                try:
                    timestamp = datetime.fromisoformat(str(row.get("timestamp")))
                    pm25 = float(row.get("PM25"))
                except (TypeError, ValueError):
                    continue
                if timestamp.year != self.snapshot_year:
                    continue
                hour_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
                snapshots.setdefault(hour_timestamp, {}).setdefault(station_id, []).append(pm25)
                timestamps.append(hour_timestamp)

        if timestamps:
            self._snapshot_date_range = (
                min(timestamps).date().isoformat(),
                max(timestamps).date().isoformat(),
            )
        else:
            self._snapshot_date_range = (None, None)

        return {
            timestamp: {
                station_id: round(sum(values) / len(values), 1)
                for station_id, values in station_values.items()
                if values
            }
            for timestamp, station_values in snapshots.items()
        }

    def _get_snapshot_index(self) -> dict[datetime, dict[str, float]]:
        if self._snapshot_index is not None:
            return self._snapshot_index
        with self._load_lock:
            if self._snapshot_index is None:
                self._snapshot_index = self._load_snapshot_index()
        return self._snapshot_index

    @staticmethod
    def _parse_snapshot_timestamp(snapshot_date: str, hour: int, snapshot_year: int) -> datetime:
        try:
            parsed_date = date.fromisoformat(snapshot_date)
        except ValueError as exc:
            raise ValueError("date debe venir en formato YYYY-MM-DD.") from exc
        if parsed_date.year != snapshot_year:
            raise ValueError(f"Por ahora el mapa PM2.5 historico solo acepta fechas del año {snapshot_year}.")
        if hour < 0 or hour > 23:
            raise ValueError("hour debe estar entre 0 y 23.")
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, hour)

    def station_snapshot(self, snapshot_date: str, hour: int) -> Pm25SnapshotResponse:
        coords = self._load_station_coords()
        timestamp = self._parse_snapshot_timestamp(snapshot_date, hour, self.snapshot_year)
        values = self._get_snapshot_index().get(timestamp, {})
        stations: list[Pm25StationCondition] = []
        for station_id, pm25 in values.items():
            if station_id not in coords:
                continue
            station_name, latitude, longitude = coords[station_id]
            stations.append(
                Pm25StationCondition(
                    station_id=station_id,
                    station_name=station_name,
                    lat=latitude,
                    lon=longitude,
                    pm25=pm25,
                    category=_category(pm25),
                )
            )
        stations.sort(key=lambda item: item.station_name)
        average_pm25 = round(sum(item.pm25 for item in stations) / len(stations), 1) if stations else None
        start_date, end_date = self._snapshot_date_range
        return Pm25SnapshotResponse(
            available=bool(stations),
            requested_at=timestamp.strftime("%Y-%m-%d %H:00:00"),
            stations=stations,
            average_pm25=average_pm25,
            date_range={"start": start_date, "end": end_date},
            method=(
                f"Lectura historica real por estacion para la fecha y hora seleccionadas, "
                f"filtrada al año {self.snapshot_year}; sin interpolacion ni pronostico."
            ),
            data_source=self.hourly_path.name,
        )

    @staticmethod
    def _hour_value(profile: StationProfile, departure_hour: float | None) -> float:
        target_hour = int(departure_hour or 0) % 24
        return profile.hourly_pm25.get(target_hour, profile.overall_pm25)

    def _hour_pm25_range(self, departure_hour: float | None) -> tuple[float, float] | None:
        values = [self._hour_value(profile, departure_hour) for profile in self._get_profiles()]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            return None
        return min(values), max(values)

    def local_pm25_range(self, timestamps: Iterable[datetime] | None = None) -> tuple[float, float] | None:
        timestamp_filter = None
        if timestamps is not None:
            timestamp_filter = {
                item.replace(minute=0, second=0, microsecond=0)
                for item in timestamps
                if isinstance(item, datetime)
            }
        values: list[float] = []
        for timestamp, station_values in self._get_snapshot_index().items():
            if timestamp_filter is not None and timestamp not in timestamp_filter:
                continue
            timestamp_values = [value for value in station_values.values() if math.isfinite(value)]
            if timestamp_values:
                values.append(sum(timestamp_values) / len(timestamp_values))
        if not values:
            for profile in self._get_profiles():
                values.extend(value for value in profile.hourly_pm25.values() if math.isfinite(value))
        if not values:
            return None
        return min(values), max(values)

    def _ranked_station_values(
        self,
        lat: float,
        lon: float,
        departure_hour: float | None,
    ) -> list[tuple[StationProfile, float, float]]:
        profiles = self._get_profiles()
        ranked = [
            (
                profile,
                _haversine_km(lat, lon, profile.latitude, profile.longitude),
                self._hour_value(profile, departure_hour),
            )
            for profile in profiles
        ]
        return [
            item
            for item in sorted(ranked, key=lambda item: item[1])
            if math.isfinite(item[1]) and math.isfinite(item[2])
        ]

    @staticmethod
    def _idw_pm25(ranked: list[tuple[StationProfile, float, float]]) -> float | None:
        contributors = ranked[:PM25_IDW_STATION_COUNT]
        if not contributors:
            return None
        weighted_sum = 0.0
        total_weight = 0.0
        for _, distance_km, pm25 in contributors:
            weight = 1.0 / ((distance_km + PM25_IDW_EPSILON_KM) ** PM25_IDW_POWER)
            weighted_sum += pm25 * weight
            total_weight += weight
        if total_weight <= 0:
            return contributors[0][2]
        return weighted_sum / total_weight

    def estimate_point_pm25(
        self,
        lat: float,
        lon: float,
        departure_hour: float | None,
    ) -> float | None:
        ranked = self._ranked_station_values(lat, lon, departure_hour)
        if not ranked:
            return None
        return self._idw_pm25(ranked)

    def route_cost_factor(
        self,
        lat: float,
        lon: float,
        departure_hour: float | None,
    ) -> float:
        pm25 = self.estimate_point_pm25(lat, lon, departure_hour)
        if pm25 is None:
            return 1.0
        hour_range = self._hour_pm25_range(departure_hour)
        if hour_range is None:
            return 1.0
        min_pm25, max_pm25 = hour_range
        absolute_score = _clamp(
            (pm25 - PM25_CLEAN_BASELINE) / max(PM25_HIGH_REFERENCE - PM25_CLEAN_BASELINE, 1e-9)
        )
        relative_score = _clamp((pm25 - min_pm25) / max(max_pm25 - min_pm25, 5.0))
        normalized = PM25_ABSOLUTE_SCORE_WEIGHT * absolute_score + PM25_RELATIVE_SCORE_WEIGHT * relative_score
        # Clean air is the neutral routing baseline. Only higher exposure adds cost;
        # otherwise every edge in a clean area receives an unrelated distance bonus.
        return round(1.0 + _clamp(normalized) * PM25_ROUTE_PENALTY_STRENGTH, 3)

    def estimate_route_exposure(
        self,
        geometry: Iterable[RoutePoint],
        departure_hour: float | None,
    ) -> Pm25Exposure | None:
        profiles = self._get_profiles()
        points = list(geometry)
        if not profiles or not points:
            return None

        station_hits: dict[str, tuple[StationProfile, float, int]] = {}
        pm25_values: list[float] = []
        sampled_points = _sample_points(points)

        for point in sampled_points:
            ranked = self._ranked_station_values(point.lat, point.lon, departure_hour)
            if not ranked:
                continue
            pm25 = self._idw_pm25(ranked)
            if pm25 is None:
                continue
            pm25_values.append(pm25)
            for profile, distance_km, _ in ranked[:PM25_IDW_STATION_COUNT]:
                previous = station_hits.get(profile.station_id)
                if previous is None:
                    station_hits[profile.station_id] = (profile, distance_km, 1)
                else:
                    station_hits[profile.station_id] = (profile, min(previous[1], distance_km), previous[2] + 1)

        if not pm25_values:
            return None

        average = round(sum(pm25_values) / len(pm25_values), 1)
        stations = [
            Pm25StationExposure(
                station_id=profile.station_id,
                station_name=profile.station_name,
                distance_km=round(distance_km, 2),
                pm25=round(self._hour_value(profile, departure_hour), 1),
                sample_points=sample_count,
            )
            for profile, distance_km, sample_count in station_hits.values()
        ]
        stations.sort(key=lambda item: (-item.sample_points, item.distance_km))
        return Pm25Exposure(
            available=True,
            average_pm25=average,
            category=_category(average),
            stations=stations[:5],
            method="Promedio historico horario PM2.5 core interpolado por IDW con las 3 estaciones mas cercanas; usado como costo ambiental, no como pronostico en tiempo real.",
            data_source=self.hourly_path.name,
        )


@lru_cache(maxsize=1)
def get_air_quality_service() -> AirQualityService:
    return AirQualityService()
