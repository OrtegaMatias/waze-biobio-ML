# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Iterable

from ..schemas.routes import Pm25Exposure, Pm25StationExposure, RoutePoint

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[4]
PM25_HOURLY_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_pm25_core_hourly_clean.csv"
PM25_STATION_SUMMARY_PATH = ROOT_DIR / "data_analysis" / "station_summary_gran_concepcion_core.csv"
MAX_ROUTE_SAMPLE_POINTS = 80


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
    if pm25 < 15:
        return "Baja"
    if pm25 < 35:
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


class AirQualityService:
    def __init__(
        self,
        hourly_path: Path = PM25_HOURLY_PATH,
        station_summary_path: Path = PM25_STATION_SUMMARY_PATH,
    ) -> None:
        self.hourly_path = hourly_path
        self.station_summary_path = station_summary_path
        self._profiles: list[StationProfile] | None = None
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

    def estimate_route_exposure(
        self,
        geometry: Iterable[RoutePoint],
        departure_hour: float | None,
    ) -> Pm25Exposure | None:
        profiles = self._get_profiles()
        points = list(geometry)
        if not profiles or not points:
            return None

        target_hour = int(departure_hour or 0) % 24
        station_hits: dict[str, tuple[StationProfile, float, int]] = {}
        pm25_values: list[float] = []
        sampled_points = _sample_points(points)

        for point in sampled_points:
            nearest = min(
                profiles,
                key=lambda station: _haversine_km(point.lat, point.lon, station.latitude, station.longitude),
            )
            distance_km = _haversine_km(point.lat, point.lon, nearest.latitude, nearest.longitude)
            pm25 = nearest.hourly_pm25.get(target_hour, nearest.overall_pm25)
            pm25_values.append(pm25)
            previous = station_hits.get(nearest.station_id)
            if previous is None:
                station_hits[nearest.station_id] = (nearest, distance_km, 1)
            else:
                station_hits[nearest.station_id] = (nearest, min(previous[1], distance_km), previous[2] + 1)

        if not pm25_values:
            return None

        average = round(sum(pm25_values) / len(pm25_values), 1)
        stations = [
            Pm25StationExposure(
                station_id=profile.station_id,
                station_name=profile.station_name,
                distance_km=round(distance_km, 2),
                pm25=round(profile.hourly_pm25.get(target_hour, profile.overall_pm25), 1),
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
            method="Promedio historico horario por estacion PM2.5 core mas cercana a puntos de la ruta.",
            data_source=self.hourly_path.name,
        )


@lru_cache(maxsize=1)
def get_air_quality_service() -> AirQualityService:
    return AirQualityService()
