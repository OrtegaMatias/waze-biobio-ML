from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "data" / "air_quality" / "sinca_biobio_hourly_2021plus"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
STATIONS_METADATA_PATH = DATASET_DIR / "stations_metadata.csv"
OUTPUT_PROCESSED_DIR = ROOT_DIR / "data_processed"
OUTPUT_ANALYSIS_DIR = ROOT_DIR / "data_analysis"

VALID_START = pd.Timestamp("2021-01-01 00:00:00")
VALID_END = pd.Timestamp.now().floor("h")
MAX_REASONABLE_WIND_SPEED = 80.0

CANDIDATE_COMMUNES = {
    "concepcion",
    "talcahuano",
    "hualpen",
    "san pedro de la paz",
    "chiguayante",
    "coronel",
    "lota",
    "penco",
    "tome",
    "hualqui",
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", "" if value is None else str(value))
    stripped = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(stripped.strip().lower().split())


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["rows"] = pd.to_numeric(manifest["rows"], errors="coerce").fillna(0).astype(int)
    manifest["csv_abspath"] = manifest["csv_path"].map(lambda rel: ROOT_DIR / str(rel))
    manifest["station_id"] = manifest["station_id"].astype(str)
    return manifest


def load_stations_metadata() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_METADATA_PATH)
    stations["station_id"] = stations["station_id"].astype(str)
    stations["commune_norm"] = stations["commune"].map(normalize_text)
    return stations


def load_parameter_hourly(manifest: pd.DataFrame, parameter_label: str, value_label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    parameter_manifest = manifest[(manifest["parameter_label"] == parameter_label) & (manifest["rows"] > 0)].copy()

    for row in parameter_manifest.to_dict("records"):
        frame = pd.read_csv(
            Path(row["csv_abspath"]),
            usecols=["station_id", "station_name", "datetime_local", "preferred_value"],
        )
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["datetime_local"], errors="coerce")
        frame[value_label] = pd.to_numeric(frame["preferred_value"], errors="coerce")
        frame = frame[frame["timestamp"].notna()]
        frame = frame[frame["timestamp"].between(VALID_START, VALID_END, inclusive="both")]
        frame = frame.drop(columns=["datetime_local", "preferred_value"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "station_id", "station_name", value_label])

    hourly = pd.concat(frames, ignore_index=True)
    hourly["station_id"] = hourly["station_id"].astype(str)
    hourly = hourly.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp"],
        keep="last",
    )
    hourly = hourly.dropna(subset=[value_label]).reset_index(drop=True)
    return hourly[["timestamp", "station_id", "station_name", value_label]]


def add_wind_components(hourly: pd.DataFrame) -> pd.DataFrame:
    wind = hourly.copy()
    wind["wind_speed"] = pd.to_numeric(wind["wind_speed"], errors="coerce")
    wind["wind_direction_deg"] = pd.to_numeric(wind["wind_direction_deg"], errors="coerce")
    valid_speed = wind["wind_speed"].between(0, MAX_REASONABLE_WIND_SPEED, inclusive="both")
    wind.loc[~valid_speed, "wind_speed"] = np.nan

    valid_direction = wind["wind_speed"].notna() & wind["wind_direction_deg"].between(0, 360, inclusive="left")
    radians = np.deg2rad(wind.loc[valid_direction, "wind_direction_deg"])
    speed = wind.loc[valid_direction, "wind_speed"]

    wind["wind_u"] = np.nan
    wind["wind_v"] = np.nan
    wind.loc[valid_direction, "wind_u"] = -speed * np.sin(radians)
    wind.loc[valid_direction, "wind_v"] = -speed * np.cos(radians)
    wind = wind.dropna(subset=["wind_speed"]).reset_index(drop=True)
    return wind


def vector_direction_from_uv(u: pd.Series | np.ndarray, v: pd.Series | np.ndarray) -> np.ndarray:
    direction = (np.degrees(np.arctan2(-np.asarray(u), -np.asarray(v))) + 360.0) % 360.0
    return direction


def cardinal_direction(degrees: float | int | None) -> str:
    if degrees is None or pd.isna(degrees):
        return ""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((float(degrees) + 22.5) // 45) % 8
    return labels[index]


def build_wind_hourly(manifest: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    candidate_ids = set(stations.loc[stations["commune_norm"].isin(CANDIDATE_COMMUNES), "station_id"])
    speed = load_parameter_hourly(manifest, parameter_label="wind_speed", value_label="wind_speed")
    direction = load_parameter_hourly(
        manifest,
        parameter_label="wind_direction",
        value_label="wind_direction_deg",
    )

    speed = speed[speed["station_id"].isin(candidate_ids)].copy()
    direction = direction[direction["station_id"].isin(candidate_ids)].copy()
    wind = speed.merge(
        direction[["timestamp", "station_id", "wind_direction_deg"]],
        on=["timestamp", "station_id"],
        how="left",
    )
    wind = wind.merge(
        stations[["station_id", "commune", "latitude", "longitude"]],
        on="station_id",
        how="left",
    )
    wind = add_wind_components(wind)
    wind["wind_direction_cardinal"] = wind["wind_direction_deg"].map(cardinal_direction)
    wind["date"] = wind["timestamp"].dt.date.astype(str)
    wind["hour"] = wind["timestamp"].dt.strftime("%H:00")
    columns = [
        "timestamp",
        "date",
        "hour",
        "station_id",
        "station_name",
        "commune",
        "latitude",
        "longitude",
        "wind_speed",
        "wind_direction_deg",
        "wind_direction_cardinal",
        "wind_u",
        "wind_v",
    ]
    wind["wind_speed"] = wind["wind_speed"].round(4)
    wind["wind_direction_deg"] = wind["wind_direction_deg"].round(2)
    wind["wind_u"] = wind["wind_u"].round(4)
    wind["wind_v"] = wind["wind_v"].round(4)
    return wind[columns].sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def aggregate_network_wind(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "date",
                "hour",
                "wind_speed_mean",
                "wind_speed_median",
                "wind_direction_vector_deg",
                "wind_direction_cardinal",
                "wind_u_mean",
                "wind_v_mean",
                "wind_station_count",
                "wind_direction_station_count",
            ]
        )

    grouped = (
        hourly.groupby("timestamp", dropna=False)
        .agg(
            wind_speed_mean=("wind_speed", "mean"),
            wind_speed_median=("wind_speed", "median"),
            wind_u_mean=("wind_u", "mean"),
            wind_v_mean=("wind_v", "mean"),
            wind_station_count=("wind_speed", "count"),
            wind_direction_station_count=("wind_direction_deg", "count"),
        )
        .reset_index()
    )
    has_vector = grouped["wind_u_mean"].notna() & grouped["wind_v_mean"].notna()
    grouped["wind_direction_vector_deg"] = np.nan
    grouped.loc[has_vector, "wind_direction_vector_deg"] = vector_direction_from_uv(
        grouped.loc[has_vector, "wind_u_mean"],
        grouped.loc[has_vector, "wind_v_mean"],
    )
    grouped["wind_direction_cardinal"] = grouped["wind_direction_vector_deg"].map(cardinal_direction)
    grouped["date"] = grouped["timestamp"].dt.date.astype(str)
    grouped["hour"] = grouped["timestamp"].dt.strftime("%H:00")

    numeric_columns = [
        "wind_speed_mean",
        "wind_speed_median",
        "wind_direction_vector_deg",
        "wind_u_mean",
        "wind_v_mean",
    ]
    grouped[numeric_columns] = grouped[numeric_columns].round(4)
    columns = [
        "timestamp",
        "date",
        "hour",
        "wind_speed_mean",
        "wind_speed_median",
        "wind_direction_vector_deg",
        "wind_direction_cardinal",
        "wind_u_mean",
        "wind_v_mean",
        "wind_station_count",
        "wind_direction_station_count",
    ]
    return grouped[columns].sort_values("timestamp").reset_index(drop=True)


def build_pm25_wind_hourly(wind_hourly: pd.DataFrame) -> pd.DataFrame:
    pm25_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_hourly_clean.csv"
    if not pm25_path.exists():
        return pd.DataFrame()

    pm25 = pd.read_csv(pm25_path)
    pm25["timestamp"] = pd.to_datetime(pm25["timestamp"], errors="coerce")
    pm25["station_id"] = pm25["station_id"].astype(str)
    wind_for_join = wind_hourly.drop(columns=["station_name", "date", "hour"], errors="ignore")
    joined = pm25.merge(wind_for_join, on=["timestamp", "station_id"], how="left")
    return joined.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def build_congestion_pm25_wind_hourly(network_wind: pd.DataFrame) -> pd.DataFrame:
    congestion_path = OUTPUT_ANALYSIS_DIR / "congestion_aggregated_gran_concepcion.csv"
    pm25_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_hourly_clean.csv"
    if not congestion_path.exists() or not pm25_path.exists():
        return pd.DataFrame()

    congestion = pd.read_csv(congestion_path)
    congestion["timestamp"] = pd.to_datetime(congestion["periodo_hora"], errors="coerce")

    pm25 = pd.read_csv(pm25_path)
    pm25["timestamp"] = pd.to_datetime(pm25["timestamp"], errors="coerce")
    pm25_network = (
        pm25.groupby("timestamp", dropna=False)
        .agg(PM25_mean=("PM25", "mean"), PM25_median=("PM25", "median"), PM25_station_count=("PM25", "count"))
        .reset_index()
    )
    pm25_network[["PM25_mean", "PM25_median"]] = pm25_network[["PM25_mean", "PM25_median"]].round(4)
    joined = congestion.merge(pm25_network, on="timestamp", how="left")
    joined = joined.merge(network_wind.drop(columns=["date", "hour"], errors="ignore"), on="timestamp", how="left")
    return joined.sort_values("timestamp").reset_index(drop=True)


def build_wind_station_summary(wind_hourly: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    if wind_hourly.empty:
        return pd.DataFrame()

    summary = (
        wind_hourly.groupby(["station_id", "station_name"], dropna=False)
        .agg(
            commune=("commune", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            wind_records=("wind_speed", "count"),
            wind_direction_records=("wind_direction_deg", "count"),
            first_wind_timestamp=("timestamp", "min"),
            last_wind_timestamp=("timestamp", "max"),
            wind_speed_mean=("wind_speed", "mean"),
            wind_speed_median=("wind_speed", "median"),
        )
        .reset_index()
    )
    summary["has_wind_direction"] = summary["wind_direction_records"] > 0
    summary[["wind_speed_mean", "wind_speed_median"]] = summary[["wind_speed_mean", "wind_speed_median"]].round(4)
    return summary.sort_values(["commune", "station_name"]).reset_index(drop=True)


def ensure_output_dirs() -> None:
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_output_dirs()
    manifest = load_manifest()
    stations = load_stations_metadata()

    wind_hourly = build_wind_hourly(manifest=manifest, stations=stations)
    network_wind = aggregate_network_wind(wind_hourly)
    pm25_wind = build_pm25_wind_hourly(wind_hourly)
    congestion_pm25_wind = build_congestion_pm25_wind_hourly(network_wind)
    wind_summary = build_wind_station_summary(wind_hourly=wind_hourly, stations=stations)

    wind_hourly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_wind_hourly_clean.csv"
    network_wind_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_wind_network_hourly.csv"
    pm25_wind_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_wind_hourly_clean.csv"
    congestion_pm25_wind_path = OUTPUT_ANALYSIS_DIR / "congestion_pm25_wind_hourly_gran_concepcion.csv"
    wind_summary_path = OUTPUT_ANALYSIS_DIR / "wind_station_summary_gran_concepcion.csv"

    wind_hourly.to_csv(wind_hourly_path, index=False)
    network_wind.to_csv(network_wind_path, index=False)
    pm25_wind.to_csv(pm25_wind_path, index=False)
    congestion_pm25_wind.to_csv(congestion_pm25_wind_path, index=False)
    wind_summary.to_csv(wind_summary_path, index=False)

    payload = {
        "wind_hourly_rows": int(len(wind_hourly)),
        "wind_stations": int(wind_hourly["station_id"].nunique()) if not wind_hourly.empty else 0,
        "network_wind_rows": int(len(network_wind)),
        "pm25_wind_rows": int(len(pm25_wind)),
        "pm25_rows_with_exact_station_wind": int(pm25_wind["wind_speed"].notna().sum()) if not pm25_wind.empty else 0,
        "congestion_pm25_wind_rows": int(len(congestion_pm25_wind)),
        "congestion_rows_with_wind": (
            int(congestion_pm25_wind["wind_speed_mean"].notna().sum()) if not congestion_pm25_wind.empty else 0
        ),
        "wind_hourly_path": str(wind_hourly_path.relative_to(ROOT_DIR)),
        "network_wind_path": str(network_wind_path.relative_to(ROOT_DIR)),
        "pm25_wind_path": str(pm25_wind_path.relative_to(ROOT_DIR)),
        "congestion_pm25_wind_path": str(congestion_pm25_wind_path.relative_to(ROOT_DIR)),
        "wind_summary_path": str(wind_summary_path.relative_to(ROOT_DIR)),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
