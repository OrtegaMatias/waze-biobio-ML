from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "data" / "air_quality" / "sinca_biobio_hourly_2021plus"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
STATIONS_METADATA_PATH = DATASET_DIR / "stations_metadata.csv"
OUTPUT_PROCESSED_DIR = ROOT_DIR / "data_processed"
OUTPUT_ANALYSIS_DIR = ROOT_DIR / "data_analysis"

RAIN_YEAR = 2025
RAIN_PARAMETER_LABEL = "rain"
MAX_REASONABLE_RAIN_MM_H = 150.0

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
    manifest["station_id"] = manifest["station_id"].astype(str)
    manifest["csv_abspath"] = manifest["csv_path"].map(lambda rel: ROOT_DIR / str(rel))
    return manifest


def load_stations_metadata() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_METADATA_PATH)
    stations["station_id"] = stations["station_id"].astype(str)
    stations["commune_norm"] = stations["commune"].map(normalize_text)
    return stations


def load_rain_hourly(manifest: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    candidate_ids = set(stations.loc[stations["commune_norm"].isin(CANDIDATE_COMMUNES), "station_id"])
    rain_manifest = manifest[
        (manifest["parameter_label"].astype(str).str.lower() == RAIN_PARAMETER_LABEL)
        & (manifest["rows"] > 0)
        & (manifest["station_id"].isin(candidate_ids))
    ].copy()

    frames: list[pd.DataFrame] = []
    for row in rain_manifest.to_dict("records"):
        csv_path = Path(row["csv_abspath"])
        if not csv_path.exists():
            continue
        frame = pd.read_csv(
            csv_path,
            usecols=["station_id", "station_name", "datetime_local", "preferred_value"],
        )
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["datetime_local"], errors="coerce")
        frame["rain_mm"] = pd.to_numeric(frame["preferred_value"], errors="coerce")
        frame = frame[frame["timestamp"].notna()]
        frame = frame[frame["timestamp"].dt.year == RAIN_YEAR]
        frame = frame.drop(columns=["datetime_local", "preferred_value"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["timestamp", "station_id", "station_name", "commune", "latitude", "longitude", "rain_mm"]
        )

    rain = pd.concat(frames, ignore_index=True)
    rain["station_id"] = rain["station_id"].astype(str)
    rain["rain_mm"] = pd.to_numeric(rain["rain_mm"], errors="coerce")
    rain = rain.dropna(subset=["rain_mm"]).copy()
    rain = rain[rain["rain_mm"].between(0.0, MAX_REASONABLE_RAIN_MM_H, inclusive="both")]
    rain = rain.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp"],
        keep="last",
    )
    rain = rain.merge(
        stations[["station_id", "commune", "latitude", "longitude"]],
        on="station_id",
        how="left",
    )
    rain["rain_mm"] = rain["rain_mm"].round(3)
    return rain[
        ["timestamp", "station_id", "station_name", "commune", "latitude", "longitude", "rain_mm"]
    ].reset_index(drop=True)


def build_rain_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "station_id",
                "station_name",
                "commune",
                "latitude",
                "longitude",
                "rain_mm",
                "wet_hours",
                "n_obs",
            ]
        )
    daily = hourly.copy()
    daily["date"] = pd.to_datetime(daily["timestamp"]).dt.date.astype(str)
    grouped = (
        daily.groupby(["date", "station_id", "station_name", "commune", "latitude", "longitude"], dropna=False)
        .agg(
            rain_mm=("rain_mm", "sum"),
            wet_hours=("rain_mm", lambda values: int((values > 0).sum())),
            n_obs=("rain_mm", "count"),
        )
        .reset_index()
    )
    grouped["rain_mm"] = grouped["rain_mm"].round(3)
    return grouped.sort_values(["date", "station_id"]).reset_index(drop=True)


def build_network_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(columns=["timestamp", "rain_mm_mean", "rain_mm_max", "wet_station_count", "station_count"])
    network = (
        hourly.groupby("timestamp", dropna=False)
        .agg(
            rain_mm_mean=("rain_mm", "mean"),
            rain_mm_max=("rain_mm", "max"),
            wet_station_count=("rain_mm", lambda values: int((values > 0).sum())),
            station_count=("rain_mm", "count"),
        )
        .reset_index()
    )
    network[["rain_mm_mean", "rain_mm_max"]] = network[["rain_mm_mean", "rain_mm_max"]].round(3)
    return network.sort_values("timestamp").reset_index(drop=True)


def build_station_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "commune",
                "latitude",
                "longitude",
                "first_timestamp",
                "last_timestamp",
                "n_obs",
                "wet_hours",
                "rain_mm_total",
            ]
        )
    summary = (
        hourly.groupby(["station_id", "station_name", "commune", "latitude", "longitude"], dropna=False)
        .agg(
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            n_obs=("rain_mm", "count"),
            wet_hours=("rain_mm", lambda values: int((values > 0).sum())),
            rain_mm_total=("rain_mm", "sum"),
        )
        .reset_index()
    )
    summary["rain_mm_total"] = summary["rain_mm_total"].round(3)
    return summary.sort_values(["commune", "station_name"]).reset_index(drop=True)


def main() -> None:
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    stations = load_stations_metadata()
    hourly = load_rain_hourly(manifest=manifest, stations=stations)
    daily = build_rain_daily(hourly)
    network_hourly = build_network_hourly(hourly)
    summary = build_station_summary(hourly)

    hourly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_rain_hourly_clean.csv"
    daily_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_rain_daily.csv"
    network_hourly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_rain_network_hourly.csv"
    summary_path = OUTPUT_ANALYSIS_DIR / "rain_station_summary_gran_concepcion.csv"

    hourly.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)
    network_hourly.to_csv(network_hourly_path, index=False)
    summary.to_csv(summary_path, index=False)

    report = {
        "year": RAIN_YEAR,
        "hourly_rows": int(len(hourly)),
        "daily_rows": int(len(daily)),
        "network_hourly_rows": int(len(network_hourly)),
        "station_count": int(hourly["station_id"].nunique()) if not hourly.empty else 0,
        "date_range": {
            "start": str(pd.to_datetime(hourly["timestamp"]).min()) if not hourly.empty else None,
            "end": str(pd.to_datetime(hourly["timestamp"]).max()) if not hourly.empty else None,
        },
        "outputs": {
            "hourly": str(hourly_path.relative_to(ROOT_DIR)),
            "daily": str(daily_path.relative_to(ROOT_DIR)),
            "network_hourly": str(network_hourly_path.relative_to(ROOT_DIR)),
            "summary": str(summary_path.relative_to(ROOT_DIR)),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
