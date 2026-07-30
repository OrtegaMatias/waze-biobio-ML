from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PM25_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_pm25_core_hourly_clean.csv"
WIND_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_wind_network_hourly.csv"
CONGESTION_PATH = ROOT_DIR / "data_analysis" / "congestion_clean_gran_concepcion_core.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data_analysis" / "environmental_normalization_v1.json"
REFERENCE_START = pd.Timestamp("2021-01-01 00:00:00")
REFERENCE_END_EXCLUSIVE = pd.Timestamp("2025-01-01 00:00:00")
CONGESTION_REFERENCE_START = pd.Timestamp("2025-03-13 00:00:00")
CONGESTION_REFERENCE_END_EXCLUSIVE = pd.Timestamp("2025-08-23 00:00:00")
REFERENCE_VERSION = "environmental-normalization-v1"


def _reference_values(path: Path, value_column: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["timestamp", value_column])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    valid = (
        frame["timestamp"].notna()
        & frame["timestamp"].between(
            REFERENCE_START,
            REFERENCE_END_EXCLUSIVE,
            inclusive="left",
        )
        & values.notna()
    )
    return values.loc[valid]


def _variable_payload(values: pd.Series, *, unit: str, scope: str) -> dict:
    if values.empty:
        raise ValueError(f"No hay observaciones validas para {scope}.")
    percentiles = values.quantile([0.10, 0.50, 0.90])
    if float(percentiles.loc[0.90]) <= float(percentiles.loc[0.10]):
        raise ValueError(f"El rango P10-P90 es insuficiente para {scope}.")
    return {
        "unit": unit,
        "scope": scope,
        "sample_size": int(len(values)),
        "p10": round(float(percentiles.loc[0.10]), 5),
        "p50": round(float(percentiles.loc[0.50]), 5),
        "p90": round(float(percentiles.loc[0.90]), 5),
    }


def _congestion_values(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["segment_id", "velocidad_kmh", "duracion_min", "datetime_inicio"],
    )
    frame["datetime_inicio"] = pd.to_datetime(frame["datetime_inicio"], errors="coerce")
    frame = frame[
        frame["datetime_inicio"].between(
            CONGESTION_REFERENCE_START,
            CONGESTION_REFERENCE_END_EXCLUSIVE,
            inclusive="left",
        )
    ]
    return frame.drop_duplicates(subset=["segment_id"], keep="first")


def build_reference(
    pm25_path: Path = PM25_PATH,
    wind_path: Path = WIND_PATH,
    congestion_path: Path = CONGESTION_PATH,
) -> dict:
    congestion = _congestion_values(congestion_path)
    return {
        "version": REFERENCE_VERSION,
        "reference_period": {
            "start": REFERENCE_START.isoformat(),
            "end_exclusive": REFERENCE_END_EXCLUSIVE.isoformat(),
        },
        "congestion_reference_period": {
            "start": CONGESTION_REFERENCE_START.isoformat(),
            "end_exclusive": CONGESTION_REFERENCE_END_EXCLUSIVE.isoformat(),
        },
        "variables": {
            "pm25": _variable_payload(
                _reference_values(pm25_path, "PM25"),
                unit="ug/m3",
                scope="gran_concepcion_core_stations",
            ),
            "wind_speed": _variable_payload(
                _reference_values(wind_path, "wind_speed_mean"),
                unit="m/s",
                scope="gran_concepcion_network_hourly",
            ),
            "congestion_speed_kmh": _variable_payload(
                pd.to_numeric(congestion["velocidad_kmh"], errors="coerce").dropna(),
                unit="km/h",
                scope="gran_concepcion_core_congestion_events",
            ),
            "congestion_duration_min": _variable_payload(
                pd.to_numeric(congestion["duracion_min"], errors="coerce").dropna(),
                unit="min",
                scope="gran_concepcion_core_congestion_events",
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye la referencia historica fija de la capa ambiental.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    payload = build_reference()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
