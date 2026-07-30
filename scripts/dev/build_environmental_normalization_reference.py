from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
PM25_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_pm25_core_hourly_clean.csv"
WIND_PATH = ROOT_DIR / "data_processed" / "gran_concepcion_wind_network_hourly.csv"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data_analysis" / "environmental_normalization_v1.json"
REFERENCE_START = pd.Timestamp("2021-01-01 00:00:00")
REFERENCE_END_EXCLUSIVE = pd.Timestamp("2025-01-01 00:00:00")
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


def build_reference(pm25_path: Path = PM25_PATH, wind_path: Path = WIND_PATH) -> dict:
    return {
        "version": REFERENCE_VERSION,
        "reference_period": {
            "start": REFERENCE_START.isoformat(),
            "end_exclusive": REFERENCE_END_EXCLUSIVE.isoformat(),
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
