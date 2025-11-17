# -*- coding: utf-8 -*-
"""
Genera datasets de ratings sintéticos a partir del road_network.
Permite cubrir todas las vías con distintos perfiles de usuario (velocidad, distancia, tiempo, balanceado).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ROAD_NETWORK_PATH = PROCESSED_DIR / "road_network.csv"
OUTPUTS: Dict[str, Path] = {
    "regional": PROCESSED_DIR / "user_ratings.csv",
    "concepcion": PROCESSED_DIR / "user_ratings_concepcion.csv",
}

CONCEPCION_BBOX: Tuple[float, float, float, float] = (-36.95, -36.7, -73.2, -72.9)


def _normalize(series: pd.Series) -> pd.Series:
    clean = series.fillna(series.median() if not series.dropna().empty else 0.0)
    min_val = float(clean.min())
    max_val = float(clean.max())
    if min_val == max_val:
        return pd.Series(0.5, index=clean.index)
    return (clean - min_val) / (max_val - min_val)


def _to_rating(score: pd.Series) -> pd.Series:
    return (score.clip(0, 1) * 4 + 1).clip(1.0, 5.0)


def load_road_network(region: str) -> pd.DataFrame:
    if not ROAD_NETWORK_PATH.exists():
        raise FileNotFoundError(f"No existe {ROAD_NETWORK_PATH}")
    df = pd.read_csv(ROAD_NETWORK_PATH)
    for column in ("velocidad_kmh", "distancia_km", "duracion_hrs", "lat", "lon"):
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["via"] = df["via"].astype(str).str.strip()
    df = df[df["via"] != ""]
    if region == "concepcion":
        lat_min, lat_max, lon_min, lon_max = CONCEPCION_BBOX
        df = df[
            df["lat"].between(lat_min, lat_max)
            & df["lon"].between(lon_min, lon_max)
        ]
    if df.empty:
        raise ValueError(f"No se encontraron vías para la región '{region}'.")
    grouped = (
        df.groupby("via", as_index=False)
        .agg(
            velocidad_kmh=("velocidad_kmh", "mean"),
            distancia_km=("distancia_km", "mean"),
            duracion_hrs=("duracion_hrs", "mean"),
        )
        .reset_index(drop=True)
    )
    return grouped


def build_ratings(region: str) -> pd.DataFrame:
    roads = load_road_network(region)
    speed_norm = _normalize(roads["velocidad_kmh"])
    distance_norm = _normalize(roads["distancia_km"])
    duration_norm = _normalize(roads["duracion_hrs"])

    ratings_map = {
        "speed_hunter": _to_rating(speed_norm),
        "short_trip": _to_rating(1 - distance_norm),
        "time_saver": _to_rating(1 - duration_norm),
        "balanced_traveler": np.clip(
            (_to_rating(speed_norm) + _to_rating(1 - distance_norm) + _to_rating(1 - duration_norm)) / 3,
            1.0,
            5.0,
        ),
    }

    rows = []
    vias = roads["via"].tolist()
    for user_id, scores in ratings_map.items():
        for via, rating in zip(vias, scores):
            rows.append(
                {
                    "user_id": user_id,
                    "via": via,
                    "rating": round(float(rating), 2),
                }
            )
    ratings_df = pd.DataFrame(rows).sort_values(["user_id", "via"]).reset_index(drop=True)
    return ratings_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera user_ratings sintéticos basados en road_network.csv")
    parser.add_argument(
        "--mode",
        choices=["regional", "concepcion"],
        default="regional",
        help="Subset del road_network a considerar.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida opcional. Por defecto sobrescribe el archivo oficial según el modo.",
    )
    args = parser.parse_args()
    output_path = Path(args.output) if args.output else OUTPUTS[args.mode]
    ratings_df = build_ratings(args.mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(output_path, index=False)
    print(f"Escribí {len(ratings_df)} ratings en {output_path}")


if __name__ == "__main__":
    main()
