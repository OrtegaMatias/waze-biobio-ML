# -*- coding: utf-8 -*-
"""
Genera user_ratings basados en datos historicos de congestiones y accidentes.
Vias con menos incidentes reciben ratings mas altos.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from algorithms.recommenders.geo_profiles import canonicalize_data_profile, filter_dataframe_for_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ACCIDENTS_PATH = RAW_DIR / "ACCIDENTES.csv"
CONGESTIONS_PATH = RAW_DIR / "CONGESTIONES.csv"
ROAD_NETWORK_PATH = PROCESSED_DIR / "road_network.csv"

OUTPUTS: Dict[str, Path] = {
    "gran_concepcion": PROCESSED_DIR / "user_ratings.csv",
    "gran_concepcion_core": PROCESSED_DIR / "user_ratings_concepcion.csv",
}


def load_incidents() -> pd.DataFrame:
    incidents = []

    if ACCIDENTS_PATH.exists():
        acc = pd.read_csv(ACCIDENTS_PATH)
        acc["tipo_evento"] = "Accidente"
        incidents.append(acc)

    if CONGESTIONS_PATH.exists():
        cong = pd.read_csv(CONGESTIONS_PATH)
        cong["tipo_evento"] = "Congestion"
        incidents.append(cong)

    if not incidents:
        raise FileNotFoundError("No se encontraron archivos de incidentes")

    df = pd.concat(incidents, ignore_index=True)
    df["via"] = df["via"].astype(str).str.strip()
    df = df[df["via"] != ""]
    df["duracion_hrs"] = pd.to_numeric(df.get("duracion_hrs", 0), errors="coerce").fillna(0)
    return df


def calculate_via_safety_scores(incidents: pd.DataFrame, region: str = "gran_concepcion") -> pd.DataFrame:
    region = canonicalize_data_profile(region)
    incidents = filter_dataframe_for_profile(incidents, region)

    via_stats = incidents.groupby("via").agg(
        {
            "tipo_evento": "count",
            "duracion_hrs": "mean",
        }
    ).rename(columns={"tipo_evento": "incident_count", "duracion_hrs": "avg_duration"})

    accident_counts = incidents[incidents["tipo_evento"] == "Accidente"].groupby("via").size()
    congestion_counts = incidents[incidents["tipo_evento"] == "Congestion"].groupby("via").size()

    via_stats["accident_count"] = accident_counts
    via_stats["congestion_count"] = congestion_counts
    via_stats = via_stats.fillna(0)
    via_stats["danger_score"] = (
        via_stats["accident_count"] * 3.0
        + via_stats["congestion_count"] * 1.0
        + via_stats["avg_duration"] * 10.0
    )

    return via_stats.reset_index()


def normalize_to_ratings(via_stats: pd.DataFrame) -> pd.DataFrame:
    if via_stats.empty:
        return via_stats

    min_danger = via_stats["danger_score"].min()
    max_danger = via_stats["danger_score"].max()

    if min_danger == max_danger:
        via_stats["normalized_danger"] = 0.5
    else:
        via_stats["normalized_danger"] = (
            (via_stats["danger_score"] - min_danger) / (max_danger - min_danger)
        )

    via_stats["safety_rating"] = (1 - via_stats["normalized_danger"]) * 4 + 1
    via_stats["safety_rating"] = via_stats["safety_rating"].clip(1.0, 5.0)

    return via_stats


def load_all_vias(region: str) -> pd.DataFrame:
    region = canonicalize_data_profile(region)
    if not ROAD_NETWORK_PATH.exists():
        raise FileNotFoundError(f"No existe {ROAD_NETWORK_PATH}")

    df = pd.read_csv(ROAD_NETWORK_PATH)
    df["via"] = df["via"].astype(str).str.strip()
    df = df[df["via"] != ""]
    df["velocidad_kmh"] = pd.to_numeric(df.get("velocidad_kmh", 0), errors="coerce").fillna(40)
    df["distancia_km"] = pd.to_numeric(df.get("distancia_km", 0), errors="coerce").fillna(1)
    df = filter_dataframe_for_profile(df, region)

    via_attrs = df.groupby("via").agg(
        {
            "velocidad_kmh": "mean",
            "distancia_km": "mean",
        }
    ).reset_index()

    return via_attrs


def build_user_profiles(via_stats: pd.DataFrame, all_vias: pd.DataFrame) -> pd.DataFrame:
    full_data = all_vias.merge(
        via_stats[["via", "danger_score", "accident_count", "congestion_count"]],
        on="via",
        how="left",
    )

    full_data["danger_score"] = full_data["danger_score"].fillna(0)
    full_data["accident_count"] = full_data["accident_count"].fillna(0)
    full_data["congestion_count"] = full_data["congestion_count"].fillna(0)

    np.random.seed(42)

    full_data["velocidad_norm"] = (
        (full_data["velocidad_kmh"] - full_data["velocidad_kmh"].min())
        / (full_data["velocidad_kmh"].max() - full_data["velocidad_kmh"].min() + 0.01)
    )
    full_data["distancia_norm"] = (
        (full_data["distancia_km"] - full_data["distancia_km"].min())
        / (full_data["distancia_km"].max() - full_data["distancia_km"].min() + 0.01)
    )
    full_data["danger_norm"] = (
        (full_data["danger_score"] - full_data["danger_score"].min())
        / (full_data["danger_score"].max() - full_data["danger_score"].min() + 0.01)
    )

    danger_p33 = full_data["danger_score"].quantile(0.33)
    danger_p66 = full_data["danger_score"].quantile(0.66)
    velocity_p50 = full_data["velocidad_kmh"].quantile(0.50)

    profiles = {}

    profiles["safety_focused"] = (
        5.0 - (full_data["danger_norm"] * 3.5) + np.random.normal(0, 0.4, len(full_data))
    ).clip(1.0, 5.0)

    profiles["risk_taker"] = (
        2.5
        + full_data["velocidad_norm"] * 2.5
        + (1 - full_data["danger_norm"]) * 0.5
        + np.random.normal(0, 0.5, len(full_data))
    ).clip(1.0, 5.0)

    moderate_ratings = np.ones(len(full_data)) * 3.0
    mask_mid = (full_data["danger_score"] >= danger_p33) & (full_data["danger_score"] <= danger_p66)
    moderate_ratings[mask_mid] += 1.0
    mask_very_dangerous = full_data["danger_score"] > danger_p66
    moderate_ratings[mask_very_dangerous] -= 0.8
    mask_fast = full_data["velocidad_kmh"] > velocity_p50
    moderate_ratings[mask_fast] += 0.5
    profiles["moderate_risk"] = (pd.Series(moderate_ratings) + np.random.normal(0, 0.6, len(full_data))).clip(
        1.0,
        5.0,
    )

    demo_ratings = np.ones(len(full_data)) * 3.0
    mask_good = (full_data["velocidad_norm"] > 0.6) & (full_data["danger_norm"] < 0.3)
    demo_ratings[mask_good] += 1.5
    mask_bad = (full_data["velocidad_norm"] < 0.4) | (full_data["danger_norm"] > 0.7)
    demo_ratings[mask_bad] -= 1.0
    mask_accidents = full_data["accident_count"] > 0
    demo_ratings[mask_accidents] -= 0.8
    profiles["usuario_demo"] = (pd.Series(demo_ratings) + np.random.normal(0, 0.7, len(full_data))).clip(
        1.0,
        5.0,
    )

    rows = []
    for user_id, ratings in profiles.items():
        user_data = pd.DataFrame(
            {
                "via": full_data["via"],
                "rating": ratings,
                "danger_score": full_data["danger_score"],
                "velocidad_kmh": full_data["velocidad_kmh"],
            }
        )

        if user_id == "safety_focused":
            mask = (user_data["rating"] >= 4.5) | (user_data["rating"] <= 2.5)
            selected = user_data[mask].sample(frac=0.6, random_state=42)
        elif user_id == "risk_taker":
            mask = (user_data["velocidad_kmh"] > 50) | (user_data["rating"] >= 4.0)
            selected = user_data[mask].sample(frac=0.5, random_state=43)
        elif user_id == "moderate_risk":
            selected = user_data.sample(frac=0.4, random_state=44)
        else:
            mask = (user_data["rating"] >= 3.5) | (user_data["rating"] <= 1.5)
            selected = user_data[mask].sample(frac=0.45, random_state=45)

        for _, row in selected.iterrows():
            rows.append(
                {
                    "user_id": user_id,
                    "via": row["via"],
                    "rating": round(float(row["rating"]), 2),
                }
            )

    return pd.DataFrame(rows).sort_values(["user_id", "via"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera user_ratings basados en datos historicos de incidentes")
    parser.add_argument(
        "--mode",
        choices=["gran_concepcion", "gran_concepcion_core", "regional", "concepcion"],
        default="gran_concepcion",
        help="Subset geografico a considerar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida opcional",
    )
    args = parser.parse_args()

    mode = canonicalize_data_profile(args.mode)
    output_path = Path(args.output) if args.output else OUTPUTS[mode]

    print("Cargando incidentes...")
    incidents = load_incidents()
    print(f"  Total incidentes: {len(incidents)}")

    print("Calculando scores de seguridad por via...")
    via_stats = calculate_via_safety_scores(incidents, region=mode)
    print(f"  Vias con incidentes: {len(via_stats)}")

    print("Normalizando a ratings...")
    via_stats = normalize_to_ratings(via_stats)

    print("\nCargando todas las vias de la red...")
    all_vias = load_all_vias(mode)
    print(f"  Total vias en red: {len(all_vias)}")

    print("Generando perfiles de usuario...")
    ratings_df = build_user_profiles(via_stats, all_vias)
    print(f"  Total ratings generados: {len(ratings_df)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(output_path, index=False)
    print(f"\nEscribi {len(ratings_df)} ratings en {output_path}")

    print("\nMuestra de ratings por perfil:")
    for user in ratings_df["user_id"].unique():
        user_ratings = ratings_df[ratings_df["user_id"] == user]["rating"]
        print(f"  {user}: promedio={user_ratings.mean():.2f}, std={user_ratings.std():.2f}")


if __name__ == "__main__":
    main()
