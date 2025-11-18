# -*- coding: utf-8 -*-
"""
Genera user_ratings basados en datos históricos de congestiones y accidentes.
Vías con MENOS incidentes reciben ratings MÁS ALTOS.
Esto permite que UBCF/IBCF recomienden rutas más seguras.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ACCIDENTS_PATH = RAW_DIR / "ACCIDENTES.csv"
CONGESTIONS_PATH = RAW_DIR / "CONGESTIONES.csv"
ROAD_NETWORK_PATH = PROCESSED_DIR / "road_network.csv"

OUTPUTS: Dict[str, Path] = {
    "regional": PROCESSED_DIR / "user_ratings.csv",
    "concepcion": PROCESSED_DIR / "user_ratings_concepcion.csv",
}

CONCEPCION_BBOX = (-36.95, -36.7, -73.2, -72.9)


def load_incidents() -> pd.DataFrame:
    """Carga y combina datos de accidentes y congestiones"""
    incidents = []

    if ACCIDENTS_PATH.exists():
        acc = pd.read_csv(ACCIDENTS_PATH)
        acc["tipo_evento"] = "Accidente"
        incidents.append(acc)

    if CONGESTIONS_PATH.exists():
        cong = pd.read_csv(CONGESTIONS_PATH)
        cong["tipo_evento"] = "Congestión"
        incidents.append(cong)

    if not incidents:
        raise FileNotFoundError("No se encontraron archivos de incidentes")

    df = pd.concat(incidents, ignore_index=True)

    # Limpiar y normalizar
    df["via"] = df["via"].astype(str).str.strip()
    df = df[df["via"] != ""]

    # Convertir duracion_hrs a numérico
    df["duracion_hrs"] = pd.to_numeric(df.get("duracion_hrs", 0), errors="coerce").fillna(0)

    return df


def calculate_via_safety_scores(incidents: pd.DataFrame, region: str = "regional") -> pd.DataFrame:
    """
    Calcula puntajes de seguridad para cada vía basándose en incidentes.

    Métricas:
    - Frecuencia de incidentes (menos es mejor)
    - Severidad (duración promedio de incidentes)
    - Tipo (accidentes pesan más que congestiones)
    """

    # Filtrar por región si es necesario
    if region == "concepcion" and "lat" in incidents.columns and "lon" in incidents.columns:
        lat_min, lat_max, lon_min, lon_max = CONCEPCION_BBOX
        incidents["lat"] = pd.to_numeric(incidents["lat"], errors="coerce")
        incidents["lon"] = pd.to_numeric(incidents["lon"], errors="coerce")
        incidents = incidents[
            incidents["lat"].between(lat_min, lat_max) &
            incidents["lon"].between(lon_min, lon_max)
        ]

    # Agrupar por vía
    via_stats = incidents.groupby("via").agg({
        "tipo_evento": "count",  # Frecuencia total
        "duracion_hrs": "mean",   # Severidad promedio
    }).rename(columns={"tipo_evento": "incident_count", "duracion_hrs": "avg_duration"})

    # Contar accidentes y congestiones por separado
    accident_counts = incidents[incidents["tipo_evento"] == "Accidente"].groupby("via").size()
    congestion_counts = incidents[incidents["tipo_evento"] == "Congestión"].groupby("via").size()

    via_stats["accident_count"] = accident_counts
    via_stats["congestion_count"] = congestion_counts
    via_stats = via_stats.fillna(0)

    # Calcular score de peligrosidad (más alto = más peligroso)
    # Accidentes pesan 3x más que congestiones
    via_stats["danger_score"] = (
        via_stats["accident_count"] * 3.0 +
        via_stats["congestion_count"] * 1.0 +
        via_stats["avg_duration"] * 10.0  # Duración también afecta
    )

    return via_stats.reset_index()


def normalize_to_ratings(via_stats: pd.DataFrame) -> pd.DataFrame:
    """Convierte scores de peligro a ratings de 1-5 (invertidos)"""

    if via_stats.empty:
        return via_stats

    # Normalizar danger_score a [0, 1]
    min_danger = via_stats["danger_score"].min()
    max_danger = via_stats["danger_score"].max()

    if min_danger == max_danger:
        via_stats["normalized_danger"] = 0.5
    else:
        via_stats["normalized_danger"] = (
            (via_stats["danger_score"] - min_danger) / (max_danger - min_danger)
        )

    # Convertir a rating: peligro alto = rating bajo
    # normalized_danger: 0 (seguro) → 1 (peligroso)
    # rating: 5 (bueno) → 1 (malo)
    via_stats["safety_rating"] = (1 - via_stats["normalized_danger"]) * 4 + 1
    via_stats["safety_rating"] = via_stats["safety_rating"].clip(1.0, 5.0)

    return via_stats


def load_all_vias(region: str) -> pd.DataFrame:
    """Carga todas las vías de la red con atributos adicionales para mayor variabilidad"""
    if not ROAD_NETWORK_PATH.exists():
        raise FileNotFoundError(f"No existe {ROAD_NETWORK_PATH}")

    df = pd.read_csv(ROAD_NETWORK_PATH)
    df["via"] = df["via"].astype(str).str.strip()
    df = df[df["via"] != ""]

    # Convertir columnas numéricas
    df["velocidad_kmh"] = pd.to_numeric(df.get("velocidad_kmh", 0), errors="coerce").fillna(40)
    df["distancia_km"] = pd.to_numeric(df.get("distancia_km", 0), errors="coerce").fillna(1)

    if region == "concepcion" and "lat" in df.columns and "lon" in df.columns:
        lat_min, lat_max, lon_min, lon_max = CONCEPCION_BBOX
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df[
            df["lat"].between(lat_min, lat_max) &
            df["lon"].between(lon_min, lon_max)
        ]

    # Agregar estadísticas promedio por vía
    via_attrs = df.groupby("via").agg({
        "velocidad_kmh": "mean",
        "distancia_km": "mean",
    }).reset_index()

    return via_attrs


def build_user_profiles(via_stats: pd.DataFrame, all_vias: pd.DataFrame) -> pd.DataFrame:
    """
    Crea perfiles de usuario con PATRONES REALMENTE DIFERENTES para que
    UBCF e IBCF generen recomendaciones distintas.

    Estrategia: Crear preferencias OPUESTAS y usar diferentes combinaciones de features.

    Perfiles:
    - safety_focused: SOLO seguridad, ignora velocidad
    - risk_taker: INVERSO - prefiere vías peligrosas pero rápidas
    - moderate_risk: Usa percentiles para preferencias específicas
    - usuario_demo: Mezcla con alta variabilidad aleatoria
    """

    # Merge con todas las vías y atributos
    full_data = all_vias.merge(via_stats[["via", "danger_score", "accident_count", "congestion_count"]], on="via", how="left")

    # Rellenar valores faltantes
    full_data["danger_score"] = full_data["danger_score"].fillna(0)
    full_data["accident_count"] = full_data["accident_count"].fillna(0)
    full_data["congestion_count"] = full_data["congestion_count"].fillna(0)

    # Normalizar features
    np.random.seed(42)  # Para reproducibilidad

    full_data["velocidad_norm"] = (full_data["velocidad_kmh"] - full_data["velocidad_kmh"].min()) / (full_data["velocidad_kmh"].max() - full_data["velocidad_kmh"].min() + 0.01)
    full_data["distancia_norm"] = (full_data["distancia_km"] - full_data["distancia_km"].min()) / (full_data["distancia_km"].max() - full_data["distancia_km"].min() + 0.01)
    full_data["danger_norm"] = (full_data["danger_score"] - full_data["danger_score"].min()) / (full_data["danger_score"].max() - full_data["danger_score"].min() + 0.01)

    # Calcular percentiles de peligro para crear grupos
    danger_p33 = full_data["danger_score"].quantile(0.33)
    danger_p66 = full_data["danger_score"].quantile(0.66)
    velocity_p50 = full_data["velocidad_kmh"].quantile(0.50)

    # Crear perfiles con PREFERENCIAS OPUESTAS
    profiles = {}

    # SAFETY FOCUSED: Penaliza FUERTEMENTE el peligro, no le importa la velocidad
    # Este perfil tiene correlación NEGATIVA con danger_score
    profiles["safety_focused"] = (
        5.0 - (full_data["danger_norm"] * 3.5) +  # Peligro alto = rating muy bajo
        np.random.normal(0, 0.4, len(full_data))  # Alta variabilidad aleatoria
    ).clip(1.0, 5.0)

    # RISK TAKER: Prefiere velocidad, tolera peligro (OPUESTO a safety_focused)
    # Este perfil tiene correlación POSITIVA con velocidad y BAJA con seguridad
    profiles["risk_taker"] = (
        2.5 +  # Base neutral-baja
        full_data["velocidad_norm"] * 2.5 +  # Velocidad alta = rating alto
        (1 - full_data["danger_norm"]) * 0.5 +  # Peligro afecta POCO
        np.random.normal(0, 0.5, len(full_data))  # Alta variabilidad
    ).clip(1.0, 5.0)

    # MODERATE RISK: Usa PERCENTILES para crear preferencias por grupos
    # Prefiere vías del grupo medio (ni muy peligrosas ni muy seguras)
    moderate_ratings = np.ones(len(full_data)) * 3.0  # Base

    # Bonus para vías en el rango medio de peligro
    mask_mid = (full_data["danger_score"] >= danger_p33) & (full_data["danger_score"] <= danger_p66)
    moderate_ratings[mask_mid] += 1.0

    # Penaliza vías MUY peligrosas
    mask_very_dangerous = full_data["danger_score"] > danger_p66
    moderate_ratings[mask_very_dangerous] -= 0.8

    # Bonus por velocidad moderada
    mask_fast = full_data["velocidad_kmh"] > velocity_p50
    moderate_ratings[mask_fast] += 0.5

    profiles["moderate_risk"] = (
        pd.Series(moderate_ratings) +
        np.random.normal(0, 0.6, len(full_data))  # MUY alta variabilidad
    ).clip(1.0, 5.0)

    # USUARIO DEMO: Mezcla ALEATORIA con patrones diferentes por tipo de vía
    # Usa combinación no lineal de features
    demo_ratings = np.ones(len(full_data)) * 3.0

    # Grupo 1: Vías rápidas Y seguras (bonus alto)
    mask_good = (full_data["velocidad_norm"] > 0.6) & (full_data["danger_norm"] < 0.3)
    demo_ratings[mask_good] += 1.5

    # Grupo 2: Vías lentas O peligrosas (penalización)
    mask_bad = (full_data["velocidad_norm"] < 0.4) | (full_data["danger_norm"] > 0.7)
    demo_ratings[mask_bad] -= 1.0

    # Grupo 3: Solo accidentes (penalización extra)
    mask_accidents = full_data["accident_count"] > 0
    demo_ratings[mask_accidents] -= 0.8

    profiles["usuario_demo"] = (
        pd.Series(demo_ratings) +
        np.random.normal(0, 0.7, len(full_data))  # MÁXIMA variabilidad aleatoria
    ).clip(1.0, 5.0)

    # Generar ratings SPARSE: cada usuario solo califica vías según sus preferencias
    # Esto es CRÍTICO para que UBCF e IBCF generen resultados diferentes
    rows = []

    for user_id, ratings in profiles.items():
        # Convertir a DataFrame temporal para filtrar
        user_data = pd.DataFrame({
            'via': full_data['via'],
            'rating': ratings,
            'danger_score': full_data['danger_score'],
            'velocidad_kmh': full_data['velocidad_kmh'],
        })

        # Cada perfil califica diferentes tipos de vías (crear SPARSITY)
        if user_id == "safety_focused":
            # Solo califica vías que considera "seguras" o "muy peligrosas" (para evitarlas)
            # 60% de las vías con ratings >= 4.5 O ratings <= 2.5
            mask = (user_data['rating'] >= 4.5) | (user_data['rating'] <= 2.5)
            selected = user_data[mask].sample(frac=0.6, random_state=42)

        elif user_id == "risk_taker":
            # Califica principalmente vías rápidas y algunas lentas (para evitarlas)
            # 50% de las vías con velocidad > 50 km/h O rating >= 4.0
            mask = (user_data['velocidad_kmh'] > 50) | (user_data['rating'] >= 4.0)
            selected = user_data[mask].sample(frac=0.5, random_state=43)

        elif user_id == "moderate_risk":
            # Califica una mezcla equilibrada de diferentes tipos
            # 40% aleatorio de todas las vías
            selected = user_data.sample(frac=0.4, random_state=44)

        else:  # usuario_demo
            # Califica principalmente vías con ratings extremos (altos o bajos)
            # 45% de vías con rating >= 3.5 O rating <= 1.5
            mask = (user_data['rating'] >= 3.5) | (user_data['rating'] <= 1.5)
            selected = user_data[mask].sample(frac=0.45, random_state=45)

        # Agregar ratings seleccionados
        for _, row in selected.iterrows():
            rows.append({
                "user_id": user_id,
                "via": row['via'],
                "rating": round(float(row['rating']), 2),
            })

    return pd.DataFrame(rows).sort_values(["user_id", "via"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera user_ratings basados en datos históricos de incidentes"
    )
    parser.add_argument(
        "--mode",
        choices=["regional", "concepcion"],
        default="regional",
        help="Subset geográfico a considerar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida opcional",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else OUTPUTS[args.mode]

    print(f"Cargando incidentes...")
    incidents = load_incidents()
    print(f"  Total incidentes: {len(incidents)}")

    print(f"Calculando scores de seguridad por vía...")
    via_stats = calculate_via_safety_scores(incidents, region=args.mode)
    print(f"  Vías con incidentes: {len(via_stats)}")

    print(f"Normalizando a ratings...")
    via_stats = normalize_to_ratings(via_stats)

    # Mostrar estadísticas
    print(f"\nEstadísticas de peligrosidad:")
    print(f"  Vía más peligrosa: {via_stats.loc[via_stats['danger_score'].idxmax(), 'via']}")
    print(f"  Vía más segura: {via_stats.loc[via_stats['danger_score'].idxmin(), 'via']}")
    print(f"  Rating promedio: {via_stats['safety_rating'].mean():.2f}")
    print(f"  Rating mínimo: {via_stats['safety_rating'].min():.2f}")
    print(f"  Rating máximo: {via_stats['safety_rating'].max():.2f}")

    print(f"\nCargando todas las vías de la red...")
    all_vias = load_all_vias(args.mode)
    print(f"  Total vías en red: {len(all_vias)}")

    print(f"Generando perfiles de usuario...")
    ratings_df = build_user_profiles(via_stats, all_vias)
    print(f"  Total ratings generados: {len(ratings_df)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ratings_df.to_csv(output_path, index=False)
    print(f"\n✅ Escribí {len(ratings_df)} ratings en {output_path}")

    # Mostrar muestra de ratings por perfil
    print("\nMuestra de ratings por perfil:")
    for user in ratings_df["user_id"].unique():
        user_ratings = ratings_df[ratings_df["user_id"] == user]["rating"]
        print(f"  {user}: promedio={user_ratings.mean():.2f}, std={user_ratings.std():.2f}")


if __name__ == "__main__":
    main()
