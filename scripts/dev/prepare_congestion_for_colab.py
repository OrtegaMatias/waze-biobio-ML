from __future__ import annotations

import argparse
from pathlib import Path
import unicodedata

import pandas as pd

PROFILE_ALIASES = {
    "regional": "gran_concepcion",
    "concepcion": "gran_concepcion_core",
}

PROFILE_COMMUNES = {
    "gran_concepcion": {
        "chiguayante",
        "concepcion",
        "coronel",
        "hualpen",
        "hualqui",
        "lota",
        "penco",
        "san pedro de la paz",
        "talcahuano",
        "tome",
    },
    "gran_concepcion_core": {
        "chiguayante",
        "concepcion",
        "hualpen",
        "penco",
        "san pedro de la paz",
        "talcahuano",
    },
}


def normalize_geo_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.strip().split())
    if not text:
        return None
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    lowered = ascii_text.lower()
    return lowered or None


def canonicalize_data_profile(profile: str) -> str:
    normalized = normalize_geo_text(profile)
    if normalized is None:
        raise ValueError("Perfil de datos vacio o invalido.")
    normalized = PROFILE_ALIASES.get(normalized, normalized)
    if normalized not in PROFILE_COMMUNES:
        raise ValueError(f"Perfil de datos no soportado: {profile}")
    return normalized


def filter_dataframe_for_profile(
    df: pd.DataFrame,
    profile: str,
    commune_column: str = "comuna",
) -> pd.DataFrame:
    if df.empty or commune_column not in df.columns:
        return df
    canonical_profile = canonicalize_data_profile(profile)
    normalized_communes = df[commune_column].map(normalize_geo_text)
    return df[normalized_communes.isin(PROFILE_COMMUNES[canonical_profile])].reset_index(drop=True)


def build_outputs(
    input_path: Path,
    output_dir: Path,
    profile: str | None = None,
) -> tuple[Path, Path]:
    df = pd.read_csv(input_path)
    suffix = ""
    if profile:
        canonical_profile = canonicalize_data_profile(profile)
        df = filter_dataframe_for_profile(df, canonical_profile)
        suffix = f"_{canonical_profile}"
    original_columns = df.columns.tolist()

    # Minimal cleanup: remove exact duplicates without changing the base logic.
    df = df.drop_duplicates().copy()

    # Preserve original columns and add derived fields for analysis.
    df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["fecha_dia_dt"] = pd.to_datetime(df["fecha_dia"], errors="coerce")
    df["hora_inicio_time"] = pd.to_datetime(df["hora_inicio"], format="%H:%M", errors="coerce")
    df["hora_fin_time"] = pd.to_datetime(df["hora_fin"], format="%H:%M", errors="coerce")
    df["datetime_inicio"] = pd.to_datetime(
        df["fecha"].astype(str) + " " + df["hora_inicio"].astype(str),
        errors="coerce",
    )
    df["datetime_fin"] = pd.to_datetime(
        df["fecha"].astype(str) + " " + df["hora_fin"].astype(str),
        errors="coerce",
    )
    df["duracion_min"] = df["duracion_hrs"] * 60

    clean_columns = original_columns + [
        "fecha_dt",
        "fecha_dia_dt",
        "hora_inicio_time",
        "hora_fin_time",
        "datetime_inicio",
        "datetime_fin",
        "duracion_min",
    ]
    clean_df = df.loc[:, clean_columns].copy()

    # Hourly aggregation at event level to avoid duplicating by coordinate.
    events = clean_df.sort_values("row_idx").drop_duplicates(subset=["segment_id"], keep="first").copy()
    events["periodo_hora"] = events["datetime_inicio"].dt.floor("h")

    aggregated_df = (
        events.groupby("periodo_hora", dropna=False)
        .agg(
            cantidad_eventos_congestion=("segment_id", "nunique"),
            velocidad_promedio_kmh=("velocidad_kmh", "mean"),
            duracion_promedio_min=("duracion_min", "mean"),
            distancia_promedio_km=("distancia_km", "mean"),
            archivos_origen_n=("archivo_origen", "nunique"),
            comunas_n=("comuna", "nunique"),
        )
        .reset_index()
        .sort_values("periodo_hora")
    )
    aggregated_df["fecha"] = aggregated_df["periodo_hora"].dt.date.astype("string")
    aggregated_df["hora"] = aggregated_df["periodo_hora"].dt.strftime("%H:%M")
    aggregated_columns = [
        "periodo_hora",
        "fecha",
        "hora",
        "cantidad_eventos_congestion",
        "velocidad_promedio_kmh",
        "duracion_promedio_min",
        "distancia_promedio_km",
        "archivos_origen_n",
        "comunas_n",
    ]
    aggregated_df = aggregated_df.loc[:, aggregated_columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = output_dir / f"congestion_clean{suffix}.csv"
    aggregated_path = output_dir / f"congestion_aggregated{suffix}.csv"

    clean_df.to_csv(clean_path, index=False)
    aggregated_df.to_csv(aggregated_path, index=False)

    return clean_path, aggregated_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare congestion datasets for Google Colab analysis."
    )
    parser.add_argument(
        "--input",
        default=r"data\raw\CONGESTIONES_biobio_2025_03_08.csv",
        help="Path to the base congestion CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"data_analysis",
        help="Directory where clean and aggregated CSVs will be written.",
    )
    parser.add_argument(
        "--profile",
        choices=["regional", "concepcion", "gran_concepcion", "gran_concepcion_core"],
        default=None,
        help="Optional geographic filter. If omitted, keeps regional coverage.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    clean_path, aggregated_path = build_outputs(input_path, output_dir, profile=args.profile)

    print(f"input={input_path}")
    print(f"profile={args.profile or 'regional_full'}")
    print(f"clean={clean_path}")
    print(f"aggregated={aggregated_path}")


if __name__ == "__main__":
    main()
