# -*- coding: utf-8 -*-
"""
Utilidades para cargar y transformar los datos del Biobío.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from sklearn.neighbors import BallTree

ROOT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT_DIR / "data" / "cache"
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ACCIDENT_PATH = RAW_DIR / "ACCIDENTES.csv"
CONGESTION_PATH = RAW_DIR / "CONGESTIONES.csv"
USER_RATINGS_PATH = PROCESSED_DIR / "user_ratings.csv"
ROAD_NETWORK_PATH = PROCESSED_DIR / "road_network.csv"
_CURRENT_USER_RATINGS_PATH = USER_RATINGS_PATH

HOUR_BUCKETS: List[Tuple[int, int, str]] = [
    (0, 6, "Madrugada (00-05h)"),
    (6, 10, "Punta AM (06-09h)"),
    (10, 16, "Horario Medio (10-15h)"),
    (16, 21, "Punta PM (16-20h)"),
    (21, 24, "Nocturno (21-23h)"),
]

ACCIDENT_PENALTY = 1.75
CONGESTION_PENALTY = 1.35
PENALTY_RADIUS_M = 60
EARTH_RADIUS_M = 6_371_000


def _cache_artifact(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _cache_metadata(name: str) -> Path:
    return CACHE_DIR / f"{name}.meta.json"


def _read_cache_signature(path: Path) -> list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("signature")
    except Exception:
        return None


def _store_cache_signature(path: Path, signature) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": list(signature)}))


def _load_cached_dataframe(name: str, signature) -> pd.DataFrame | None:
    artifact = _cache_artifact(name)
    meta = _cache_metadata(name)
    if not artifact.exists():
        return None
    stored_signature = _read_cache_signature(meta)
    if stored_signature != list(signature):
        return None
    try:
        return pd.read_pickle(artifact)
    except Exception:
        return None


def _store_cached_dataframe(name: str, signature, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _cache_artifact(name)
    df.to_pickle(artifact)
    _store_cache_signature(_cache_metadata(name), signature)


def _file_signature(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def data_version() -> Tuple[float, float, float]:
    """Sello temporal para invalidar caches cuando cambian los datos base."""
    return (
        _file_signature(ACCIDENT_PATH),
        _file_signature(CONGESTION_PATH),
        _file_signature(ROAD_NETWORK_PATH),
    )


def _hour_bucket(hour: float) -> str:
    hour = 0 if pd.isna(hour) else hour
    for start, end, label in HOUR_BUCKETS:
        if start <= hour < end:
            return label
    return "No definido"


def hour_bucket(hour: float) -> str:
    return _hour_bucket(hour)


def _duration_bucket(hours: float) -> str:
    if pd.isna(hours):
        return "Desconocida"
    if hours < 0.25:
        return "Duración muy corta"
    if hours < 0.5:
        return "Duración corta"
    if hours < 1:
        return "Duración media"
    if hours < 2:
        return "Duración prolongada"
    return "Duración crítica"


def _speed_bucket(speed: float) -> str:
    if pd.isna(speed):
        return "Velocidad desconocida"
    if speed <= 15:
        return "Velocidad crítica"
    if speed <= 25:
        return "Velocidad lenta"
    if speed <= 40:
        return "Velocidad moderada"
    return "Velocidad fluida"


def _clean_string(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.lower() == "nan":
        return None
    return cleaned


def _safe_title(value: str, fallback: str = "Sin dato") -> str:
    cleaned = _clean_string(value)
    if not cleaned:
        return fallback
    return cleaned.title()


def _get_value(row: pd.Series, column: str, fallback: str) -> str:
    value = row.get(column, fallback)
    if isinstance(value, str):
        cleaned = _clean_string(value)
        if cleaned is None:
            return fallback
        return cleaned
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    return str(value)


def _build_tokens(row: pd.Series) -> Set[str]:
    dia_semana = _get_value(row, "dia_semana", "Monday")
    dia_tipo = "Fin de semana" if dia_semana in {"Saturday", "Sunday"} else "Día laboral"
    tokens = {
        f"evento={_get_value(row, 'tipo_evento', 'Sin evento')}",
        f"comuna={_get_value(row, 'comuna', 'Sin comuna')}",
        f"franja={_get_value(row, 'franja_horaria', 'No definido')}",
        f"duracion={_get_value(row, 'duracion_categoria', 'Desconocida')}",
        f"velocidad={_get_value(row, 'velocidad_categoria', 'Velocidad desconocida')}",
        f"dia={dia_tipo}",
        f"via={_get_value(row, 'via_categoria', 'Otras vías')}",
    }
    return {token for token in tokens if "nan" not in token.lower()}


def _build_penalty_lookup(events: pd.DataFrame):
    valid = events.dropna(subset=["lat", "lon"])
    if valid.empty:
        return None
    coords = np.radians(valid[["lat", "lon"]].astype(float).to_numpy())
    penalties = valid["penalty_factor"].astype(float).to_numpy()
    tree = BallTree(coords, metric="haversine")
    return tree, penalties


def _apply_penalties(reference: pd.DataFrame, lookup) -> pd.DataFrame:
    if reference.empty:
        return reference
    ref = reference.copy()
    ref["penalty_factor"] = ref["penalty_factor"].fillna(1.0)
    if not lookup:
        return ref
    coords = ref[["lat", "lon"]].astype(float).to_numpy()
    mask = np.isfinite(coords).all(axis=1)
    if not mask.any():
        return ref
    tree, penalties = lookup
    radius = PENALTY_RADIUS_M / EARTH_RADIUS_M
    ref_coords = np.radians(coords[mask])
    neighbor_indices = tree.query_radius(ref_coords, r=radius, return_distance=False)
    penalized = ref["penalty_factor"].to_numpy()
    for idx_ref, neighbors in zip(np.where(mask)[0], neighbor_indices):
        if len(neighbors):
            penalized[idx_ref] = max(penalized[idx_ref], penalties[neighbors].max())
    ref["penalty_factor"] = penalized
    return ref


def _prepare_dataframe(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    tipo_evento = {
        "accidente": "Accidente",
        "congestion": "Congestión",
        "referencia": "Referencia",
    }.get(label, label.title())
    df["tipo_evento"] = tipo_evento
    df["penalty_factor"] = 1.0
    if tipo_evento == "Accidente":
        df["penalty_factor"] = ACCIDENT_PENALTY
    elif tipo_evento == "Congestión":
        df["penalty_factor"] = CONGESTION_PENALTY
    df["duracion_hrs"] = pd.to_numeric(df["duracion_hrs"], errors="coerce")
    df["distancia_km"] = pd.to_numeric(df["distancia_km"], errors="coerce")
    df["velocidad_kmh"] = pd.to_numeric(df["velocidad_kmh"], errors="coerce")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["hora_num"] = (
        pd.to_datetime(df["hora_inicio"], format="%H:%M", errors="coerce").dt.hour.fillna(0).astype(int)
    )
    df["franja_horaria"] = df["hora_num"].apply(_hour_bucket)
    df["dia_semana"] = df["fecha"].dt.day_name()
    df["duracion_categoria"] = df["duracion_hrs"].apply(_duration_bucket)
    df["velocidad_categoria"] = df["velocidad_kmh"].apply(_speed_bucket)
    df["comuna"] = df["comuna"].apply(_safe_title)
    df["via"] = df["via"].apply(_safe_title, args=("Sin vía",))
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["oneway"] = df.get("oneway", False)
    df["oneway"] = df["oneway"].fillna(False).astype(bool)
    if "indice_coord" in df.columns:
        df["indice_coord"] = pd.to_numeric(df["indice_coord"], errors="coerce").fillna(0).astype(int)
        df = df.sort_values(["segment_id", "indice_coord"]).reset_index(drop=True)
    else:
        df = df.sort_values(["segment_id"]).reset_index(drop=True)
    df["segment_seq"] = df.groupby("segment_id").cumcount()
    return df


def load_raw_events() -> pd.DataFrame:
    return _load_raw_events(data_version())


@lru_cache(maxsize=1)
def _load_raw_events(signature: Tuple[float, float, float]) -> pd.DataFrame:
    cached = _load_cached_dataframe("raw_events", signature)
    if cached is not None:
        return cached
    accidentes = _prepare_dataframe(pd.read_csv(ACCIDENT_PATH), label="accidente")
    congestiones = _prepare_dataframe(pd.read_csv(CONGESTION_PATH), label="congestion")
    reference = load_reference_network()
    event_frames = [df for df in (accidentes, congestiones) if not df.empty]
    events = (
        pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame(columns=accidentes.columns)
    )
    combined_frames = []
    if not events.empty:
        combined_frames.append(events)
    if not reference.empty:
        penalty_lookup = _build_penalty_lookup(events) if not events.empty else None
        reference = _apply_penalties(reference, penalty_lookup)
        combined_frames.append(reference)
    if not combined_frames:
        return pd.DataFrame(columns=accidentes.columns)
    eventos = pd.concat(combined_frames, ignore_index=True)
    eventos = eventos.sort_values(["segment_id", "segment_seq"]).reset_index(drop=True)
    _store_cached_dataframe("raw_events", signature, eventos)
    return eventos


def load_reference_network(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else ROAD_NETWORK_PATH
    signature = _file_signature(csv_path)
    return _load_reference_network(str(csv_path), signature)


@lru_cache(maxsize=4)
def _load_reference_network(path_str: str, signature: float) -> pd.DataFrame:
    csv_path = Path(path_str)
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    required = {
        "segment_id",
        "indice_coord",
        "lat",
        "lon",
        "comuna",
        "via",
        "distancia_km",
        "velocidad_kmh",
        "duracion_hrs",
        "fecha",
        "hora_inicio",
        "hora_fin",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El archivo {csv_path} no contiene las columnas requeridas: {missing}")
    df["oneway"] = df.get("oneway", False)
    return _prepare_dataframe(df, label="referencia")


def load_segment_summary() -> pd.DataFrame:
    return _load_segment_summary(data_version())


@lru_cache(maxsize=1)
def _load_segment_summary(signature: Tuple[float, float, float]) -> pd.DataFrame:
    cached = _load_cached_dataframe("segment_summary", signature)
    if cached is not None:
        return cached
    eventos = _load_raw_events(signature)
    eventos = eventos[eventos["tipo_evento"] != "Referencia"]
    segmentos = (
        eventos.groupby("segment_id")
        .agg(
            {
                "tipo_evento": "first",
                "comuna": "first",
                "via": "first",
                "franja_horaria": "first",
                "duracion_categoria": "first",
                "velocidad_categoria": "first",
                "dia_semana": "first",
                "duracion_hrs": "mean",
                "velocidad_kmh": "mean",
                "lat": "mean",
                "lon": "mean",
            }
        )
        .reset_index()
    )
    top_vias = segmentos["via"].value_counts().head(30).index
    segmentos["via_categoria"] = np.where(
        segmentos["via"].isin(top_vias), segmentos["via"], "Otras vías"
    )
    segmentos["tokens"] = segmentos.apply(lambda row: frozenset(_build_tokens(row)), axis=1)
    _store_cached_dataframe("segment_summary", signature, segmentos)
    return segmentos


def load_transactions() -> pd.DataFrame:
    return _load_transactions(data_version())


@lru_cache(maxsize=1)
def _load_transactions(signature: Tuple[float, float, float]) -> pd.DataFrame:
    cached = _load_cached_dataframe("transactions", signature)
    if cached is not None:
        return cached
    segmentos = _load_segment_summary(signature)
    transactions = segmentos["tokens"].apply(list).tolist()
    encoder = TransactionEncoder()
    matrix = encoder.fit(transactions).transform(transactions)
    df = pd.DataFrame(matrix, columns=encoder.columns_)
    _store_cached_dataframe("transactions", signature, df)
    return df


def set_user_ratings_path(path: Path) -> None:
    global _CURRENT_USER_RATINGS_PATH
    new_path = Path(path)
    if not new_path.exists():
        raise FileNotFoundError(f"No existe el archivo de ratings en {new_path}")
    if new_path == _CURRENT_USER_RATINGS_PATH:
        return
    _CURRENT_USER_RATINGS_PATH = new_path
    _load_user_ratings.cache_clear()


def get_user_ratings_path() -> Path:
    return _CURRENT_USER_RATINGS_PATH


def load_user_ratings(path: Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else _CURRENT_USER_RATINGS_PATH
    signature = _file_signature(csv_path)
    return _load_user_ratings(str(csv_path), signature)


@lru_cache(maxsize=4)
def _load_user_ratings(csv_path_str: str, signature: float) -> pd.DataFrame:
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de ratings en {csv_path}. "
            "Crea data/processed/user_ratings.csv con columnas: user_id,via,rating."
        )
    ratings = pd.read_csv(csv_path)
    expected_cols = {"user_id", "via", "rating"}
    if not expected_cols.issubset(ratings.columns):
        raise ValueError(f"El archivo de ratings debe contener las columnas {expected_cols}.")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "via", "rating"])
    ratings["via"] = ratings["via"].apply(_safe_title)
    return ratings.reset_index(drop=True)


def available_options() -> dict:
    segmentos = load_segment_summary()
    accident_ratio = (segmentos["tipo_evento"] == "Accidente").mean()
    raw_events = load_raw_events()
    lat_series = raw_events["lat"].dropna()
    lon_series = raw_events["lon"].dropna()
    bounds = {
        "lat_min": float(lat_series.min()) if not lat_series.empty else -90.0,
        "lat_max": float(lat_series.max()) if not lat_series.empty else 90.0,
        "lon_min": float(lon_series.min()) if not lon_series.empty else -180.0,
        "lon_max": float(lon_series.max()) if not lon_series.empty else 180.0,
    }
    options = {
        "event_types": sorted(segmentos["tipo_evento"].unique()),
        "communes": sorted(segmentos["comuna"].unique()),
        "franjas": sorted(segmentos["franja_horaria"].unique()),
        "durations": sorted(segmentos["duracion_categoria"].unique()),
        "velocities": sorted(segmentos["velocidad_categoria"].unique()),
        "vias": sorted(segmentos["via_categoria"].unique()),
        "total_events": int(len(segmentos)),
        "total_vias": int(segmentos["via_categoria"].nunique()),
        "accident_ratio": round(float(accident_ratio), 4),
        "bounds": bounds,
    }
    return options
