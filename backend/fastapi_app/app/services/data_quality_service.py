# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd

from algorithms.recommenders import data_loader
from algorithms.recommenders.geo_profiles import canonicalize_data_profile, filter_dataframe_for_profile

_COMMUNE_PATTERN = r"^[A-Z]-\d|^\d+$|,|Region del Biobio"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _load_profiled_raw(path, profile: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return filter_dataframe_for_profile(df, profile)


def _load_profiled_network_communes(_profile: str) -> pd.Series:
    network = data_loader.load_reference_network()
    if network.empty:
        return pd.Series(dtype=str)
    return network.get("comuna", pd.Series(dtype=str))


def inspect_data_quality() -> Dict[str, Any]:
    return _inspect_data_quality(data_loader.data_version())


@lru_cache(maxsize=2)
def _inspect_data_quality(signature) -> Dict[str, Any]:
    profile = canonicalize_data_profile(data_loader.get_data_profile())
    congestions = _load_profiled_raw(data_loader.CONGESTION_PATH, profile)

    warnings: List[str] = []
    notes: List[str] = []

    duplicate_sources = False
    combined = congestions.copy()
    total_rows = len(combined)
    via_series = combined.get("via", pd.Series(dtype=str))
    via_text = via_series.astype(str)
    missing_via = (
        via_series.isna().sum()
        + via_text.str.strip().eq("").sum()
        + via_text.str.lower().eq("nan").sum()
    )
    missing_via_ratio = _safe_ratio(int(missing_via), total_rows)
    if missing_via_ratio > 0.01:
        warnings.append(f"Hay un {missing_via_ratio * 100:.1f}% de registros sin via utilizable.")

    date_series = pd.to_datetime(combined.get("fecha"), errors="coerce")
    valid_dates = date_series.dropna()
    date_start = str(valid_dates.min().date()) if not valid_dates.empty else None
    date_end = str(valid_dates.max().date()) if not valid_dates.empty else None
    unique_days = int(valid_dates.dt.normalize().nunique()) if not valid_dates.empty else 0
    if unique_days and unique_days < 45:
        warnings.append(
            f"La cobertura temporal del perfil activo es corta: {unique_days} dias entre {date_start} y {date_end}."
        )

    if not data_loader.ROAD_NETWORK_PATH.exists():
        warnings.append(
            "No se encontro road_network.csv en data/processed; la demo sigue operativa, pero omite validaciones de comunas de la red vial."
        )

    network_communes = _load_profiled_network_communes(profile)
    network_communes = network_communes.dropna().astype(str).str.strip()
    anomalous_communes = sorted(
        {
            commune
            for commune in network_communes.unique()
            if commune and pd.Series([commune]).str.contains(_COMMUNE_PATTERN, regex=True, case=False).iloc[0]
        }
    )
    if anomalous_communes:
        warnings.append("La red vial del perfil activo conserva algunas etiquetas de comuna anomalas.")
        notes.append(f"Ejemplos: {', '.join(anomalous_communes[:5])}")

    status = "ok"
    if warnings:
        status = "warning"
    if total_rows == 0:
        status = "error"
        warnings.append("No se encontraron incidentes en el perfil activo.")

    return {
        "status": status,
        "dataset_profile": profile,
        "duplicate_incident_sources": duplicate_sources,
        "date_range": {"start": date_start, "end": date_end, "days": unique_days},
        "missing_via_ratio": missing_via_ratio,
        "anomalous_communes": anomalous_communes[:10],
        "raw_counts": {
            "accidents": 0,
            "congestions": int(len(congestions)),
            "combined": int(total_rows),
        },
        "warnings": warnings,
        "notes": notes,
    }
