# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd

from algorithms.recommenders import data_loader

_COMMUNE_PATTERN = r"^[A-Z]-\d|^\d+$|,|Región del Biobío"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _load_profiled_raw(path, profile: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if profile == "concepcion" and {"lat", "lon"}.issubset(df.columns):
        lat_min, lat_max, lon_min, lon_max = data_loader.CONCEPCION_BBOX
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df[
            df["lat"].between(lat_min, lat_max)
            & df["lon"].between(lon_min, lon_max)
        ].reset_index(drop=True)
    return df


def _load_profiled_network_communes(profile: str) -> pd.Series:
    network = pd.read_csv(data_loader.ROAD_NETWORK_PATH)
    if profile == "concepcion" and {"lat", "lon"}.issubset(network.columns):
        lat_min, lat_max, lon_min, lon_max = data_loader.CONCEPCION_BBOX
        network["lat"] = pd.to_numeric(network["lat"], errors="coerce")
        network["lon"] = pd.to_numeric(network["lon"], errors="coerce")
        network = network[
            network["lat"].between(lat_min, lat_max)
            & network["lon"].between(lon_min, lon_max)
        ].reset_index(drop=True)
    return network.get("comuna", pd.Series(dtype=str))


def inspect_data_quality() -> Dict[str, Any]:
    return _inspect_data_quality(data_loader.data_version())


@lru_cache(maxsize=2)
def _inspect_data_quality(signature) -> Dict[str, Any]:
    profile = data_loader.get_data_profile()
    accidents = _load_profiled_raw(data_loader.ACCIDENT_PATH, profile)
    congestions = _load_profiled_raw(data_loader.CONGESTION_PATH, profile)

    warnings: List[str] = []
    notes: List[str] = []

    duplicate_sources = (
        accidents.shape == congestions.shape
        and list(accidents.columns) == list(congestions.columns)
        and accidents.equals(congestions)
    )
    if duplicate_sources:
        warnings.append(
            "ACCIDENTES.csv y CONGESTIONES.csv son idénticos en el perfil activo; la app los presenta como incidentes históricos agregados."
        )

    combined = pd.concat([accidents, congestions], ignore_index=True) if not accidents.empty or not congestions.empty else pd.DataFrame()
    total_rows = len(combined)
    via_series = combined.get("via", pd.Series(dtype=str))
    via_text = via_series.astype(str)
    missing_via = via_series.isna().sum() + via_text.str.strip().eq("").sum() + via_text.str.lower().eq("nan").sum()
    missing_via_ratio = _safe_ratio(int(missing_via), total_rows)
    if missing_via_ratio > 0.01:
        warnings.append(f"Hay un {missing_via_ratio * 100:.1f}% de registros sin vía utilizable.")

    date_series = pd.to_datetime(combined.get("fecha"), errors="coerce")
    valid_dates = date_series.dropna()
    date_start = str(valid_dates.min().date()) if not valid_dates.empty else None
    date_end = str(valid_dates.max().date()) if not valid_dates.empty else None
    unique_days = int(valid_dates.dt.normalize().nunique()) if not valid_dates.empty else 0
    if unique_days and unique_days < 45:
        warnings.append(
            f"La cobertura temporal del perfil activo es corta: {unique_days} días entre {date_start} y {date_end}."
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
        if profile == "regional":
            warnings.append("La red vial contiene etiquetas de comuna anómalas en cobertura regional.")
        else:
            warnings.append("La red vial del perfil activo conserva algunas etiquetas de comuna anómalas.")
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
            "accidents": int(len(accidents)),
            "congestions": int(len(congestions)),
            "combined": int(total_rows),
        },
        "warnings": warnings,
        "notes": notes,
    }
