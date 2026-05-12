from __future__ import annotations

import unicodedata
from typing import Final

import pandas as pd


PROFILE_ALIASES: Final[dict[str, str]] = {
    "regional": "gran_concepcion",
    "concepcion": "gran_concepcion",
}

PROFILE_COMMUNES: Final[dict[str, frozenset[str]]] = {
    "gran_concepcion": frozenset(
        {
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
        }
    ),
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


def _matches_profile_commune(value: str | None, profile: str) -> bool:
    if value is None:
        return False
    communes = PROFILE_COMMUNES[profile]
    if value in communes:
        return True
    first_component = value.split(",", 1)[0].strip()
    if first_component in communes:
        return True
    return any(value.startswith(f"{commune},") for commune in communes)


def filter_dataframe_for_profile(
    df: pd.DataFrame,
    profile: str,
    commune_column: str = "comuna",
) -> pd.DataFrame:
    if df.empty or commune_column not in df.columns:
        return df
    canonical_profile = canonicalize_data_profile(profile)
    normalized_communes = df[commune_column].map(normalize_geo_text)
    return df[normalized_communes.map(lambda value: _matches_profile_commune(value, canonical_profile))].reset_index(drop=True)
