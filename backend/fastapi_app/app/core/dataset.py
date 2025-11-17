# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from algorithms.recommenders import data_loader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_PROFILES: Dict[str, Dict[str, Path | str]] = {
    "regional": {
        "label": "Cobertura regional",
        "ratings_path": PROCESSED_DIR / "user_ratings.csv",
    },
    "concepcion": {
        "label": "Solo Concepción",
        "ratings_path": PROCESSED_DIR / "user_ratings_concepcion.csv",
    },
}

_current_profile = "regional"


def _apply_profile(profile: str) -> None:
    info = DATA_PROFILES[profile]
    data_loader.set_user_ratings_path(Path(info["ratings_path"]))


_apply_profile(_current_profile)


def available_profiles() -> List[Tuple[str, str]]:
    return [(key, value["label"]) for key, value in DATA_PROFILES.items()]


def get_profile() -> str:
    return _current_profile


def get_profile_label(profile: str | None = None) -> str:
    key = profile or _current_profile
    return str(DATA_PROFILES.get(key, {}).get("label", key))


def set_profile(profile: str) -> None:
    global _current_profile
    if profile not in DATA_PROFILES:
        raise ValueError(f"Perfil de datos no soportado: {profile}")
    if profile == _current_profile:
        return
    _apply_profile(profile)
    _current_profile = profile
