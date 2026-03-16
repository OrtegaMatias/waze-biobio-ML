# -*- coding: utf-8 -*-
from __future__ import annotations

TRAVELER_PROFILES = {
    "safety_focused": {
        "label": "🛡️ Seguridad",
        "short_label": "Seguridad",
        "description": "Perfil sintético que evita vías con mayor exposición histórica.",
        "intent": "Ideal para mostrar reducción de riesgo aunque la ruta crezca un poco.",
    },
    "usuario_demo": {
        "label": "⚖️ Equilibrado",
        "short_label": "Equilibrado",
        "description": "Balancea rapidez y exposición usando reglas sintéticas intermedias.",
        "intent": "Es el mejor perfil para una demo general y comparación de tradeoffs.",
    },
    "moderate_risk": {
        "label": "🚗 Moderado",
        "short_label": "Moderado",
        "description": "Tolera algo de exposición a cambio de trayectos más directos.",
        "intent": "Útil para mostrar que no siempre conviene el camino más conservador.",
    },
    "risk_taker": {
        "label": "⚡ Rápido",
        "short_label": "Rápido",
        "description": "Prioriza velocidad y acepta más riesgo histórico en la simulación.",
        "intent": "Sirve para contrastar claramente con el perfil de seguridad.",
    },
}

PROFILE_ORDER = ["safety_focused", "usuario_demo", "moderate_risk", "risk_taker"]

VARIANT_META = {
    "reference": {"label": "🔵 Ruta base", "story": "Camino base sin sesgo colaborativo."},
    "ubcf": {"label": "🟢 Perfil por usuarios", "story": "Usa similitud entre perfiles sintéticos."},
    "ibcf": {"label": "🟠 Perfil por vías", "story": "Usa similitud entre vías y patrones históricos."},
    "personalized": {"label": "⚪ Compat legacy", "story": "Compatibilidad hacia atrás."},
}

COMPARISON_LABELS = {
    "fastest_variant": "Más rápida",
    "safest_variant": "Más segura",
    "lowest_exposure_variant": "Menor exposición",
    "best_balance_variant": "Mejor balance",
}
