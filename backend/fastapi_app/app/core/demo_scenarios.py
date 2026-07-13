# -*- coding: utf-8 -*-
from __future__ import annotations

DEMO_SCENARIOS = [
    {
        "id": "centro_punta_am",
        "title": "Centro de Concepción en punta AM",
        "description": "Caso urbano denso para comparar ruta mas corta versus ruta con menor exposición histórica.",
        "origin": {"lat": -36.8267, "lon": -73.0498},
        "destination": {"lat": -36.8114, "lon": -73.0490},
        "day_of_week": "Wednesday",
        "departure_hour": 8.0,
        "profile": "safety_focused",
        "recommended_focus": "Mostrar cómo una ruta apenas más larga puede reducir retrasos y exposición.",
    },
    {
        "id": "cruce_san_pedro_pm",
        "title": "Cruce hacia San Pedro en punta PM",
        "description": "Escenario típico de hora punta para resaltar impacto del horario y del perfil equilibrado.",
        "origin": {"lat": -36.8321, "lon": -73.0510},
        "destination": {"lat": -36.8437, "lon": -73.1038},
        "day_of_week": "Friday",
        "departure_hour": 18.0,
        "profile": "usuario_demo",
        "recommended_focus": "Comparar el balance entre rapidez y exposición en el cruce del río.",
    },
    {
        "id": "acceso_talcahuano",
        "title": "Acceso a Talcahuano",
        "description": "Trayecto intercomunal para mostrar que la estrategia colaborativa cambia las vías sugeridas.",
        "origin": {"lat": -36.7975, "lon": -73.0675},
        "destination": {"lat": -36.7167, "lon": -73.1162},
        "day_of_week": "Thursday",
        "departure_hour": 17.0,
        "profile": "moderate_risk",
        "recommended_focus": "Explicar por qué UBCF e IBCF favorecen vías distintas.",
    },
    {
        "id": "trayecto_nocturno",
        "title": "Trayecto nocturno balanceado",
        "description": "Caso con menor presión horaria para evidenciar cuándo la ruta mas corta sigue siendo suficiente.",
        "origin": {"lat": -36.8200, "lon": -73.0440},
        "destination": {"lat": -36.8502, "lon": -73.1292},
        "day_of_week": "Saturday",
        "departure_hour": 22.0,
        "profile": "risk_taker",
        "recommended_focus": "Mostrar que no siempre conviene desviar si la exposición histórica baja.",
    },
]
