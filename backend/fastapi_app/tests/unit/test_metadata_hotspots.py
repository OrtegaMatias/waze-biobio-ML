# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from backend.fastapi_app.app import main


def test_build_hotspot_points_handles_missing_times(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "tipo_evento": "Congestión",
                "lat": -36.8,
                "lon": -73.0,
                "hora_inicio": "",
                "hora_fin": None,
                "dia_semana": "Monday",
                "franja_horaria": "Punta AM (06-09h)",
                "segment_id": "seg-1",
                "velocidad_kmh": 15,
            }
        ]
    )

    monkeypatch.setattr(main.data_loader, "load_raw_events", lambda: df)

    points = main._build_hotspot_points()

    assert len(points) == 1
    assert points[0]["hora_inicio_float"] is None
    assert points[0]["hora_fin_float"] is None
