import pandas as pd

from backend.fastapi_app.app.services.recommendation_service import RecommendationService


def _segment_row(commune: str, event_type: str, tokens):
    return {
        "via_categoria": "Ruta Azul",
        "comuna": commune,
        "tipo_evento": event_type,
        "duracion_hrs": 0.5,
        "velocidad_kmh": 18.0,
        "lat": -36.82,
        "lon": -73.05,
        "tokens": frozenset(tokens),
    }


def test_summarize_segment_prioritizes_user_tokens():
    service = RecommendationService()
    segmentos = pd.DataFrame(
        [
            _segment_row("Concepción", "Accidente", {"comuna=Concepción", "evento=Accidente"}),
            _segment_row("Coronel", "Congestión", {"comuna=Coronel", "evento=Congestión"}),
        ]
    )

    summary = service._summarize_segment(segmentos, "Ruta Azul", {"comuna=Concepción"})

    assert summary["events"] == 1
    assert summary["communes"] == ["Concepción"]
    assert summary["accident_ratio"] == 1.0


def test_summarize_segment_falls_back_when_no_overlap():
    service = RecommendationService()
    segmentos = pd.DataFrame(
        [
            _segment_row("Concepción", "Accidente", {"comuna=Concepción", "evento=Accidente"}),
            _segment_row("Coronel", "Congestión", {"comuna=Coronel", "evento=Congestión"}),
        ]
    )

    summary = service._summarize_segment(segmentos, "Ruta Azul", {"comuna=Mulchén"})

    assert summary["events"] == 2
    assert sorted(summary["communes"]) == ["Concepción", "Coronel"]
