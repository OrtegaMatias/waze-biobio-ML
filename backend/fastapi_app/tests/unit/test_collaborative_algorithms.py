# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd

from algorithms.recommenders import collaborative


def _sample_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("alice", "Ruta Azul", 5),
            ("alice", "Ruta Roja", 2),
            ("bob", "Ruta Azul", 4),
            ("bob", "Ruta Verde", 5),
            ("carla", "Ruta Roja", 4),
            ("carla", "Ruta Verde", 1),
            ("david", "Ruta Roja", 5),
            ("david", "Ruta Verde", 4),
        ],
        columns=["user_id", "via", "rating"],
    )


def test_ubcf_generates_personalized_scores() -> None:
    model = collaborative.CollaborativeFilteringRecommender(strategy="ubcf")
    model.fit(_sample_ratings())

    recs = model.recommend(user_id="carla", known_vias=[], top_n=3)

    assert recs, "UBCF debe retornar recomendaciones personalizadas"
    assert recs[0].strategy == "ubcf"
    assert recs[0].via == "Ruta Azul"


def test_ibcf_generates_personalized_scores() -> None:
    model = collaborative.CollaborativeFilteringRecommender(strategy="ibcf")
    model.fit(_sample_ratings())

    recs = model.recommend(user_id="carla", known_vias=[], top_n=3)

    assert recs, "IBCF debe retornar recomendaciones personalizadas"
    assert recs[0].strategy == "ibcf"
    assert recs[0].via == "Ruta Azul"


def test_model_refits_when_ratings_signature_changes(monkeypatch) -> None:
    signature_state = {"value": ("sig1", 1)}

    def fake_signature(self):
        return signature_state["value"]

    fit_calls = {"count": 0}
    original_fit = collaborative.CollaborativeFilteringRecommender.fit

    def wrapped_fit(self, ratings=None):
        fit_calls["count"] += 1
        return original_fit(self, ratings)

    monkeypatch.setattr(
        collaborative.CollaborativeFilteringRecommender,
        "_ratings_file_signature",
        fake_signature,
    )
    monkeypatch.setattr(
        collaborative.CollaborativeFilteringRecommender,
        "fit",
        wrapped_fit,
    )
    monkeypatch.setattr(collaborative.data_loader, "load_user_ratings", lambda: _sample_ratings())

    model = collaborative.CollaborativeFilteringRecommender(strategy="ubcf")
    model.recommend(user_id="carla", known_vias=[], top_n=2)

    signature_state["value"] = ("sig2", 1)
    model.recommend(user_id="carla", known_vias=[], top_n=2)

    assert fit_calls["count"] == 2
