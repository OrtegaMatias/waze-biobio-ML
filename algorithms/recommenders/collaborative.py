# -*- coding: utf-8 -*-
"""
Filtrado colaborativo basado en usuarios (UBCF) o ítems (IBCF) usando cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from . import data_loader


@dataclass
class CFRecommendation:
    via: str
    estimated_rating: float
    strategy: str


class CollaborativeFilteringRecommender:
    def __init__(self, strategy: Literal["ubcf", "ibcf"] = "ubcf") -> None:
        self.strategy = strategy
        self._ratings: pd.DataFrame | None = None
        self._popularity: pd.Series | None = None
        self._user_item: pd.DataFrame | None = None
        self._item_user: pd.DataFrame | None = None
        self._user_sim: pd.DataFrame | None = None
        self._item_sim: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame | None = None) -> None:
        ratings_df = ratings if ratings is not None else data_loader.load_user_ratings()
        if ratings_df.empty:
            raise ValueError("No existen calificaciones de usuarios para entrenar el modelo.")
        self._ratings = ratings_df.copy()
        self._popularity = (
            ratings_df.groupby("via")["rating"].mean().sort_values(ascending=False)
        )
        user_item = ratings_df.pivot_table(index="user_id", columns="via", values="rating")
        self._user_item = user_item
        self._item_user = user_item.transpose()
        self._user_sim = self._compute_similarity(user_item)
        self._item_sim = self._compute_similarity(self._item_user)

    def _ensure_fitted(self) -> None:
        if (
            self._ratings is None
            or self._user_item is None
            or self._item_user is None
            or self._user_sim is None
            or self._item_sim is None
        ):
            self.fit()

    def recommend(
        self, user_id: str, known_vias: Sequence[str] | None = None, top_n: int = 5
    ) -> List[CFRecommendation]:
        self._ensure_fitted()
        assert self._ratings is not None
        known_vias = set(known_vias or [])
        user_history = set(self._ratings[self._ratings["user_id"] == user_id]["via"])
        if not user_history:
            return self._fallback_recommendations(exclude=known_vias, top_n=top_n)

        if self.strategy == "ubcf":
            scores = self._score_user_based(user_id)
        else:
            scores = self._score_item_based(user_id)

        recs: List[CFRecommendation] = []
        for via, score in scores.items():
            if via in user_history or via in known_vias or np.isnan(score):
                continue
            recs.append(CFRecommendation(via=via, estimated_rating=float(score), strategy=self.strategy))
        recs.sort(key=lambda item: item.estimated_rating, reverse=True)
        if not recs:
            return self._fallback_recommendations(exclude=known_vias, top_n=top_n)
        return recs[:top_n]

    def _score_user_based(self, user_id: str) -> pd.Series:
        assert self._user_item is not None and self._user_sim is not None
        if user_id not in self._user_item.index:
            return pd.Series(dtype=float)
        similarities = self._user_sim.loc[user_id]
        ratings = self._user_item.fillna(0)
        weighted_sum = similarities.values @ ratings.values
        sim_sums = similarities.values @ (ratings.values > 0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.divide(weighted_sum, sim_sums, out=np.zeros_like(weighted_sum), where=sim_sums != 0)
        return pd.Series(scores, index=ratings.columns)

    def _score_item_based(self, user_id: str) -> pd.Series:
        assert self._user_item is not None and self._item_sim is not None
        if user_id not in self._user_item.index:
            return pd.Series(dtype=float)
        user_vector = self._user_item.loc[user_id].fillna(0)
        similarities = self._item_sim[user_vector.index]
        weighted_sum = similarities.values @ user_vector.values
        sim_sums = (similarities.values > 0).astype(float) @ (user_vector.values > 0).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.divide(weighted_sum, sim_sums, out=np.zeros_like(weighted_sum), where=sim_sums != 0)
        return pd.Series(scores, index=similarities.index)

    def _fallback_recommendations(
        self, exclude: Sequence[str], top_n: int
    ) -> List[CFRecommendation]:
        if self._popularity is None:
            return []
        recs: List[CFRecommendation] = []
        for via, rating in self._popularity.items():
            if via in exclude:
                continue
            recs.append(
                CFRecommendation(via=via, estimated_rating=float(rating), strategy="popular")
            )
            if len(recs) >= top_n:
                break
        return recs

    @staticmethod
    def _compute_similarity(matrix: pd.DataFrame) -> pd.DataFrame:
        filled = matrix.fillna(0)
        if filled.shape[0] == 0:
            return pd.DataFrame()
        sim = cosine_similarity(filled)
        sim_df = pd.DataFrame(sim, index=filled.index, columns=filled.index)
        np.fill_diagonal(sim_df.values, 0.0)
        return sim_df
