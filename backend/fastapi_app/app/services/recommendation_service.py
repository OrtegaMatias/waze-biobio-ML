# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List

from ..schemas.recommendations import (
    CollaborativeRecommendation,
    CollaborativeRequest,
    PlaygroundRequest,
)
from algorithms.recommenders import collaborative, data_loader

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self) -> None:
        logger.info("Inicializando RecommendationService")
        self._cf_models = {
            "ubcf": collaborative.CollaborativeFilteringRecommender(strategy="ubcf"),
            "ibcf": collaborative.CollaborativeFilteringRecommender(strategy="ibcf"),
        }

    def available_options(self) -> dict:
        logger.debug("Leyendo opciones disponibles para filtros")
        return data_loader.available_options()

    def collaborative_recommendations(
        self, payload: CollaborativeRequest
    ) -> List[CollaborativeRecommendation]:
        logger.info("Colaborativo strategy=%s user=%s", payload.strategy, payload.user_id)
        cf_model = self._cf_models[payload.strategy]
        recs = cf_model.recommend(
            user_id=payload.user_id,
            known_vias=payload.known_vias,
            top_n=payload.limit,
        )
        logger.info("Colaborativo retornó %d opciones", len(recs))
        return [
            CollaborativeRecommendation(
                via=item.via,
                estimated_rating=item.estimated_rating,
                strategy=item.strategy,
            )
            for item in recs
        ]

    def playground_recommendations(
        self, payload: PlaygroundRequest
    ) -> Dict[str, List[CollaborativeRecommendation]]:
        strategies = payload.strategies or ["ubcf", "ibcf"]
        unique_strategies = []
        for strategy in strategies:
            if strategy in self._cf_models and strategy not in unique_strategies:
                unique_strategies.append(strategy)
        if not unique_strategies:
            unique_strategies = ["ubcf", "ibcf"]
        results: Dict[str, List[CollaborativeRecommendation]] = {}
        for strategy in unique_strategies:
            logger.info("Playground strategy=%s user=%s", strategy, payload.user_id)
            cf_model = self._cf_models[strategy]
            recs = cf_model.recommend(
                user_id=payload.user_id,
                known_vias=payload.known_vias,
                top_n=payload.limit,
            )
            results[strategy] = [
                CollaborativeRecommendation(
                    via=item.via,
                    estimated_rating=item.estimated_rating,
                    strategy=item.strategy,
                )
                for item in recs
            ]
        return results


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    return RecommendationService()
