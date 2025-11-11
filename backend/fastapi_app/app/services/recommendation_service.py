# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Set

import numpy as np

from ..schemas.recommendations import (
    AssociationRecommendation,
    AssociationRequest,
    CollaborativeRecommendation,
    CollaborativeRequest,
)
from algorithms.recommenders import association, collaborative, data_loader

logger = logging.getLogger(__name__)


class RecommendationService:
    def __init__(self) -> None:
        logger.info("Inicializando RecommendationService")
        self._association_model = association.AssociationRuleRecommender()
        self._cf_models = {
            "ubcf": collaborative.CollaborativeFilteringRecommender(strategy="ubcf"),
            "ibcf": collaborative.CollaborativeFilteringRecommender(strategy="ibcf"),
        }

    def available_options(self) -> dict:
        logger.debug("Leyendo opciones disponibles para filtros")
        return data_loader.available_options()

    def association_recommendations(self, payload: AssociationRequest) -> List[AssociationRecommendation]:
        logger.info(
            "Generando reglas de asociación (events=%s, communes=%d)",
            ",".join(payload.event_types),
            len(payload.communes),
        )
        self._association_model.min_support = payload.min_support
        self._association_model.min_confidence = payload.min_confidence
        self._association_model.fit()
        tokens = self._build_tokens(payload)
        rules = self._association_model.recommend(tokens, top_n=payload.limit)
        segmentos = data_loader.load_segment_summary()
        recommendations: List[AssociationRecommendation] = []
        for rule in rules:
            via_tokens = [token for token in rule.consequents if token.startswith("via=")]
            for via_token in via_tokens:
                via_value = via_token.split("=", 1)[1]
                summary = self._summarize_segment(segmentos, via_value, tokens)
                if summary is None:
                    continue
                recommendations.append(
                    AssociationRecommendation(
                        via=via_value,
                        communes=summary["communes"],
                        events=summary["events"],
                        accident_ratio=summary["accident_ratio"],
                        duration_avg=summary["duration_avg"],
                        speed_avg=summary["speed_avg"],
                        lat=summary["lat"],
                        lon=summary["lon"],
                        confidence=rule.confidence,
                        lift=rule.lift,
                        support=rule.support,
                        rule=self._stringify_rule(rule),
                    )
                )
                if len(recommendations) >= payload.limit:
                    break
            if len(recommendations) >= payload.limit:
                break
        return recommendations

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

    def _build_tokens(self, payload: AssociationRequest) -> Set[str]:
        tokens: Set[str] = set()
        for evt in payload.event_types:
            tokens.add(f"evento={evt}")
        for comuna in payload.communes:
            tokens.add(f"comuna={comuna}")
        for franja in payload.franjas:
            tokens.add(f"franja={franja}")
        for duration in payload.durations:
            tokens.add(f"duracion={duration}")
        for vel in payload.velocities:
            tokens.add(f"velocidad={vel}")
        if payload.day_type:
            tokens.add(f"dia={payload.day_type}")
        if payload.via_preferred:
            tokens.add(f"via={payload.via_preferred}")
        return tokens

    def _summarize_segment(self, segmentos, via_value: str, tokens: Set[str]):
        via_mask = segmentos["via_categoria"] == via_value
        subset = segmentos[via_mask]
        if subset.empty:
            return None
        if tokens and "tokens" in subset.columns:
            filtered = subset[subset["tokens"].apply(lambda seg_tokens: bool(seg_tokens & tokens))]
            if not filtered.empty:
                subset = filtered
        communes = subset["comuna"].value_counts().head(3).index.tolist()
        accidents_ratio = (subset["tipo_evento"] == "Accidente").mean()
        lat_mean = subset["lat"].mean()
        lon_mean = subset["lon"].mean()
        return {
            "communes": communes,
            "events": int(len(subset)),
            "accident_ratio": round(accidents_ratio, 3),
            "duration_avg": round(subset["duracion_hrs"].mean(), 2),
            "speed_avg": round(subset["velocidad_kmh"].mean(), 2),
            "lat": float(lat_mean) if not np.isnan(lat_mean) else None,
            "lon": float(lon_mean) if not np.isnan(lon_mean) else None,
        }

    def _stringify_rule(self, rule: association.AssociationRule) -> str:
        def format_token(token: str) -> str:
            label, value = token.split("=", 1)
            return f"{label.capitalize()}={value}"

        antecedents = ", ".join(sorted(format_token(token) for token in rule.antecedents))
        consequents = ", ".join(sorted(format_token(token) for token in rule.consequents))
        return f"{antecedents} ⇒ {consequents}"


@lru_cache(maxsize=1)
def get_recommendation_service() -> RecommendationService:
    return RecommendationService()
