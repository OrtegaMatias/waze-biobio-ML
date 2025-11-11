# -*- coding: utf-8 -*-
"""
Recomendador basado en reglas de asociación.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set
import warnings

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from . import data_loader


@dataclass
class AssociationRule:
    antecedents: Set[str]
    consequents: Set[str]
    support: float
    confidence: float
    lift: float

    def as_dict(self) -> dict:
        return {
            "antecedents": sorted(self.antecedents),
            "consequents": sorted(self.consequents),
            "support": self.support,
            "confidence": self.confidence,
            "lift": self.lift,
        }


class AssociationRuleRecommender:
    """Envuelve la lógica de Apriori + filtros para tokens de usuario."""

    def __init__(self, min_support: float = 0.02, min_confidence: float = 0.4) -> None:
        self.min_support = min_support
        self.min_confidence = min_confidence
        self._rules: pd.DataFrame | None = None

    def fit(self, transactions: pd.DataFrame | None = None) -> None:
        df = transactions if transactions is not None else data_loader.load_transactions()
        frequent_itemsets = apriori(df, min_support=self.min_support, use_colnames=True)
        if frequent_itemsets.empty:
            self._rules = pd.DataFrame()
            return
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module="mlxtend.frequent_patterns.association_rules",
            )
            rules = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=self.min_confidence
            )
        if not rules.empty:
            rules.replace([np.inf, -np.inf], np.nan, inplace=True)
            rules = rules.dropna(subset=["support", "confidence", "lift"], how="any")
        if rules.empty:
            self._rules = pd.DataFrame()
            return
        rules["antecedents"] = rules["antecedents"].apply(frozenset)
        rules["consequents"] = rules["consequents"].apply(frozenset)
        self._rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(
            drop=True
        )

    @property
    def rules(self) -> pd.DataFrame:
        if self._rules is None:
            self.fit()
        return self._rules.copy()

    def recommend(self, user_tokens: Set[str], top_n: int = 5) -> List[AssociationRule]:
        rules = self.rules
        if rules.empty or not user_tokens:
            return []
        mask = rules["antecedents"].apply(lambda antecedent: antecedent.issubset(user_tokens))
        filtered = rules[mask]
        recs: List[AssociationRule] = []
        for _, row in filtered.head(200).iterrows():
            recs.append(
                AssociationRule(
                    antecedents=set(row["antecedents"]),
                    consequents=set(row["consequents"]),
                    support=float(row["support"]),
                    confidence=float(row["confidence"]),
                    lift=float(row["lift"]),
                )
            )
            if len(recs) >= top_n:
                break
        return recs
