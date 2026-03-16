# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List

from .demo_content import COMPARISON_LABELS, VARIANT_META


def total_duration(variant: dict) -> float:
    return float((variant or {}).get("estimated_duration_min", 0.0) + (variant or {}).get("extra_delay_min", 0.0))


def comparison_highlights(route_result: dict) -> List[dict]:
    comparison = route_result.get("comparison") or {}
    items = []
    for key, label in COMPARISON_LABELS.items():
        variant = comparison.get(key)
        if not variant:
            continue
        items.append(
            {
                "title": label,
                "variant": variant,
                "label": VARIANT_META.get(variant, {}).get("label", variant),
            }
        )
    return items


def comparison_deltas(route_result: dict) -> Dict[str, dict]:
    deltas = {}
    for item in (route_result.get("comparison") or {}).get("deltas", []):
        variant = item.get("variant")
        if variant:
            deltas[variant] = item
    return deltas


def variant_summary_cards(route_result: dict) -> List[dict]:
    cards = []
    for key in ["reference", "ubcf", "ibcf"]:
        variant = route_result.get(key) or {}
        exposure = variant.get("incident_exposure") or {}
        cards.append(
            {
                "key": key,
                "label": VARIANT_META.get(key, {}).get("label", key),
                "story": VARIANT_META.get(key, {}).get("story", ""),
                "distance_km": variant.get("distance_km", 0.0),
                "base_minutes": variant.get("estimated_duration_min", 0.0),
                "extra_minutes": variant.get("extra_delay_min", 0.0),
                "total_minutes": total_duration(variant),
                "risk_score": variant.get("risk_score", 0.0),
                "matched_incidents": exposure.get("matched_incident_segments", 0),
                "why_changed": variant.get("why_changed") or [],
                "top_penalized_segments": variant.get("top_penalized_segments") or [],
                "top_preferred_vias": variant.get("top_preferred_vias") or [],
            }
        )
    return cards
