from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

from algorithms.recommenders.routing import GraphNode, haversine_km


DEFAULT_URBAN_SPEED_KMH = 40.0
MAX_FALLBACK_URBAN_SPEED_KMH = 50.0
MIN_DRIVING_SPEED_KMH = 5.0
MAX_VALID_SPECIFIC_SPEED_KMH = 130.0
FLUENCY_CONGESTION_WEIGHT = 2.0
FLUENCY_STOP_MIN_PER_KM = 1.5
ENVIRONMENT_CONGESTION_WEIGHT = 1.25
ENVIRONMENT_PM25_WEIGHT = 0.70
ENVIRONMENT_ADVERSE_WEIGHT = 0.75
ENVIRONMENT_MIN_EDGE_FACTOR = 0.35


ROAD_CLASS_SPEED_KMH: dict[str, float] = {
    "motorway": 50.0,
    "motorway_link": 50.0,
    "trunk": 50.0,
    "trunk_link": 50.0,
    "primary": 50.0,
    "primary_link": 45.0,
    "secondary": 45.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 35.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "unclassified": 30.0,
}


@dataclass(frozen=True)
class EdgeCostBreakdown:
    base_time_min: float
    congestion_delay_min: float = 0.0
    congestion_penalty_min: float = 0.0
    stop_penalty_min: float = 0.0
    pm25_penalty_min: float = 0.0
    adverse_environment_penalty_min: float = 0.0
    urban_benefit_min: float = 0.0

    @property
    def travel_time_min(self) -> float:
        return self.base_time_min + self.congestion_delay_min

    @property
    def optimization_cost_min(self) -> float:
        return max(
            0.0,
            self.travel_time_min
            + self.congestion_penalty_min
            + self.stop_penalty_min
            + self.pm25_penalty_min
            + self.adverse_environment_penalty_min
            - self.urban_benefit_min,
        )


@dataclass(frozen=True)
class RouteCostTotals:
    base_time_min: float = 0.0
    congestion_delay_min: float = 0.0
    congestion_penalty_min: float = 0.0
    stop_penalty_min: float = 0.0
    pm25_penalty_min: float = 0.0
    adverse_environment_penalty_min: float = 0.0
    urban_benefit_min: float = 0.0
    optimization_cost_min: float = 0.0

    @property
    def travel_time_min(self) -> float:
        return self.base_time_min + self.congestion_delay_min


def _finite(value: float | int | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def free_flow_speed_kmh(node: GraphNode) -> float:
    """Resolve a vehicle speed without treating congestion observations as defaults."""

    specific = _finite(node.velocidad_kmh)
    if specific is not None and MIN_DRIVING_SPEED_KMH <= specific <= MAX_VALID_SPECIFIC_SPEED_KMH:
        return specific
    road_class = str(getattr(node, "road_class", "") or "").strip().lower()
    fallback = ROAD_CLASS_SPEED_KMH.get(road_class, DEFAULT_URBAN_SPEED_KMH)
    return min(MAX_FALLBACK_URBAN_SPEED_KMH, max(MIN_DRIVING_SPEED_KMH, fallback))


class RoutingCostModel:
    """Build non-negative, minute-equivalent edge costs for the three route goals."""

    def __init__(
        self,
        *,
        congestion_scores: Mapping[str, float] | None = None,
        congestion_speeds_kmh: Mapping[str, float] | None = None,
        segment_lengths_km: Mapping[str, float] | None = None,
        pm25_factor: Callable[[GraphNode], float] | None = None,
        wellbeing_factor: Callable[[GraphNode], float] | None = None,
        adverse_environment_factor: Callable[[GraphNode], float] | None = None,
    ) -> None:
        self.congestion_scores = congestion_scores or {}
        self.congestion_speeds_kmh = congestion_speeds_kmh or {}
        self.segment_lengths_km = segment_lengths_km or {}
        self.pm25_factor = pm25_factor
        self.wellbeing_factor = wellbeing_factor
        self.adverse_environment_factor = adverse_environment_factor

    @staticmethod
    def edge_distance_km(source: GraphNode, target: GraphNode, supplied_distance_km: float | None = None) -> float:
        supplied = _finite(supplied_distance_km)
        if supplied is not None and supplied > 0:
            return supplied
        return max(0.0, haversine_km(source.lat, source.lon, target.lat, target.lon))

    def _congestion_score(self, source: GraphNode, target: GraphNode) -> float:
        values = [
            _finite(self.congestion_scores.get(source.node_id)),
            _finite(self.congestion_scores.get(target.node_id)),
            _finite(self.congestion_scores.get(source.segment_id)),
            _finite(self.congestion_scores.get(target.segment_id)),
        ]
        return max(0.0, min(1.0, max((value for value in values if value is not None), default=0.0)))

    def _observed_speed(self, source: GraphNode, target: GraphNode) -> float | None:
        values = [
            _finite(self.congestion_speeds_kmh.get(source.node_id)),
            _finite(self.congestion_speeds_kmh.get(target.node_id)),
            _finite(self.congestion_speeds_kmh.get(source.segment_id)),
            _finite(self.congestion_speeds_kmh.get(target.segment_id)),
        ]
        valid = [value for value in values if value is not None and value >= MIN_DRIVING_SPEED_KMH]
        return min(valid) if valid else None

    def _travel_components(
        self,
        source: GraphNode,
        target: GraphNode,
        distance_km: float,
    ) -> tuple[float, float, float]:
        free_speed = max(
            MIN_DRIVING_SPEED_KMH,
            (free_flow_speed_kmh(source) + free_flow_speed_kmh(target)) / 2.0,
        )
        base_time = distance_km / free_speed * 60.0
        congestion_score = self._congestion_score(source, target)
        observed_speed = self._observed_speed(source, target)
        if congestion_score <= 0:
            effective_speed = free_speed
        elif observed_speed is not None:
            effective_speed = min(free_speed, observed_speed)
        else:
            effective_speed = max(MIN_DRIVING_SPEED_KMH, free_speed * (1.0 - 0.75 * congestion_score))
        contextual_time = distance_km / max(MIN_DRIVING_SPEED_KMH, effective_speed) * 60.0
        return base_time, max(0.0, contextual_time - base_time), congestion_score

    def fastest(self, source: GraphNode, target: GraphNode, distance_km: float) -> EdgeCostBreakdown:
        distance = self.edge_distance_km(source, target, distance_km)
        base_time, delay, _score = self._travel_components(source, target, distance)
        return EdgeCostBreakdown(base_time_min=base_time, congestion_delay_min=delay)

    def fluent(self, source: GraphNode, target: GraphNode, distance_km: float) -> EdgeCostBreakdown:
        distance = self.edge_distance_km(source, target, distance_km)
        base_time, delay, score = self._travel_components(source, target, distance)
        contextual_time = base_time + delay
        segment_length = max(
            distance,
            _finite(self.segment_lengths_km.get(target.segment_id)) or distance,
        )
        proportional_stop = FLUENCY_STOP_MIN_PER_KM * score * distance / max(segment_length, 1e-9)
        return EdgeCostBreakdown(
            base_time_min=base_time,
            congestion_delay_min=delay,
            congestion_penalty_min=contextual_time * score * FLUENCY_CONGESTION_WEIGHT,
            stop_penalty_min=proportional_stop,
        )

    def environmental(self, source: GraphNode, target: GraphNode, distance_km: float) -> EdgeCostBreakdown:
        distance = self.edge_distance_km(source, target, distance_km)
        base_time, delay, score = self._travel_components(source, target, distance)
        contextual_time = base_time + delay
        pm25_factor = max(1.0, _finite(self.pm25_factor(target)) or 1.0) if self.pm25_factor else 1.0
        adverse_factor = (
            max(1.0, _finite(self.adverse_environment_factor(target)) or 1.0)
            if self.adverse_environment_factor
            else 1.0
        )
        wellbeing_factor = (
            max(ENVIRONMENT_MIN_EDGE_FACTOR, min(1.0, _finite(self.wellbeing_factor(target)) or 1.0))
            if self.wellbeing_factor
            else 1.0
        )
        congestion_penalty = contextual_time * score * ENVIRONMENT_CONGESTION_WEIGHT
        pm25_penalty = contextual_time * (pm25_factor - 1.0) * ENVIRONMENT_PM25_WEIGHT
        adverse_penalty = contextual_time * (adverse_factor - 1.0) * ENVIRONMENT_ADVERSE_WEIGHT
        gross_cost = contextual_time + congestion_penalty + pm25_penalty + adverse_penalty
        requested_benefit = contextual_time * (1.0 - wellbeing_factor)
        max_benefit = gross_cost * (1.0 - ENVIRONMENT_MIN_EDGE_FACTOR)
        benefit = min(max_benefit, max(0.0, requested_benefit))
        return EdgeCostBreakdown(
            base_time_min=base_time,
            congestion_delay_min=delay,
            congestion_penalty_min=congestion_penalty,
            pm25_penalty_min=pm25_penalty,
            adverse_environment_penalty_min=adverse_penalty,
            urban_benefit_min=benefit,
        )


def sum_breakdowns(items: list[EdgeCostBreakdown]) -> RouteCostTotals:
    return RouteCostTotals(
        base_time_min=sum(item.base_time_min for item in items),
        congestion_delay_min=sum(item.congestion_delay_min for item in items),
        congestion_penalty_min=sum(item.congestion_penalty_min for item in items),
        stop_penalty_min=sum(item.stop_penalty_min for item in items),
        pm25_penalty_min=sum(item.pm25_penalty_min for item in items),
        adverse_environment_penalty_min=sum(item.adverse_environment_penalty_min for item in items),
        urban_benefit_min=sum(item.urban_benefit_min for item in items),
        optimization_cost_min=sum(item.optimization_cost_min for item in items),
    )
