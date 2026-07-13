import { describe, expect, it } from "vitest";

import { buildRouteInsightMessages } from "./PlanPage";
import type { EnvironmentalImpactResponse, PlanRouteCard } from "../types";

function route(overrides: Partial<PlanRouteCard> = {}): PlanRouteCard {
  return {
    key: "fastest",
    label: "Llegar antes",
    badges: [{ key: "fastest", label: "Llegar antes" }],
    duration_min: 10,
    distance_km: 4,
    delay_min: 0,
    congestion_score: 10,
    risk_level: "low",
    summary: "Ruta de prueba",
    geometry: [],
    top_alerts: [],
    why_changed: [],
    top_penalized_segments: [],
    top_preferred_vias: [],
    incident_exposure: {
      total_incident_segments: 0,
      matched_incident_segments: 0,
      congestion_segments: 0,
      accident_segments: 0,
      exposure_minutes: 0,
    },
    ...overrides,
  };
}

function pm25(category: "Baja" | "Media" | "Alta", average_pm25: number) {
  return {
    available: true,
    average_pm25,
    category,
    stations: [],
    method: "test",
    data_source: "test",
  };
}

function urbanWellbeing(overrides: Partial<NonNullable<PlanRouteCard["urban_wellbeing"]>> = {}) {
  return {
    available: true,
    score: 35,
    green_ratio: 0.3,
    blue_ratio: 0,
    tree_ratio: 0,
    public_space_ratio: 0,
    sustainability_ratio: 0,
    nearby_feature_count: 1,
    nearby_buffer_m: 100,
    top_features: [
      {
        feature_id: "laguna-los-patos",
        name: "Laguna Los Patos",
        category: "blue_space" as const,
        subtype: "lagoon",
        distance_m: 40,
        source: "test",
        base_weight: 1,
      },
    ],
    method: "test",
    data_source: "test",
    ...overrides,
  };
}

function weather(
  overrides: Partial<NonNullable<EnvironmentalImpactResponse["summary"]["weather"]>> = {},
): EnvironmentalImpactResponse["summary"]["weather"] {
  return {
    pm25: 12,
    pm25_min: 8,
    pm25_max: 55,
    rain_mm: 0,
    has_rain: false,
    rain_label: "Sin lluvia",
    wind_speed: 4,
    wind_speed_kmh: 14.4,
    wind_speed_min: 0,
    wind_speed_max: 10,
    wind_speed_min_kmh: 0,
    wind_speed_max_kmh: 36,
    wind_label: "Viento moderado",
    global_radiation: null,
    sky_label: "Sin dato",
    ...overrides,
  };
}

describe("buildRouteInsightMessages", () => {
  it("returns the three always-on base messages after selecting a route", () => {
    const selected = route({
      congestion_score: 55,
      risk_level: "high",
      pm25_exposure: pm25("Alta", 58),
    });

    const result = buildRouteInsightMessages("fastest", selected, [selected], weather());

    expect(result.map((message) => message.id)).toEqual([
      "base-air-quality",
      "base-congestion",
      "base-travel-time",
    ]);
    expect(result.map((message) => message.type)).toEqual(["air", "congestion", "time"]);
    expect(result[0].detail).toMatch(/calidad del aire/i);
    expect(result[0].detail).not.toMatch(/PM2\.5/);
    expect(result[0].detail).not.toMatch(/congestion/i);
    expect(result[1].detail).toMatch(/congest/i);
    expect(result[1].detail).not.toMatch(/PM2\.5/);
  });

  it("recommends the environmental route when it takes the same time or up to three extra minutes", () => {
    const fastest = route({
      key: "fastest",
      duration_min: 10,
      pm25_exposure: pm25("Alta", 56),
    });
    const healthiest = route({
      key: "healthiest",
      label: "Menor exposición ambiental",
      badges: [{ key: "healthiest", label: "Menor exposición ambiental" }],
      duration_min: 13,
      pm25_exposure: pm25("Media", 22),
    });

    const result = buildRouteInsightMessages("fastest", fastest, [fastest, healthiest], weather());
    const recommendation = result.find((message) => message.type === "recommendation");

    expect(recommendation).toMatchObject({
      type: "recommendation",
      action: {
        label: "Seleccionar ruta saludable",
        targetRouteId: "healthiest",
      },
    });
  });

  it("does not force a recommendation when the healthy route is slower and does not reduce exposure", () => {
    const fastest = route({
      key: "fastest",
      duration_min: 10,
      pm25_exposure: pm25("Alta", 56),
    });
    const healthiest = route({
      key: "healthiest",
      label: "Menor exposición ambiental",
      badges: [{ key: "healthiest", label: "Menor exposición ambiental" }],
      duration_min: 18,
      pm25_exposure: pm25("Alta", 56),
    });

    const result = buildRouteInsightMessages("fastest", fastest, [fastest, healthiest], weather());

    expect(result.some((message) => message.type === "recommendation")).toBe(false);
  });

  it("recommends the healthy route when it reduces exposure even if it is slower", () => {
    const fastest = route({
      key: "fastest",
      duration_min: 10,
      pm25_exposure: pm25("Alta", 56),
    });
    const healthiest = route({
      key: "healthiest",
      label: "Menor exposición ambiental",
      badges: [{ key: "healthiest", label: "Menor exposición ambiental" }],
      duration_min: 18,
      pm25_exposure: pm25("Media", 22),
    });

    const result = buildRouteInsightMessages("fastest", fastest, [fastest, healthiest], weather());

    expect(result.find((message) => message.id === "recommend-healthiest-lower-exposure")).toMatchObject({
      action: {
        label: "Seleccionar ruta saludable",
        targetRouteId: "healthiest",
      },
    });
  });

  it("adds one weather message only when rain is present", () => {
    const selected = route();
    const result = buildRouteInsightMessages(
      "least_congested",
      selected,
      [selected],
      weather({ has_rain: true, rain_label: "Lluvia", wind_label: "Viento fuerte", wind_speed_kmh: 30 }),
    );

    expect(result.filter((message) => message.type === "weather")).toHaveLength(1);
    expect(result.some((message) => message.id === "weather-rain")).toBe(true);
  });

  it("shows an informational healthy environment message for the selected healthy route", () => {
    const selected = route({
      key: "healthiest",
      label: "Menor exposición ambiental",
      badges: [{ key: "healthiest", label: "Menor exposición ambiental" }],
      pm25_exposure: pm25("Baja", 12),
      urban_wellbeing: urbanWellbeing(),
    });

    const result = buildRouteInsightMessages("healthiest", selected, [selected], weather());
    const healthyMessage = result.find((message) => message.id === "healthy-route-environment");

    expect(healthyMessage).toMatchObject({ type: "route_attribute" });
    expect(healthyMessage?.detail).toContain("Laguna Los Patos");
    expect(healthyMessage?.detail).toContain("borde de agua");
    expect(healthyMessage?.action).toBeUndefined();
  });

  it("explains vehicle exposure when the selected environmental route has no urban contribution", () => {
    const selected = route({
      key: "healthiest",
      label: "Menor exposicion ambiental",
      badges: [{ key: "healthiest", label: "Menor exposicion ambiental" }],
      pm25_exposure: pm25("Baja", 12),
      urban_wellbeing: urbanWellbeing({
        score: 0,
        nearby_feature_count: 0,
        top_features: [],
      }),
    });

    const result = buildRouteInsightMessages("healthiest", selected, [selected], weather());
    const healthyMessage = result.find((message) => message.id === "healthy-route-environment");

    expect(healthyMessage).toMatchObject({
      title: "Entorno saludable",
      priority: "medium",
    });
    expect(healthyMessage?.detail).toMatch(/No se encontro una alternativa con aporte urbano favorable/i);
    expect(healthyMessage?.detail).toMatch(/menor exposicion vehicular/i);
  });
});
