import { describe, expect, it } from "vitest";

import {
  congestionMeaning,
  cyclewayEvidenceSegments,
  environmentalMeaning,
  environmentalQueryBox,
  levelRangeLabel,
  routeAutoFitBounds,
  routeColor,
  shouldPickMapPoint,
  wellbeingEvidenceFeatureIds,
} from "./PlanningMap";
import type { CyclewayFeature, PlanRouteCard } from "../types";

function route(): PlanRouteCard {
  return {
    key: "fastest",
    label: "Llegar antes",
    badges: [],
    duration_min: 10,
    distance_km: 3,
    delay_min: 0,
    congestion_score: 0,
    risk_level: "low",
    summary: "Ruta de prueba",
    geometry: [
      { lat: -36.82, lon: -73.06 },
      { lat: -36.8, lon: -73.01 },
    ],
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
  };
}

describe("routeAutoFitBounds", () => {
  it("does not request a map movement before a route exists", () => {
    expect(routeAutoFitBounds(null)).toBeNull();
  });

  it("frames only the calculated route geometry", () => {
    expect(routeAutoFitBounds(route())).toEqual([-73.06, -36.82, -73.01, -36.8]);
  });
});

describe("routeColor", () => {
  it("uses blue for the fastest route and teal for the least congested route", () => {
    expect(routeColor(route(), 0)).toBe("#2563eb");
    expect(routeColor({ ...route(), key: "least_congested" }, 0)).toBe("#0f766e");
  });
});

describe("wellbeingEvidenceFeatureIds", () => {
  it("returns only the urban features that explain the selected healthy route", () => {
    const healthyRoute: PlanRouteCard = {
      ...route(),
      key: "healthiest",
      label: "Menor exposicion ambiental",
      badges: [{ key: "healthiest", label: "Menor exposicion ambiental" }],
      urban_wellbeing: {
        available: true,
        score: 31.5,
        green_ratio: 0.2,
        blue_ratio: 0,
        tree_ratio: 0.1,
        public_space_ratio: 0,
        sustainability_ratio: 0,
        nearby_feature_count: 1,
        nearby_buffer_m: 65,
        top_features: [
          {
            feature_id: "parque-ecuador",
            name: "Parque Ecuador",
            category: "green_space",
            subtype: "park",
            distance_m: 18,
            source: "OpenStreetMap/Overpass",
            base_weight: 1,
          },
        ],
        method: "Cobertura del corredor de ruta por elementos de bienestar urbano.",
        data_source: "OpenStreetMap/Overpass",
      },
    };

    expect(wellbeingEvidenceFeatureIds(healthyRoute)).toEqual(["parque-ecuador"]);
    expect(wellbeingEvidenceFeatureIds({ ...healthyRoute, key: "fastest", badges: [] })).toEqual([]);
  });
});

describe("cyclewayEvidenceSegments", () => {
  it("returns only the cycleway segment that runs beside the selected route", () => {
    const selectedRoute: PlanRouteCard = {
      ...route(),
      geometry: [
        { lat: -36.0, lon: -73.001 },
        { lat: -36.0, lon: -72.999 },
      ],
    };
    const cycleway: CyclewayFeature = {
      type: "Feature",
      properties: {
        local_id: "long-cycleway",
        category: "local_verified_cycleway",
      },
      geometry: {
        type: "LineString",
        coordinates: [
          [-73.001, -36.0],
          [-73.0005, -36.0],
          [-72.999, -36.0],
          [-72.998, -35.998],
          [-72.997, -35.996],
        ],
      },
    };

    expect(cyclewayEvidenceSegments(cycleway, selectedRoute)).toEqual([
      [
        [-73.001, -36.0],
        [-73.0005, -36.0],
        [-72.999, -36.0],
      ],
    ]);
  });
});

describe("environmentalQueryBox", () => {
  it("expands environmental hit detection around the pointer", () => {
    expect(environmentalQueryBox([100, 80])).toEqual([
      [90, 70],
      [110, 90],
    ]);
  });

  it("supports a custom hit tolerance", () => {
    expect(environmentalQueryBox([100, 80], 4)).toEqual([
      [96, 76],
      [104, 84],
    ]);
  });
});

describe("shouldPickMapPoint", () => {
  it("does not place route points while inspection mode is active", () => {
    expect(shouldPickMapPoint(true, 0)).toBe(false);
  });

  it("places route points only when no environmental feature was clicked", () => {
    expect(shouldPickMapPoint(false, 0)).toBe(true);
    expect(shouldPickMapPoint(false, 1)).toBe(false);
  });
});

describe("levelRangeLabel", () => {
  it("explains the three color thresholds", () => {
    expect(levelRangeLabel("low")).toBe("menos de 35%");
    expect(levelRangeLabel("medium")).toBe("entre 35% y menos de 65%");
    expect(levelRangeLabel("high")).toBe("65% o mas");
  });
});

describe("plain-language map explanations", () => {
  it("explains congestion using severity and age", () => {
    expect(congestionMeaning("high", 0)).toContain("mucho taco");
    expect(congestionMeaning("medium", 1)).toContain("efecto ya es menor");
  });

  it("explains environmental impact as potential exposure", () => {
    expect(environmentalMeaning("high")).toContain("presión ambiental potencial alta");
    expect(environmentalMeaning("low")).toContain("presión potencial baja");
  });
});
