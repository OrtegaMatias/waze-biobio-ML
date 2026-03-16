export type ReadinessStatus = {
  status: "ready" | "warming" | "error";
  ready: boolean;
  message: string;
  dataset_profile: string;
  bootstrap: {
    status: "idle" | "running" | "completed" | "error";
    message: string;
    percent: number;
    routing_nodes: number;
    routing_segments: number;
    duration_ms: number;
    dataset_profile: string;
    quality?: DataQualitySummary | null;
  };
};

export type DataQualitySummary = {
  status: "ok" | "warning" | "error";
  dataset_profile: string;
  duplicate_incident_sources: boolean;
  date_range: {
    start: string | null;
    end: string | null;
    days: number;
  };
  missing_via_ratio: number;
  anomalous_communes: string[];
  raw_counts: {
    accidents: number;
    congestions: number;
    combined: number;
  };
  warnings: string[];
  notes: string[];
};

export type DatasetStatus = {
  current: string;
  current_label: string;
  available: Array<{ key: string; label: string }>;
};

export type DemoScenario = {
  id: string;
  title: string;
  description: string;
  origin: { lat: number; lon: number };
  destination: { lat: number; lon: number };
  day_of_week: string;
  departure_hour: number;
  profile: TravelerProfileId;
  recommended_focus: string;
};

export type TravelerProfileId =
  | "safety_focused"
  | "usuario_demo"
  | "moderate_risk"
  | "risk_taker";

export type RecommendationItem = {
  via: string;
  estimated_rating: number;
  strategy: string;
};

export type RouteVariant = {
  distance_km: number;
  estimated_duration_min: number;
  extra_delay_min: number;
  risk_score: number;
  geometry: Array<{
    lat: number;
    lon: number;
  }>;
  why_changed: string[];
  top_penalized_segments: Array<{
    segment_id: string;
    via: string;
    comuna: string;
    event_type: string;
    impact_score: number;
    reason: string;
  }>;
  top_preferred_vias: Array<{
    via: string;
    factor: number;
    reason: string;
  }>;
  incident_exposure: {
    total_incident_segments: number;
    matched_incident_segments: number;
    congestion_segments: number;
    accident_segments: number;
    exposure_minutes: number;
  };
};

export type RouteResponse = {
  reference: RouteVariant;
  ubcf: RouteVariant;
  ibcf: RouteVariant;
  personalized: RouteVariant | null;
  comparison: {
    fastest_variant: string;
    safest_variant: string;
    lowest_exposure_variant: string;
    best_balance_variant: string;
    deltas: Array<{
      variant: string;
      distance_delta_km: number;
      total_duration_delta_min: number;
      risk_delta: number;
      exposure_delta: number;
    }>;
  };
};
