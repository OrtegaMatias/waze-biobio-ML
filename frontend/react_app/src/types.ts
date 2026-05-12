export type RoutePoint = {
  lat: number;
  lon: number;
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

export type DatasetStatus = {
  current: string;
  current_label: string;
  available: Array<{ key: string; label: string }>;
};

export type DemoScenario = {
  id: string;
  title: string;
  description: string;
  origin: RoutePoint;
  destination: RoutePoint;
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

export type TravelStyle = "safe" | "balanced" | "fast";

export type RecommendationItem = {
  via: string;
  estimated_rating: number;
  strategy: string;
};

export type IncidentExposure = {
  total_incident_segments: number;
  matched_incident_segments: number;
  congestion_segments: number;
  accident_segments: number;
  exposure_minutes: number;
};

export type Pm25StationExposure = {
  station_id: string;
  station_name: string;
  distance_km: number;
  pm25: number;
  sample_points: number;
};

export type Pm25Exposure = {
  available: boolean;
  average_pm25: number;
  category: "Baja" | "Media" | "Alta";
  stations: Pm25StationExposure[];
  method: string;
  data_source: string;
};

export type SegmentImpact = {
  segment_id: string;
  via: string;
  comuna: string;
  event_type: string;
  impact_score: number;
  reason: string;
};

export type PreferredViaImpact = {
  via: string;
  factor: number;
  reason: string;
};

export type RouteVariant = {
  distance_km: number;
  estimated_duration_min: number;
  extra_delay_min: number;
  risk_score: number;
  geometry: RoutePoint[];
  why_changed: string[];
  top_penalized_segments: SegmentImpact[];
  top_preferred_vias: PreferredViaImpact[];
  incident_exposure: IncidentExposure;
  pm25_exposure?: Pm25Exposure | null;
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

export type PlaceResult = {
  id: string;
  label: string;
  lat: number;
  lon: number;
  bbox: number[] | null;
};

export type HotspotPoint = {
  lat: number;
  lon: number;
  weight: number;
  day?: string | null;
  bucket?: string | null;
  segment_id?: string | null;
  hora_inicio_float?: number | null;
  hora_fin_float?: number | null;
};

export type RouteBadge = {
  key: "base" | "least_congestion" | "healthiest" | "recommended" | "fastest" | "least_exposure";
  label: string;
};

export type PlanRouteAlert = {
  title: string;
  detail: string;
  severity: "low" | "medium" | "high";
};

export type PlanRouteSummary = {
  eta_total_min: number;
  distance_km: number;
  delay_min: number;
  alerts_on_route: number;
  main_reason: string;
};

export type PlanRouteCard = {
  key: string;
  label: string;
  badges: RouteBadge[];
  duration_min: number;
  distance_km: number;
  delay_min: number;
  risk_level: "low" | "medium" | "high";
  summary: string;
  geometry: RoutePoint[];
  top_alerts: PlanRouteAlert[];
  why_changed: string[];
  top_penalized_segments: SegmentImpact[];
  top_preferred_vias: PreferredViaImpact[];
  incident_exposure: IncidentExposure;
  pm25_exposure?: Pm25Exposure | null;
};

export type PlanRouteResponse = {
  selected_route_key: string;
  routes: PlanRouteCard[];
  summary: PlanRouteSummary;
  alerts: PlanRouteAlert[];
  hotspots: HotspotPoint[];
  map_bounds: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
};
