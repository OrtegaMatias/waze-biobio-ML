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

export type CongestionDateCoverage = {
  start: string | null;
  end: string | null;
  available_dates: string[];
  missing_dates: string[];
  rain_dates: string[];
  available_days: number;
  calendar_days: number;
  data_source: string;
};

export type CongestionHourAvailability = {
  date: string;
  available_hours: number[];
  count: number;
  data_source: string;
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

export type Pm25StationCondition = {
  station_id: string;
  station_name: string;
  lat: number;
  lon: number;
  pm25: number;
  category: "Baja" | "Media" | "Alta";
};

export type Pm25SnapshotResponse = {
  available: boolean;
  requested_at: string;
  stations: Pm25StationCondition[];
  average_pm25: number | null;
  date_range: {
    start?: string | null;
    end?: string | null;
  };
  method: string;
  data_source: string;
};

export type CyclewayCoverage = {
  available: boolean;
  coverage_ratio: number;
  nearby_cycleway_km: number;
  route_km: number;
  nearby_buffer_m: number;
  has_high_coverage: boolean;
  data_source: string;
};

export type AirQualityLevel = "good" | "moderate" | "elevated" | "very_high" | "unknown";

export type AirQualityInsight = {
  level: AirQualityLevel;
  label: string;
  value: string;
  headline: string;
  messages: string[];
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
  urban_wellbeing?: UrbanWellbeingAnalysis | null;
  healthy_route_score?: number | null;
};

export type RouteResponse = {
  reference: RouteVariant;
  least_congestion?: RouteVariant | null;
  ubcf: RouteVariant;
  ibcf: RouteVariant;
  healthiest?: RouteVariant | null;
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

export type EnvironmentalImpactPoint = {
  lat: number;
  lon: number;
  score: number;
  level: "low" | "medium" | "high";
  congestion_score: number;
  congestion_level: "low" | "medium" | "high";
  pm25?: number | null;
  rain_mm?: number | null;
  wind_speed?: number | null;
  segment_id: string;
  via?: string | null;
  comuna?: string | null;
  message: string;
};

export type EnvironmentalImpactResponse = {
  summary: {
    available: boolean;
    requested_at: string;
    point_count: number;
    dominant_level: "low" | "medium" | "high" | "none";
    weather?: {
      pm25?: number | null;
      pm25_min?: number | null;
      pm25_max?: number | null;
      rain_mm?: number | null;
      has_rain?: boolean | null;
      rain_label: "Sin lluvia" | "Llovizna" | "Lluvia" | "Lluvia fuerte" | "Sin dato";
      wind_speed?: number | null;
      wind_speed_kmh?: number | null;
      wind_speed_min?: number | null;
      wind_speed_max?: number | null;
      wind_speed_min_kmh?: number | null;
      wind_speed_max_kmh?: number | null;
      wind_label: "Viento suave" | "Viento moderado" | "Viento fuerte" | "Sin dato";
      global_radiation?: number | null;
      sky_label: "Despejado" | "Parcial" | "Nublado" | "Oscuro" | "Sin dato";
    };
    messages: string[];
    method: string;
    data_source: string;
  };
  points: EnvironmentalImpactPoint[];
  zones?: {
    type: "FeatureCollection";
    features: Array<{
      type: "Feature";
      properties: Record<string, unknown>;
      geometry: {
        type: string;
        coordinates: unknown;
      };
    }>;
  };
  congestion_lines?: {
    type: "FeatureCollection";
    features: Array<{
      type: "Feature";
      properties: Record<string, unknown>;
      geometry: {
        type: string;
        coordinates: unknown;
      };
    }>;
  };
};

export type CyclewayFeature = {
  type: "Feature";
  properties: {
    osm_id?: number;
    minvu_id?: number;
    minvu_path_index?: number;
    local_id?: string;
    name?: string;
    category?: string;
    highway?: string;
    cycleway?: string;
    cycleway_left?: string;
    cycleway_right?: string;
    cycleway_both?: string;
    bicycle?: string;
    source?: string;
    source_detail?: string;
    source_url?: string;
    comuna?: string;
    start?: string;
    end?: string;
    km?: number;
    stage?: string;
    stage_detail?: string;
  };
  geometry: {
    type: "LineString";
    coordinates: Array<[number, number]>;
  };
};

export type CyclewayCollection = {
  type: "FeatureCollection";
  name: string;
  features: CyclewayFeature[];
};

export type UrbanWellbeingCategory =
  | "green_space"
  | "blue_space"
  | "tree_cover"
  | "public_space"
  | "sustainability"
  | "cycleway";

export type UrbanWellbeingFeature = {
  type: "Feature";
  properties: {
    feature_id?: string;
    osm_id?: number;
    name?: string;
    category?: UrbanWellbeingCategory;
    subtype?: string;
    base_weight?: number;
    source?: string;
  };
  geometry: {
    type: string;
    coordinates: unknown;
  };
};

export type UrbanWellbeingCollection = {
  type: "FeatureCollection";
  name: string;
  features: UrbanWellbeingFeature[];
};

export type WellbeingFeatureImpact = {
  feature_id: string;
  name: string;
  category: UrbanWellbeingCategory;
  subtype: string;
  distance_m: number;
  source: string;
  base_weight: number;
};

export type UrbanWellbeingAnalysis = {
  available: boolean;
  score: number;
  green_ratio: number;
  blue_ratio: number;
  tree_ratio: number;
  public_space_ratio: number;
  sustainability_ratio: number;
  cycleway_ratio?: number;
  nearby_feature_count: number;
  nearby_buffer_m: number;
  top_features: WellbeingFeatureImpact[];
  method: string;
  data_source: string;
};

export type RouteBadge = {
  key: "base" | "least_congestion" | "healthiest" | "recommended" | "fastest" | "least_exposure";
  label: string;
};

export type RouteType = "fastest" | "least_congested" | "healthiest";

export type PlanRouteAlert = {
  title: string;
  detail: string;
  severity: "low" | "medium" | "high";
};

export type ContextualMobilityMessage = {
  id: string;
  title: string;
  detail: string;
  mode: "walk" | "bike" | "active_mobility";
  priority: "low" | "medium" | "high";
};

export type MobilityMessageType =
  | "air"
  | "congestion"
  | "time"
  | "route_attribute"
  | "weather"
  | "recommendation";

export type MobilityGuidanceMessage = {
  id: string;
  title: string;
  detail: string;
  type: MobilityMessageType;
  priority: "low" | "medium" | "high";
  action?: {
    label: string;
    targetRouteId: string;
  };
};

export type ActiveMobilityEstimate = {
  auto_min: number;
  bike_min: number;
  walk_min: number;
  bike_extra_min: number;
  walk_extra_min: number;
  bike_speed_kmh: number;
  walk_speed_kmh: number;
};

export type PlanRouteSummary = {
  eta_total_min: number;
  distance_km: number;
  delay_min: number;
  alerts_on_route: number;
  main_reason: string;
};

export type RouteCongestionCoverage = {
  route_m: number;
  congested_m: number;
  high_m: number;
  medium_m: number;
  low_m: number;
  congested_pct: number;
  high_pct: number;
  medium_pct: number;
  low_pct: number;
  primary_via?: string | null;
};

export type PlanRouteCard = {
  key: RouteType;
  label: string;
  badges: RouteBadge[];
  duration_min: number;
  distance_km: number;
  delay_min: number;
  congestion_score: number;
  risk_level: "low" | "medium" | "high";
  summary: string;
  geometry: RoutePoint[];
  road_geometry?: RoutePoint[];
  access_geometry?: RoutePoint[][];
  top_alerts: PlanRouteAlert[];
  why_changed: string[];
  top_penalized_segments: SegmentImpact[];
  top_preferred_vias: PreferredViaImpact[];
  congestion_coverage?: RouteCongestionCoverage;
  incident_exposure: IncidentExposure;
  pm25_exposure?: Pm25Exposure | null;
  urban_wellbeing?: UrbanWellbeingAnalysis | null;
  healthy_route_score?: number | null;
  cycleway_coverage?: CyclewayCoverage | null;
  bicycle_suggestion?: string | null;
  contextual_messages?: ContextualMobilityMessage[];
  active_mobility_estimate?: ActiveMobilityEstimate | null;
};

export type PlanRouteResponse = {
  selected_route_key: RouteType;
  routes: PlanRouteCard[];
  routes_by_type: Record<RouteType, PlanRouteCard>;
  summary: PlanRouteSummary;
  alerts: PlanRouteAlert[];
  contextual_messages?: ContextualMobilityMessage[];
  hotspots: HotspotPoint[];
  map_bounds: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
};
