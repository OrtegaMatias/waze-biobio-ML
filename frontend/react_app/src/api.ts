import type {
  DatasetStatus,
  DemoScenario,
  ReadinessStatus,
  RecommendationItem,
  RouteResponse,
  TravelerProfileId,
} from "./types";

export const BACKEND_URL =
  import.meta.env.VITE_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
    ...init,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return (await response.json()) as T;
}

export function getReadiness(): Promise<ReadinessStatus> {
  return requestJson<ReadinessStatus>("/readyz");
}

export function startBootstrap(): Promise<unknown> {
  return requestJson("/system/bootstrap", {
    method: "POST",
    body: JSON.stringify({}),
  });
}

export function getDatasetStatus(): Promise<DatasetStatus> {
  return requestJson<DatasetStatus>("/system/dataset");
}

export function setDataset(profile: string): Promise<DatasetStatus> {
  return requestJson<DatasetStatus>("/system/dataset", {
    method: "POST",
    body: JSON.stringify({ profile }),
  });
}

export async function getScenarios(): Promise<DemoScenario[]> {
  const response = await requestJson<{ scenarios: DemoScenario[] }>("/system/demo-scenarios");
  return response.scenarios;
}

export async function getRecommendations(profile: TravelerProfileId): Promise<Record<string, RecommendationItem[]>> {
  return requestJson<Record<string, RecommendationItem[]>>("/recommendations/playground", {
    method: "POST",
    body: JSON.stringify({
      user_id: profile,
      known_vias: [],
      limit: 10,
      strategies: ["ubcf", "ibcf"],
    }),
  });
}

export function buildPreferencePayload(
  recommendations: Record<string, RecommendationItem[]>,
): { ubcf_preferences: Array<{ via: string; weight: number }>; ibcf_preferences: Array<{ via: string; weight: number }> } {
  const build = (items: RecommendationItem[] = []) =>
    items.slice(0, 6).map((item) => ({
      via: item.via,
      weight: Number((item.estimated_rating / 5).toFixed(3)),
    }));
  return {
    ubcf_preferences: build(recommendations.ubcf),
    ibcf_preferences: build(recommendations.ibcf),
  };
}

export async function generateRouteComparison(args: {
  origin: { lat: number; lon: number };
  destination: { lat: number; lon: number };
  day_of_week: string;
  departure_hour: number;
  avoid_congestion: boolean;
  avoid_accidents: boolean;
  profile: TravelerProfileId;
}): Promise<RouteResponse> {
  const recommendations = await getRecommendations(args.profile);
  const preferences = buildPreferencePayload(recommendations);
  return requestJson<RouteResponse>("/routes/optimal", {
    method: "POST",
    body: JSON.stringify({
      origin: args.origin,
      destination: args.destination,
      preferences: [],
      ubcf_preferences: preferences.ubcf_preferences,
      ibcf_preferences: preferences.ibcf_preferences,
      day_of_week: args.day_of_week,
      departure_hour: args.departure_hour,
      avoid_congestion: args.avoid_congestion,
      avoid_accidents: args.avoid_accidents,
    }),
  });
}
