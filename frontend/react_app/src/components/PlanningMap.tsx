import { useEffect, useEffectEvent, useRef, useState } from "react";
import type { MouseEvent, PointerEvent } from "react";
import type { StyleSpecification } from "maplibre-gl";

import type {
  CyclewayFeature,
  EnvironmentalImpactResponse,
  HotspotPoint,
  PlanRouteCard,
  RoutePoint,
  RouteType,
  UrbanWellbeingFeature,
  UrbanWellbeingCategory,
} from "../types";

type PinKey = "origin" | "destination";

type PlanningMapProps = {
  enabled: boolean;
  styleUrl: string;
  mapboxToken: string;
  routes: PlanRouteCard[];
  selectedRouteKey: RouteType | null;
  hotspots: HotspotPoint[];
  environmentalImpact: EnvironmentalImpactResponse | null;
  environmentalImpactLoading?: boolean;
  showImpactCard?: boolean;
  cycleways: CyclewayFeature[];
  wellbeingFeatures: UrbanWellbeingFeature[];
  inspectMode: boolean;
  showCycleways: boolean;
  wellbeingVisibility: WellbeingVisibility;
  origin: RoutePoint | null;
  destination: RoutePoint | null;
  activePin: PinKey;
  onPickPoint: (pin: PinKey, point: RoutePoint) => void;
  onMarkerDrag: (pin: PinKey, point: RoutePoint) => void;
};

type RouteTooltipState = {
  route: PlanRouteCard;
  x: number;
  y: number;
};

type LayerTooltipState = {
  kind: "congestion" | "environment";
  properties: Record<string, unknown>;
  x: number;
  y: number;
};

type SelectedLayerInfo = {
  congestion: Record<string, unknown> | null;
  environment: Record<string, unknown> | null;
  anchor: { x: number; y: number };
};

export type WellbeingVisibility = Record<UrbanWellbeingCategory, boolean>;

type PanelPosition = {
  left: number;
  top: number;
};

type PanelDragState = {
  pointerId: number;
  offsetX: number;
  offsetY: number;
};

const BASE_ROUTE_COLOR = "#2563eb";
const LEAST_CONGESTED_ROUTE_COLOR = "#0f766e";
const HEALTHIEST_ROUTE_COLOR = "#16a34a";
const FALLBACK_ROUTE_COLORS = ["#ea580c", "#7c3aed", "#0891b2"];
const OVERLAP_ROUTE_OFFSET_PX = 9;
const ENVIRONMENTAL_ZONE_SOURCE = "environmental-impact-zones";
const ENVIRONMENTAL_ZONE_FILL_LAYER = "environmental-impact-zone-fill";
const ENVIRONMENTAL_ZONE_OUTLINE_LAYER = "environmental-impact-zone-outline";
const ENVIRONMENTAL_CONGESTION_LINE_SOURCE = "environmental-congestion-lines";
const ENVIRONMENTAL_CONGESTION_LINE_CASING_LAYER = "environmental-congestion-line-casing";
const ENVIRONMENTAL_CONGESTION_LINE_LAYER = "environmental-congestion-line";
const ENVIRONMENTAL_POINT_SOURCE = "environmental-impact-points";
const ENVIRONMENTAL_HEATMAP_LAYER = "environmental-impact-heatmap";
const ENVIRONMENTAL_HIT_TOLERANCE_PX = 10;
const ENVIRONMENTAL_CLICK_TOLERANCE_PX = 18;
const PLANNED_ROUTE_SOURCE = "planned-routes";
const PLANNED_ROUTE_ACCESS_SOURCE = "planned-route-access";
const PLANNED_ROUTE_CASING_LAYER = "planned-route-casing";
const PLANNED_ROUTE_LINE_LAYER = "planned-route-line";
const PLANNED_ROUTE_ACCESS_CASING_LAYER = "planned-route-access-casing";
const PLANNED_ROUTE_ACCESS_LINE_LAYER = "planned-route-access-line";
export const WELLBEING_LAYER_OPTIONS: Array<{
  category: UrbanWellbeingCategory;
  label: string;
  description: string;
}> = [
  {
    category: "green_space",
    label: "Parques y áreas verdes",
    description: "Manchas verdes: parques, jardines, bosques urbanos, humedales y áreas recreativas.",
  },
  {
    category: "blue_space",
    label: "Lagos, lagunas y cursos de agua",
    description: "Manchas o líneas azules: lagunas, ríos, canales, esteros y otros cuerpos de agua.",
  },
  {
    category: "tree_cover",
    label: "Sectores arbolados",
    description: "Líneas verde oscuro: hileras de árboles registradas en OpenStreetMap.",
  },
  {
    category: "public_space",
    label: "Plazas y espacios públicos",
    description: "Manchas amarillas: plazas y espacios urbanos abiertos asociados a permanencia y encuentro.",
  },
  {
    category: "sustainability",
    label: "Puntos de reciclaje",
    description: "Puntos morados: lugares registrados para reciclaje u otros servicios de sostenibilidad.",
  },
];

export const DEFAULT_WELLBEING_VISIBILITY: WellbeingVisibility = {
  green_space: false,
  blue_space: false,
  tree_cover: false,
  public_space: false,
  sustainability: false,
  cycleway: false,
};
const CYCLEWAY_EVIDENCE_DISPLAY_BUFFER_M = 35;
const CARTO_LIGHT_TILES = [
  "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
  "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
  "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
  "https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
];

function localBasicStyle(includeTiles = true): StyleSpecification {
  const sources: StyleSpecification["sources"] = includeTiles
    ? {
        "carto-light": {
          type: "raster",
          tiles: CARTO_LIGHT_TILES,
          tileSize: 256,
          maxzoom: 19,
          attribution: "OpenStreetMap contributors, CARTO",
        },
      }
    : {};
  const layers: StyleSpecification["layers"] = [
    {
      id: "background",
      type: "background",
      paint: {
        "background-color": "#e7f0ef",
      },
    },
  ];

  if (includeTiles) {
    layers.push({
      id: "carto-light",
      type: "raster",
      source: "carto-light",
      paint: {
        "raster-opacity": 0.96,
      },
    });
  }

  return {
    version: 8,
    sources,
    layers,
  };
}

function normalizeStyleUrl(styleUrl: string, token: string): string | StyleSpecification {
  if (!styleUrl || styleUrl === "local-basic") {
    return localBasicStyle();
  }
  if (styleUrl.startsWith("mapbox://styles/") && !token) {
    return localBasicStyle();
  }
  if (styleUrl.startsWith("mapbox://styles/")) {
    return `https://api.mapbox.com/styles/v1/${styleUrl.replace("mapbox://styles/", "")}?access_token=${token}`;
  }
  return styleUrl;
}

function isRecoverableMapError(event: any): boolean {
  if (event?.sourceId === "carto-light") {
    return true;
  }
  const message = String(event?.error?.message ?? event?.message ?? "").toLowerCase();
  const status = String(event?.error?.status ?? event?.status ?? "");
  if (message.includes(ENVIRONMENTAL_ZONE_SOURCE) || message.includes("environmental-impact")) {
    return true;
  }
  return (
    status === "401" ||
    status === "403" ||
    status === "404" ||
    message.includes("401") ||
    message.includes("403") ||
    message.includes("404") ||
    message.includes("unauthorized") ||
    message.includes("forbidden") ||
    message.includes("not found") ||
    message.includes("tile") ||
    message.includes("raster") ||
    message.includes("image") ||
    message.includes("failed to fetch") ||
    message.includes("network")
  );
}

function shouldFallbackToLocalStyle(event: any): boolean {
  const message = String(event?.error?.message ?? event?.message ?? "").toLowerCase();
  const status = String(event?.error?.status ?? event?.status ?? "");
  return (
    status === "401" ||
    status === "403" ||
    status === "404" ||
    message.includes("access token") ||
    message.includes("unauthorized") ||
    message.includes("forbidden") ||
    message.includes("not found") ||
    message.includes("failed to fetch") ||
    message.includes("network")
  );
}

function addMapboxToken(url: string, token: string): string {
  if (!token || !url.includes("mapbox.com")) {
    return url;
  }
  try {
    const next = new URL(url);
    if (!next.searchParams.has("access_token")) {
      next.searchParams.set("access_token", token);
    }
    return next.toString();
  } catch {
    return url;
  }
}

function routeHasBadge(route: PlanRouteCard, badgeKey: string): boolean {
  return route.badges.some((badge) => badge.key === badgeKey);
}

function cyclewayFeatureId(feature: CyclewayFeature, index: number): string {
  const properties = feature.properties;
  if (properties.local_id) {
    return `cycleway-local-${properties.local_id}`;
  }
  if (properties.minvu_id !== undefined) {
    return `cycleway-minvu-${properties.minvu_id}-${properties.minvu_path_index ?? index}`;
  }
  if (properties.osm_id !== undefined) {
    return `cycleway-osm-${properties.osm_id}`;
  }
  return `cycleway-feature-${index}`;
}

export function wellbeingEvidenceFeatureIds(route: PlanRouteCard | null): string[] {
  if (!route || !(route.key === "healthiest" || routeHasBadge(route, "healthiest"))) {
    return [];
  }
  const wellbeing = route.urban_wellbeing;
  if (!wellbeing?.available || wellbeing.score <= 0) {
    return [];
  }
  const ids = new Set<string>();
  for (const feature of wellbeing.top_features ?? []) {
    if (feature.feature_id) {
      ids.add(feature.feature_id);
    }
  }
  return [...ids];
}

export function routeColor(route: PlanRouteCard, fallbackIndex: number): string {
  if (route.key === "fastest" || routeHasBadge(route, "base") || routeHasBadge(route, "fastest")) {
    return BASE_ROUTE_COLOR;
  }
  if (
    route.key === "least_congested" ||
    routeHasBadge(route, "least_congestion") ||
    routeHasBadge(route, "least_exposure")
  ) {
    return LEAST_CONGESTED_ROUTE_COLOR;
  }
  if (route.key === "healthiest" || routeHasBadge(route, "healthiest")) {
    return HEALTHIEST_ROUTE_COLOR;
  }
  return FALLBACK_ROUTE_COLORS[fallbackIndex % FALLBACK_ROUTE_COLORS.length];
}

function routeRoadGeometry(route: PlanRouteCard): RoutePoint[] {
  if (route.road_geometry && route.road_geometry.length > 1) {
    return route.road_geometry;
  }
  return route.geometry.length >= 4 ? route.geometry.slice(1, -1) : route.geometry;
}

function routeAccessGeometry(route: PlanRouteCard): RoutePoint[][] {
  if (route.access_geometry?.length) {
    return route.access_geometry.filter((segment) => segment.length > 1);
  }
  if (route.geometry.length < 4) {
    return [];
  }
  return [route.geometry.slice(0, 2), route.geometry.slice(-2)];
}

function lonLatToMeters(lon: number, lat: number, referenceLat: number): { x: number; y: number } {
  return {
    x: lon * 111_320 * Math.cos((referenceLat * Math.PI) / 180),
    y: lat * 110_540,
  };
}

function pointToSegmentDistanceMeters(
  point: { x: number; y: number },
  start: { x: number; y: number },
  end: { x: number; y: number },
): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= 0) {
    return Math.hypot(point.x - start.x, point.y - start.y);
  }
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
  const projected = { x: start.x + t * dx, y: start.y + t * dy };
  return Math.hypot(point.x - projected.x, point.y - projected.y);
}

function distanceToRouteMeters(coordinate: [number, number], routeGeometry: RoutePoint[], referenceLat: number): number {
  if (routeGeometry.length < 2) {
    return Number.POSITIVE_INFINITY;
  }
  const point = lonLatToMeters(coordinate[0], coordinate[1], referenceLat);
  let best = Number.POSITIVE_INFINITY;
  for (let index = 1; index < routeGeometry.length; index += 1) {
    const startPoint = routeGeometry[index - 1];
    const endPoint = routeGeometry[index];
    const start = lonLatToMeters(startPoint.lon, startPoint.lat, referenceLat);
    const end = lonLatToMeters(endPoint.lon, endPoint.lat, referenceLat);
    best = Math.min(best, pointToSegmentDistanceMeters(point, start, end));
  }
  return best;
}

export function cyclewayEvidenceSegments(
  feature: CyclewayFeature,
  selectedRoute: PlanRouteCard | null,
  bufferMeters = CYCLEWAY_EVIDENCE_DISPLAY_BUFFER_M,
): Array<Array<[number, number]>> {
  if (!selectedRoute || feature.geometry.coordinates.length < 2) {
    return [];
  }
  const routeGeometry = routeRoadGeometry(selectedRoute);
  if (routeGeometry.length < 2) {
    return [];
  }
  const referenceLat =
    routeGeometry.reduce((sum, point) => sum + point.lat, 0) / routeGeometry.length;
  const evidenceSegments: Array<Array<[number, number]>> = [];
  let current: Array<[number, number]> = [];

  for (let index = 1; index < feature.geometry.coordinates.length; index += 1) {
    const start = feature.geometry.coordinates[index - 1];
    const end = feature.geometry.coordinates[index];
    const midpoint: [number, number] = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
    const startIsNearRoute = distanceToRouteMeters(start, routeGeometry, referenceLat) <= bufferMeters;
    const endIsNearRoute = distanceToRouteMeters(end, routeGeometry, referenceLat) <= bufferMeters;
    const midpointIsNearRoute = distanceToRouteMeters(midpoint, routeGeometry, referenceLat) <= bufferMeters;
    const segmentIsNearRoute =
      midpointIsNearRoute || (startIsNearRoute && endIsNearRoute);

    if (segmentIsNearRoute) {
      if (!current.length) {
        current = [start, end];
      } else {
        current.push(end);
      }
      continue;
    }

    if (current.length > 1) {
      evidenceSegments.push(current);
    }
    current = [];
  }

  if (current.length > 1) {
    evidenceSegments.push(current);
  }

  return evidenceSegments;
}

function buildPlannedRouteData(routes: PlanRouteCard[], selectedRouteKey: RouteType | null) {
  const mainFeatures: Array<{
    type: "Feature";
    properties: {
      key: RouteType;
      label: string;
      color: string;
      selected: boolean;
      sort: number;
    };
    geometry: {
      type: "LineString";
      coordinates: number[][];
    };
  }> = [];
  const accessFeatures: Array<{
    type: "Feature";
    properties: {
      key: RouteType;
      label: string;
      color: string;
      selected: boolean;
      sort: number;
    };
    geometry: {
      type: "LineString";
      coordinates: number[][];
    };
  }> = [];

  routes
    .filter((route) => route.geometry.length > 1)
    .forEach((route, index) => {
      const selected = selectedRouteKey ? route.key === selectedRouteKey : true;
      const baseProperties = {
        key: route.key,
        label: route.label,
        color: routeColor(route, index),
        selected,
        sort: selected ? 2 : 1,
      };
      const toCoordinates = (points: RoutePoint[]) => points.map((point) => [point.lon, point.lat]);
      const roadGeometry = routeRoadGeometry(route);
      const accessGeometry = routeAccessGeometry(route);

      if (roadGeometry.length < 2) {
        mainFeatures.push({
          type: "Feature",
          properties: baseProperties,
          geometry: {
            type: "LineString",
            coordinates: toCoordinates(route.geometry),
          },
        });
        return;
      }

      mainFeatures.push({
        type: "Feature",
        properties: baseProperties,
        geometry: {
          type: "LineString",
          coordinates: toCoordinates(roadGeometry),
        },
      });
      accessGeometry.forEach((segment) => {
        accessFeatures.push({
          type: "Feature",
          properties: baseProperties,
          geometry: {
            type: "LineString",
            coordinates: toCoordinates(segment),
          },
        });
      });
    });

  return {
    main: {
      type: "FeatureCollection" as const,
      features: mainFeatures,
    },
    access: {
      type: "FeatureCollection" as const,
      features: accessFeatures,
    },
  };
}

function collectBounds(points: RoutePoint[]): [number, number, number, number] | null {
  if (!points.length) {
    return null;
  }
  const lats = points.map((point) => point.lat);
  const lons = points.map((point) => point.lon);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}

export function routeAutoFitBounds(route: PlanRouteCard | null): [number, number, number, number] | null {
  return collectBounds(route?.geometry ?? []);
}

export function environmentalQueryBox(
  point: [number, number],
  tolerance = ENVIRONMENTAL_HIT_TOLERANCE_PX,
): [[number, number], [number, number]] {
  return [
    [point[0] - tolerance, point[1] - tolerance],
    [point[0] + tolerance, point[1] + tolerance],
  ];
}

export function shouldPickMapPoint(inspectMode: boolean, environmentalFeatureCount: number): boolean {
  return !inspectMode && environmentalFeatureCount === 0;
}

function routeGeometrySignature(route: PlanRouteCard): string {
  return route.geometry.map((point) => `${point.lat.toFixed(6)},${point.lon.toFixed(6)}`).join("|");
}

function routeOverlapOffsets(routes: PlanRouteCard[]): Map<string, number> {
  const groups = new Map<string, PlanRouteCard[]>();
  routes.forEach((route) => {
    const signature = routeGeometrySignature(route);
    groups.set(signature, [...(groups.get(signature) ?? []), route]);
  });
  const offsets = new Map<string, number>();
  groups.forEach((group) => {
    if (group.length === 1) {
      offsets.set(group[0].key, 0);
      return;
    }
    const center = (group.length - 1) / 2;
    group.forEach((route, index) => {
      offsets.set(route.key, (index - center) * OVERLAP_ROUTE_OFFSET_PX);
    });
  });
  return offsets;
}

function offsetProjectedPoints(points: Array<{ x: number; y: number }>, offset: number): Array<{ x: number; y: number }> {
  if (!offset || points.length < 2) {
    return points;
  }
  return points.map((point, index) => {
    const previous = points[Math.max(0, index - 1)];
    const next = points[Math.min(points.length - 1, index + 1)];
    const dx = next.x - previous.x;
    const dy = next.y - previous.y;
    const length = Math.hypot(dx, dy) || 1;
    return {
      x: point.x + (-dy / length) * offset,
      y: point.y + (dx / length) * offset,
    };
  });
}

function projectRoutePath(map: any, geometry: RoutePoint[], offset = 0): string {
  if (geometry.length < 2) {
    return "";
  }
  const projectedPoints = geometry.map((point) => {
    const projected = map.project([point.lon, point.lat]);
    return { x: projected.x, y: projected.y };
  });
  return offsetProjectedPoints(projectedPoints, offset)
    .map((point, index) => {
      return `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`;
    })
    .join(" ");
}

function projectCoordinatePath(map: any, coordinates: Array<[number, number]>): string {
  if (coordinates.length < 2) {
    return "";
  }
  return coordinates
    .map(([lon, lat], index) => {
      const projected = map.project([lon, lat]);
      return `${index === 0 ? "M" : "L"} ${projected.x.toFixed(1)} ${projected.y.toFixed(1)}`;
    })
    .join(" ");
}

function isCoordinatePair(value: unknown): value is [number, number] {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number" &&
    Number.isFinite(value[0]) &&
    Number.isFinite(value[1])
  );
}

function projectCoordinateTree(map: any, coordinates: unknown, closePath = false): string[] {
  if (!Array.isArray(coordinates)) {
    return [];
  }
  if (coordinates.every(isCoordinatePair)) {
    const path = projectCoordinatePath(map, coordinates);
    return path ? [`${path}${closePath ? " Z" : ""}`] : [];
  }
  return coordinates.flatMap((item) => projectCoordinateTree(map, item, closePath));
}

function impactClassFromLevel(value: unknown): "impact-low" | "impact-medium" | "impact-high" {
  if (value === "low") {
    return "impact-low";
  }
  if (value === "high") {
    return "impact-high";
  }
  return "impact-medium";
}

function hotspotRadius(weight: number | null | undefined): number {
  return Math.max(5, Math.min(14, 5 + Number(weight ?? 0.5) * 5));
}

function buildEnvironmentalZoneData(environmentalImpact: EnvironmentalImpactResponse | null) {
  if (environmentalImpact?.summary.available && environmentalImpact.zones?.features?.length) {
    return environmentalImpact.zones;
  }
  return { type: "FeatureCollection", features: [] };
}

function buildEnvironmentalCongestionLineData(environmentalImpact: EnvironmentalImpactResponse | null) {
  if (environmentalImpact?.summary.available && environmentalImpact.congestion_lines?.features?.length) {
    return environmentalImpact.congestion_lines;
  }
  return { type: "FeatureCollection", features: [] };
}

function buildEnvironmentalPointData(environmentalImpact: EnvironmentalImpactResponse | null) {
  return {
    type: "FeatureCollection" as const,
    features: (environmentalImpact?.points ?? []).map((point) => ({
      type: "Feature" as const,
      properties: {
        level: point.level,
        score: point.score,
        congestion_score: point.congestion_score,
        congestion_level: point.congestion_level,
        pm25: point.pm25,
        rain_mm: point.rain_mm,
        wind_speed: point.wind_speed,
        segment_id: point.segment_id,
        via: point.via,
        comuna: point.comuna,
        message: point.message,
      },
      geometry: {
        type: "Point" as const,
        coordinates: [point.lon, point.lat],
      },
    })),
  };
}


function localRangePercent(value?: number | null, min?: number | null, max?: number | null, invert = false): number | null {
  if (value == null || min == null || max == null || max <= min) {
    return null;
  }
  const normalized = Math.max(0, Math.min(1, (value - min) / (max - min)));
  return Math.round((invert ? 1 - normalized : normalized) * 100);
}

function environmentalCardLevel(percent: number | null): "low" | "medium" | "high" | "none" {
  if (percent == null) {
    return "none";
  }
  if (percent < 34) {
    return "low";
  }
  if (percent < 67) {
    return "medium";
  }
  return "high";
}

function impactLevelLabel(level: "low" | "medium" | "high" | "none"): string {
  if (level === "low") {
    return "Bajo";
  }
  if (level === "medium") {
    return "Moderado";
  }
  if (level === "high") {
    return "Alto";
  }
  return "Sin estimacion";
}

function shortLevelLabel(value: unknown): string {
  if (value === "low") return "Bajo";
  if (value === "medium") return "Moderado";
  if (value === "high") return "Alto";
  return "Sin dato";
}

function environmentalColorLabel(level: unknown): string {
  if (level === "high") return "rojo";
  if (level === "medium") return "naranjo";
  return "verde";
}

export function levelRangeLabel(value: unknown): string {
  if (value === "low") return "menos de 35%";
  if (value === "medium") return "entre 35% y menos de 65%";
  if (value === "high") return "65% o mas";
  return "sin rango disponible";
}

export function congestionMeaning(level: unknown, lagHours: unknown): string {
  const previous = Number(lagHours) > 0;
  if (level === "high") {
    return previous
      ? "Hubo mucho taco antes de la hora seleccionada. Su efecto disminuye con el tiempo."
      : "Hay mucho taco en este tramo. El tránsito es muy lento o la demora es prolongada.";
  }
  if (level === "medium") {
    return previous
      ? "Hubo taco recientemente, aunque su efecto ya es menor."
      : "Hay taco moderado y el viaje puede tardar más de lo habitual.";
  }
  return previous
    ? "El taco ocurrió antes y tiene poca influencia en la hora seleccionada."
    : "Hay algo de congestión, pero el tránsito mantiene mayor fluidez.";
}

function congestionContextExplanation(
  speed: unknown,
  duration: unknown,
  lagHours: unknown,
): string {
  const speedValue = Number(speed);
  const durationValue = Number(duration);
  const parts: string[] = [];
  if (Number.isFinite(speedValue)) {
    parts.push(`En este tramo los vehículos avanzan cerca de ${speedValue.toFixed(1)} km/h`);
  }
  if (Number.isFinite(durationValue)) {
    parts.push(`esta condición de congestión se mantuvo registrada durante aproximadamente ${durationValue.toFixed(0)} minutos`);
  }
  const message = parts.length ? `${parts.join(" y ")}.` : "No hay más detalles disponibles para este tramo.";
  const previous = Number(lagHours);
  return Number.isFinite(previous) && previous > 0
    ? `${message} Este registro corresponde a ${previous} h antes de la hora seleccionada.`
    : message;
}

export function environmentalMeaning(level: unknown): string {
  if (level === "high") {
    return "La congestión y las condiciones del aire generan una presión ambiental potencial alta en esta zona.";
  }
  if (level === "medium") {
    return "La combinación de tráfico, PM2.5 y ventilación produce una presión ambiental potencial moderada.";
  }
  return "El aporte estimado del tráfico y las condiciones ambientales producen una presión potencial baja.";
}

function combinedLayerExplanation(congestionLevel: unknown, environmentalLevel: unknown): string {
  if (congestionLevel === "high" && environmentalLevel !== "high") {
    return "Hay taco importante, pero las condiciones ambientales reducen parte de su impacto potencial.";
  }
  if (congestionLevel === "low" && environmentalLevel !== "low") {
    return "La congestion es baja, pero PM2.5 o baja dispersion mantienen un impacto potencial relevante.";
  }
  if (congestionLevel === "high" && environmentalLevel === "high") {
    return "Coinciden congestion alta y condiciones ambientales desfavorables.";
  }
  if (congestionLevel === "low" && environmentalLevel === "low") {
    return "Coinciden congestion baja y condiciones ambientales favorables.";
  }
  return "El trafico y el impacto ambiental potencial se leen por separado.";
}

function formatConditionValue(value?: number | null, unit = ""): string {
  if (value == null) {
    return "Sin dato";
  }
  return `${value.toFixed(1)}${unit ? ` ${unit}` : ""}`;
}

function relativeConditionLabel(value?: number | null, min?: number | null, max?: number | null): string {
  if (value == null || min == null || max == null || max <= min) return "sin comparación disponible";
  const position = (value - min) / (max - min);
  if (position < 0.35) return "bajo para este horario";
  if (position < 0.65) return "medio para este horario";
  return "alto para este horario";
}

function environmentalContextExplanation(
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
  congestionLevel: unknown,
): string[] {
  const messages: string[] = [];
  const congestion = shortLevelLabel(congestionLevel).toLowerCase();
  if (congestionLevel) {
    messages.push(`La congestión asociada es ${congestion}, por lo que aporta emisiones del tráfico a esta zona.`);
  }

  if (weather?.pm25 != null) {
    const comparison = relativeConditionLabel(weather.pm25, weather.pm25_min, weather.pm25_max);
    messages.push(
      `El PM2.5 es de ${weather.pm25.toFixed(1)} µg/m³ (${comparison}, según los datos locales disponibles).`,
    );
  }

  if (weather?.wind_speed_kmh != null) {
    const comparison = relativeConditionLabel(
      weather.wind_speed_kmh,
      weather.wind_speed_min_kmh,
      weather.wind_speed_max_kmh,
    );
    const effect = comparison.startsWith("alto")
      ? "Este viento ayuda a dispersar las partículas."
      : comparison.startsWith("medio")
        ? "Este viento ayuda a dispersar parte de las partículas."
        : "Al haber poco viento, las partículas pueden permanecer más concentradas cerca del tráfico.";
    messages.push(`El viento es de ${weather.wind_speed_kmh.toFixed(1)} km/h (${comparison}). ${effect}`);
  }

  if (weather?.has_rain) {
    messages.push(`${weather.rain_label}: la lluvia puede ayudar a reducir la concentración de partículas.`);
  } else if (weather?.rain_label === "Sin lluvia") {
    messages.push("No hay lluvia que ayude a reducir la concentración de partículas.");
  }
  return messages;
}

export function PlanningMap({
  enabled,
  styleUrl,
  mapboxToken,
  routes,
  selectedRouteKey,
  hotspots,
  environmentalImpact,
  environmentalImpactLoading = false,
  showImpactCard = true,
  cycleways,
  wellbeingFeatures,
  inspectMode,
  showCycleways,
  wellbeingVisibility,
  origin,
  destination,
  activePin,
  onPickPoint,
  onMarkerDrag,
}: PlanningMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const impactCardRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<any>(null);
  const moduleRef = useRef<any>(null);
  const markersRef = useRef<Record<PinKey, any | null>>({ origin: null, destination: null });
  const environmentalPopupRef = useRef<any>(null);
  const activePinRef = useRef<PinKey>(activePin);
  const inspectModeRef = useRef(false);
  const mapReadyRef = useRef(false);
  const overlayFrameRef = useRef<number | null>(null);
  const bearingFrameRef = useRef<number | null>(null);
  const lastFitBoundsKeyRef = useRef<string | null>(null);
  const impactDragRef = useRef<PanelDragState | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [mapStyleRevision, setMapStyleRevision] = useState(0);
  const [mapBearing, setMapBearing] = useState(0);
  const [viewRevision, setViewRevision] = useState(0);
  const [routeTooltip, setRouteTooltip] = useState<RouteTooltipState | null>(null);
  const [layerTooltip, setLayerTooltip] = useState<LayerTooltipState | null>(null);
  const [selectedLayerInfo, setSelectedLayerInfo] = useState<SelectedLayerInfo | null>(null);
  const [impactCardCollapsed, setImpactCardCollapsed] = useState(false);
  const [impactInfoOpen, setImpactInfoOpen] = useState(false);
  const [impactCardPosition, setImpactCardPosition] = useState<PanelPosition | null>(null);
  const [environmentalLayerStatus, setEnvironmentalLayerStatus] = useState<"loading" | "ready" | "empty" | "error">(
    "loading",
  );
  const [environmentalLayerError, setEnvironmentalLayerError] = useState<string | null>(null);
  const selectedRoute = routes.find((route) => route.key === selectedRouteKey) ?? routes[0] ?? null;
  const selectedWellbeingFeatureIds = wellbeingEvidenceFeatureIds(selectedRoute);
  const selectedWellbeingFeatureIdSet = new Set(selectedWellbeingFeatureIds);
  const overlayWidth = containerRef.current?.clientWidth ?? 0;
  const overlayHeight = containerRef.current?.clientHeight ?? 0;
  const tooltipLeft = Math.min(Math.max((routeTooltip?.x ?? 0) + 14, 12), Math.max(12, overlayWidth - 250));
  const tooltipTop = Math.min(Math.max((routeTooltip?.y ?? 0) + 14, 12), Math.max(12, overlayHeight - 156));
  const layerTooltipLeft = Math.min(Math.max((layerTooltip?.x ?? 0) + 16, 12), Math.max(12, overlayWidth - 330));
  const layerTooltipTop = Math.min(Math.max((layerTooltip?.y ?? 0) + 16, 12), Math.max(12, overlayHeight - 245));
  const overlapOffsets = routeOverlapOffsets(routes);
  const projectionMap = mapReady ? mapRef.current : null;
  const projectedRoutes =
    projectionMap
      ? routes
          .map((route, index) => {
            const visibleGeometry = routeRoadGeometry(route);
            return {
              color: routeColor(route, index),
              offset: overlapOffsets.get(route.key) ?? 0,
              path: projectRoutePath(projectionMap, visibleGeometry, overlapOffsets.get(route.key) ?? 0),
              route,
              selected: selectedRouteKey ? route.key === selectedRouteKey : true,
            };
          })
          .filter((item) => item.path)
          .sort((a, b) => Number(a.selected) - Number(b.selected))
      : [];
  const projectedRouteVisuals =
    projectionMap
      ? routes
          .map((route, index) => {
            const offset = overlapOffsets.get(route.key) ?? 0;
            const selected = selectedRouteKey ? route.key === selectedRouteKey : true;
            const mainGeometry = routeRoadGeometry(route);
            return {
              accessPaths: routeAccessGeometry(route)
                .map((segment) => projectRoutePath(projectionMap, segment, offset))
                .filter(Boolean),
              color: routeColor(route, index),
              mainPath: projectRoutePath(projectionMap, mainGeometry, offset),
              route,
              selected,
            };
          })
          .filter((item) => item.mainPath)
          .sort((a, b) => Number(a.selected) - Number(b.selected))
      : [];
  const projectedHotspots =
    projectionMap && !environmentalImpact?.summary.available
      ? hotspots.map((point, index) => {
          const projected = projectionMap.project([point.lon, point.lat]);
          return {
            key: `${point.segment_id ?? "hotspot"}-${index}`,
            r: hotspotRadius(point.weight),
            x: projected.x,
            y: projected.y,
          };
        })
      : [];
  const projectedCycleways =
    projectionMap
      ? cycleways
          .flatMap((feature, featureIndex) => {
            const featureId = cyclewayFeatureId(feature, featureIndex);
            const evidence = selectedWellbeingFeatureIdSet.has(featureId);
            if (!showCycleways && !evidence) {
              return [];
            }
            const coordinateSegments = evidence
              ? cyclewayEvidenceSegments(feature, selectedRoute)
              : [feature.geometry.coordinates];
            return {
              key: featureId,
              category: feature.properties.category ?? "cycling_infrastructure",
              evidence,
              paths: coordinateSegments.map((coordinates) => projectCoordinatePath(projectionMap, coordinates)),
            };
          })
          .flatMap((item) =>
            item.paths
              .filter(Boolean)
              .map((path, pathIndex) => ({
                key: `${item.key}-${pathIndex}`,
                category: item.category,
                evidence: item.evidence,
                path,
              })),
          )
      : [];
  const projectedWellbeingPaths =
    projectionMap
      ? wellbeingFeatures.flatMap((feature, featureIndex) => {
          const category = feature.properties.category ?? "green_space";
          const featureId = feature.properties.feature_id ?? "";
          const evidence = Boolean(featureId && selectedWellbeingFeatureIdSet.has(featureId));
          if (!wellbeingVisibility[category] && !evidence) {
            return [];
          }
          const coordinates = feature.geometry.coordinates;
          if (isCoordinatePair(coordinates)) {
            return [];
          }
          return projectCoordinateTree(projectionMap, coordinates, feature.geometry.type.includes("Polygon")).map(
            (path, pathIndex) => ({
              key: `wellbeing-${featureIndex}-${pathIndex}`,
              category,
              evidence,
              path,
            }),
          );
        })
      : [];
  const projectedWellbeingPoints =
    projectionMap
      ? wellbeingFeatures.flatMap((feature, featureIndex) => {
          const category = feature.properties.category ?? "sustainability";
          const featureId = feature.properties.feature_id ?? "";
          const evidence = Boolean(featureId && selectedWellbeingFeatureIdSet.has(featureId));
          if (!wellbeingVisibility[category] && !evidence) {
            return [];
          }
          if (!isCoordinatePair(feature.geometry.coordinates)) {
            return [];
          }
          const projected = projectionMap.project(feature.geometry.coordinates);
          return [
            {
              key: `wellbeing-point-${featureIndex}`,
              category,
              evidence,
              x: projected.x,
              y: projected.y,
            },
          ];
        })
      : [];
  const projectedEnvironmentalZones =
    projectionMap && environmentalImpact?.summary.available
      ? (environmentalImpact.zones?.features ?? []).flatMap((feature, featureIndex) =>
          projectCoordinateTree(projectionMap, feature.geometry.coordinates, true).map((path, pathIndex) => ({
            key: `environment-zone-${featureIndex}-${pathIndex}`,
            path,
            levelClass: impactClassFromLevel(feature.properties.level),
            properties: feature.properties,
            selected: selectedLayerInfo?.environment?.zone_id === feature.properties.zone_id,
          })),
        )
      : [];
  const projectedEnvironmentalLines =
    projectionMap && environmentalImpact?.summary.available
      ? (environmentalImpact.congestion_lines?.features ?? []).flatMap((feature, featureIndex) =>
          projectCoordinateTree(projectionMap, feature.geometry.coordinates).map((path, pathIndex) => ({
            key: `congestion-line-${featureIndex}-${pathIndex}`,
            path,
            levelClass: impactClassFromLevel(feature.properties.level),
            recent: feature.properties.recency === "reciente",
            properties: feature.properties,
          })),
        )
      : [];
  const weather = environmentalImpact?.summary.weather;
  const pm25Percent = localRangePercent(weather?.pm25, weather?.pm25_min, weather?.pm25_max);
  const windPercent = localRangePercent(
    weather?.wind_speed_kmh,
    weather?.wind_speed_min_kmh,
    weather?.wind_speed_max_kmh,
  );
  const dominantLevel = environmentalImpact?.summary.dominant_level ?? "none";
  const cardLevel = dominantLevel === "none" ? environmentalCardLevel(pm25Percent) : dominantLevel;
  const impactCardStyle = impactCardPosition
    ? { left: impactCardPosition.left, right: "auto", top: impactCardPosition.top }
    : undefined;
  const layerPopupStyle = selectedLayerInfo
    ? (() => {
        const margin = 12;
        const gap = 14;
        const width = Math.min(440, Math.max(300, overlayWidth - margin * 2));
        const estimatedHeight = 350;
        let left = selectedLayerInfo.anchor.x + gap;
        let top = selectedLayerInfo.anchor.y + gap;
        if (left + width > overlayWidth - margin) {
          left = selectedLayerInfo.anchor.x - width - gap;
        }
        if (top + estimatedHeight > overlayHeight - margin) {
          top = selectedLayerInfo.anchor.y - estimatedHeight - gap;
        }
        return {
          left: Math.max(margin, left),
          right: "auto",
          top: Math.max(margin, top),
          bottom: "auto",
          width,
        };
      })()
    : undefined;

  void viewRevision;

  const handlePickPoint = useEffectEvent((pin: PinKey, point: RoutePoint) => {
    onPickPoint(pin, point);
  });

  const handleMarkerDrag = useEffectEvent((pin: PinKey, point: RoutePoint) => {
    onMarkerDrag(pin, point);
  });

  function updateRouteTooltip(route: PlanRouteCard, event: PointerEvent<SVGPathElement>) {
    const bounds = containerRef.current?.getBoundingClientRect();
    if (!bounds) {
      return;
    }
    setRouteTooltip({
      route,
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });
  }

  function updateLayerTooltipFromPointer(event: PointerEvent<HTMLDivElement>) {
    if (!inspectMode) {
      return;
    }
    const map = mapRef.current;
    const bounds = containerRef.current?.getBoundingClientRect();
    if (!map || !mapReady || !bounds) {
      return;
    }
    const point: [number, number] = [event.clientX - bounds.left, event.clientY - bounds.top];
    const layers = [ENVIRONMENTAL_CONGESTION_LINE_LAYER, ENVIRONMENTAL_ZONE_FILL_LAYER].filter((layerId) =>
      map.getLayer(layerId),
    );
    if (!layers.length) {
      setLayerTooltip(null);
      return;
    }
    const features = map.queryRenderedFeatures(environmentalQueryBox(point), { layers }) ?? [];
    const feature =
      features.find((candidate: any) => candidate.layer?.id === ENVIRONMENTAL_CONGESTION_LINE_LAYER) ??
      features.find((candidate: any) => candidate.layer?.id === ENVIRONMENTAL_ZONE_FILL_LAYER);
    if (!feature) {
      map.getCanvas().style.cursor = "";
      setLayerTooltip(null);
      return;
    }
    map.getCanvas().style.cursor = "pointer";
    setLayerTooltip({
      kind: feature.layer?.id === ENVIRONMENTAL_CONGESTION_LINE_LAYER ? "congestion" : "environment",
      properties: feature.properties ?? {},
      x: point[0],
      y: point[1],
    });
  }

  function selectLayerInfoFromClick(event: MouseEvent<HTMLDivElement>) {
    if (!inspectMode) {
      return;
    }
    if (
      (event.target as HTMLElement).closest(
        "button, .map-impact-card, .layer-details-panel, .map-guide-panel, .maplibregl-popup",
      )
    ) {
      return;
    }
    const map = mapRef.current;
    const bounds = containerRef.current?.getBoundingClientRect();
    if (!map || !mapReady || !bounds) {
      return;
    }
    const point: [number, number] = [event.clientX - bounds.left, event.clientY - bounds.top];
    const layers = [ENVIRONMENTAL_CONGESTION_LINE_LAYER, ENVIRONMENTAL_ZONE_FILL_LAYER].filter((layerId) =>
      map.getLayer(layerId),
    );
    if (!layers.length) {
      return;
    }
    const features = map.queryRenderedFeatures(environmentalQueryBox(point, ENVIRONMENTAL_CLICK_TOLERANCE_PX), {
      layers,
    }) ?? [];
    const congestion = features.find((feature: any) => feature.layer?.id === ENVIRONMENTAL_CONGESTION_LINE_LAYER);
    const environment = features.find((feature: any) => feature.layer?.id === ENVIRONMENTAL_ZONE_FILL_LAYER);
    if (!congestion && !environment) {
      setSelectedLayerInfo(null);
      return;
    }
    setSelectedLayerInfo({
      congestion: congestion?.properties ?? null,
      environment: environment?.properties ?? null,
      anchor: { x: point[0], y: point[1] },
    });
  }

  function selectedLayerAnchor(event: MouseEvent<SVGPathElement>) {
    const bounds = containerRef.current?.getBoundingClientRect();
    return bounds
      ? { x: event.clientX - bounds.left, y: event.clientY - bounds.top }
      : { x: 18, y: 18 };
  }

  function selectEnvironmentalLine(properties: Record<string, unknown>, event: MouseEvent<SVGPathElement>) {
    const segmentId = String(properties.segment_id ?? "");
    const environment =
      environmentalImpact?.zones?.features.find(
        (feature) =>
          Array.isArray(feature.properties.segment_ids) && feature.properties.segment_ids.includes(segmentId),
      )?.properties ??
      environmentalImpact?.zones?.features.find(
        (feature) => feature.properties.level === properties.environmental_level,
      )?.properties ?? null;
    setSelectedLayerInfo({ congestion: properties, environment, anchor: selectedLayerAnchor(event) });
    setLayerTooltip(null);
  }

  function selectEnvironmentalZone(properties: Record<string, unknown>, event: MouseEvent<SVGPathElement>) {
    setSelectedLayerInfo({ congestion: null, environment: properties, anchor: selectedLayerAnchor(event) });
    setLayerTooltip(null);
  }

  function clampImpactCardPosition(position: PanelPosition): PanelPosition {
    const container = containerRef.current;
    const card = impactCardRef.current;
    if (!container || !card) {
      return position;
    }
    const margin = 12;
    const maxLeft = Math.max(margin, container.clientWidth - card.offsetWidth - margin);
    const maxTop = Math.max(margin, container.clientHeight - card.offsetHeight - margin);
    return {
      left: Math.min(Math.max(position.left, margin), maxLeft),
      top: Math.min(Math.max(position.top, margin), maxTop),
    };
  }

  function moveImpactCard(clientX: number, clientY: number) {
    const container = containerRef.current;
    const drag = impactDragRef.current;
    if (!container || !drag) {
      return;
    }
    const bounds = container.getBoundingClientRect();
    setImpactCardPosition(
      clampImpactCardPosition({
        left: clientX - bounds.left - drag.offsetX,
        top: clientY - bounds.top - drag.offsetY,
      }),
    );
  }

  function startImpactCardDrag(event: PointerEvent<HTMLDivElement>) {
    if ((event.target as HTMLElement).closest("button")) {
      return;
    }
    const container = containerRef.current;
    const card = impactCardRef.current;
    if (!container || !card) {
      return;
    }
    const containerBounds = container.getBoundingClientRect();
    const cardBounds = card.getBoundingClientRect();
    impactDragRef.current = {
      pointerId: event.pointerId,
      offsetX: event.clientX - cardBounds.left,
      offsetY: event.clientY - cardBounds.top,
    };
    setImpactCardPosition(
      clampImpactCardPosition({
        left: cardBounds.left - containerBounds.left,
        top: cardBounds.top - containerBounds.top,
      }),
    );
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function dragImpactCard(event: PointerEvent<HTMLDivElement>) {
    if (impactDragRef.current?.pointerId !== event.pointerId) {
      return;
    }
    moveImpactCard(event.clientX, event.clientY);
  }

  function stopImpactCardDrag(event: PointerEvent<HTMLDivElement>) {
    if (impactDragRef.current?.pointerId !== event.pointerId) {
      return;
    }
    impactDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function resetNorth() {
    mapRef.current?.easeTo({ bearing: 0, pitch: 0, duration: 350 });
  }

  useEffect(() => {
    activePinRef.current = activePin;
  }, [activePin]);

  useEffect(() => {
    inspectModeRef.current = inspectMode;
    if (!inspectMode) {
      setLayerTooltip(null);
      setSelectedLayerInfo(null);
      if (mapRef.current) {
        mapRef.current.getCanvas().style.cursor = "";
      }
    }
  }, [inspectMode]);

  useEffect(() => {
    setViewRevision((current) => current + 1);
  }, [cycleways, environmentalImpact, hotspots, routes, selectedRouteKey, showCycleways, wellbeingFeatures, wellbeingVisibility]);

  useEffect(() => {
    if (!impactCardPosition) {
      return;
    }
    function clampOnResize() {
      setImpactCardPosition((current) => (current ? clampImpactCardPosition(current) : current));
    }
    window.addEventListener("resize", clampOnResize);
    return () => {
      window.removeEventListener("resize", clampOnResize);
    };
  }, [impactCardPosition]);

  useEffect(() => {
    setImpactCardPosition((current) => (current ? clampImpactCardPosition(current) : current));
  }, [impactCardCollapsed]);

  useEffect(() => {
    if (!enabled || !containerRef.current || mapRef.current) {
      return;
    }
    let cancelled = false;
    let usingFallbackStyle = false;

    async function setupMap() {
      try {
        const refreshOverlay = () => {
          if (overlayFrameRef.current !== null) {
            return;
          }
          overlayFrameRef.current = window.requestAnimationFrame(() => {
            overlayFrameRef.current = null;
            setViewRevision((current) => current + 1);
          });
        };
        const refreshBearing = () => {
          if (bearingFrameRef.current !== null) {
            return;
          }
          bearingFrameRef.current = window.requestAnimationFrame(() => {
            bearingFrameRef.current = null;
            const bearing = mapRef.current?.getBearing?.() ?? 0;
            setMapBearing(Number((((bearing % 360) + 360) % 360).toFixed(1)));
          });
        };

        const maplibre = await import("maplibre-gl");
        if (cancelled || !containerRef.current) {
          return;
        }
        moduleRef.current = maplibre;
        const map = new maplibre.Map({
          container: containerRef.current,
          style: normalizeStyleUrl(styleUrl, mapboxToken),
          center: [-73.05, -36.82],
          zoom: 13.2,
          maxZoom: 18,
          attributionControl: false,
          fadeDuration: 80,
          refreshExpiredTiles: false,
          transformRequest: (url: string) => ({ url: addMapboxToken(url, mapboxToken) }),
        });
        mapRef.current = map;
        map.on("load", () => {
          mapReadyRef.current = true;
          setMapReady(true);
          refreshBearing();
          refreshOverlay();
        });
        map.on("style.load", () => {
          if (!cancelled) {
            setMapStyleRevision((current) => current + 1);
            refreshOverlay();
          }
        });
        map.on("move", refreshOverlay);
        map.on("moveend", refreshOverlay);
        map.on("zoom", refreshOverlay);
        map.on("zoomend", refreshOverlay);
        map.on("rotate", refreshOverlay);
        map.on("rotate", refreshBearing);
        map.on("rotateend", refreshBearing);
        map.on("resize", refreshOverlay);
        map.on("click", (event: any) => {
          const interactiveLayers = [
            ENVIRONMENTAL_ZONE_FILL_LAYER,
            ENVIRONMENTAL_CONGESTION_LINE_LAYER,
          ].filter(
            (layerId) => map.getLayer(layerId),
          );
          const environmentalFeatureCount = interactiveLayers.length
            ? map.queryRenderedFeatures(event.point, { layers: interactiveLayers }).length
            : 0;
          if (!shouldPickMapPoint(inspectModeRef.current, environmentalFeatureCount)) {
            return;
          }
          handlePickPoint(activePinRef.current, { lat: event.lngLat.lat, lon: event.lngLat.lng });
        });
        map.on("error", (event: any) => {
          if (!mapReadyRef.current && !usingFallbackStyle && shouldFallbackToLocalStyle(event)) {
            usingFallbackStyle = true;
            setMapError(null);
            try {
              map.setStyle(localBasicStyle());
            } catch (fallbackError) {
              console.warn("No se pudo aplicar el estilo local de respaldo.", fallbackError);
            }
            return;
          }
          if (isRecoverableMapError(event)) {
            return;
          }
          console.warn("MapLibre emitio un error no bloqueante.", event?.error ?? event);
          setMapError(null);
        });
      } catch (error) {
        setMapError(error instanceof Error ? error.message : "No se pudo inicializar el mapa.");
      }
    }

    setupMap();

    return () => {
      cancelled = true;
      mapReadyRef.current = false;
      setMapReady(false);
      setRouteTooltip(null);
      markersRef.current.origin?.remove();
      markersRef.current.destination?.remove();
      environmentalPopupRef.current?.remove();
      environmentalPopupRef.current = null;
      markersRef.current = { origin: null, destination: null };
      if (overlayFrameRef.current !== null) {
        window.cancelAnimationFrame(overlayFrameRef.current);
        overlayFrameRef.current = null;
      }
      if (bearingFrameRef.current !== null) {
        window.cancelAnimationFrame(bearingFrameRef.current);
        bearingFrameRef.current = null;
      }
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, [enabled, mapboxToken, styleUrl]);

  useEffect(() => {
    if (!enabled || !mapRef.current || !moduleRef.current) {
      return;
    }
    const { Marker } = moduleRef.current;
    (["origin", "destination"] as const).forEach((pin) => {
      const point = pin === "origin" ? origin : destination;
      if (!point) {
        markersRef.current[pin]?.remove();
        markersRef.current[pin] = null;
        return;
      }
      const marker =
        markersRef.current[pin] ?? new Marker({ color: pin === "origin" ? "#0f766e" : "#dc2626", draggable: true });
      marker.setLngLat([point.lon, point.lat]);
      if (!markersRef.current[pin]) {
        marker.addTo(mapRef.current);
        marker.on("dragend", () => {
          const lngLat = marker.getLngLat();
          handleMarkerDrag(pin, { lat: lngLat.lat, lon: lngLat.lng });
        });
      }
      markersRef.current[pin] = marker;
    });
  }, [destination, enabled, origin]);

  useEffect(() => {
    const map = mapRef.current;
    if (!enabled || !mapReady || !map) {
      return;
    }

    try {
      const data = buildEnvironmentalZoneData(environmentalImpact);
      const lineData = buildEnvironmentalCongestionLineData(environmentalImpact);
      const pointData = buildEnvironmentalPointData(environmentalImpact);
      setEnvironmentalLayerStatus("loading");
      setEnvironmentalLayerError(null);

      const initialPointSource = map.getSource(ENVIRONMENTAL_POINT_SOURCE);
      if (initialPointSource?.setData) {
        initialPointSource.setData(pointData);
      } else {
        map.addSource(ENVIRONMENTAL_POINT_SOURCE, {
          type: "geojson",
          data: pointData,
        });
      }
      if (!map.getLayer(ENVIRONMENTAL_HEATMAP_LAYER)) {
        map.addLayer({
          id: ENVIRONMENTAL_HEATMAP_LAYER,
          type: "heatmap",
          source: ENVIRONMENTAL_POINT_SOURCE,
          maxzoom: 16,
          paint: {
            "heatmap-weight": ["interpolate", ["linear"], ["get", "score"], 0, 0.25, 100, 1],
            "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 8, 0.9, 15, 1.5],
            "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 8, 18, 15, 34],
            "heatmap-opacity": 0.001,
            "heatmap-color": [
              "interpolate",
              ["linear"],
              ["heatmap-density"],
              0,
              "rgba(34,197,94,0)",
              0.25,
              "rgba(34,197,94,0.7)",
              0.5,
              "rgba(250,204,21,0.78)",
              0.75,
              "rgba(249,115,22,0.82)",
              1,
              "rgba(220,38,38,0.88)",
            ],
          },
        });
      }
      setEnvironmentalLayerStatus(pointData.features.length ? "ready" : "empty");

      const existingSource = map.getSource(ENVIRONMENTAL_ZONE_SOURCE);
      if (existingSource?.setData) {
        existingSource.setData(data);
      } else {
        map.addSource(ENVIRONMENTAL_ZONE_SOURCE, {
          type: "geojson",
          data,
        });
      }
      const existingLineSource = map.getSource(ENVIRONMENTAL_CONGESTION_LINE_SOURCE);
      if (existingLineSource?.setData) {
        existingLineSource.setData(lineData);
      } else {
        map.addSource(ENVIRONMENTAL_CONGESTION_LINE_SOURCE, {
          type: "geojson",
          data: lineData,
        });
      }
      const existingPointSource = map.getSource(ENVIRONMENTAL_POINT_SOURCE);
      if (existingPointSource?.setData) {
        existingPointSource.setData(pointData);
      } else {
        map.addSource(ENVIRONMENTAL_POINT_SOURCE, {
          type: "geojson",
          data: pointData,
        });
      }

      if (!map.getLayer(ENVIRONMENTAL_ZONE_FILL_LAYER)) {
        map.addLayer({
          id: ENVIRONMENTAL_ZONE_FILL_LAYER,
          type: "fill",
          source: ENVIRONMENTAL_ZONE_SOURCE,
          paint: {
            "fill-color": [
              "match",
              ["get", "level"],
              "low",
              "#16a34a",
              "medium",
              "#f59e0b",
              "high",
              "#dc2626",
              "#64748b",
            ],
            "fill-opacity": 0.001,
            "fill-antialias": true,
          },
        });
      }

      if (!map.getLayer(ENVIRONMENTAL_ZONE_OUTLINE_LAYER)) {
        map.addLayer({
          id: ENVIRONMENTAL_ZONE_OUTLINE_LAYER,
          type: "line",
          source: ENVIRONMENTAL_ZONE_SOURCE,
          paint: {
            "line-color": [
              "match",
              ["get", "level"],
              "low",
              "#15803d",
              "medium",
              "#d97706",
              "high",
              "#b91c1c",
              "#475569",
            ],
            "line-opacity": 0.001,
            "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.1, 15, 2.2],
          },
        });
      }

      if (!map.getLayer(ENVIRONMENTAL_CONGESTION_LINE_CASING_LAYER)) {
        map.addLayer({
          id: ENVIRONMENTAL_CONGESTION_LINE_CASING_LAYER,
          type: "line",
          source: ENVIRONMENTAL_CONGESTION_LINE_SOURCE,
          paint: {
            "line-color": "#111827",
            "line-opacity": 0.001,
            "line-width": ["interpolate", ["linear"], ["zoom"], 10, 5.2, 15, 9.2],
          },
          layout: {
            "line-cap": "round",
            "line-join": "round",
          },
        });
      }

      if (!map.getLayer(ENVIRONMENTAL_CONGESTION_LINE_LAYER)) {
        map.addLayer({
          id: ENVIRONMENTAL_CONGESTION_LINE_LAYER,
          type: "line",
          source: ENVIRONMENTAL_CONGESTION_LINE_SOURCE,
          paint: {
            "line-color": [
              "match",
              ["get", "level"],
              "low",
              "#22c55e",
              "medium",
              "#f59e0b",
              "high",
              "#ef4444",
              "#64748b",
            ],
            "line-opacity": 0.001,
            "line-width": ["interpolate", ["linear"], ["zoom"], 10, 3.2, 15, 6.2],
          },
          layout: {
            "line-cap": "round",
            "line-join": "round",
          },
        });
      }
      if (map.getLayer(ENVIRONMENTAL_HEATMAP_LAYER)) {
        map.moveLayer(ENVIRONMENTAL_HEATMAP_LAYER);
      }
      if (map.getLayer(ENVIRONMENTAL_ZONE_FILL_LAYER)) {
        map.moveLayer(ENVIRONMENTAL_ZONE_FILL_LAYER);
      }
      if (map.getLayer(ENVIRONMENTAL_ZONE_OUTLINE_LAYER)) {
        map.moveLayer(ENVIRONMENTAL_ZONE_OUTLINE_LAYER);
      }
      if (map.getLayer(ENVIRONMENTAL_CONGESTION_LINE_CASING_LAYER)) {
        map.moveLayer(ENVIRONMENTAL_CONGESTION_LINE_CASING_LAYER);
      }
      if (map.getLayer(ENVIRONMENTAL_CONGESTION_LINE_LAYER)) {
        map.moveLayer(ENVIRONMENTAL_CONGESTION_LINE_LAYER);
      }
      const showEnvironmentalPopup = (event: any) => {
        const feature = event.features?.[0];
        if (!feature || !moduleRef.current?.Popup) {
          return;
        }
        const properties = feature.properties ?? {};
        const content = document.createElement("div");
        content.className = "environmental-popup-content";
        const title = document.createElement("strong");
        const isCongestionLine = properties.layer_kind === "congestion";
        title.textContent =
          properties.via || (isCongestionLine ? "Congestion historica del tramo" : "Impacto ambiental potencial");
        content.appendChild(title);
        [
          [isCongestionLine ? "Congestion" : "Impacto potencial", properties.level],
          ["Velocidad", properties.speed_kmh != null ? `${Number(properties.speed_kmh).toFixed(1)} km/h` : null],
          ["Duracion", properties.duration_min != null ? `${Number(properties.duration_min).toFixed(1)} min` : null],
          ["Persistencia", properties.lag_hours != null ? `${Number(properties.lag_hours)} h` : null],
          ["Observaciones consolidadas", properties.observation_count],
          ["PM2.5", properties.pm25 != null ? `${Number(properties.pm25).toFixed(1)} ug/m3` : null],
          ["Viento", properties.wind_speed != null ? `${Number(properties.wind_speed).toFixed(1)} m/s` : null],
          ["Lluvia", properties.rain_mm != null ? `${Number(properties.rain_mm).toFixed(1)} mm` : null],
          ["Segmentos", properties.segment_count],
        ].forEach(([label, value]) => {
          if (value == null || value === "") {
            return;
          }
          const row = document.createElement("span");
          row.textContent = `${label}: ${value}`;
          content.appendChild(row);
        });
        if (properties.message) {
          const message = document.createElement("p");
          message.textContent = String(properties.message);
          content.appendChild(message);
        }
        environmentalPopupRef.current?.remove();
        environmentalPopupRef.current = new moduleRef.current.Popup({
          closeButton: true,
          closeOnClick: true,
          maxWidth: "300px",
          offset: 12,
        })
          .setLngLat(event.lngLat)
          .setDOMContent(content)
          .addTo(map);
      };
      const setEnvironmentalCursor = () => {
        map.getCanvas().style.cursor = "pointer";
      };
      const clearEnvironmentalCursor = () => {
        map.getCanvas().style.cursor = "";
        setLayerTooltip(null);
      };
      map.on("click", ENVIRONMENTAL_ZONE_FILL_LAYER, showEnvironmentalPopup);
      map.on("click", ENVIRONMENTAL_CONGESTION_LINE_LAYER, showEnvironmentalPopup);
      map.on("mouseenter", ENVIRONMENTAL_ZONE_FILL_LAYER, setEnvironmentalCursor);
      map.on("mouseenter", ENVIRONMENTAL_CONGESTION_LINE_LAYER, setEnvironmentalCursor);
      map.on("mouseleave", ENVIRONMENTAL_ZONE_FILL_LAYER, clearEnvironmentalCursor);
      map.on("mouseleave", ENVIRONMENTAL_CONGESTION_LINE_LAYER, clearEnvironmentalCursor);

      return () => {
        map.off("click", ENVIRONMENTAL_ZONE_FILL_LAYER, showEnvironmentalPopup);
        map.off("click", ENVIRONMENTAL_CONGESTION_LINE_LAYER, showEnvironmentalPopup);
        map.off("mouseenter", ENVIRONMENTAL_ZONE_FILL_LAYER, setEnvironmentalCursor);
        map.off("mouseenter", ENVIRONMENTAL_CONGESTION_LINE_LAYER, setEnvironmentalCursor);
        map.off("mouseleave", ENVIRONMENTAL_ZONE_FILL_LAYER, clearEnvironmentalCursor);
        map.off("mouseleave", ENVIRONMENTAL_CONGESTION_LINE_LAYER, clearEnvironmentalCursor);
      };
    } catch (error) {
      console.warn("No se pudo renderizar la capa zonal ambiental.", error);
      if (map.getLayer(ENVIRONMENTAL_HEATMAP_LAYER)) {
        setEnvironmentalLayerStatus("ready");
        setEnvironmentalLayerError("Algunos detalles ambientales no pudieron mostrarse.");
      } else {
        setEnvironmentalLayerStatus("error");
        setEnvironmentalLayerError(error instanceof Error ? error.message : "Error desconocido al dibujar la capa.");
      }
    }
  }, [enabled, environmentalImpact, mapReady, mapStyleRevision]);

  useEffect(() => {
    const map = mapRef.current;
    if (!enabled || !mapReady || !map) {
      return;
    }

    try {
      const data = buildPlannedRouteData(routes, selectedRouteKey);
      const existingSource = map.getSource(PLANNED_ROUTE_SOURCE);
      if (existingSource?.setData) {
        existingSource.setData(data.main);
      } else {
        map.addSource(PLANNED_ROUTE_SOURCE, {
          type: "geojson",
          data: data.main,
        });
      }

      const existingAccessSource = map.getSource(PLANNED_ROUTE_ACCESS_SOURCE);
      if (existingAccessSource?.setData) {
        existingAccessSource.setData(data.access);
      } else {
        map.addSource(PLANNED_ROUTE_ACCESS_SOURCE, {
          type: "geojson",
          data: data.access,
        });
      }

      if (!map.getLayer(PLANNED_ROUTE_ACCESS_CASING_LAYER)) {
        map.addLayer({
          id: PLANNED_ROUTE_ACCESS_CASING_LAYER,
          type: "line",
          source: PLANNED_ROUTE_ACCESS_SOURCE,
          layout: {
            "line-cap": "round",
            "line-join": "round",
          },
          paint: {
            "line-color": "#ffffff",
            "line-dasharray": [0.2, 1.4],
            "line-opacity": ["case", ["get", "selected"], 0.9, 0.28],
            "line-width": [
              "interpolate",
              ["linear"],
              ["zoom"],
              10,
              ["case", ["get", "selected"], 4, 3],
              16,
              ["case", ["get", "selected"], 7, 5],
            ],
          },
        });
      }

      if (!map.getLayer(PLANNED_ROUTE_ACCESS_LINE_LAYER)) {
        map.addLayer({
          id: PLANNED_ROUTE_ACCESS_LINE_LAYER,
          type: "line",
          source: PLANNED_ROUTE_ACCESS_SOURCE,
          layout: {
            "line-cap": "round",
            "line-join": "round",
          },
          paint: {
            "line-color": ["get", "color"],
            "line-dasharray": [0.2, 1.4],
            "line-opacity": ["case", ["get", "selected"], 0.78, 0.24],
            "line-width": [
              "interpolate",
              ["linear"],
              ["zoom"],
              10,
              ["case", ["get", "selected"], 1.5, 1],
              16,
              ["case", ["get", "selected"], 2.5, 1.75],
            ],
          },
        });
      }

      if (!map.getLayer(PLANNED_ROUTE_CASING_LAYER)) {
        map.addLayer({
          id: PLANNED_ROUTE_CASING_LAYER,
          type: "line",
          source: PLANNED_ROUTE_SOURCE,
          layout: {
            "line-cap": "round",
            "line-join": "round",
          },
          paint: {
            "line-color": "#ffffff",
            "line-opacity": ["case", ["get", "selected"], 0.96, 0.38],
            "line-width": ["interpolate", ["linear"], ["zoom"], 10, ["case", ["get", "selected"], 8, 5], 16, ["case", ["get", "selected"], 14, 9]],
          },
        });
      }

      if (!map.getLayer(PLANNED_ROUTE_LINE_LAYER)) {
        map.addLayer({
          id: PLANNED_ROUTE_LINE_LAYER,
          type: "line",
          source: PLANNED_ROUTE_SOURCE,
          layout: {
            "line-cap": "round",
            "line-join": "round",
          },
          paint: {
            "line-color": ["get", "color"],
            "line-opacity": ["case", ["get", "selected"], 1, 0.32],
            "line-width": ["interpolate", ["linear"], ["zoom"], 10, ["case", ["get", "selected"], 5, 3], 16, ["case", ["get", "selected"], 9, 6]],
          },
        });
      }

      if (map.getLayer(PLANNED_ROUTE_CASING_LAYER)) {
        map.moveLayer(PLANNED_ROUTE_CASING_LAYER);
      }
      if (map.getLayer(PLANNED_ROUTE_LINE_LAYER)) {
        map.moveLayer(PLANNED_ROUTE_LINE_LAYER);
      }
      if (map.getLayer(PLANNED_ROUTE_ACCESS_CASING_LAYER)) {
        map.moveLayer(PLANNED_ROUTE_ACCESS_CASING_LAYER);
      }
      if (map.getLayer(PLANNED_ROUTE_ACCESS_LINE_LAYER)) {
        map.moveLayer(PLANNED_ROUTE_ACCESS_LINE_LAYER);
      }
    } catch (error) {
      console.warn("No se pudieron dibujar las rutas planificadas.", error);
    }
  }, [enabled, mapReady, mapStyleRevision, routes, selectedRouteKey]);

  useEffect(() => {
    if (!enabled || !mapRef.current || !moduleRef.current) {
      return;
    }
    const boundsSource = routeAutoFitBounds(selectedRoute);
    if (!boundsSource) {
      return;
    }
    const boundsKey = boundsSource.map((value) => value.toFixed(6)).join("|");
    if (lastFitBoundsKeyRef.current === boundsKey) {
      return;
    }
    lastFitBoundsKeyRef.current = boundsKey;
    const [minLon, minLat, maxLon, maxLat] = boundsSource;
    const { LngLatBounds } = moduleRef.current;
    const bounds = new LngLatBounds([minLon, minLat], [maxLon, maxLat]);
    mapRef.current.fitBounds(bounds, { padding: 56, maxZoom: 14, duration: 0 });
    setViewRevision((current) => current + 1);
  }, [enabled, selectedRoute]);

  if (!enabled) {
    return (
      <div className="planner-map fallback-map">
        <div className="map-placeholder">
          <strong>Mapa no disponible</strong>
          <p>Configura `VITE_MAP_STYLE_URL` para habilitar el mapa interactivo.</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`planner-map ${inspectMode ? "inspect-mode" : ""}`}
      onClick={selectLayerInfoFromClick}
      onPointerLeave={() => {
        if (mapRef.current) {
          mapRef.current.getCanvas().style.cursor = "";
        }
        setLayerTooltip(null);
      }}
      onPointerMove={updateLayerTooltipFromPointer}
    >
      <div ref={containerRef} className="planner-map-canvas" />
      {mapReady && overlayWidth > 0 && overlayHeight > 0 ? (
        <svg
          className={`route-svg-overlay ${inspectMode ? "inspect-mode" : ""}`}
          role="img"
          aria-label="Rutas calculadas sobre el mapa"
          viewBox={`0 0 ${overlayWidth} ${overlayHeight}`}
        >
          {projectedHotspots.map((point) => (
            <circle className="hotspot-svg-point" cx={point.x} cy={point.y} key={point.key} r={point.r} />
          ))}
          {projectedCycleways.map((item) => (
            <path
              className={`cycleway-svg-line cycleway-${item.category} ${item.evidence ? "cycleway-evidence" : ""}`}
              d={item.path}
              key={`cycleway-${item.key}`}
            />
          ))}
          {projectedWellbeingPaths.map((item) => (
            <path
              className={`wellbeing-svg-feature wellbeing-${item.category} ${item.evidence ? "wellbeing-evidence" : ""}`}
              d={item.path}
              key={item.key}
            />
          ))}
          {projectedWellbeingPoints.map((item) => (
            <circle
              className={`wellbeing-svg-point wellbeing-${item.category} ${item.evidence ? "wellbeing-evidence" : ""}`}
              cx={item.x}
              cy={item.y}
              key={item.key}
              r={item.evidence ? 7 : item.category === "sustainability" ? 4 : 5}
            />
          ))}
          {projectedEnvironmentalZones.map((item) => (
            <path
              className={`environmental-impact-zone ${item.levelClass} ${item.selected ? "selected" : ""}`}
              d={item.path}
              key={item.key}
            />
          ))}
          {inspectMode
            ? projectedEnvironmentalZones.map((item) => (
                <path
                  className="environmental-zone-hit-area"
                  d={item.path}
                  key={`${item.key}-hit`}
                  onClick={(event) => {
                    event.stopPropagation();
                    selectEnvironmentalZone(item.properties, event);
                  }}
                />
              ))
            : null}
          {projectedEnvironmentalLines.map((item) => (
            <path
              className={`environmental-congestion-line ${item.levelClass} ${item.recent ? "recent" : ""} ${
                selectedLayerInfo?.congestion?.segment_id === item.properties.segment_id ? "selected" : ""
              }`}
              d={item.path}
              key={item.key}
            />
          ))}
          {inspectMode
            ? projectedEnvironmentalLines.map((item) => (
                <path
                  className="environmental-line-hit-area"
                  d={item.path}
                  key={`${item.key}-hit`}
                  onClick={(event) => {
                    event.stopPropagation();
                    selectEnvironmentalLine(item.properties, event);
                  }}
                />
              ))
            : null}
          {projectedRouteVisuals.map((item) => (
            <g key={`${item.route.key}-visual`}>
              {item.mainPath ? (
                <path
                  className={`route-svg-shadow ${item.selected ? "selected" : ""}`}
                  d={item.mainPath}
                />
              ) : null}
              {item.accessPaths.map((path, index) => (
                <path
                  className={`route-svg-access-shadow ${item.selected ? "selected" : ""}`}
                  d={path}
                  key={`${item.route.key}-access-shadow-${index}`}
                />
              ))}
              {item.mainPath ? (
                <path
                  className={`route-svg-line ${item.selected ? "selected" : ""}`}
                  d={item.mainPath}
                  stroke={item.color}
                />
              ) : null}
              {item.accessPaths.map((path, index) => (
                <path
                  className={`route-svg-access ${item.selected ? "selected" : ""}`}
                  d={path}
                  key={`${item.route.key}-access-${index}`}
                  stroke={item.color}
                />
              ))}
            </g>
          ))}
          {projectedRoutes.map((item) => (
            <path
              className="route-svg-hit-area"
              d={item.path}
              key={`${item.route.key}-hit`}
              onPointerEnter={(event) => updateRouteTooltip(item.route, event)}
              onPointerMove={(event) => updateRouteTooltip(item.route, event)}
              onPointerLeave={() => setRouteTooltip(null)}
            />
          ))}
        </svg>
      ) : null}
      {routeTooltip ? (
        <div className="route-tooltip" style={{ left: tooltipLeft, top: tooltipTop }}>
          <strong>{routeTooltip.route.label}</strong>
          <div className="route-tooltip-grid">
            <span>Tiempo</span>
            <b>{routeTooltip.route.duration_min.toFixed(1)} min</b>
            <span>Distancia</span>
            <b>{routeTooltip.route.distance_km.toFixed(2)} km</b>
          </div>
        </div>
      ) : null}
      {layerTooltip ? (
        <div
          className={`layer-tooltip layer-tooltip-${layerTooltip.kind}`}
          role="status"
          style={{ left: layerTooltipLeft, top: layerTooltipTop }}
        >
          {layerTooltip.kind === "congestion" ? (
            <>
              <div className="layer-tooltip-heading">
                <span className="layer-symbol line-symbol" />
                <div>
                  <small>Linea: trafico historico</small>
                  <strong>{String(layerTooltip.properties.via || "Tramo congestionado")}</strong>
                </div>
              </div>
              <div className="layer-tooltip-comparison">
                <span>
                  <small>Congestion</small>
                  <strong>{shortLevelLabel(layerTooltip.properties.level)}</strong>
                </span>
                <span>
                  <small>Nube ambiental</small>
                  <strong>{shortLevelLabel(layerTooltip.properties.environmental_level)}</strong>
                </span>
              </div>
              <div className="layer-tooltip-metrics">
                <span>
                  Velocidad:{" "}
                  <b>
                    {layerTooltip.properties.speed_kmh != null
                      ? `${Number(layerTooltip.properties.speed_kmh).toFixed(1)} km/h`
                      : "Sin dato"}
                  </b>
                </span>
                <span>
                  Duracion:{" "}
                  <b>
                    {layerTooltip.properties.duration_min != null
                      ? `${Number(layerTooltip.properties.duration_min).toFixed(1)} min`
                      : "Sin dato"}
                  </b>
                </span>
                <span>
                  Persistencia: <b>{Number(layerTooltip.properties.lag_hours ?? 0)} h</b>
                </span>
                {Number(layerTooltip.properties.observation_count ?? 1) > 1 ? (
                  <span>
                    Observaciones consolidadas: <b>{Number(layerTooltip.properties.observation_count)}</b>
                  </span>
                ) : null}
              </div>
              <p>
                {combinedLayerExplanation(
                  layerTooltip.properties.level,
                  layerTooltip.properties.environmental_level,
                )}
              </p>
              <small className="layer-tooltip-action">Haz clic para ver mas detalle.</small>
            </>
          ) : (
            <>
              <div className="layer-tooltip-heading">
                <span className="layer-symbol cloud-symbol" />
                <div>
                  <small>Nube: impacto ambiental potencial</small>
                  <strong>{shortLevelLabel(layerTooltip.properties.level)}</strong>
                </div>
              </div>
              <p>
                Aparece alrededor de tramos con congestion y estima su impacto potencial segun PM2.5, viento, lluvia
                y persistencia.
              </p>
              <div className="layer-tooltip-metrics">
                <span>PM2.5 observado: <b>{formatConditionValue(weather?.pm25, "ug/m3")}</b></span>
                <span>Viento: <b>{formatConditionValue(weather?.wind_speed_kmh, "km/h")}</b></span>
                <span>Lluvia: <b>{weather?.rain_label ?? "Sin dato"}</b></span>
              </div>
              <small className="layer-tooltip-action">Haz clic para ver mas detalle.</small>
            </>
          )}
        </div>
      ) : null}
      {selectedLayerInfo ? (
        <aside
          className={`layer-details-panel ${selectedLayerInfo.congestion ? "" : "environment-only"}`}
          aria-label="Detalle contextual de la capa seleccionada"
          style={layerPopupStyle}
        >
          <div className="layer-details-header">
            <div>
              {selectedLayerInfo.congestion ? <small>Información del mapa</small> : null}
              <strong>{selectedLayerInfo.congestion ? "¿Qué ocurre en esta zona?" : "Nube ambiental"}</strong>
            </div>
            <button type="button" aria-label="Cerrar detalle" onClick={() => setSelectedLayerInfo(null)}>
              x
            </button>
          </div>
          <div className="layer-details-grid">
            {selectedLayerInfo.congestion ? <section className="layer-details-section congestion-details">
              <small>Línea de congestión</small>
              <strong>{String(selectedLayerInfo.congestion.via || "Tramo congestionado")}</strong>
              <span className="layer-level-label">Congestión {selectedLayerInfo.congestion.level === "high" ? "alta" : selectedLayerInfo.congestion.level === "medium" ? "media" : "baja"}</span>
              <div className="color-explanation">
                <p>{congestionMeaning(selectedLayerInfo.congestion.level, selectedLayerInfo.congestion.lag_hours)}</p>
                <p>{congestionContextExplanation(selectedLayerInfo.congestion.speed_kmh, selectedLayerInfo.congestion.duration_min, selectedLayerInfo.congestion.lag_hours)}</p>
              </div>
            </section> : null}
            <section className="layer-details-section environment-details">
              {selectedLayerInfo.congestion ? <small>Nube ambiental</small> : null}
              {selectedLayerInfo.environment ? (
                <>
                  <strong>
                    {Array.isArray(selectedLayerInfo.environment.vias) && selectedLayerInfo.environment.vias.length
                      ? selectedLayerInfo.environment.vias.join(", ")
                      : "Zona asociada al tráfico"}
                  </strong>
                  <span className="layer-level-label">{shortLevelLabel(selectedLayerInfo.environment.level)}</span>
                  <div className="color-explanation">
                    <p>{environmentalMeaning(selectedLayerInfo.environment.level)}</p>
                  </div>
                  <div className="layer-simple-metrics environment-metrics">
                    <span><small>Focos actuales</small><b>{Number(selectedLayerInfo.environment.current_focus_count ?? 0)}</b></span>
                    <span><small>Memoria reciente</small><b>{Number(selectedLayerInfo.environment.memory_focus_count ?? 0)}</b></span>
                    <span><small>Presión local</small><b>{formatConditionValue(Number(selectedLayerInfo.environment.score_avg), "%")}</b></span>
                  </div>
                  {Number(selectedLayerInfo.environment.memory_focus_count ?? 0) > 0 ? (
                    <p className="environment-memory-note">
                      Incluye influencia reducida de congestiones finalizadas hasta {Number(selectedLayerInfo.environment.memory_max_lag_hours ?? 1)} h antes. Estas congestiones no se dibujan como líneas.
                    </p>
                  ) : null}
                  <div className="layer-simple-metrics environment-metrics">
                    <span><small>Partículas PM2.5</small><b>{formatConditionValue(weather?.pm25, "µg/m³")}</b><em>{relativeConditionLabel(weather?.pm25, weather?.pm25_min, weather?.pm25_max)}</em></span>
                    <span><small>Viento</small><b>{formatConditionValue(weather?.wind_speed_kmh, "km/h")}</b><em>{relativeConditionLabel(weather?.wind_speed_kmh, weather?.wind_speed_min_kmh, weather?.wind_speed_max_kmh)}</em></span>
                    <span><small>Lluvia</small><b>{weather?.rain_label ?? "Sin dato"}</b></span>
                  </div>
                  <details className="layer-calculation-details">
                    <summary>¿Por qué aparece {environmentalColorLabel(selectedLayerInfo.environment.level)}?</summary>
                    <div className="environment-context-explanation">
                      {environmentalContextExplanation(weather, selectedLayerInfo.congestion?.level).map((message) => (
                        <p key={message}>{message}</p>
                      ))}
                    </div>
                  </details>
                </>
              ) : (
                <span>No hay nube ambiental en este punto.</span>
              )}
            </section>
          </div>
          {selectedLayerInfo.congestion ? (
            <p>
              {combinedLayerExplanation(
                selectedLayerInfo.congestion.level,
                selectedLayerInfo.congestion.environmental_level,
              )}
            </p>
          ) : null}
        </aside>
      ) : null}
      <div className="map-overlay">
        {inspectMode ? <span className="map-chip">Detalles activos: toca una linea o nube</span> : null}
        {mapError ? <span className="map-chip warning">{mapError}</span> : null}
      </div>
      <button
        className="map-compass"
        type="button"
        aria-label={`Orientacion del mapa ${Math.round(mapBearing)} grados. Volver al norte`}
        title="Volver al norte"
        onClick={resetNorth}
      >
        <span className="compass-dial" style={{ transform: `rotate(${-mapBearing}deg)` }}>
          <span className="compass-north">N</span>
          <span className="compass-needle" />
        </span>
        <small>{Math.round(mapBearing)}°</small>
      </button>
      {false ? (
        <aside className="map-guide-panel" aria-label="Explorar mapa">
          <div className="map-guide-content">
            {false ? (
              <>
                <h3>Que significa una linea de congestion</h3>
                <div className="guide-key-message">
                  El color responde: <b>¿que tan importante es esta congestion para la hora seleccionada?</b>
                </div>
                <div className="guide-color-scale">
                  <span className="guide-low"><b>Verde</b> Afectacion menor</span>
                  <span className="guide-medium"><b>Naranjo</b> Congestion relevante</span>
                  <span className="guide-high"><b>Rojo</b> Congestion severa</span>
                </div>
                <p>
                  Una linea puede ser roja porque circula muy lento, porque duro bastante tiempo o por ambas razones.
                  Solo se muestran congestiones activas durante la hora seleccionada.
                </p>
                <ul>
                  <li>Linea continua: congestion activa durante la hora seleccionada.</li>
                  <li>Una linea mas lenta puede tener menor color si duro poco.</li>
                </ul>
                <details className="guide-technical">
                  <summary>Ver criterio tecnico y porcentajes</summary>
                  <p>
                    La app calcula un puntaje de velocidad y otro de duracion, conserva el mayor efecto y aplica el
                    peso temporal: hora actual 100%, una hora anterior 35% y dos horas anteriores 15%.
                  </p>
                  <div className="guide-color-scale">
                    <span className="guide-low">Verde: menos de 35%</span>
                    <span className="guide-medium">Naranjo: 35% a menos de 65%</span>
                    <span className="guide-high">Rojo: 65% o mas</span>
                  </div>
                </details>
                <div className="guide-source-note">
                  <strong>Fuente y alcance</strong>
                  <span>
                    Las lineas provienen de registros historicos procesados por la app. La formula de severidad,
                    cortes de color y pesos temporales son metodologia propia.
                  </span>
                </div>
              </>
            ) : null}
            {false ? (
              <>
                <h3>Que significa una nube ambiental</h3>
                <div className="guide-key-message">
                  La nube responde: <b>¿donde una congestion podria generar mayor exposicion ambiental?</b>
                </div>
                <div className="guide-cause-chain">
                  <span>Congestion</span><b>+</b><span>PM2.5</span><b>+</b><span>Poco viento</span><b>-</b><span>Lluvia</span>
                </div>
                <div className="guide-color-scale">
                  <span className="guide-low"><b>Verde</b> Menor acumulacion potencial</span>
                  <span className="guide-medium"><b>Naranjo</b> Exposicion potencial relevante</span>
                  <span className="guide-high"><b>Rojo</b> Condiciones desfavorables coinciden</span>
                </div>
                <p>
                  No mide emisiones exactas de cada vehiculo ni reemplaza un indice oficial de calidad del aire. Los
                  colores de la nube y de la linea son independientes.
                </p>
                <details className="guide-technical">
                  <summary>Ver calculo tecnico</summary>
                  <div className="guide-weight-grid">
                    <span><b>45%</b> PM2.5 relativo</span>
                    <span><b>30%</b> congestion</span>
                    <span><b>25%</b> baja ventilacion</span>
                  </div>
                  <p>La lluvia resta alivio al puntaje. Los colores usan los cortes 35% y 65%.</p>
                </details>
                <div className="guide-source-note">
                  <strong>Fuente y alcance</strong>
                  <span>
                    La nube, sus ponderaciones y colores son una estimacion propia de la app; no corresponden a un
                    indice oficial de calidad del aire.
                  </span>
                </div>
              </>
            ) : null}
            {false ? (
              <>
                <h3>Como construye la informacion</h3>
                <ol className="guide-process">
                  <li><b>Observa congestion historica:</b> velocidad, duracion, calle y hora.</li>
                  <li><b>Calcula severidad:</b> combina baja velocidad, duracion y antiguedad.</li>
                  <li><b>Lee el ambiente:</b> usa PM2.5 medido, viento y lluvia observados.</li>
                  <li><b>Estima impacto potencial:</b> construye nubes solo alrededor de congestiones existentes.</li>
                </ol>
                <p>
                  La app no inventa congestiones entre calles. La linea es observada; la nube es una estimacion
                  construida a partir de esa congestion y del ambiente.
                </p>
                <div className="guide-reader-help">
                  <strong>Para consultar un elemento</strong>
                  <span>Activa <b>Ver detalles al tocar el mapa</b> y haz clic en una linea o nube.</span>
                </div>
                <div className="guide-source-note">
                  <strong>Datos utilizados en esta vista</strong>
                  <span>{environmentalImpact?.summary.data_source ?? "Fuentes no disponibles para esta seleccion."}</span>
                  <span>
                    El procesamiento, la severidad de congestion y la estimacion ambiental son metodologia propia de
                    esta aplicacion.
                  </span>
                </div>
              </>
            ) : null}
          </div>
        </aside>
      ) : null}
      {showImpactCard && environmentalImpact ? (
        <div
          className={`map-impact-card impact-${cardLevel} ${impactCardCollapsed ? "collapsed" : ""}`}
          ref={impactCardRef}
          style={impactCardStyle}
        >
          <div
            className="map-card-header draggable"
            onPointerCancel={stopImpactCardDrag}
            onPointerDown={startImpactCardDrag}
            onPointerMove={dragImpactCard}
            onPointerUp={stopImpactCardDrag}
            title="Arrastrar panel"
          >
            <div>
              <small>Impacto ambiental del trafico</small>
              <strong>{impactLevelLabel(dominantLevel)}</strong>
            </div>
            <button
              className={`map-card-info ${impactInfoOpen ? "active" : ""}`}
              type="button"
              aria-label={impactInfoOpen ? "Ocultar explicacion de la nube ambiental" : "Explicar nube ambiental"}
              aria-expanded={impactInfoOpen}
              onClick={() => {
                setImpactCardCollapsed(false);
                setImpactInfoOpen((current) => !current);
              }}
            >
              i
            </button>
            <button
              className="map-card-toggle"
              type="button"
              aria-label={impactCardCollapsed ? "Mostrar condiciones ambientales" : "Ocultar condiciones ambientales"}
              aria-expanded={!impactCardCollapsed}
              onClick={() => setImpactCardCollapsed((current) => !current)}
            >
              {impactCardCollapsed ? "+" : "-"}
            </button>
          </div>
          {!impactCardCollapsed ? (
            <div className="environment-conditions" aria-label="Impacto ambiental potencial asociado al trafico">
              {impactInfoOpen ? (
                <div className="impact-card-info-panel">
                  <strong>Que significa este bloque</strong>
                  <p>
                    Resume las condiciones usadas para estimar donde la congestion podria tener mayor impacto
                    ambiental. No representa emisiones exactas ni reemplaza un indice oficial.
                  </p>
                  <dl>
                    <dt>Nivel superior</dt>
                    <dd>Resume el impacto potencial dominante de las nubes mostradas.</dd>
                    <dt>PM2.5</dt>
                    <dd>
                      Lectura: hacia la derecha indica mayor concentracion respecto al historial de esa hora. Efecto:
                      una concentracion mayor aumenta la exposicion ambiental potencial.
                    </dd>
                    <dt>Viento</dt>
                    <dd>
                      Rangos: suave bajo 20 km/h, moderado entre 20 y 39 km/h y fuerte desde 39 km/h. Efecto: poco
                      viento dificulta la dispersion; mas viento favorece la dispersion.
                    </dd>
                    <dt>Lluvia</dt>
                    <dd>
                      Rangos: llovizna 0,1-2 mm, lluvia 2-10 mm y lluvia fuerte sobre 10 mm. Efecto: puede reducir
                      particulas suspendidas, aunque puede dificultar el transito.
                    </dd>
                  </dl>
                </div>
              ) : null}
              <div className="condition-row">
                <div className="condition-title">
                  <span>PM2.5</span>
                  <strong>{formatConditionValue(weather?.pm25, "ug/m3")}</strong>
                </div>
                <div className="condition-bar pm25-bar">
                  <span className="condition-marker" style={{ left: `${pm25Percent ?? 0}%` }} />
                </div>
              </div>
              <div className="condition-row">
                <div className="condition-title">
                  <span>Viento</span>
                  <strong>
                    {formatConditionValue(weather?.wind_speed_kmh, "km/h")} · {weather?.wind_label ?? "Sin dato"}
                  </strong>
                </div>
                <div className="condition-bar wind-bar">
                  <span className="condition-marker" style={{ left: `${windPercent ?? 0}%` }} />
                </div>
              </div>
              <div className="condition-rain">
                <span>Lluvia</span>
                <strong>{weather?.rain_label ?? "Sin dato"}</strong>
              </div>
              <div className={`environmental-layer-status status-${environmentalLayerStatus}`} role="status">
                <span>
                  {environmentalLayerStatus === "loading"
                    ? "Dibujando capa..."
                    : environmentalLayerStatus === "ready"
                      ? `Nube potencial y congestion: ${environmentalImpact.congestion_lines?.features.length ?? 0} tramos`
                      : environmentalLayerStatus === "empty"
                        ? "No hay tramos congestionados para esta fecha"
                        : "No se pudo dibujar la capa"}
                </span>
                {environmentalImpact.points.length ? (
                  <span>Activa “Ver detalles al tocar el mapa” y haz clic en una linea o nube</span>
                ) : null}
              </div>
              {environmentalLayerError ? <p className="environmental-layer-error">{environmentalLayerError}</p> : null}
              <span>{environmentalImpact.summary.requested_at}</span>
            </div>
          ) : null}
        </div>
      ) : null}
      {showImpactCard && environmentalImpactLoading && !environmentalImpact ? (
        <div className="map-impact-card impact-none loading" role="status">
          <div className="map-card-header">
            <div>
              <small>Capa ambiental</small>
              <strong>Cargando datos...</strong>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
