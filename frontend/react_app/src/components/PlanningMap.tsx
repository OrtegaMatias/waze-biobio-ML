import { useEffect, useEffectEvent, useRef, useState } from "react";
import type { FeatureCollection, LineString, Point } from "geojson";
import type { GeoJSONSource, StyleSpecification } from "maplibre-gl";

import type { HotspotPoint, PlanRouteCard, RoutePoint } from "../types";

type PinKey = "origin" | "destination";

type PlanningMapProps = {
  enabled: boolean;
  styleUrl: string;
  mapboxToken: string;
  routes: PlanRouteCard[];
  selectedRouteKey: string | null;
  hotspots: HotspotPoint[];
  origin: RoutePoint | null;
  destination: RoutePoint | null;
  activePin: PinKey;
  onPickPoint: (pin: PinKey, point: RoutePoint) => void;
  onMarkerDrag: (pin: PinKey, point: RoutePoint) => void;
};

type RouteOverlayPath = {
  key: string;
  color: string;
  path: string;
  selected: boolean;
  translate: string;
};

const BASE_ROUTE_COLOR = "#0f766e";
const LEAST_CONGESTED_ROUTE_COLOR = "#2563eb";
const HEALTHIEST_ROUTE_COLOR = "#16a34a";
const FALLBACK_ROUTE_COLORS = ["#ea580c", "#7c3aed", "#0891b2"];
const CARTO_LIGHT_TILES = [
  "https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
  "https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
  "https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
  "https://d.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
];

function normalizeStyleUrl(styleUrl: string, token: string): string | StyleSpecification {
  if (!styleUrl || styleUrl === "local-basic") {
    return {
      version: 8,
      sources: {
        "carto-light": {
          type: "raster",
          tiles: CARTO_LIGHT_TILES,
          tileSize: 256,
          maxzoom: 19,
          attribution: "OpenStreetMap contributors, CARTO",
        },
      },
      layers: [
        {
          id: "background",
          type: "background",
          paint: {
            "background-color": "#e7f0ef",
          },
        },
        {
          id: "carto-light",
          type: "raster",
          source: "carto-light",
          paint: {
            "raster-opacity": 0.96,
          },
        },
      ],
    };
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
  return (
    message.includes("tile") ||
    message.includes("raster") ||
    message.includes("image") ||
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

function routeColor(route: PlanRouteCard, fallbackIndex: number): string {
  if (routeHasBadge(route, "base")) {
    return BASE_ROUTE_COLOR;
  }
  if (routeHasBadge(route, "least_congestion") || routeHasBadge(route, "least_exposure")) {
    return LEAST_CONGESTED_ROUTE_COLOR;
  }
  if (routeHasBadge(route, "healthiest")) {
    return HEALTHIEST_ROUTE_COLOR;
  }
  return FALLBACK_ROUTE_COLORS[fallbackIndex % FALLBACK_ROUTE_COLORS.length];
}

function routeOffset(route: PlanRouteCard): number {
  if (routeHasBadge(route, "base")) {
    return -4;
  }
  if (routeHasBadge(route, "least_congestion") || routeHasBadge(route, "least_exposure")) {
    return 4;
  }
  if (routeHasBadge(route, "healthiest")) {
    return 0;
  }
  return 0;
}

function buildRouteGeoJson(routes: PlanRouteCard[], selectedRouteKey: string | null): FeatureCollection<LineString> {
  return {
    type: "FeatureCollection",
    features: routes
      .filter((route) => route.geometry.length > 1)
      .map((route, index) => ({
        type: "Feature" as const,
        geometry: {
          type: "LineString" as const,
          coordinates: route.geometry.map((point) => [point.lon, point.lat]),
        },
        properties: {
          key: route.key,
          color: routeColor(route, index),
          offset: routeOffset(route),
          selected: route.key === selectedRouteKey ? 1 : 0,
        },
      })),
  };
}

function buildHotspotGeoJson(hotspots: HotspotPoint[]): FeatureCollection<Point> {
  return {
    type: "FeatureCollection",
    features: hotspots.map((point) => ({
      type: "Feature" as const,
      geometry: {
        type: "Point" as const,
        coordinates: [point.lon, point.lat] as [number, number],
      },
      properties: {
        weight: point.weight ?? 0.5,
      },
    })),
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

function syncMapData(
  map: any,
  routes: PlanRouteCard[],
  selectedRouteKey: string | null,
  hotspots: HotspotPoint[],
) {
  const routeSource = map.getSource("plan-routes") as GeoJSONSource | undefined;
  if (routeSource) {
    routeSource.setData(buildRouteGeoJson(routes, selectedRouteKey));
  }
  const hotspotSource = map.getSource("plan-hotspots") as GeoJSONSource | undefined;
  if (hotspotSource) {
    hotspotSource.setData(buildHotspotGeoJson(hotspots));
  }
}

function buildProjectedRoutePath(map: any, route: PlanRouteCard): string {
  if (route.geometry.length < 2) {
    return "";
  }
  return route.geometry
    .map((point, index) => {
      const projected = map.project([point.lon, point.lat]);
      return `${index === 0 ? "M" : "L"} ${projected.x.toFixed(1)} ${projected.y.toFixed(1)}`;
    })
    .join(" ");
}

function buildProjectedRouteOverlays(
  map: any,
  routes: PlanRouteCard[],
  selectedRouteKey: string | null,
): RouteOverlayPath[] {
  return routes
    .filter((route) => route.geometry.length > 1)
    .map((route, index) => {
      const offset = routeOffset(route);
      return {
        key: route.key,
        color: routeColor(route, index),
        path: buildProjectedRoutePath(map, route),
        selected: route.key === selectedRouteKey,
        translate: `translate(${offset} ${-offset})`,
      };
    })
    .filter((route) => route.path)
    .sort((a, b) => Number(a.selected) - Number(b.selected));
}

export function PlanningMap({
  enabled,
  styleUrl,
  mapboxToken,
  routes,
  selectedRouteKey,
  hotspots,
  origin,
  destination,
  activePin,
  onPickPoint,
  onMarkerDrag,
}: PlanningMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<any>(null);
  const moduleRef = useRef<any>(null);
  const markersRef = useRef<Record<PinKey, any | null>>({ origin: null, destination: null });
  const activePinRef = useRef<PinKey>(activePin);
  const latestDataRef = useRef({ routes, selectedRouteKey, hotspots });
  const animationRef = useRef<number | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [routeOverlayPaths, setRouteOverlayPaths] = useState<RouteOverlayPath[]>([]);
  const selectedRoute = routes.find((route) => route.key === selectedRouteKey) ?? routes[0] ?? null;

  const handlePickPoint = useEffectEvent((pin: PinKey, point: RoutePoint) => {
    onPickPoint(pin, point);
  });

  const handleMarkerDrag = useEffectEvent((pin: PinKey, point: RoutePoint) => {
    onMarkerDrag(pin, point);
  });

  useEffect(() => {
    activePinRef.current = activePin;
  }, [activePin]);

  useEffect(() => {
    latestDataRef.current = { routes, selectedRouteKey, hotspots };
  }, [hotspots, routes, selectedRouteKey]);

  const updateRouteOverlay = useEffectEvent(() => {
    if (!mapRef.current) {
      setRouteOverlayPaths([]);
      return;
    }
    setRouteOverlayPaths(
      buildProjectedRouteOverlays(
        mapRef.current,
        latestDataRef.current.routes,
        latestDataRef.current.selectedRouteKey,
      ),
    );
  });

  useEffect(() => {
    if (!enabled || !containerRef.current || mapRef.current) {
      return;
    }
    let cancelled = false;

    async function setupMap() {
      try {
        const maplibre = await import("maplibre-gl");
        if (cancelled || !containerRef.current) {
          return;
        }
        moduleRef.current = maplibre;
        const map = new maplibre.Map({
          container: containerRef.current,
          style: normalizeStyleUrl(styleUrl, mapboxToken),
          center: [-73.05, -36.82],
          zoom: 12,
          maxZoom: 18,
          attributionControl: false,
          fadeDuration: 80,
          refreshExpiredTiles: false,
          transformRequest: (url: string) => ({ url: addMapboxToken(url, mapboxToken) }),
        });
        mapRef.current = map;
        map.on("load", () => {
          if (!map.getSource("plan-routes")) {
            map.addSource("plan-routes", { type: "geojson", data: buildRouteGeoJson([], null) });
            map.addLayer({
              id: "plan-routes",
              type: "line",
              source: "plan-routes",
              layout: {
                "line-cap": "butt",
                "line-join": "miter",
              },
              paint: {
                "line-color": ["get", "color"],
                "line-width": ["case", ["==", ["get", "selected"], 1], 7, 4],
                "line-offset": ["get", "offset"],
                "line-opacity": 0.92,
              },
            });
          }
          if (!map.getSource("plan-hotspots")) {
            map.addSource("plan-hotspots", { type: "geojson", data: buildHotspotGeoJson([]) });
            map.addLayer({
              id: "plan-hotspots",
              type: "circle",
              source: "plan-hotspots",
              paint: {
                "circle-radius": ["interpolate", ["linear"], ["get", "weight"], 0.1, 5, 2, 12],
                "circle-color": "#f97316",
                "circle-opacity": 0.28,
                "circle-stroke-width": 1,
                "circle-stroke-color": "#fb923c",
              },
            });
          }
          syncMapData(
            map,
            latestDataRef.current.routes,
            latestDataRef.current.selectedRouteKey,
            latestDataRef.current.hotspots,
          );
          updateRouteOverlay();
        });
        const queueOverlayUpdate = () => {
          if (animationRef.current !== null) {
            return;
          }
          animationRef.current = window.requestAnimationFrame(() => {
            animationRef.current = null;
            updateRouteOverlay();
          });
        };
        map.on("move", queueOverlayUpdate);
        map.on("zoom", queueOverlayUpdate);
        map.on("resize", queueOverlayUpdate);
        map.on("click", (event: any) => {
          handlePickPoint(activePinRef.current, { lat: event.lngLat.lat, lon: event.lngLat.lng });
        });
        map.on("error", (event: any) => {
          if (isRecoverableMapError(event)) {
            return;
          }
          setMapError("No se pudo inicializar el mapa con la configuracion actual.");
        });
      } catch (error) {
        setMapError(error instanceof Error ? error.message : "No se pudo inicializar el mapa.");
      }
    }

    setupMap();

    return () => {
      cancelled = true;
      if (animationRef.current !== null) {
        window.cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      markersRef.current.origin?.remove();
      markersRef.current.destination?.remove();
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, [enabled, mapboxToken, styleUrl]);

  useEffect(() => {
    if (!enabled || !mapRef.current) {
      return;
    }
    syncMapData(mapRef.current, routes, selectedRouteKey, hotspots);
    updateRouteOverlay();
  }, [enabled, hotspots, routes, selectedRouteKey]);

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
      const marker = markersRef.current[pin] ?? new Marker({ color: pin === "origin" ? "#0f766e" : "#dc2626", draggable: true });
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
  }, [destination, enabled, handleMarkerDrag, origin]);

  useEffect(() => {
    if (!enabled || !mapRef.current || !moduleRef.current) {
      return;
    }
    const boundsSource =
      collectBounds(selectedRoute?.geometry ?? []) ??
      collectBounds([origin, destination].filter(Boolean) as RoutePoint[]);
    if (!boundsSource) {
      return;
    }
    const [minLon, minLat, maxLon, maxLat] = boundsSource;
    const { LngLatBounds } = moduleRef.current;
    const bounds = new LngLatBounds([minLon, minLat], [maxLon, maxLat]);
    mapRef.current.fitBounds(bounds, { padding: 56, maxZoom: 14, duration: 700 });
    updateRouteOverlay();
  }, [destination, enabled, origin, selectedRoute]);

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
    <div className="planner-map">
      <div ref={containerRef} className="planner-map-canvas" />
      {routeOverlayPaths.length ? (
        <svg className="route-svg-overlay" aria-hidden="true">
          {routeOverlayPaths.map((route) => (
            <g key={route.key} transform={route.translate}>
              <path className="route-svg-shadow" d={route.path} />
              <path
                className={`route-svg-line ${route.selected ? "selected" : ""}`}
                d={route.path}
                style={{ stroke: route.color }}
              />
            </g>
          ))}
        </svg>
      ) : null}
      <div className="map-overlay">
        <span className="map-chip">Toque el mapa para mover {activePin === "origin" ? "origen" : "destino"}</span>
        {mapError ? <span className="map-chip warning">{mapError}</span> : null}
      </div>
    </div>
  );
}
