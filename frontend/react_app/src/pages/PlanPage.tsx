import { startTransition, useDeferredValue, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { planRoute, reversePlace, getReadiness, searchPlaces, startBootstrap } from "../api";
import { PlanningMap } from "../components/PlanningMap";
import type {
  PlaceResult,
  PlanRouteResponse,
  ReadinessStatus,
  RoutePoint,
  TravelStyle,
} from "../types";

type PinKey = "origin" | "destination";

type PlaceSelection = {
  id: string;
  label: string;
  point: RoutePoint;
  bbox: number[] | null;
};

type PlannerState = {
  origin: PlaceSelection;
  destination: PlaceSelection;
  day_of_week: string;
  departure_hour: number;
  travel_style: TravelStyle;
  avoid_congestion: boolean;
};

const RECENT_PLACES_KEY = "wbm_recent_places";
const DEFAULT_MAP_STYLE_URL = "local-basic";
const DAY_OPTIONS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const STYLE_OPTIONS: Array<{ key: TravelStyle; label: string; description: string }> = [
  { key: "safe", label: "Safe", description: "Prioriza menos zonas con congestion historica." },
  { key: "balanced", label: "Balanced", description: "Busca el mejor balance general." },
  { key: "fast", label: "Fast", description: "Apreta el tiempo total al maximo." },
];

function makeSelection(id: string, label: string, point: RoutePoint, bbox: number[] | null = null): PlaceSelection {
  return { id, label, point, bbox };
}

function readRecentPlaces(): PlaceResult[] {
  try {
    const stored = window.localStorage.getItem(RECENT_PLACES_KEY);
    if (!stored) {
      return [];
    }
    const parsed = JSON.parse(stored) as PlaceResult[];
    return Array.isArray(parsed) ? parsed.slice(0, 6) : [];
  } catch {
    return [];
  }
}

function writeRecentPlaces(items: PlaceResult[]) {
  window.localStorage.setItem(RECENT_PLACES_KEY, JSON.stringify(items.slice(0, 6)));
}

function upsertRecentPlace(current: PlaceResult[], next: PlaceResult): PlaceResult[] {
  const deduped = current.filter((item) => item.id !== next.id);
  return [next, ...deduped].slice(0, 6);
}

function selectionFromPlace(place: PlaceResult): PlaceSelection {
  return makeSelection(place.id, place.label, { lat: place.lat, lon: place.lon }, place.bbox);
}

function buildManualLabel(point: RoutePoint): string {
  return `Punto manual (${point.lat.toFixed(4)}, ${point.lon.toFixed(4)})`;
}

function routeHasBadge(route: PlanRouteResponse["routes"][number], badgeKey: string): boolean {
  return route.badges.some((badge) => badge.key === badgeKey);
}

function formatRouteExposure(route: PlanRouteResponse["routes"][number]): string {
  const exposure = route.incident_exposure;
  return `${exposure.matched_incident_segments} zonas / ${exposure.exposure_minutes.toFixed(1)} min`;
}

function formatPm25Exposure(route: PlanRouteResponse["routes"][number] | null): string {
  const exposure = route?.pm25_exposure;
  if (!exposure) {
    return "No disponible";
  }
  return `${exposure.average_pm25.toFixed(1)} ug/m3 | ${exposure.category}`;
}

function routeGeometryKey(route: PlanRouteResponse["routes"][number] | null): string {
  return route?.geometry.map((point) => `${point.lat.toFixed(5)},${point.lon.toFixed(5)}`).join("|") ?? "";
}

export function PlanPage() {
  const mapStyleUrl = (import.meta.env.VITE_MAP_STYLE_URL ?? DEFAULT_MAP_STYLE_URL).trim();
  const mapboxToken = (import.meta.env.VITE_MAPBOX_TOKEN ?? "").trim();
  const mapEnabled = Boolean(mapStyleUrl);
  const geocodingEnabled = Boolean(mapboxToken);

  const [readiness, setReadiness] = useState<ReadinessStatus | null>(null);
  const [planner, setPlanner] = useState<PlannerState>({
    origin: makeSelection("origin-default", "Plaza Peru, Concepcion", { lat: -36.8271, lon: -73.0496 }),
    destination: makeSelection("destination-default", "Mall del Centro, Concepcion", { lat: -36.826, lon: -73.0504 }),
    day_of_week: "Wednesday",
    departure_hour: 8,
    travel_style: "balanced",
    avoid_congestion: true,
  });
  const [queries, setQueries] = useState<Record<PinKey, string>>({
    origin: "Plaza Peru, Concepcion",
    destination: "Mall del Centro, Concepcion",
  });
  const [suggestions, setSuggestions] = useState<Record<PinKey, PlaceResult[]>>({
    origin: [],
    destination: [],
  });
  const [recentPlaces, setRecentPlaces] = useState<PlaceResult[]>(() => readRecentPlaces());
  const [activePin, setActivePin] = useState<PinKey>("origin");
  const [plan, setPlan] = useState<PlanRouteResponse | null>(null);
  const deferredPlan = useDeferredValue(plan);
  const [selectedRouteKey, setSelectedRouteKey] = useState<string | null>(null);
  const [busy, setBusy] = useState({ refresh: false, planning: false, geolocate: false });
  const [error, setError] = useState<string | null>(null);
  const selectedRoute =
    deferredPlan?.routes.find((route) => route.key === selectedRouteKey) ?? deferredPlan?.routes[0] ?? null;

  async function refreshBootState(forceWarmup: boolean) {
    setBusy((current) => ({ ...current, refresh: true }));
    setError(null);
    try {
      if (forceWarmup) {
        await startBootstrap();
      }
      const ready = await getReadiness();
      setReadiness(ready);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo revisar el estado del backend.");
    } finally {
      setBusy((current) => ({ ...current, refresh: false }));
    }
  }

  useEffect(() => {
    refreshBootState(true);
  }, []);

  useEffect(() => {
    if (!readiness || readiness.ready || busy.refresh) {
      return;
    }
    const timer = window.setTimeout(() => {
      refreshBootState(false);
    }, 2500);
    return () => window.clearTimeout(timer);
  }, [busy.refresh, readiness]);

  useEffect(() => {
    if (!geocodingEnabled) {
      setSuggestions({ origin: [], destination: [] });
      return;
    }
    const entries = (Object.entries(queries) as Array<[PinKey, string]>).map(([pin, query]) => ({ pin, query }));
    const timer = window.setTimeout(async () => {
      await Promise.all(
        entries.map(async ({ pin, query }) => {
          const normalized = query.trim();
          if (normalized.length < 3 || normalized === planner[pin].label) {
            setSuggestions((current) => ({
              ...current,
              [pin]: normalized.length === 0 ? recentPlaces : [],
            }));
            return;
          }
          try {
            const results = await searchPlaces(normalized, 5);
            setSuggestions((current) => ({ ...current, [pin]: results }));
          } catch {
            setSuggestions((current) => ({ ...current, [pin]: [] }));
          }
        }),
      );
    }, 280);
    return () => window.clearTimeout(timer);
  }, [geocodingEnabled, planner, queries, recentPlaces]);

  function applySelection(pin: PinKey, selection: PlaceSelection) {
    setPlanner((current) => ({ ...current, [pin]: selection }));
    setQueries((current) => ({ ...current, [pin]: selection.label }));
    setSuggestions((current) => ({ ...current, [pin]: [] }));
    setPlan(null);
    setSelectedRouteKey(null);
  }

  async function resolveLabel(pin: PinKey, point: RoutePoint) {
    if (!geocodingEnabled) {
      applySelection(pin, makeSelection(`${pin}-manual`, buildManualLabel(point), point));
      return;
    }
    try {
      const result = await reversePlace(point.lat, point.lon);
      if (result) {
        applySelection(pin, selectionFromPlace(result));
        return;
      }
    } catch {
      // noop: fallback below
    }
    applySelection(pin, makeSelection(`${pin}-manual`, buildManualLabel(point), point));
  }

  function handlePointInput(pin: PinKey, axis: "lat" | "lon", rawValue: string) {
    const parsed = Number(rawValue);
    if (Number.isNaN(parsed)) {
      return;
    }
    setPlanner((current) => ({
      ...current,
      [pin]: {
        ...current[pin],
        point: {
          ...current[pin].point,
          [axis]: parsed,
        },
      },
    }));
    setPlan(null);
  }

  function handleSuggestionSelect(pin: PinKey, place: PlaceResult) {
    const nextRecentPlaces = upsertRecentPlace(recentPlaces, place);
    setRecentPlaces(nextRecentPlaces);
    writeRecentPlaces(nextRecentPlaces);
    applySelection(pin, selectionFromPlace(place));
  }

  function handleSwap() {
    setPlanner((current) => ({
      ...current,
      origin: current.destination,
      destination: current.origin,
    }));
    setQueries((current) => ({
      origin: current.destination,
      destination: current.origin,
    }));
    setPlan(null);
    setSelectedRouteKey(null);
  }

  function handleGeolocate() {
    if (!navigator.geolocation) {
      setError("Tu navegador no expone geolocalizacion.");
      return;
    }
    setBusy((current) => ({ ...current, geolocate: true }));
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const point = {
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        };
        void resolveLabel("origin", point);
        setBusy((current) => ({ ...current, geolocate: false }));
      },
      () => {
        setBusy((current) => ({ ...current, geolocate: false }));
        setError("No fue posible tomar tu ubicacion actual.");
      },
      { enableHighAccuracy: true, timeout: 7000, maximumAge: 0 },
    );
  }

  async function handlePlan() {
    setBusy((current) => ({ ...current, planning: true }));
    setError(null);
    try {
      const response = await planRoute({
        origin: planner.origin.point,
        destination: planner.destination.point,
        day_of_week: planner.day_of_week,
        departure_hour: planner.departure_hour,
        travel_style: planner.travel_style,
        avoid_congestion: planner.avoid_congestion,
        avoid_accidents: false,
      });
      startTransition(() => {
        setPlan(response);
        setSelectedRouteKey(response.selected_route_key);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo planificar el viaje.");
    } finally {
      setBusy((current) => ({ ...current, planning: false }));
    }
  }

  const routeCards = deferredPlan?.routes ?? [];
  const baseRoute = routeCards.find((route) => routeHasBadge(route, "base")) ?? routeCards[0] ?? null;
  const leastCongestedRoute =
    routeCards.find((route) => routeHasBadge(route, "least_congestion") || routeHasBadge(route, "least_exposure")) ??
    routeCards[0] ??
    null;
  const healthiestRoute = routeCards.find((route) => routeHasBadge(route, "healthiest")) ?? routeCards[0] ?? null;
  const routesCoincide =
    routeCards.length === 1 ||
    (baseRoute !== null &&
      leastCongestedRoute !== null &&
      routeGeometryKey(baseRoute) === routeGeometryKey(leastCongestedRoute));

  return (
    <main className="plan-shell">
      <section className="topbar topbar-product">
        <div>
          <p className="eyebrow">Movilidad clara</p>
          <h1>Planifica tu viaje con congestion historica</h1>
          <p className="lead">
            Define origen y destino, compara alternativas claras y revisa zonas con mayor congestion historica antes de salir.
          </p>
        </div>
        <Link className="secondary-link" to="/demo">
          Ir al modo demo
        </Link>
      </section>

      <section className="search-shell sticky-shell">
        <div className="search-grid">
          {(["origin", "destination"] as const).map((pin) => (
            <label className="search-field" key={pin}>
              <span>{pin === "origin" ? "Origen" : "Destino"}</span>
              <input
                type="text"
                value={queries[pin]}
                placeholder={pin === "origin" ? "Buscar punto de partida" : "Buscar destino"}
                onFocus={() => {
                  setActivePin(pin);
                  if (geocodingEnabled && queries[pin].trim().length === 0) {
                    setSuggestions((current) => ({ ...current, [pin]: recentPlaces }));
                  }
                }}
                onChange={(event) => {
                  setActivePin(pin);
                  setQueries((current) => ({ ...current, [pin]: event.target.value }));
                }}
                disabled={!geocodingEnabled}
              />
              {geocodingEnabled && suggestions[pin].length ? (
                <div className="suggestion-list">
                  {suggestions[pin].map((place) => (
                    <button
                      className="suggestion-item"
                      key={`${pin}-${place.id}`}
                      type="button"
                      onClick={() => handleSuggestionSelect(pin, place)}
                    >
                      <strong>{place.label}</strong>
                      <span>
                        {place.lat.toFixed(4)}, {place.lon.toFixed(4)}
                      </span>
                    </button>
                  ))}
                </div>
              ) : null}
            </label>
          ))}
        </div>

        <div className="search-actions">
          <button className="ghost-button" type="button" onClick={handleSwap}>
            Intercambiar
          </button>
          <button className="ghost-button" type="button" onClick={handleGeolocate} disabled={busy.geolocate}>
            {busy.geolocate ? "Ubicando..." : "Usar mi ubicacion"}
          </button>
          <button
            className="primary-button"
            type="button"
            onClick={handlePlan}
            disabled={busy.planning || !readiness?.ready}
          >
            {busy.planning ? "Planificando..." : "Planificar viaje"}
          </button>
        </div>
      </section>

      {readiness && !readiness.ready ? (
        <section className="banner info-banner" role="status">
          <strong>{readiness.status === "error" ? "Backend con problema" : "Preparando rutas"}</strong>
          <span>{readiness.message}</span>
        </section>
      ) : null}

      {error ? (
        <section className="banner error-banner" role="alert">
          <strong>No se pudo completar la accion.</strong>
          <span>{error}</span>
        </section>
      ) : null}

      <section className="planner-layout">
        <div className="map-stage">
          <PlanningMap
            enabled={mapEnabled}
            styleUrl={mapStyleUrl}
            mapboxToken={mapboxToken}
            routes={routeCards}
            selectedRouteKey={selectedRouteKey}
            hotspots={deferredPlan?.hotspots ?? []}
            origin={planner.origin.point}
            destination={planner.destination.point}
            activePin={activePin}
            onPickPoint={(pin, point) => {
              setActivePin(pin);
              void resolveLabel(pin, point);
            }}
            onMarkerDrag={(pin, point) => {
              setActivePin(pin);
              void resolveLabel(pin, point);
            }}
          />
        </div>

        <aside className="results-sheet">
          <section className="panel product-panel">
            <div className="section-header">
              <div>
                <div className="eyebrow">Viaje</div>
                <h2>Preferencias simples</h2>
              </div>
              <div className="panel-tag">{planner.day_of_week}</div>
            </div>

            <div className="style-grid">
              {STYLE_OPTIONS.map((style) => (
                <button
                  key={style.key}
                  type="button"
                  className={`style-card ${planner.travel_style === style.key ? "active" : ""}`}
                  onClick={() => setPlanner((current) => ({ ...current, travel_style: style.key }))}
                >
                  <strong>{style.label}</strong>
                  <span>{style.description}</span>
                </button>
              ))}
            </div>

            <div className="field-grid">
              <label>
                <span>Dia</span>
                <select
                  value={planner.day_of_week}
                  onChange={(event) =>
                    setPlanner((current) => ({
                      ...current,
                      day_of_week: event.target.value,
                    }))
                  }
                >
                  {DAY_OPTIONS.map((day) => (
                    <option key={day} value={day}>
                      {day}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Hora de salida</span>
                <input
                  type="range"
                  min={0}
                  max={23}
                  step={1}
                  value={planner.departure_hour}
                  onChange={(event) =>
                    setPlanner((current) => ({
                      ...current,
                      departure_hour: Number(event.target.value),
                    }))
                  }
                />
                <small>{planner.departure_hour}:00</small>
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={planner.avoid_congestion}
                  onChange={(event) =>
                    setPlanner((current) => ({
                      ...current,
                      avoid_congestion: event.target.checked,
                    }))
                  }
                />
                <span>Evitar zonas con mayor congestion historica</span>
              </label>
            </div>

            <div className="manual-grid">
              {(["origin", "destination"] as const).map((pin) => (
                <article className="manual-card" key={pin}>
                  <div className="card-title-row">
                    <h3>{pin === "origin" ? "Origen" : "Destino"}</h3>
                    <button className="text-button" type="button" onClick={() => setActivePin(pin)}>
                      Editar en mapa
                    </button>
                  </div>
                  <p className="muted">{planner[pin].label}</p>
                  <div className="point-grid">
                    <label>
                      <span>Lat</span>
                      <input
                        type="number"
                        step="0.0001"
                        value={planner[pin].point.lat}
                        onChange={(event) => handlePointInput(pin, "lat", event.target.value)}
                        onBlur={() => resolveLabel(pin, planner[pin].point)}
                      />
                    </label>
                    <label>
                      <span>Lon</span>
                      <input
                        type="number"
                        step="0.0001"
                        value={planner[pin].point.lon}
                        onChange={(event) => handlePointInput(pin, "lon", event.target.value)}
                        onBlur={() => resolveLabel(pin, planner[pin].point)}
                      />
                    </label>
                  </div>
                </article>
              ))}
            </div>
          </section>

          {deferredPlan ? (
            <>
              <section className="panel product-panel summary-panel">
                <div className="section-header">
                  <div>
                    <div className="eyebrow">Resumen del viaje</div>
                  <h2>{selectedRoute?.label ?? "Ruta seleccionada"}</h2>
                  </div>
                  <div className={`risk-pill risk-${selectedRoute?.risk_level ?? "low"}`}>
                    {selectedRoute?.risk_level ?? "low"}
                  </div>
                </div>
                <div className="summary-grid">
                  <article className="summary-card">
                    <span className="metric-label">ETA total</span>
                    <strong className="metric-value">{(selectedRoute?.duration_min ?? deferredPlan.summary.eta_total_min).toFixed(1)} min</strong>
                  </article>
                  <article className="summary-card">
                    <span className="metric-label">Distancia</span>
                    <strong className="metric-value">{(selectedRoute?.distance_km ?? deferredPlan.summary.distance_km).toFixed(1)} km</strong>
                  </article>
                  <article className="summary-card">
                    <span className="metric-label">Demora estimada</span>
                    <strong className="metric-value">{(selectedRoute?.delay_min ?? deferredPlan.summary.delay_min).toFixed(1)} min</strong>
                  </article>
                  <article className="summary-card">
                    <span className="metric-label">Zonas historicas en ruta</span>
                    <strong className="metric-value">
                      {selectedRoute?.incident_exposure.matched_incident_segments ?? deferredPlan.summary.alerts_on_route}
                    </strong>
                  </article>
                  <article className="summary-card">
                    <span className="metric-label">Exposicion PM2.5 estimada</span>
                    <strong className="metric-value">{formatPm25Exposure(selectedRoute)}</strong>
                  </article>
                </div>
                <p className="explanation-copy">
                  {selectedRoute?.why_changed[0] ?? deferredPlan.summary.main_reason}
                </p>
              </section>

              {baseRoute && leastCongestedRoute && healthiestRoute ? (
                <section className="panel product-panel route-compare-panel">
                  <div className="section-header">
                    <div>
                      <div className="eyebrow">Comparacion clave</div>
                      <h2>Base, menor congestion y mas saludable</h2>
                    </div>
                  </div>

                  {routesCoincide ? (
                    <p className="coincidence-note">
                      Para este trayecto, algunas alternativas pueden compartir el mismo trazado.
                    </p>
                  ) : null}

                  <div className="route-compare-grid">
                    {[
                      { title: "Ruta base", route: baseRoute },
                      { title: "Menor congestion", route: leastCongestedRoute },
                      { title: "Mas saludable", route: healthiestRoute },
                    ].map(({ title, route }) => (
                      <article className="route-compare-card" key={`${title}-${route.key}`}>
                        <div className="card-title-row">
                          <h3>{title}</h3>
                          <button className="text-button" type="button" onClick={() => setSelectedRouteKey(route.key)}>
                            Ver en mapa
                          </button>
                        </div>
                        <div className="compare-metrics">
                          <span>
                            <strong>{route.distance_km.toFixed(1)} km</strong>
                            Distancia
                          </span>
                          <span>
                            <strong>{route.duration_min.toFixed(1)} min</strong>
                            Tiempo estimado
                          </span>
                          <span>
                            <strong>{route.delay_min.toFixed(1)} min</strong>
                            Penalizacion
                          </span>
                          <span>
                            <strong>{formatRouteExposure(route)}</strong>
                            Exposicion historica
                          </span>
                          <span>
                            <strong>{formatPm25Exposure(route)}</strong>
                            PM2.5 estimado
                          </span>
                        </div>
                      </article>
                    ))}
                  </div>
                </section>
              ) : null}

              <section className="panel product-panel">
                <div className="section-header">
                  <div>
                    <div className="eyebrow">Alternativas</div>
                    <h2>Escoge una ruta estable</h2>
                  </div>
                </div>
                <div className="route-card-list">
                  {routeCards.map((route) => (
                    <button
                      key={route.key}
                      type="button"
                      className={`route-card ${selectedRouteKey === route.key ? "selected" : ""}`}
                      onClick={() => setSelectedRouteKey(route.key)}
                    >
                      <div className="card-title-row">
                        <h3>{route.label}</h3>
                        <span className="panel-tag">{route.duration_min.toFixed(1)} min</span>
                      </div>
                      <div className="badge-row">
                        {route.badges.map((badge) => (
                          <span className="badge-pill" key={`${route.key}-${badge.key}`}>
                            {badge.label}
                          </span>
                        ))}
                      </div>
                      <p>{route.summary}</p>
                      <div className="stats-inline">
                        <span>{route.distance_km.toFixed(1)} km</span>
                        <span>demora {route.delay_min.toFixed(1)} min</span>
                        <span>{route.incident_exposure.matched_incident_segments} zonas historicas</span>
                        <span>PM2.5 {formatPm25Exposure(route)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </section>

              <section className="panel product-panel">
                <div className="section-header">
                  <div>
                    <div className="eyebrow">Detalle</div>
                    <h2>Lo mas importante del trayecto</h2>
                  </div>
                </div>

                <div className="detail-stack">
                  <article className="detail-card">
                    <h3>Congestion destacada</h3>
                    {selectedRoute?.top_alerts.length ? (
                      <div className="alert-list">
                        {selectedRoute.top_alerts.map((alert) => (
                          <div className={`alert-card alert-${alert.severity}`} key={`${alert.title}-${alert.detail}`}>
                            <strong>{alert.title}</strong>
                            <p>{alert.detail}</p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No hay zonas de congestion historica relevantes para este trayecto.</p>
                    )}
                  </article>

                  <article className="detail-card">
                    <h3>Zonas con mayor congestion historica</h3>
                    {deferredPlan.hotspots.length ? (
                      <div className="mini-list">
                        {deferredPlan.hotspots.slice(0, 5).map((hotspot, index) => (
                          <div className="mini-list-item" key={`${hotspot.segment_id ?? index}-${index}`}>
                            <strong>Segmento {hotspot.segment_id ?? index + 1}</strong>
                            <span>
                              {hotspot.day || "Sin dia"} | {hotspot.bucket || "Sin franja"}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No hay zonas relevantes para este horario.</p>
                    )}
                  </article>

                  <article className="detail-card">
                    <h3>Exposicion PM2.5 estimada</h3>
                    {selectedRoute?.pm25_exposure ? (
                      <div className="mini-list">
                        <div className="mini-list-item">
                          <strong>{formatPm25Exposure(selectedRoute)}</strong>
                          <span>{selectedRoute.pm25_exposure.method}</span>
                        </div>
                        {selectedRoute.pm25_exposure.stations.slice(0, 3).map((station) => (
                          <div className="mini-list-item" key={station.station_id}>
                            <strong>{station.station_name}</strong>
                            <span>
                              {station.pm25.toFixed(1)} ug/m3 | {station.distance_km.toFixed(1)} km aprox.
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No hay datos PM2.5 disponibles para estimar esta ruta.</p>
                    )}
                  </article>

                  <article className="detail-card">
                    <h3>Razon de recomendacion</h3>
                    <ul className="reason-list">
                      {(selectedRoute?.why_changed ?? []).map((reason) => (
                        <li key={reason}>{reason}</li>
                      ))}
                    </ul>
                  </article>

                  <article className="detail-card">
                    <h3>Vias clave</h3>
                    {(selectedRoute?.top_preferred_vias ?? []).length ? (
                      <div className="mini-list">
                        {selectedRoute?.top_preferred_vias.map((via) => (
                          <div className="mini-list-item" key={`${via.via}-${via.factor}`}>
                            <strong>{via.via}</strong>
                            <span>{via.reason}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No hay vias destacadas para esta alternativa.</p>
                    )}
                  </article>
                </div>
              </section>
            </>
          ) : (
            <section className="panel product-panel empty-panel">
              <div className="eyebrow">Listo para salir</div>
              <h2>Empieza con dos puntos y un estilo de viaje</h2>
              <p>
                La app prioriza Concepcion, muestra rutas pre-viaje y explica la recomendacion con lenguaje simple.
              </p>
            </section>
          )}
        </aside>
      </section>
    </main>
  );
}
