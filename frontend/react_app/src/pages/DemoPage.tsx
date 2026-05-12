import { startTransition, useDeferredValue, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import {
  generateRouteComparison,
  getDatasetStatus,
  getReadiness,
  getScenarios,
  startBootstrap,
} from "../api";
import { ComparisonPanel } from "../components/ComparisonPanel";
import { StatusPanel } from "../components/StatusPanel";
import type { DatasetStatus, DemoScenario, ReadinessStatus, RouteResponse, TravelerProfileId } from "../types";

type FormState = {
  scenarioId: string;
  origin: { lat: number; lon: number };
  destination: { lat: number; lon: number };
  day_of_week: string;
  departure_hour: number;
  profile: TravelerProfileId;
  avoid_congestion: boolean;
};

const PROFILE_LABELS: Record<TravelerProfileId, string> = {
  safety_focused: "Seguridad",
  usuario_demo: "Equilibrado",
  moderate_risk: "Moderado",
  risk_taker: "Rapido",
};

const EMPTY_FORM: FormState = {
  scenarioId: "",
  origin: { lat: -36.8267, lon: -73.0498 },
  destination: { lat: -36.8114, lon: -73.049 },
  day_of_week: "Wednesday",
  departure_hour: 8,
  profile: "usuario_demo",
  avoid_congestion: true,
};

function applyScenarioToForm(scenario: DemoScenario): FormState {
  return {
    scenarioId: scenario.id,
    origin: scenario.origin,
    destination: scenario.destination,
    day_of_week: scenario.day_of_week,
    departure_hour: scenario.departure_hour,
    profile: scenario.profile,
    avoid_congestion: true,
  };
}

export function DemoPage() {
  const [readiness, setReadiness] = useState<ReadinessStatus | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [scenarios, setScenarios] = useState<DemoScenario[]>([]);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const deferredRoute = useDeferredValue(route);
  const [busy, setBusy] = useState<{ refresh: boolean; route: boolean }>({
    refresh: false,
    route: false,
  });
  const [error, setError] = useState<string | null>(null);

  async function refreshBootState(forceWarmup: boolean) {
    setBusy((current) => ({ ...current, refresh: true }));
    setError(null);
    try {
      if (forceWarmup) {
        await startBootstrap();
      }
      const [ready, dataset, scenarioList] = await Promise.all([
        getReadiness(),
        getDatasetStatus(),
        getScenarios(),
      ]);
      startTransition(() => {
        setReadiness(ready);
        setDatasetStatus(dataset);
        setScenarios(scenarioList);
        if (scenarioList.length && !form.scenarioId) {
          setForm(applyScenarioToForm(scenarioList[0]));
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo cargar el estado inicial.");
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
  }, [readiness, busy.refresh]);

  async function handleGenerateRoute() {
    setBusy((current) => ({ ...current, route: true }));
    setError(null);
    try {
      const result = await generateRouteComparison({
        origin: form.origin,
        destination: form.destination,
        day_of_week: form.day_of_week,
        departure_hour: form.departure_hour,
        avoid_congestion: form.avoid_congestion,
        avoid_accidents: false,
        profile: form.profile,
      });
      startTransition(() => {
        setRoute(result);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo generar la comparacion.");
    } finally {
      setBusy((current) => ({ ...current, route: false }));
    }
  }

  return (
    <main className="app-shell demo-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Modo demo</p>
          <h1>Explicacion academica de rutas</h1>
          <p className="lead">
            La app principal vive en la home. Esta vista conserva escenarios curados, perfiles sinteticos y el
            lenguaje explicable del proyecto original, centrado en congestion historica.
          </p>
        </div>
        <Link className="secondary-link" to="/">
          Volver a planificador
        </Link>
      </section>

      <StatusPanel readiness={readiness} onRefresh={() => refreshBootState(true)} busy={busy.refresh} />

      {error ? (
        <section className="panel error-panel" role="alert">
          <div className="eyebrow">Error</div>
          <p>{error}</p>
        </section>
      ) : null}

      <section className="panel control-panel">
        <div className="section-header">
          <div>
            <div className="eyebrow">Escenario + comparacion</div>
            <h2>Flujo principal de la aplicacion academica</h2>
          </div>
          <div className="action-row">
            <span className="control-inline">
              <span>Perfil de datos</span>
              <strong>{datasetStatus?.current_label ?? "Gran Concepcion"}</strong>
            </span>
          </div>
        </div>

        <div className="control-grid">
          <article className="panel inset-panel">
            <h3>Escenario curado</h3>
            <select
              aria-label="Escenario curado"
              value={form.scenarioId}
              onChange={(event) => {
                const nextScenario = scenarios.find((scenario) => scenario.id === event.target.value);
                if (nextScenario) {
                  setForm(applyScenarioToForm(nextScenario));
                  setRoute(null);
                }
              }}
            >
              {scenarios.map((scenario) => (
                <option key={scenario.id} value={scenario.id}>
                  {scenario.title}
                </option>
              ))}
            </select>
            <p className="muted">
              {scenarios.find((scenario) => scenario.id === form.scenarioId)?.description ??
                "Selecciona un caso de demo."}
            </p>
            <p className="focus-note">
              {scenarios.find((scenario) => scenario.id === form.scenarioId)?.recommended_focus ?? ""}
            </p>
          </article>

          <article className="panel inset-panel">
            <h3>Configuracion</h3>
            <div className="field-grid">
              <label>
                <span>Perfil simulado</span>
                <select
                  value={form.profile}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      profile: event.target.value as TravelerProfileId,
                    }))
                  }
                >
                  {Object.entries(PROFILE_LABELS).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Dia</span>
                <select
                  value={form.day_of_week}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      day_of_week: event.target.value,
                    }))
                  }
                >
                  {["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].map((day) => (
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
                  value={form.departure_hour}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      departure_hour: Number(event.target.value),
                    }))
                  }
                />
                <small>{form.departure_hour}:00</small>
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={form.avoid_congestion}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      avoid_congestion: event.target.checked,
                    }))
                  }
                />
                <span>Evitar zonas con mayor congestion historica</span>
              </label>
            </div>
          </article>
        </div>

        <div className="route-points">
          {(["origin", "destination"] as const).map((key) => (
            <article className="point-card" key={key}>
              <h4>{key === "origin" ? "Origen" : "Destino"}</h4>
              <div className="point-grid">
                <label>
                  <span>Lat</span>
                  <input
                    type="number"
                    step="0.0001"
                    value={form[key].lat}
                    onChange={(event) =>
                      setForm((current) => ({
                        ...current,
                        [key]: { ...current[key], lat: Number(event.target.value) },
                      }))
                    }
                  />
                </label>
                <label>
                  <span>Lon</span>
                  <input
                    type="number"
                    step="0.0001"
                    value={form[key].lon}
                    onChange={(event) =>
                      setForm((current) => ({
                        ...current,
                        [key]: { ...current[key], lon: Number(event.target.value) },
                      }))
                    }
                  />
                </label>
              </div>
            </article>
          ))}
        </div>

        <button
          className="primary-button"
          onClick={handleGenerateRoute}
          disabled={busy.route || !readiness?.ready}
        >
          {busy.route ? "Generando comparacion..." : "Generar comparacion"}
        </button>
      </section>

      {deferredRoute ? <ComparisonPanel route={deferredRoute} /> : null}
    </main>
  );
}
