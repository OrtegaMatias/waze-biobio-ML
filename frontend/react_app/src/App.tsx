import { startTransition, useDeferredValue, useEffect, useState } from "react";

import {
  generateRouteComparison,
  getDatasetStatus,
  getReadiness,
  getScenarios,
  setDataset,
  startBootstrap,
} from "./api";
import { ComparisonPanel } from "./components/ComparisonPanel";
import { StatusPanel } from "./components/StatusPanel";
import type { DatasetStatus, DemoScenario, ReadinessStatus, RouteResponse, TravelerProfileId } from "./types";

type FormState = {
  scenarioId: string;
  origin: { lat: number; lon: number };
  destination: { lat: number; lon: number };
  day_of_week: string;
  departure_hour: number;
  profile: TravelerProfileId;
  avoid_congestion: boolean;
  avoid_accidents: boolean;
};

const PROFILE_LABELS: Record<TravelerProfileId, string> = {
  safety_focused: "🛡️ Seguridad",
  usuario_demo: "⚖️ Equilibrado",
  moderate_risk: "🚗 Moderado",
  risk_taker: "⚡ Rápido",
};

const EMPTY_FORM: FormState = {
  scenarioId: "",
  origin: { lat: -36.8267, lon: -73.0498 },
  destination: { lat: -36.8114, lon: -73.0490 },
  day_of_week: "Wednesday",
  departure_hour: 8,
  profile: "usuario_demo",
  avoid_congestion: true,
  avoid_accidents: false,
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
    avoid_accidents: false,
  };
}

export default function App() {
  const [readiness, setReadiness] = useState<ReadinessStatus | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [scenarios, setScenarios] = useState<DemoScenario[]>([]);
  const [form, setForm] = useState<FormState>(EMPTY_FORM);
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const deferredRoute = useDeferredValue(route);
  const [busy, setBusy] = useState<{ refresh: boolean; route: boolean; dataset: boolean }>({
    refresh: false,
    route: false,
    dataset: false,
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

  async function handleDatasetChange(profile: string) {
    setBusy((current) => ({ ...current, dataset: true }));
    setError(null);
    try {
      const nextDataset = await setDataset(profile);
      setDatasetStatus(nextDataset);
      setRoute(null);
      await refreshBootState(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo cambiar el perfil.");
    } finally {
      setBusy((current) => ({ ...current, dataset: false }));
    }
  }

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
        avoid_accidents: form.avoid_accidents,
        profile: form.profile,
      });
      startTransition(() => {
        setRoute(result);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo generar la comparación.");
    } finally {
      setBusy((current) => ({ ...current, route: false }));
    }
  }

  return (
    <main className="app-shell">
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
            <div className="eyebrow">Escenario + comparación</div>
            <h2>Flujo principal de React v1</h2>
          </div>
          <div className="action-row">
            <label className="control-inline">
              <span>Perfil de datos</span>
              <select
                value={datasetStatus?.current ?? "concepcion"}
                onChange={(event) => handleDatasetChange(event.target.value)}
                disabled={busy.dataset}
              >
                {(datasetStatus?.available ?? []).map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
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
            <h3>Configuración</h3>
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
                <span>Día</span>
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
                <span>Evitar congestiones históricas</span>
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={form.avoid_accidents}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      avoid_accidents: event.target.checked,
                    }))
                  }
                />
                <span>Evitar accidentes históricos</span>
              </label>
            </div>
          </article>
        </div>

        <div className="route-points">
          <article className="point-card">
            <h4>Origen</h4>
            <div className="point-grid">
              <label>
                <span>Lat</span>
                <input
                  type="number"
                  step="0.0001"
                  value={form.origin.lat}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      origin: { ...current.origin, lat: Number(event.target.value) },
                    }))
                  }
                />
              </label>
              <label>
                <span>Lon</span>
                <input
                  type="number"
                  step="0.0001"
                  value={form.origin.lon}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      origin: { ...current.origin, lon: Number(event.target.value) },
                    }))
                  }
                />
              </label>
            </div>
          </article>
          <article className="point-card">
            <h4>Destino</h4>
            <div className="point-grid">
              <label>
                <span>Lat</span>
                <input
                  type="number"
                  step="0.0001"
                  value={form.destination.lat}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      destination: { ...current.destination, lat: Number(event.target.value) },
                    }))
                  }
                />
              </label>
              <label>
                <span>Lon</span>
                <input
                  type="number"
                  step="0.0001"
                  value={form.destination.lon}
                  onChange={(event) =>
                    setForm((current) => ({
                      ...current,
                      destination: { ...current.destination, lon: Number(event.target.value) },
                    }))
                  }
                />
              </label>
            </div>
          </article>
        </div>

        <button
          className="primary-button"
          onClick={handleGenerateRoute}
          disabled={busy.route || !readiness?.ready}
        >
          {busy.route ? "Generando comparación..." : "Generar comparación"}
        </button>
      </section>

      {deferredRoute ? <ComparisonPanel route={deferredRoute} /> : null}
    </main>
  );
}
