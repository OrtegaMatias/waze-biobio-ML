import { useEffect, useRef } from "react";

import type { PlanRouteCard, RouteOptimizationTrace, RouteType } from "../types";

type InternalRoutingCostsDialogProps = {
  routes: PlanRouteCard[];
  onClose: () => void;
};

const ROUTE_LABELS: Record<RouteType, string> = {
  fastest: "Llegar antes",
  least_congested: "Circulación más fluida",
  healthiest: "Menor exposición ambiental",
};

const OBJECTIVE_LABELS: Record<RouteOptimizationTrace["objective"], string> = {
  fastest: "Tiempo de viaje",
  fluent: "Fluidez",
  environmental: "Exposición ambiental",
};

const COST_ROWS: Array<{
  key: keyof RouteOptimizationTrace;
  label: string;
  sign?: "positive" | "negative";
}> = [
  { key: "base_time_min", label: "Tiempo base" },
  { key: "congestion_delay_min", label: "Retraso por congestión", sign: "positive" },
  { key: "congestion_penalty_min", label: "Penalización por congestión", sign: "positive" },
  { key: "stop_penalty_min", label: "Penalización por detenciones", sign: "positive" },
  { key: "pm25_penalty_min", label: "Penalización por PM2.5", sign: "positive" },
  {
    key: "adverse_environment_penalty_min",
    label: "Penalización ambiental adversa",
    sign: "positive",
  },
  { key: "urban_benefit_min", label: "Beneficio urbano", sign: "negative" },
];

function formatCost(value: number, sign?: "positive" | "negative"): string {
  const magnitude = Math.abs(value).toFixed(3);
  if (sign === "positive" && value > 0) {
    return `+${magnitude}`;
  }
  if (sign === "negative" && value > 0) {
    return `-${magnitude}`;
  }
  return magnitude;
}

export function InternalRoutingCostsDialog({ routes, onClose }: InternalRoutingCostsDialogProps) {
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    closeButtonRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="internal-costs-overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <section
        className="internal-costs-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="internal-costs-title"
      >
        <header className="internal-costs-header">
          <div>
            <p className="eyebrow">Diagnóstico de desarrollo</p>
            <h2 id="internal-costs-title">Costos internos de las rutas</h2>
            <p>
              Minutos equivalentes que usa el algoritmo para comparar caminos. No corresponden a dinero ni al
              tiempo real del viaje.
            </p>
          </div>
          <button
            ref={closeButtonRef}
            className="internal-costs-close"
            type="button"
            aria-label="Cerrar costos internos"
            onClick={onClose}
          >
            x
          </button>
        </header>

        <div className="internal-costs-warning" role="note">
          Uso interno solamente. Este diagnóstico no forma parte de la aplicación final.
        </div>

        <div className="internal-costs-grid">
          {routes.map((route) => {
            const trace = route.optimization_trace;
            return (
              <article className="internal-costs-route" key={route.key}>
                <div className="internal-costs-route-heading">
                  <div>
                    <span>{ROUTE_LABELS[route.key]}</span>
                    <strong>{trace ? OBJECTIVE_LABELS[trace.objective] : "Sin traza disponible"}</strong>
                  </div>
                  {trace ? <small>{trace.logical_segment_count} segmentos</small> : null}
                </div>

                {trace ? (
                  <>
                    <dl className="internal-costs-breakdown">
                      {COST_ROWS.map((row) => (
                        <div key={row.key}>
                          <dt>{row.label}</dt>
                          <dd>{formatCost(Number(trace[row.key]), row.sign)}</dd>
                        </div>
                      ))}
                    </dl>
                    <div className="internal-costs-total">
                      <span>Costo total de optimización</span>
                      <strong>{trace.optimization_cost_min.toFixed(3)}</strong>
                    </div>
                    <small className="internal-costs-version">Modelo: {trace.cost_model_version}</small>
                  </>
                ) : (
                  <p className="internal-costs-empty">La respuesta de esta ruta no incluyó métricas internas.</p>
                )}
              </article>
            );
          })}
        </div>
      </section>
    </div>
  );
}
