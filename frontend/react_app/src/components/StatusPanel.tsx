import type { ReadinessStatus } from "../types";

type StatusPanelProps = {
  readiness: ReadinessStatus | null;
  onRefresh: () => void;
  busy: boolean;
};

export function StatusPanel({ readiness, onRefresh, busy }: StatusPanelProps) {
  const quality = readiness?.bootstrap.quality;
  return (
    <section className="panel hero-panel">
      <div className="eyebrow">Estado del backend</div>
      <h1>Ruta Segura Explicable · React</h1>
      <p className="lead">
        Flujo único de demo para escenario, comparación y explicación de rutas dentro de la SPA.
      </p>
      <div className="status-grid">
        <article className="status-card">
          <span className="metric-label">Estado</span>
          <strong className="metric-value">{readiness?.status ?? "loading"}</strong>
          <p>{readiness?.message ?? "Consultando backend..."}</p>
        </article>
        <article className="status-card">
          <span className="metric-label">Perfil activo</span>
          <strong className="metric-value">{readiness?.dataset_profile ?? "..."}</strong>
          <p>{readiness?.bootstrap.routing_nodes?.toLocaleString() ?? "0"} nodos cargados</p>
        </article>
        <article className="status-card">
          <span className="metric-label">Calidad de datos</span>
          <strong className="metric-value">{quality?.status ?? "..."}</strong>
          <p>{quality?.raw_counts.combined?.toLocaleString() ?? "0"} registros historicos</p>
        </article>
      </div>
      {quality?.warnings?.length ? (
        <div className="warning-box" role="status">
          <strong>Warnings activos</strong>
          <ul>
            {quality.warnings.slice(0, 3).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <button className="primary-button" onClick={onRefresh} disabled={busy}>
        {busy ? "Actualizando..." : "Refrescar estado"}
      </button>
    </section>
  );
}
