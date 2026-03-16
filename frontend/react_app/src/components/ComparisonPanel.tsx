import type { RouteResponse } from "../types";
import { RouteMap } from "./RouteMap";

const VARIANT_LABELS: Record<string, string> = {
  reference: "Ruta base",
  ubcf: "Perfil por usuarios",
  ibcf: "Perfil por vías",
};

const HIGHLIGHT_LABELS: Record<string, string> = {
  fastest_variant: "Más rápida",
  safest_variant: "Más segura",
  lowest_exposure_variant: "Menor exposición",
  best_balance_variant: "Mejor balance",
};

const HIGHLIGHT_KEYS = [
  "fastest_variant",
  "safest_variant",
  "lowest_exposure_variant",
  "best_balance_variant",
] as const;

type HighlightKey = (typeof HIGHLIGHT_KEYS)[number];

type ComparisonPanelProps = {
  route: RouteResponse;
};

export function ComparisonPanel({ route }: ComparisonPanelProps) {
  const highlights = HIGHLIGHT_KEYS.map((key) => {
    const variantKey = route.comparison[key];
    return {
      label: HIGHLIGHT_LABELS[key as HighlightKey],
      value: VARIANT_LABELS[variantKey] ?? variantKey,
    };
  });

  return (
    <section className="panel">
      <div className="section-header">
        <div>
          <div className="eyebrow">Comparación explicativa</div>
          <h2>Resultado de la simulación</h2>
        </div>
      </div>

      <div className="highlight-grid">
        {highlights.map((item) => (
          <article className="highlight-card" key={item.label}>
            <span className="metric-label">{item.label}</span>
            <strong className="metric-value">{item.value}</strong>
          </article>
        ))}
      </div>

      <RouteMap route={route} />

      <div className="comparison-grid">
        {(["reference", "ubcf", "ibcf"] as const).map((variantKey) => {
          const variant = route[variantKey];
          return (
            <article className="comparison-card" key={variantKey}>
              <div className="card-title-row">
                <h3>{VARIANT_LABELS[variantKey]}</h3>
                <span className="pill">{(variant.estimated_duration_min + variant.extra_delay_min).toFixed(1)} min</span>
              </div>
              <div className="stats-inline">
                <span>{variant.distance_km.toFixed(2)} km</span>
                <span>riesgo {variant.risk_score.toFixed(1)}/100</span>
                <span>{variant.incident_exposure.matched_incident_segments} segmentos expuestos</span>
              </div>
              <ul className="reason-list">
                {variant.why_changed.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
              {variant.top_penalized_segments.length ? (
                <>
                  <strong className="subhead">Segmentos conflictivos</strong>
                  <ul className="compact-list">
                    {variant.top_penalized_segments.slice(0, 2).map((segment) => (
                      <li key={`${segment.segment_id}-${segment.via}`}>
                        {segment.via} · {segment.event_type} · impacto {segment.impact_score.toFixed(1)}
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
              {variant.top_preferred_vias.length ? (
                <>
                  <strong className="subhead">Vías favorecidas</strong>
                  <ul className="compact-list">
                    {variant.top_preferred_vias.slice(0, 2).map((item) => (
                      <li key={`${item.via}-${item.factor}`}>
                        {item.via} · factor {item.factor.toFixed(2)}
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}
