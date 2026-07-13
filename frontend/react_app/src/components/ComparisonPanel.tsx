import type { RouteResponse } from "../types";
import { RouteMap } from "./RouteMap";

const VARIANT_LABELS: Record<string, string> = {
  reference: "Ruta mas corta",
  least_congestion: "Circulación más fluida",
  ubcf: "Perfil por usuarios",
  healthiest: "Menor exposición ambiental",
  ibcf: "Perfil por vías",
};

const HIGHLIGHT_LABELS: Record<string, string> = {
  fastest_variant: "Llegar antes",
  safest_variant: "Más segura",
  lowest_exposure_variant: "Menor exposición ambiental",
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

const DISPLAY_VARIANTS = ["reference", "least_congestion", "ubcf", "ibcf", "healthiest"] as const;

function formatPm25(
  route: RouteResponse[keyof Pick<RouteResponse, "reference" | "least_congestion" | "ubcf" | "ibcf" | "healthiest">],
): string {
  if (!route) {
    return "PM2.5 no disponible";
  }
  if (!route.pm25_exposure) {
    return "PM2.5 no disponible";
  }
  return `PM2.5 ${route.pm25_exposure.average_pm25.toFixed(1)} ug/m3 | ${route.pm25_exposure.category}`;
}

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
          <div className="eyebrow">ComparaciÃ³n explicativa</div>
          <h2>Resultado de la simulaciÃ³n</h2>
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
        {DISPLAY_VARIANTS.map((variantKey) => {
          const variant = route[variantKey];
          if (!variant) {
            return null;
          }
          return (
            <article className="comparison-card" key={variantKey}>
              <div className="card-title-row">
                <h3>{VARIANT_LABELS[variantKey]}</h3>
                <span className="pill">{(variant.estimated_duration_min + variant.extra_delay_min).toFixed(1)} min</span>
              </div>
              <div className="stats-inline">
                <span>{variant.distance_km.toFixed(2)} km</span>
                <span>riesgo {variant.risk_score.toFixed(1)}/100</span>
                <span>{variant.incident_exposure.matched_incident_segments} zonas historicas</span>
                <span>{formatPm25(variant)}</span>
              </div>
              <ul className="reason-list">
                {variant.why_changed.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
              {variant.top_penalized_segments.length ? (
                <>
                  <strong className="subhead">Zonas con mayor congestion historica</strong>
                  <ul className="compact-list">
                    {variant.top_penalized_segments.slice(0, 2).map((segment) => (
                      <li key={`${segment.segment_id}-${segment.via}`}>
                        {segment.via} Â· {segment.event_type} Â· impacto {segment.impact_score.toFixed(1)}
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
              {variant.top_preferred_vias.length ? (
                <>
                  <strong className="subhead">VÃ­as favorecidas</strong>
                  <ul className="compact-list">
                    {variant.top_preferred_vias.slice(0, 2).map((item) => (
                      <li key={`${item.via}-${item.factor}`}>
                        {item.via} Â· factor {item.factor.toFixed(2)}
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
