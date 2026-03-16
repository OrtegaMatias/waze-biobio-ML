import type { RouteResponse } from "../types";

const VARIANT_META = {
  reference: { label: "Ruta base", color: "#2563eb" },
  ubcf: { label: "Perfil por usuarios", color: "#15803d" },
  ibcf: { label: "Perfil por vías", color: "#ea580c" },
} as const;

type RouteMapProps = {
  route: RouteResponse;
};

type ProjectedPoint = {
  x: number;
  y: number;
};

export function RouteMap({ route }: RouteMapProps) {
  const routes = ["reference", "ubcf", "ibcf"]
    .map((key) => ({
      key,
      label: VARIANT_META[key as keyof typeof VARIANT_META].label,
      color: VARIANT_META[key as keyof typeof VARIANT_META].color,
      geometry: route[key as keyof Pick<RouteResponse, "reference" | "ubcf" | "ibcf">].geometry ?? [],
    }))
    .filter((item) => item.geometry.length > 1);

  if (!routes.length) {
    return null;
  }

  const allPoints = routes.flatMap((item) => item.geometry);
  const lats = allPoints.map((point) => point.lat);
  const lons = allPoints.map((point) => point.lon);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const width = 960;
  const height = 360;
  const padding = 28;
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  const latRange = Math.max(maxLat - minLat, 0.00001);
  const lonRange = Math.max(maxLon - minLon, 0.00001);

  const project = (lat: number, lon: number): ProjectedPoint => ({
    x: padding + ((lon - minLon) / lonRange) * usableWidth,
    y: padding + ((maxLat - lat) / latRange) * usableHeight,
  });

  const origin = project(routes[0].geometry[0].lat, routes[0].geometry[0].lon);
  const lastRoute = routes[0].geometry[routes[0].geometry.length - 1];
  const destination = project(lastRoute.lat, lastRoute.lon);

  return (
    <section className="route-map-shell" aria-labelledby="route-map-title">
      <div className="section-header">
        <div>
          <div className="eyebrow">Mapa comparativo</div>
          <h3 id="route-map-title">Trazado relativo de las variantes</h3>
        </div>
        <div className="map-legend" aria-label="Leyenda de rutas">
          {routes.map((item) => (
            <span className="legend-item" key={item.key}>
              <span className="legend-swatch" style={{ backgroundColor: item.color }} />
              {item.label}
            </span>
          ))}
        </div>
      </div>

      <div className="map-frame">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Mapa comparativo de rutas">
          <rect x="0" y="0" width={width} height={height} rx="24" className="map-backdrop" />
          <g className="map-grid">
            {[0.2, 0.4, 0.6, 0.8].map((ratio) => (
              <line
                key={`vertical-${ratio}`}
                x1={padding + usableWidth * ratio}
                y1={padding}
                x2={padding + usableWidth * ratio}
                y2={height - padding}
              />
            ))}
            {[0.25, 0.5, 0.75].map((ratio) => (
              <line
                key={`horizontal-${ratio}`}
                x1={padding}
                y1={padding + usableHeight * ratio}
                x2={width - padding}
                y2={padding + usableHeight * ratio}
              />
            ))}
          </g>

          {routes.map((item) => {
            const points = item.geometry
              .map((point) => project(point.lat, point.lon))
              .map((point) => `${point.x},${point.y}`)
              .join(" ");
            return (
              <polyline
                key={item.key}
                points={points}
                fill="none"
                stroke={item.color}
                strokeWidth={item.key === "reference" ? 8 : 6}
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeDasharray={item.key === "reference" ? "14 10" : undefined}
                opacity={item.key === "reference" ? 0.8 : 0.95}
              />
            );
          })}

          <circle cx={origin.x} cy={origin.y} r="8" className="map-origin" />
          <circle cx={destination.x} cy={destination.y} r="8" className="map-destination" />
        </svg>
      </div>
    </section>
  );
}
