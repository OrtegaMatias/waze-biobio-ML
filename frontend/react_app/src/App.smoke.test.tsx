import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import App from "./App";

function buildResponse(payload: unknown) {
  return {
    ok: true,
    json: async () => payload,
    text: async () => JSON.stringify(payload),
  };
}

describe("App smoke flow", () => {
  afterEach(() => {
    window.history.pushState({}, "", "/");
    vi.unstubAllGlobals();
  });

  it("renders the product planner at '/' and returns user-facing routes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;

        if (url.endsWith("/system/bootstrap")) {
          return buildResponse({ status: "running" });
        }

        if (url.endsWith("/readyz")) {
          return buildResponse({
            status: "ready",
            ready: true,
            message: "Backend listo para planificar viajes.",
            dataset_profile: "gran_concepcion",
            bootstrap: {
              status: "completed",
              message: "Infraestructura lista",
              percent: 100,
              routing_nodes: 100,
              routing_segments: 20,
              duration_ms: 1200,
              dataset_profile: "gran_concepcion",
              quality: null,
            },
          });
        }

        if (url.endsWith("/routes/plan")) {
          return buildResponse({
            selected_route_key: "least_congestion",
            routes: [
              {
                key: "base",
                label: "Ruta base",
                badges: [{ key: "base", label: "Base" }],
                duration_min: 9.5,
                distance_km: 3.0,
                delay_min: 1.4,
                risk_level: "medium",
                summary: "9.5 min en total, 1 zonas historicas en ruta.",
                geometry: [
                  { lat: -36.82, lon: -73.05 },
                  { lat: -36.816, lon: -73.047 },
                  { lat: -36.81, lon: -73.04 },
                ],
                top_alerts: [],
                why_changed: ["Esta ruta usa Dijkstra puro y prioriza el trayecto base mas directo."],
                top_penalized_segments: [],
                top_preferred_vias: [],
                incident_exposure: {
                  total_incident_segments: 1,
                  matched_incident_segments: 1,
                  congestion_segments: 1,
                  accident_segments: 0,
                  exposure_minutes: 1.4,
                },
              },
              {
                key: "least_congestion",
                label: "Menor congestion",
                badges: [{ key: "least_congestion", label: "Menor congestion" }],
                duration_min: 7.5,
                distance_km: 3.2,
                delay_min: 0,
                risk_level: "low",
                summary: "7.5 min en total, 0 zonas historicas en ruta.",
                geometry: [
                  { lat: -36.82, lon: -73.05 },
                  { lat: -36.815, lon: -73.045 },
                  { lat: -36.81, lon: -73.04 },
                ],
                top_alerts: [],
                why_changed: ["La ruta evita zonas con mayor congestion historica del horario."],
                top_penalized_segments: [],
                top_preferred_vias: [{ via: "Barros Arana", factor: 0.5, reason: "Via favorecida." }],
                incident_exposure: {
                  total_incident_segments: 1,
                  matched_incident_segments: 0,
                  congestion_segments: 1,
                  accident_segments: 0,
                  exposure_minutes: 0,
                },
              },
              {
                key: "healthiest",
                label: "Mas saludable",
                badges: [{ key: "healthiest", label: "Mas saludable" }],
                duration_min: 8.1,
                distance_km: 3.5,
                delay_min: 0.2,
                risk_level: "low",
                summary: "8.1 min en total, 0 zonas historicas en ruta.",
                geometry: [
                  { lat: -36.82, lon: -73.05 },
                  { lat: -36.818, lon: -73.047 },
                  { lat: -36.81, lon: -73.04 },
                ],
                top_alerts: [],
                why_changed: ["Pasa por menos zonas con congestion historica."],
                top_penalized_segments: [],
                top_preferred_vias: [],
                incident_exposure: {
                  total_incident_segments: 1,
                  matched_incident_segments: 0,
                  congestion_segments: 1,
                  accident_segments: 0,
                  exposure_minutes: 0,
                },
              },
            ],
            summary: {
              eta_total_min: 7.5,
              distance_km: 3.2,
              delay_min: 0,
              alerts_on_route: 0,
              main_reason: "La ruta evita zonas con mayor congestion historica del horario.",
            },
            alerts: [],
            hotspots: [
              {
                lat: -36.819,
                lon: -73.045,
                weight: 0.4,
                day: "Wednesday",
                bucket: "Punta AM (06-09h)",
                segment_id: "seg-1",
              },
            ],
            map_bounds: {
              lat_min: -36.82,
              lat_max: -36.81,
              lon_min: -73.05,
              lon_max: -73.04,
            },
          });
        }

        throw new Error(`Unexpected fetch call: ${url}`);
      }),
    );

    render(<App />);

    await waitFor(() => expect(screen.getByText(/planifica tu viaje con congestion historica/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole("button", { name: /planificar viaje/i })).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: /planificar viaje/i }));

    await waitFor(() => expect(screen.getAllByText(/menor congestion/i).length).toBeGreaterThan(0));
    expect(screen.getAllByText(/ruta base/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/mas saludable/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/la ruta evita zonas con mayor congestion historica del horario/i).length).toBeGreaterThan(0);
    expect(screen.queryByText(/busqueda enriquecida desactivada/i)).not.toBeInTheDocument();
  });

  it("keeps the academic experience available at '/demo'", async () => {
    window.history.pushState({}, "", "/demo");

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;

        if (url.endsWith("/system/bootstrap")) {
          return buildResponse({ status: "running" });
        }

        if (url.endsWith("/readyz")) {
          return buildResponse({
            status: "ready",
            ready: true,
            message: "Backend listo para planificar viajes.",
            dataset_profile: "gran_concepcion",
            bootstrap: {
              status: "completed",
              message: "Infraestructura lista",
              percent: 100,
              routing_nodes: 100,
              routing_segments: 20,
              duration_ms: 1200,
              dataset_profile: "gran_concepcion",
              quality: {
                status: "warning",
                dataset_profile: "gran_concepcion",
                duplicate_incident_sources: true,
                date_range: { start: "2025-07-01", end: "2025-07-31", days: 31 },
                missing_via_ratio: 0.08,
                anomalous_communes: [],
                raw_counts: { accidents: 1, congestions: 1, combined: 2 },
                warnings: [],
                notes: [],
              },
            },
          });
        }

        if (url.endsWith("/system/dataset")) {
          return buildResponse({
            current: "gran_concepcion",
            current_label: "Gran Concepcion",
            available: [{ key: "gran_concepcion", label: "Gran Concepcion" }],
          });
        }

        if (url.endsWith("/system/demo-scenarios")) {
          return buildResponse({
            scenarios: [
              {
                id: "demo",
                title: "Centro demo",
                description: "Caso de prueba",
                origin: { lat: -36.82, lon: -73.05 },
                destination: { lat: -36.81, lon: -73.04 },
                day_of_week: "Wednesday",
                departure_hour: 8,
                profile: "usuario_demo",
                recommended_focus: "Mostrar balance",
              },
            ],
          });
        }

        if (url.endsWith("/recommendations/playground")) {
          return buildResponse({
            ubcf: [{ via: "Barros Arana", estimated_rating: 4.5, strategy: "ubcf" }],
            ibcf: [{ via: "O'Higgins", estimated_rating: 4.2, strategy: "ibcf" }],
          });
        }

        if (url.endsWith("/routes/optimal")) {
          return buildResponse({
            reference: {
              distance_km: 3.4,
              estimated_duration_min: 10,
              extra_delay_min: 2,
              risk_score: 12,
              geometry: [
                { lat: -36.82, lon: -73.05 },
                { lat: -36.815, lon: -73.045 },
                { lat: -36.81, lon: -73.04 },
              ],
              why_changed: ["Ruta base sin sesgo colaborativo."],
              top_penalized_segments: [],
              top_preferred_vias: [],
              incident_exposure: {
                total_incident_segments: 2,
                matched_incident_segments: 1,
                congestion_segments: 1,
                accident_segments: 0,
                exposure_minutes: 2,
              },
            },
            ubcf: {
              distance_km: 3.1,
              estimated_duration_min: 8,
              extra_delay_min: 0,
              risk_score: 5,
              geometry: [
                { lat: -36.82, lon: -73.05 },
                { lat: -36.817, lon: -73.052 },
                { lat: -36.81, lon: -73.04 },
              ],
              why_changed: ["Esta variante combina congestion historica con un perfil sintetico."],
              top_penalized_segments: [],
              top_preferred_vias: [{ via: "Barros Arana", factor: 0.4, reason: "x" }],
              incident_exposure: {
                total_incident_segments: 1,
                matched_incident_segments: 0,
                congestion_segments: 1,
                accident_segments: 0,
                exposure_minutes: 0,
              },
            },
            ibcf: {
              distance_km: 3.6,
              estimated_duration_min: 9,
              extra_delay_min: 1,
              risk_score: 8,
              geometry: [
                { lat: -36.82, lon: -73.05 },
                { lat: -36.818, lon: -73.047 },
                { lat: -36.81, lon: -73.04 },
              ],
              why_changed: ["Perfil por vias."],
              top_penalized_segments: [],
              top_preferred_vias: [{ via: "O'Higgins", factor: 0.5, reason: "x" }],
              incident_exposure: {
                total_incident_segments: 1,
                matched_incident_segments: 1,
                congestion_segments: 1,
                accident_segments: 0,
                exposure_minutes: 1,
              },
            },
            personalized: null,
            comparison: {
              fastest_variant: "ubcf",
              safest_variant: "ubcf",
              lowest_exposure_variant: "ubcf",
              best_balance_variant: "ubcf",
              deltas: [],
            },
          });
        }

        throw new Error(`Unexpected fetch call: ${url}`);
      }),
    );

    render(<App />);

    await waitFor(() => expect(screen.getByText(/flujo principal de la aplicacion academica/i)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /generar comparacion/i })).toBeInTheDocument();
    expect(screen.getByText(/escenario curado/i)).toBeInTheDocument();
  });
});
