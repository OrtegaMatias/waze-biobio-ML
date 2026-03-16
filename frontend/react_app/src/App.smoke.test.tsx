import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import App from "./App";

function buildResponse(payload: unknown) {
  return {
    ok: true,
    json: async () => payload,
  };
}

describe("App smoke flow", () => {
  it("loads the scenario and generates a comparison", async () => {
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
            message: "Backend listo para generar la demo.",
            dataset_profile: "concepcion",
            bootstrap: {
              status: "completed",
              message: "Infraestructura lista",
              percent: 100,
              routing_nodes: 100,
              routing_segments: 20,
              duration_ms: 1200,
              dataset_profile: "concepcion",
              quality: {
                status: "warning",
                dataset_profile: "concepcion",
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
            current: "concepcion",
            current_label: "Solo Concepción",
            available: [
              { key: "concepcion", label: "Solo Concepción" },
              { key: "regional", label: "Cobertura regional" },
            ],
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
              why_changed: [
                "Esta variante combina simulación de incidentes históricos con un perfil de viajero sintético.",
              ],
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
              why_changed: ["Perfil por vías."],
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

    await waitFor(() => expect(screen.getByText(/flujo principal de la aplicación/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByRole("button", { name: /generar comparación/i })).toBeEnabled());

    fireEvent.click(screen.getByRole("button", { name: /generar comparación/i }));

    await waitFor(() => expect(screen.getByText(/resultado de la simulación/i)).toBeInTheDocument());
    expect(screen.getAllByText(/perfil por usuarios/i)[0]).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /mapa comparativo de rutas/i })).toBeInTheDocument();
    expect(screen.getByText(/barros arana/i)).toBeInTheDocument();
  });
});
