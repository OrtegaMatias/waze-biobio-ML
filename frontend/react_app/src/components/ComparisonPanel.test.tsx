import { render, screen } from "@testing-library/react";

import { ComparisonPanel } from "./ComparisonPanel";

describe("ComparisonPanel", () => {
  it("renders comparison highlights and route reasons", () => {
    render(
      <ComparisonPanel
        route={{
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
            why_changed: ["Ruta mas corta sin sesgo colaborativo."],
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
            distance_km: 3.8,
            estimated_duration_min: 9,
            extra_delay_min: 0,
            risk_score: 5,
            geometry: [
              { lat: -36.82, lon: -73.05 },
              { lat: -36.817, lon: -73.052 },
              { lat: -36.81, lon: -73.04 },
            ],
            why_changed: ["Esta variante combina congestion historica con un perfil de viajero sintetico."],
            top_penalized_segments: [{ segment_id: "seg-1", via: "Barros Arana", comuna: "Concepción", event_type: "Congestión", impact_score: 4.2, reason: "x" }],
            top_preferred_vias: [{ via: "O'Higgins", factor: 0.4, reason: "x" }],
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
            estimated_duration_min: 9.5,
            extra_delay_min: 1,
            risk_score: 8,
            geometry: [
              { lat: -36.82, lon: -73.05 },
              { lat: -36.818, lon: -73.047 },
              { lat: -36.81, lon: -73.04 },
            ],
            why_changed: ["Perfil por vías."],
            top_penalized_segments: [],
            top_preferred_vias: [],
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
        }}
      />,
    );

    expect(screen.getByText(/llegar antes/i)).toBeInTheDocument();
    expect(screen.getAllByText(/perfil por usuarios/i)[0]).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /mapa comparativo de rutas/i })).toBeInTheDocument();
    expect(screen.getByText(/barros arana/i)).toBeInTheDocument();
    expect(screen.getByText(/o'higgins/i)).toBeInTheDocument();
  });
});
