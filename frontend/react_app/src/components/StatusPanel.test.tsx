import { render, screen } from "@testing-library/react";

import { StatusPanel } from "./StatusPanel";

describe("StatusPanel", () => {
  it("renders ready information", () => {
    render(
      <StatusPanel
        readiness={{
          status: "ready",
          ready: true,
          message: "Backend listo para generar la demo.",
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
              warnings: ["Fuentes idénticas."],
              notes: [],
            },
          },
        }}
        onRefresh={() => {}}
        busy={false}
      />,
    );

    expect(screen.getByText(/ruta segura explicable/i)).toBeInTheDocument();
    expect(screen.getByText(/backend listo para generar la demo/i)).toBeInTheDocument();
    expect(screen.getByText(/fuentes idénticas/i)).toBeInTheDocument();
  });
});
