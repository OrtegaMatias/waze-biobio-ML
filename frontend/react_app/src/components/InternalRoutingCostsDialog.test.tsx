import { fireEvent, render, screen } from "@testing-library/react";

import type { PlanRouteCard, RouteType } from "../types";
import { InternalRoutingCostsDialog } from "./InternalRoutingCostsDialog";

function route(key: RouteType, optimizationCost: number): PlanRouteCard {
  const objective = key === "fastest" ? "fastest" : key === "least_congested" ? "fluent" : "environmental";
  return {
    key,
    label: key,
    badges: [],
    duration_min: 20,
    distance_km: 8,
    delay_min: 2,
    congestion_score: 0.4,
    risk_level: "low",
    summary: "Ruta de prueba",
    geometry: [],
    top_alerts: [],
    why_changed: [],
    top_penalized_segments: [],
    top_preferred_vias: [],
    incident_exposure: {
      total_incident_segments: 0,
      matched_incident_segments: 0,
      congestion_segments: 0,
      accident_segments: 0,
      exposure_minutes: 0,
    },
    optimization_trace: {
      objective,
      cost_model_version: "direct-edge-cost-v1",
      logical_segment_count: 12,
      base_time_min: 18.2,
      congestion_delay_min: 4.3,
      congestion_penalty_min: 1.2,
      stop_penalty_min: 0.4,
      pm25_penalty_min: 0.8,
      adverse_environment_penalty_min: 0.6,
      urban_benefit_min: 2.3,
      optimization_cost_min: optimizationCost,
    },
  };
}

describe("InternalRoutingCostsDialog", () => {
  it("identifies the diagnostic as internal and shows the three route cost traces", () => {
    render(
      <InternalRoutingCostsDialog
        routes={[
          route("fastest", 22.5),
          route("least_congested", 24.1),
          route("healthiest", 21.3),
        ]}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByRole("dialog", { name: "Costos internos de las rutas" })).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent(/no forma parte de la aplicación final/i);
    expect(screen.getByText("Llegar antes")).toBeInTheDocument();
    expect(screen.getByText("Circulación más fluida")).toBeInTheDocument();
    expect(screen.getByText("Menor exposición ambiental")).toBeInTheDocument();
    expect(screen.getAllByText("Costo total de optimización")).toHaveLength(3);
    expect(screen.getAllByText("-2.300")).toHaveLength(3);
    expect(screen.getByText("22.500")).toBeInTheDocument();
    expect(screen.getByText("24.100")).toBeInTheDocument();
    expect(screen.getByText("21.300")).toBeInTheDocument();
  });

  it("closes from the button and the Escape key", () => {
    const onClose = vi.fn();
    render(<InternalRoutingCostsDialog routes={[route("fastest", 22.5)]} onClose={onClose} />);

    fireEvent.click(screen.getByRole("button", { name: "Cerrar costos internos" }));
    fireEvent.keyDown(window, { key: "Escape" });

    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
