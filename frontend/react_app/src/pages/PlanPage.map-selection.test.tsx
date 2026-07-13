import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { PlanPage } from "./PlanPage";

vi.mock("../api", () => ({
  getCongestionDates: vi.fn(async () => ({
    available_dates: [],
    missing_dates: [],
    rain_dates: [],
  })),
  getCycleways: vi.fn(async () => ({ features: [] })),
  getEnvironmentalImpact: vi.fn(async () => null),
  getPm25Snapshot: vi.fn(async () => null),
  getReadiness: vi.fn(async () => ({ ready: true, status: "ready" })),
  getUrbanWellbeing: vi.fn(async () => ({ features: [] })),
  planRoute: vi.fn(),
  reversePlace: vi.fn(async () => null),
  searchPlaces: vi.fn(async () => []),
  startBootstrap: vi.fn(async () => ({ status: "running" })),
}));

vi.mock("../components/PlanningMap", () => ({
  PlanningMap: (props: any) => (
    <div>
      <span data-testid="origin-marker">
        {props.origin ? `${props.origin.lat},${props.origin.lon}` : "none"}
      </span>
      <span data-testid="destination-marker">
        {props.destination ? `${props.destination.lat},${props.destination.lon}` : "none"}
      </span>
      <button
        type="button"
        onClick={() => props.onPickPoint(props.activePin, { lat: -36.81, lon: -73.02 })}
      >
        Elegir primer punto
      </button>
      <button
        type="button"
        onClick={() => props.onPickPoint(props.activePin, { lat: -36.79, lon: -73.07 })}
      >
        Elegir segundo punto
      </button>
    </div>
  ),
}));

describe("PlanPage map point selection", () => {
  it("shows only the clicked marker and preserves exact click coordinates", async () => {
    render(
      <MemoryRouter>
        <PlanPage />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("origin-marker")).toHaveTextContent("none");
    expect(screen.getByTestId("destination-marker")).toHaveTextContent("none");

    fireEvent.click(screen.getByRole("button", { name: "Elegir primer punto" }));

    await waitFor(() => expect(screen.getByTestId("origin-marker")).toHaveTextContent("-36.81,-73.02"));
    expect(screen.getByTestId("destination-marker")).toHaveTextContent("none");

    fireEvent.click(screen.getByRole("button", { name: "Elegir segundo punto" }));

    await waitFor(() => expect(screen.getByTestId("destination-marker")).toHaveTextContent("-36.79,-73.07"));
    expect(screen.getByTestId("origin-marker")).toHaveTextContent("-36.81,-73.02");
  });
});
