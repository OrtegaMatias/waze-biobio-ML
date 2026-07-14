import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { planRoute } from "../api";
import { PlanPage } from "./PlanPage";

vi.mock("../api", () => ({
  getCongestionDates: vi.fn(async () => ({
    available_dates: [],
    missing_dates: [],
    rain_dates: [],
  })),
  getCongestionHours: vi.fn(async (date: string) => ({
    date,
    available_hours: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    count: 15,
    data_source: "test.csv",
  })),
  getCycleways: vi.fn(async () => ({ features: [] })),
  getEnvironmentalImpact: vi.fn(async () => null),
  getPm25Snapshot: vi.fn(async () => null),
  getReadiness: vi.fn(async () => ({ ready: true, status: "ready" })),
  getUrbanWellbeing: vi.fn(async () => ({ features: [] })),
  planRoute: vi.fn(),
  startBootstrap: vi.fn(async () => ({ status: "running" })),
}));

vi.mock("../components/PlanningMap", () => ({
  DEFAULT_WELLBEING_VISIBILITY: {
    green_space: false,
    blue_space: false,
    tree_cover: false,
    public_space: false,
    sustainability: false,
    cycleway: false,
  },
  WELLBEING_LAYER_OPTIONS: [
    { category: "green_space", label: "Parques y áreas verdes", description: "Parques" },
    { category: "blue_space", label: "Lagos, lagunas y cursos de agua", description: "Agua" },
    { category: "tree_cover", label: "Sectores arbolados", description: "Árboles" },
    { category: "public_space", label: "Plazas y espacios públicos", description: "Plazas" },
    { category: "sustainability", label: "Puntos de reciclaje", description: "Reciclaje" },
  ],
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
  beforeEach(() => {
    window.localStorage.clear();
    window.localStorage.setItem("wbm_onboarding_seen", "true");
    vi.clearAllMocks();
  });

  it("guides map selection, preserves exact coordinates and enables planning only when complete", async () => {
    render(
      <MemoryRouter>
        <PlanPage />
      </MemoryRouter>,
    );

    expect(screen.getByTestId("origin-marker")).toHaveTextContent("none");
    expect(screen.getByTestId("destination-marker")).toHaveTextContent("none");
    expect(screen.getByRole("button", { name: /marca el origen en el mapa/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /selecciona origen y destino/i })).toBeDisabled();
    expect(screen.getByRole("dialog", { name: /c.mo usar el planificador/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /cerrar ayuda/i }));
    expect(screen.queryByRole("dialog", { name: /c.mo usar el planificador/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /c.mo usar el planificador/i }));
    expect(screen.getByRole("dialog", { name: /c.mo usar el planificador/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Elegir primer punto" }));

    await waitFor(() => expect(screen.getByTestId("origin-marker")).toHaveTextContent("-36.81,-73.02"));
    expect(screen.getByTestId("destination-marker")).toHaveTextContent("none");
    expect(screen.getByRole("button", { name: /origen seleccionado/i })).toHaveTextContent("-36.81000, -73.02000");
    expect(screen.getByRole("button", { name: /marca el destino en el mapa/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/haz clic en el mapa para marcar el destino/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Elegir segundo punto" }));

    await waitFor(() => expect(screen.getByTestId("destination-marker")).toHaveTextContent("-36.79,-73.07"));
    expect(screen.getByTestId("origin-marker")).toHaveTextContent("-36.81,-73.02");
    const planButton = await screen.findByRole("button", { name: /planificar viaje/i });
    await waitFor(() => expect(planButton).toBeEnabled());

    fireEvent.click(planButton);
    await waitFor(() => expect(planRoute).toHaveBeenCalledWith(expect.objectContaining({
      origin: { lat: -36.81, lon: -73.02 },
      destination: { lat: -36.79, lon: -73.07 },
    })));
  });

  it("swaps partial and complete points together with their map markers", async () => {
    render(
      <MemoryRouter>
        <PlanPage />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Elegir primer punto" }));
    await waitFor(() => expect(screen.getByTestId("origin-marker")).toHaveTextContent("-36.81,-73.02"));

    fireEvent.click(screen.getByRole("button", { name: /intercambiar origen y destino/i }));
    expect(screen.getByTestId("origin-marker")).toHaveTextContent("none");
    expect(screen.getByTestId("destination-marker")).toHaveTextContent("-36.81,-73.02");

    fireEvent.click(screen.getByRole("button", { name: "Elegir segundo punto" }));
    await waitFor(() => expect(screen.getByTestId("origin-marker")).toHaveTextContent("-36.79,-73.07"));

    fireEvent.click(screen.getByRole("button", { name: /intercambiar origen y destino/i }));
    expect(screen.getByTestId("origin-marker")).toHaveTextContent("-36.81,-73.02");
    expect(screen.getByTestId("destination-marker")).toHaveTextContent("-36.79,-73.07");
  });
});
