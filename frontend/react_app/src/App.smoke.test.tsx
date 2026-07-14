import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import App from "./App";

vi.mock("./components/PlanningMap", () => ({
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
    <div data-testid="planning-map">
      <button
        type="button"
        onClick={() => props.onPickPoint(props.activePin, { lat: -36.8271, lon: -73.0496 })}
      >
        Marcar primer punto en mapa
      </button>
      <button
        type="button"
        onClick={() => props.onPickPoint(props.activePin, { lat: -36.826, lon: -73.0504 })}
      >
        Marcar segundo punto en mapa
      </button>
    </div>
  ),
}));

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
    window.localStorage.clear();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders the product planner at '/' and returns user-facing routes", async () => {
    vi.spyOn(window, "scrollTo").mockImplementation(() => undefined);
    const observedRequests = {
      environmentalImpactUrls: [] as string[],
      planBodies: [] as Array<Record<string, unknown>>,
    };

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
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

        if (url.endsWith("/metadata/cycleways")) {
          return buildResponse({
            type: "FeatureCollection",
            name: "gran_concepcion_cycleways",
            features: [],
          });
        }

        if (url.endsWith("/metadata/urban-wellbeing")) {
          return buildResponse({
            type: "FeatureCollection",
            name: "gran_concepcion_urban_wellbeing",
            features: [],
          });
        }

        if (url.endsWith("/metadata/congestion/dates")) {
          return buildResponse({
            start: "2025-03-13",
            end: "2025-08-22",
            available_dates: ["2025-03-13", "2025-03-14"],
            missing_dates: ["2025-03-15"],
            rain_dates: ["2025-03-14"],
            available_days: 2,
            calendar_days: 3,
            data_source: "CONGESTIONES.csv",
          });
        }

        if (url.includes("/metadata/congestion/hours")) {
          const date = new URL(url, "http://localhost").searchParams.get("date") ?? "2025-03-13";
          return buildResponse({
            date,
            available_hours: [6, 8, 10, 18],
            count: 4,
            data_source: "congestion_aggregated_gran_concepcion_core.csv",
          });
        }

        if (url.includes("/metadata/pm25/snapshot")) {
          return buildResponse({
            available: true,
            requested_at: "2025-01-01 08:00:00",
            average_pm25: 16.2,
            stations: [
              {
                station_id: "104",
                station_name: "Indura",
                lat: -36.769803,
                lon: -73.113708,
                pm25: 16.2,
                category: "Media",
              },
            ],
            date_range: { start: "2025-01-01", end: "2025-12-31" },
            method: "Lectura historica real por estacion, filtrada al año 2025.",
            data_source: "gran_concepcion_pm25_core_hourly_clean.csv",
          });
        }

        if (url.includes("/metadata/environmental-impact")) {
          observedRequests.environmentalImpactUrls.push(url);
          return buildResponse({
            summary: {
              available: true,
              requested_at: "2025-03-13 08:00:00",
              point_count: 1,
              dominant_level: "medium",
              weather: {
                pm25: 16.2,
                pm25_min: 8,
                pm25_max: 55,
                rain_mm: 0,
                has_rain: false,
                rain_label: "Sin lluvia",
                wind_speed: 2,
                wind_speed_kmh: 7.2,
                wind_speed_min: 0,
                wind_speed_max: 10,
                wind_speed_min_kmh: 0,
                wind_speed_max_kmh: 36,
                wind_label: "Viento suave",
                sky_label: "Sin dato",
              },
              messages: ["Condiciones intermedias: revisa PM2.5, viento y lluvia antes de salir."],
              method: "Condiciones ambientales de prueba.",
              data_source: "test",
            },
            points: [
              {
                lat: -36.819,
                lon: -73.045,
                score: 42,
                level: "medium",
                congestion_score: 0.5,
                congestion_level: "medium",
                pm25: 16.2,
                rain_mm: 0,
                wind_speed: 2,
                segment_id: "seg-1",
                via: "Barros Arana",
                comuna: "Concepcion",
                message: "Condiciones ambientales intermedias para movilizarse.",
              },
            ],
          });
        }

        if (url.endsWith("/routes/plan")) {
          observedRequests.planBodies.push(JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>);
          return buildResponse({
            selected_route_key: "least_congested",
            contextual_messages: [
              {
                id: "least_congestion-bike",
                title: "Buenas condiciones para bicicleta",
                detail: "Esta zona presenta baja congestion y buena cobertura ciclista cercana.",
                mode: "bike",
                priority: "high",
              },
            ],
            routes: [
              {
                key: "fastest",
                label: "Llegar antes",
                badges: [{ key: "fastest", label: "Llegar antes" }],
                duration_min: 9.5,
                distance_km: 3.0,
                delay_min: 1.4,
                congestion_score: 30,
                risk_level: "medium",
                summary: "9.5 min en total, 1 zonas historicas en ruta.",
                geometry: [
                  { lat: -36.82, lon: -73.05 },
                  { lat: -36.816, lon: -73.047 },
                  { lat: -36.81, lon: -73.04 },
                ],
                top_alerts: [],
                why_changed: ["Esta ruta usa Dijkstra puro y prioriza la ruta mas corta disponible."],
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
                key: "least_congested",
                label: "Circulación más fluida",
                badges: [{ key: "least_congestion", label: "Circulación más fluida" }],
                duration_min: 7.5,
                distance_km: 3.2,
                delay_min: 0,
                congestion_score: 5,
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
                active_mobility_estimate: {
                  auto_min: 7.5,
                  bike_min: 10.7,
                  walk_min: 40,
                  bike_extra_min: 3.2,
                  walk_extra_min: 32.5,
                  bike_speed_kmh: 18,
                  walk_speed_kmh: 4.8,
                },
                contextual_messages: [
                  {
                    id: "least_congestion-bike",
                    title: "Buenas condiciones para bicicleta",
                    detail: "Esta zona presenta baja congestion y buena cobertura ciclista cercana.",
                    mode: "bike",
                    priority: "high",
                  },
                ],
              },
              {
                key: "healthiest",
                label: "Menor exposición ambiental",
                badges: [{ key: "healthiest", label: "Menor exposición ambiental" }],
                duration_min: 8.1,
                distance_km: 3.5,
                delay_min: 0.2,
                congestion_score: 8,
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

    window.localStorage.setItem("wbm_onboarding_seen", "true");
    Object.defineProperty(window.URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:test"),
    });
    Object.defineProperty(window.URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn(),
    });
    render(<App />);

    await waitFor(() => expect(screen.getByText(/planifica tu viaje con congestion historica/i)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /marcar primer punto en mapa/i }));
    fireEvent.click(screen.getByRole("button", { name: /marcar segundo punto en mapa/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: /planificar viaje/i })).toBeEnabled());
    expect(screen.getByText(/elige fecha y hora para tu viaje/i)).toBeInTheDocument();
    expect(screen.getByText(/fecha y hora/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "16" })).toHaveClass("sunday", "blocked");
    const blockedDate = screen.getByRole("button", { name: "15" });
    expect(blockedDate).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(blockedDate);
    expect(screen.getByText(/no hay datos disponibles para esta fecha/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "13" })).toHaveClass("selected");
    const hourSelect = await screen.findByRole("combobox", { name: /elige hora de salida/i });
    await waitFor(() => expect(hourSelect).toBeEnabled());
    expect(hourSelect).toHaveValue("8");
    expect(screen.getByRole("option", { name: /09:00.*no disponible/i })).toBeDisabled();
    expect(screen.queryByText(/horas tienen registros de congesti.n para esta fecha/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /planificar viaje/i }));

    await waitFor(() => expect(screen.getAllByText(/circulación más fluida/i).length).toBeGreaterThan(0));
    expect(observedRequests.environmentalImpactUrls.length).toBeGreaterThan(0);
    const environmentalUrl = new URL(observedRequests.environmentalImpactUrls.at(-1) ?? "", "http://localhost");
    expect(environmentalUrl.searchParams.get("date")).toBe("2025-03-13");
    expect(environmentalUrl.searchParams.get("hour")).toBe("8");
    expect(observedRequests.planBodies.at(-1)?.congestion_date).toBe("2025-03-13");
    expect(observedRequests.planBodies.at(-1)?.departure_hour).toBe(8);
    expect(screen.getAllByText(/llegar antes/i).length).toBeGreaterThan(0);
    const routePreference = document.querySelector(".route-card.preference-card") as HTMLButtonElement | null;
    expect(routePreference).not.toBeNull();
    fireEvent.click(routePreference!);
    fireEvent.click(screen.getAllByRole("button", { name: /iniciar viaje/i })[0]);
    expect(screen.getByRole("complementary", { name: /visualizaci.n del recorrido planificado/i })).toBeInTheDocument();
    expect(screen.getByText(/orientaci.n ambiental prioritaria/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /ver detalles/i }));
    expect(screen.getByLabelText(/otros mensajes del recorrido/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^finalizar viaje$/i }));
    expect(screen.getByRole("dialog", { name: /llegaste a tu destino/i })).toBeInTheDocument();
    expect(screen.getByText(/viaje completado/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /planificar otro viaje/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /revisar recorrido/i }));
    expect(screen.queryByRole("dialog", { name: /llegaste a tu destino/i })).not.toBeInTheDocument();
    expect(screen.getByRole("complementary", { name: /resumen del recorrido realizado/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /planificar otro viaje/i }));
    expect(screen.queryByRole("complementary", { name: /resumen del recorrido realizado/i })).not.toBeInTheDocument();
    expect(screen.getByText(/marca el origen y el destino en el mapa/i)).toBeInTheDocument();
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
