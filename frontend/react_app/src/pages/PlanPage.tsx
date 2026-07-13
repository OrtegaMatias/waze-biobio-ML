import { type CSSProperties, startTransition, useDeferredValue, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import onboardingBeforeRouteScreen from "../assets/onboarding/03-before-route.png";
import onboardingRouteTypesScreen from "../assets/onboarding/04-route-types.png";
import onboardingRecommendationScreen from "../assets/onboarding/05-recommendation.png";
import onboardingLayersScreen from "../assets/onboarding/06-layers.png";
import {
  getCongestionDates,
  getCycleways,
  getUrbanWellbeing,
  getEnvironmentalImpact,
  getPm25Snapshot,
  planRoute,
  reversePlace,
  getReadiness,
  searchPlaces,
  startBootstrap,
} from "../api";
import {
  DEFAULT_WELLBEING_VISIBILITY,
  PlanningMap,
  WELLBEING_LAYER_OPTIONS,
  type WellbeingVisibility,
} from "../components/PlanningMap";
import type {
  CongestionDateCoverage,
  CyclewayFeature,
  EnvironmentalImpactResponse,
  MobilityGuidanceMessage,
  PlaceResult,
  PlanRouteResponse,
  Pm25SnapshotResponse,
  ReadinessStatus,
  RoutePoint,
  RouteType,
  TravelStyle,
  UrbanWellbeingCategory,
  UrbanWellbeingFeature,
} from "../types";

type PinKey = "origin" | "destination";

type PlaceSelection = {
  id: string;
  label: string;
  point: RoutePoint;
  bbox: number[] | null;
};

type PlannerState = {
  origin: PlaceSelection;
  destination: PlaceSelection;
  day_of_week: string;
  departure_hour: number;
  travel_style: TravelStyle;
  avoid_congestion: boolean;
};

type RouteGuidancePanelRect = {
  top: number;
  left: number;
  width: number;
  height: number;
};

const RECENT_PLACES_KEY = "wbm_recent_places";
const ONBOARDING_SEEN_KEY = "wbm_onboarding_seen";
const DEFAULT_MAP_STYLE_URL = "local-basic";
const DEFAULT_HISTORY_DATE = "2025-03-13";
const WEEKDAY_LABELS = ["L", "M", "M", "J", "V", "S", "D"];
const MONTH_LABELS = [
  "Enero",
  "Febrero",
  "Marzo",
  "Abril",
  "Mayo",
  "Junio",
  "Julio",
  "Agosto",
  "Septiembre",
  "Octubre",
  "Noviembre",
  "Diciembre",
];

type CalendarDay = {
  date: string;
  day: number;
  inMonth: boolean;
  hasData: boolean;
  isMissing: boolean;
  hasRain: boolean;
};

type OnboardingSlide = {
  eyebrow: string;
  title: string;
  body: string;
  image?: string;
  focus: "purpose" | "difference" | "search" | "calendar" | "plan" | "preferences" | "recommendation" | "layers";
  contentPreview?: {
    welcome: string;
    description: string;
    guidePrompt: string;
  };
  callout?: {
    label: string;
    description?: string;
    shape: "circle" | "rect";
    x: number;
    y: number;
    width: number;
    height: number;
    arrowX: number;
    arrowY: number;
  };
  mapPins?: Array<{
    tone: "origin" | "destination";
    x: number;
    y: number;
  }>;
};

const ONBOARDING_SLIDES: OnboardingSlide[] = [
  {
    eyebrow: "Bienvenida",
    title: "Bienvenidos a",
    body:
      "Esta aplicación combina movilidad, congestión vehicular y parámetros ambientales para ayudarte a programar viajes más rápidos, saludables y sustentables.",
    focus: "purpose",
    contentPreview: {
      welcome: "Bienvenidos a",
      description:
        "Aplicación de navegación urbana que integra factores de movilidad, congestión vehicular y calidad del aire para comparar rutas orientadas a llegar antes, evitar congestión y reducir la exposición ambiental.",
      guidePrompt: "Revisa el paso a paso para aprender a planificar tu viaje.",
    },
  },
  {
    eyebrow: "Primer paso",
    title: "Selecciona fecha y hora",
    body: "",
    image: onboardingBeforeRouteScreen,
    focus: "calendar",
    callout: {
      label:
        "Selecciona en el calendario el día y la hora del viaje para consultar congestión, PM2.5 y condiciones ambientales.",
      shape: "rect",
      x: 79.2,
      y: 1.2,
      width: 20.4,
      height: 63.8,
      arrowX: 77.2,
      arrowY: 36,
    },
  },
  {
    eyebrow: "Segundo paso",
    title: "Marca origen y destino",
    body:
      "Después de definir fecha y hora, marca en el mapa el punto de partida y el destino. Verás el origen en verde y el destino en rojo.",
    image: onboardingBeforeRouteScreen,
    focus: "search",
    callout: {
      label: "Haz un primer clic en el mapa para seleccionar el origen y un segundo clic para marcar el destino.",
      shape: "rect",
      x: 2.1,
      y: 15.2,
      width: 77.4,
      height: 64.5,
      arrowX: 35,
      arrowY: 83,
    },
    mapPins: [
      { tone: "origin", x: 49.7, y: 62.8 },
      { tone: "destination", x: 53.2, y: 51.7 },
    ],
  },
  {
    eyebrow: "Tercer paso",
    title: "Presiona Planifica tu viaje",
    body:
      "Con fecha, hora, origen y destino definidos, presiona Planifica tu viaje para generar las opciones de ruta disponibles.",
    image: onboardingBeforeRouteScreen,
    focus: "plan",
    callout: {
      label: "Presiona Planifica tu viaje para ver las rutas",
      shape: "rect",
      x: 20.8,
      y: 82.7,
      width: 28.7,
      height: 8.8,
      arrowX: 52,
      arrowY: 85,
    },
  },
  {
    eyebrow: "Cuarto paso",
    title: "Elige qué quieres priorizar",
    body:
      "Con el viaje y el horario definidos, compara las rutas: Llegar antes, Circulación más fluida o Menor exposición ambiental.",
    image: onboardingRouteTypesScreen,
    focus: "preferences",
    callout: {
      label: "Llegar antes, fluidez o menor exposición",
      shape: "rect",
      x: 16.7,
      y: 72.5,
      width: 56.8,
      height: 25.8,
      arrowX: 61,
      arrowY: 72,
    },
  },
  {
    eyebrow: "Recomendación explicada",
    title: "Entiende por qué se sugiere una ruta",
    body:
      "Luego revisa la explicación: la recomendación combina PM2.5, congestión, ciclovías, áreas verdes y condiciones del entorno.",
    image: onboardingRecommendationScreen,
    focus: "recommendation",
    callout: {
      label: "Lee los factores de la recomendación",
      shape: "rect",
      x: 1.2,
      y: 11.5,
      width: 26.4,
      height: 76.4,
      arrowX: 29,
      arrowY: 28,
    },
  },
  {
    eyebrow: "Sello sustentable",
    title: "Explora capas para decidir con más contexto",
    body:
      "Finalmente, presiona el botón de capas en el costado derecho del mapa para abrir este panel y revisar congestión, impacto ambiental, ciclovías y entorno.",
    image: onboardingLayersScreen,
    focus: "layers",
    callout: {
      label: "Capas de entorno y movilidad sustentable",
      shape: "rect",
      x: 63,
      y: 2,
      width: 35.5,
      height: 96,
      arrowX: 60,
      arrowY: 18,
    },
  },
];

const ROUTE_DISPLAY_NAMES: Record<RouteType, string> = {
  fastest: "Llegar antes",
  least_congested: "Circulación más fluida",
  healthiest: "Menor exposición ambiental",
};

const ROUTE_PREFERENCES: Record<
  RouteType,
  {
    icon: string;
    title: string;
    tagline: string;
    description: string;
    tone: string;
  }
> = {
  fastest: {
    icon: "\u{1F680}",
    title: ROUTE_DISPLAY_NAMES.fastest,
    tagline: "Prioriza reducir el tiempo",
    description:
      "Prioriza el trayecto con menor tiempo estimado. Puede pasar por zonas con mas congestion o exposicion ambiental.",
    tone: "time-saving",
  },
  least_congested: {
    icon: "\u{1F697}",
    title: ROUTE_DISPLAY_NAMES.least_congested,
    tagline: "Evita sectores con mas congestion",
    description:
      "Evita tramos con mayor congestion para un viaje mas continuo. No garantiza menor PM2.5 ni mejor entorno urbano.",
    tone: "traffic",
  },
  healthiest: {
    icon: "\u{1F33F}",
    title: ROUTE_DISPLAY_NAMES.healthiest,
    tagline: "Reduce la exposicion ambiental",
    description:
      "Prioriza menor PM2.5 y mejores condiciones urbanas, considerando tambien la congestion. Puede tomar mas tiempo que otras rutas.",
    tone: "healthy",
  },
};

function journeyStartMessage(routeType: RouteType): {
  title: string;
  detail: string;
  reward?: string;
  localFact?: string;
} {
  if (routeType === "fastest") {
    return {
      title: "Ruta directa",
      detail: "Prioriza avanzar sin desvio. Si la calidad del aire esta desfavorable, evita ventilacion exterior.",
      localFact: "Dato local: la exposicion a contaminantes puede aumentar cerca de tramos con alto trafico.",
    };
  }
  if (routeType === "least_congested") {
    return {
      title: "Flujo mas estable",
      detail: "Elegiste reducir detenciones. Revisa igualmente PM2.5: menos congestion no siempre implica menor exposicion.",
    };
  }
  return {
    title: "Movilidad consciente",
    detail: "Elegiste una ruta que equilibra tiempo, entorno y menor exposicion ambiental.",
    reward: "+3 estrellas de movilidad consciente",
  };
}

function routeDisplayName(routeType: RouteType): string {
  return ROUTE_DISPLAY_NAMES[routeType];
}

function isRouteType(value: string | undefined): value is RouteType {
  return value === "fastest" || value === "least_congested" || value === "healthiest";
}

const MESSAGE_TYPE_LABELS: Record<MobilityGuidanceMessage["type"], string> = {
  air: "Calidad del aire",
  congestion: "Congestión",
  time: "Tiempo",
  route_attribute: "Entorno saludable",
  weather: "Clima",
  recommendation: "Recomendación",
};

type RouteMetricKey = "duration" | "distance" | "congestion" | "pm25" | "wellbeing";
type EnvironmentConditionKey = "pm25" | "wind" | "rain";

type RouteMetricInfo = {
  label: string;
  short: string;
  represents: string;
  calculation: string[];
  variables: string[];
  limitation: string;
};

type EnvironmentConditionInfo = {
  label: string;
  short: string;
  technical: string;
  recommendation: string;
};

const ROUTE_METRIC_INFO: Record<RouteMetricKey, RouteMetricInfo> = {
  duration: {
    label: "Tiempo estimado",
    short: "Tiempo aproximado de viaje, calculado con distancia, velocidad esperada y congestion historica.",
    represents: "La duracion aproximada del recorrido seleccionado, expresada en minutos.",
    calculation: [
      "La ruta se divide en tramos de calle.",
      "Para cada tramo se estima cuanto demoraria recorrerlo segun su largo y una velocidad esperada.",
      "Si el tramo suele tener congestion en la fecha u hora elegida, se agrega una penalizacion.",
      "Finalmente se suman todos los tramos para obtener el tiempo total.",
    ],
    variables: [
      "Distancia de cada tramo",
      "Velocidad esperada",
      "Congestion historica",
      "Fecha y hora seleccionadas",
      "Preferencia de ruta elegida",
    ],
    limitation: "Es una estimacion basada en datos historicos y modelo de ruta; no es trafico en tiempo real.",
  },
  distance: {
    label: "Distancia",
    short: "Largo total del trayecto, sumando los segmentos de calle que forman la ruta.",
    represents: "La cantidad aproximada de kilometros que recorrerias desde el origen hasta el destino.",
    calculation: [
      "El algoritmo selecciona una secuencia de calles para conectar origen y destino.",
      "Cada calle se representa como uno o mas segmentos.",
      "Se calcula el largo de cada segmento usado por la ruta.",
      "Luego se suman esos largos y se muestran en kilometros.",
    ],
    variables: ["Origen", "Destino", "Segmentos de calle seleccionados", "Geometria de la red vial"],
    limitation: "Puede variar un poco respecto a mediciones de otras aplicaciones si usan otra red vial o redondeos.",
  },
  congestion: {
    label: "Congestion",
    short:
      "Bajo significa que, para la fecha y hora seleccionadas, se espera poco taco en los tramos de esta ruta. No es trafico en vivo.",
    represents:
      "Que tan probable es encontrar demoras por taco en la ruta para la fecha y hora que seleccionaste.",
    calculation: [
      "Se revisan los tramos de calle que forman la ruta.",
      "Para esos tramos se buscan los datos disponibles de la fecha y hora seleccionadas.",
      "Se resume que tan cargados aparecen esos tramos en ese horario.",
      "El resultado se muestra en una categoria simple: Bajo, Medio o Alto.",
      "Bajo: se espera un viaje mas fluido. Medio: puede haber algunas demoras. Alto: hay mayor probabilidad de taco.",
    ],
    variables: [
      "Tramos de la ruta",
      "Datos de congestion disponibles",
      "Fecha seleccionada",
      "Hora seleccionada",
    ],
    limitation:
      "No corresponde a trafico en vivo. Un accidente, corte o evento actual puede cambiar la condicion real del camino.",
  },
  pm25: {
    label: "Exposicion PM2.5",
    short: "Promedio estimado de material particulado fino sobre la ruta para la fecha y hora elegidas.",
    represents:
      "Una referencia de exposicion a PM2.5 durante el trayecto. PM2.5 son particulas muy pequenas presentes en el aire.",
    calculation: [
      "Se toman mediciones historicas de estaciones cercanas para la fecha y hora seleccionadas.",
      "Cuando no hay una estacion justo sobre la calle, se estima el valor usando estaciones cercanas.",
      "Se calcula el PM2.5 aproximado en distintos puntos de la ruta.",
      "Finalmente se promedia ese valor a lo largo del trayecto.",
    ],
    variables: [
      "Fecha seleccionada",
      "Hora seleccionada",
      "Estaciones PM2.5 cercanas",
      "Distancia entre estaciones y ruta",
      "Geometria de la ruta",
    ],
    limitation:
      "Es una estimacion espacial con datos historicos; no reemplaza una medicion personal ni un pronostico oficial.",
  },
  wellbeing: {
    label: "Bienestar urbano",
    short: "Puntaje de cercania a elementos urbanos favorables, como areas verdes, agua o ciclovias.",
    represents: "Que tan favorable puede ser el entorno de la ruta para una experiencia de viaje mas amable.",
    calculation: [
      "Se revisa que elementos urbanos hay cerca de la ruta.",
      "Se consideran elementos como areas verdes, cuerpos de agua, ciclovias y puntos de reciclaje.",
      "La ruta obtiene mas puntaje si tiene mayor cercania o cobertura de esos elementos.",
      "El resultado se resume en una escala de 0 a 100.",
    ],
    variables: [
      "Areas verdes cercanas",
      "Cuerpos de agua cercanos",
      "Ciclovias",
      "Puntos de reciclaje",
      "Distancia entre esos elementos y la ruta",
    ],
    limitation: "No significa directamente aire mas limpio; mide condiciones urbanas favorables alrededor de la ruta.",
  },
};

const ENVIRONMENT_CONDITION_INFO: Record<EnvironmentConditionKey, EnvironmentConditionInfo> = {
  pm25: {
    label: "PM2.5",
    short:
      "PM2.5 son particulas finas presentes en el aire, mucho menores que el ancho de un cabello. Pueden entrar profundamente en los pulmones.",
    technical:
      "Algunas particulas pueden alcanzar la sangre. Pueden agravar asma y enfermedades respiratorias o cardiovasculares, con mayor riesgo en ninos, adultos mayores, embarazadas y personas con enfermedades cardiacas o respiratorias.",
    recommendation:
      "Como referencia sanitaria general, la OMS recomienda no superar 15 ug/m3 como promedio de 24 horas. La app muestra valores horarios: sirven como contexto, pero no deben compararse directamente con un limite diario.",
  },
  wind: {
    label: "Viento",
    short: "Indica cuanta ventilacion hay para dispersar contaminantes.",
    technical:
      "Viento suave: menos de 20 km/h, menor dispersion. Viento moderado: 20 a menos de 39 km/h, ayuda a dispersar. Viento fuerte: desde 39 km/h, mayor dispersion.",
    recommendation:
      "La barra ordena intensidad: suave a la izquierda y fuerte a la derecha. Para colorear la nube ambiental, el viento se compara con sus minimos y maximos historicos disponibles para la misma hora.",
  },
  rain: {
    label: "Lluvia",
    short: "Resume si hubo lluvia en el horario seleccionado.",
    technical:
      "La lluvia puede ayudar a remover particulas suspendidas, pero tambien puede hacer el viaje menos comodo o mas lento.",
    recommendation:
      "En el calculo de nube ambiental, la lluvia entrega alivio al puntaje. Si hay lluvia fuerte, considera tiempo extra y prioriza seguridad.",
  },
};

function localRangePercent(value?: number | null, min?: number | null, max?: number | null): number | null {
  if (value == null || min == null || max == null || max <= min) {
    return null;
  }
  return Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));
}

function formatConditionValue(value?: number | null, unit = ""): string {
  if (value == null) {
    return "Sin dato";
  }
  return `${value.toFixed(1)}${unit ? ` ${unit}` : ""}`;
}

function makeSelection(id: string, label: string, point: RoutePoint, bbox: number[] | null = null): PlaceSelection {
  return { id, label, point, bbox };
}

function readRecentPlaces(): PlaceResult[] {
  try {
    const stored = window.localStorage.getItem(RECENT_PLACES_KEY);
    if (!stored) {
      return [];
    }
    const parsed = JSON.parse(stored) as PlaceResult[];
    return Array.isArray(parsed) ? parsed.slice(0, 6) : [];
  } catch {
    return [];
  }
}

function hasSeenOnboarding(): boolean {
  try {
    return window.localStorage.getItem(ONBOARDING_SEEN_KEY) === "true";
  } catch {
    return false;
  }
}

function rememberOnboardingSeen() {
  try {
    window.localStorage.setItem(ONBOARDING_SEEN_KEY, "true");
  } catch {
    // Storage can be unavailable in private browsing or restricted test environments.
  }
}

function writeRecentPlaces(items: PlaceResult[]) {
  window.localStorage.setItem(RECENT_PLACES_KEY, JSON.stringify(items.slice(0, 6)));
}

function upsertRecentPlace(current: PlaceResult[], next: PlaceResult): PlaceResult[] {
  const deduped = current.filter((item) => item.id !== next.id);
  return [next, ...deduped].slice(0, 6);
}

function selectionFromPlace(place: PlaceResult): PlaceSelection {
  return makeSelection(place.id, place.label, { lat: place.lat, lon: place.lon }, place.bbox);
}

function buildManualLabel(point: RoutePoint): string {
  return `Punto manual (${point.lat.toFixed(4)}, ${point.lon.toFixed(4)})`;
}

function isoDateFromLocalDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function localDateFromIso(value: string): Date {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day);
}

function dayOfWeekFromIso(value: string): string {
  return localDateFromIso(value).toLocaleDateString("en-US", { weekday: "long" });
}

function monthKeyFromIso(value: string): string {
  return value.slice(0, 7);
}

function shiftMonth(monthKey: string, delta: number): string {
  const [year, month] = monthKey.split("-").map(Number);
  const next = new Date(year, month - 1 + delta, 1);
  return `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(2, "0")}`;
}

function monthLabel(monthKey: string): string {
  const [year, month] = monthKey.split("-").map(Number);
  return `${MONTH_LABELS[month - 1]} ${year}`;
}

function buildCalendarDays(
  monthKey: string,
  availableDates: Set<string>,
  missingDates: Set<string>,
  rainDates: Set<string>,
): CalendarDay[] {
  const [year, month] = monthKey.split("-").map(Number);
  const first = new Date(year, month - 1, 1);
  const firstMondayOffset = (first.getDay() + 6) % 7;
  const start = new Date(year, month - 1, 1 - firstMondayOffset);
  return Array.from({ length: 42 }, (_, index) => {
    const date = new Date(start);
    date.setDate(start.getDate() + index);
    const iso = isoDateFromLocalDate(date);
    return {
      date: iso,
      day: date.getDate(),
      inMonth: date.getMonth() === month - 1,
      hasData: availableDates.has(iso),
      isMissing: missingDates.has(iso),
      hasRain: rainDates.has(iso),
    };
  });
}

function OnboardingPreview({
  focus,
  onIntroClick,
}: {
  focus: OnboardingSlide["focus"];
  onIntroClick?: () => void;
}) {
  const slide = ONBOARDING_SLIDES.find((item) => item.focus === focus) ?? ONBOARDING_SLIDES[0];
  if (slide.contentPreview) {
    return (
      <div className={`onboarding-preview onboarding-preview-content onboarding-focus-${focus}`}>
        <div className="onboarding-brand-panel">
          <p className="onboarding-brand-welcome">{slide.contentPreview.welcome}</p>
          <img
            className="wise-route-logo-image"
            src="/wise-route-logo.png"
            alt="WiseRouteApp"
            onError={(event) => {
              event.currentTarget.classList.add("missing-logo");
            }}
          />
          <div className="wise-route-logo-missing">
            <strong>WiseRouteApp</strong>
            <span>Falta el archivo /wise-route-logo.png</span>
          </div>
          <p>{slide.contentPreview.description}</p>
          <p>{slide.contentPreview.guidePrompt}</p>
          <button className="onboarding-inline-start" type="button" onClick={onIntroClick}>
            Ver paso a paso
          </button>
        </div>
      </div>
    );
  }

  if (!slide.image || !slide.callout) {
    return null;
  }

  const { callout } = slide;
  const calloutStyle = {
    "--callout-x": `${callout.x}%`,
    "--callout-y": `${callout.y}%`,
    "--callout-width": `${callout.width}%`,
    "--callout-height": `${callout.height}%`,
    "--callout-label-x": `${callout.arrowX}%`,
    "--callout-label-y": `${callout.arrowY}%`,
  } as CSSProperties;

  return (
    <div className={`onboarding-preview onboarding-preview-frame onboarding-focus-${focus}`} style={calloutStyle}>
      <div className="onboarding-image-stage" aria-hidden="true">
        <img className="onboarding-base-image" src={slide.image} alt="" />
        <span className={`preview-callout-ring ${callout.shape}`} />
        {slide.mapPins?.map((pin) => (
          <span
            className={`preview-map-pin ${pin.tone}`}
            key={pin.tone}
            style={{ left: `${pin.x}%`, top: `${pin.y}%` }}
          />
        ))}
        <span className="preview-callout-note">
          <strong>{callout.label}</strong>
          {callout.description ? <span>{callout.description}</span> : null}
        </span>
      </div>
    </div>
  );
}

export function getRouteMessage(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
  allRoutes: PlanRouteResponse["routes"],
): string {
  const messages = buildRouteInsightMessages(selectedRouteType, selectedRoute, allRoutes, null);
  return messages[0]?.detail ?? `Mostrando detalle de ${routeDisplayName(selectedRouteType).toLowerCase()}.`;
}

const PM25_HIGH_UG_M3 = 50;
const PM25_ELEVATED_UG_M3 = 35;
const RECOMMENDED_ROUTE_MAX_EXTRA_MIN = 3;

function durationDelta(candidate?: PlanRouteResponse["routes"][number], reference?: PlanRouteResponse["routes"][number]) {
  if (!candidate || !reference) {
    return null;
  }
  return Number((candidate.duration_min - reference.duration_min).toFixed(1));
}

function routePm25(
  route: PlanRouteResponse["routes"][number],
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
): number | null {
  if (route.pm25_exposure?.available) {
    return route.pm25_exposure.average_pm25;
  }
  return weather?.pm25 ?? null;
}

function isHighPm25(
  route: PlanRouteResponse["routes"][number],
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
): boolean {
  const pm25 = routePm25(route, weather);
  return route.pm25_exposure?.category === "Alta" || (pm25 !== null && pm25 >= PM25_HIGH_UG_M3);
}

function isElevatedPm25(
  route: PlanRouteResponse["routes"][number],
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
): boolean {
  const pm25 = routePm25(route, weather);
  return isHighPm25(route, weather) || (pm25 !== null && pm25 >= PM25_ELEVATED_UG_M3);
}

function wellbeingCategoryLabel(category: string): string {
  const labels: Record<string, string> = {
    green_space: "area verde",
    blue_space: "borde de agua",
    tree_cover: "arbolado",
    public_space: "espacio publico",
    sustainability: "sostenibilidad",
    cycleway: "ciclovia",
  };
  return labels[category] ?? "entorno urbano";
}

function routeWellbeingEvidence(route: PlanRouteResponse["routes"][number]): string | null {
  const wellbeing = route.urban_wellbeing;
  if (!wellbeing || !wellbeing.available || wellbeing.score <= 0) {
    return null;
  }
  const features = wellbeing.top_features ?? [];
  if (!features.length) {
    return `Esta ruta mantiene menor exposicion vehicular y suma un aporte urbano positivo de ${wellbeing.score.toFixed(1)} puntos.`;
  }
  const evidence = features
    .slice(0, 2)
    .map((feature) => `${feature.name} (${wellbeingCategoryLabel(feature.category)})`)
    .join(" y ");
  return `Esta ruta mantiene menor exposicion vehicular y suma entorno urbano favorable: ${evidence}.`;
}

export function airQualityMessage(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
): MobilityGuidanceMessage {
  void selectedRouteType;
  const highPm25 = isHighPm25(selectedRoute, weather);
  const elevatedPm25 = isElevatedPm25(selectedRoute, weather);
  const detail = highPm25
    ? "La calidad del aire es poco favorable. Una mayor exposición a material particulado puede aumentar molestias respiratorias e irritación en ojos o garganta, especialmente en personas sensibles."
    : elevatedPm25
      ? "La calidad del aire está en un nivel intermedio. Si eres sensible a la contaminación, conviene evitar esfuerzos intensos durante el trayecto."
      : "La calidad del aire es favorable para realizar este trayecto. Si la distancia y las condiciones lo permiten, también puede ser una buena opción para caminar o usar bicicleta.";

  return {
    id: "base-air-quality",
    title: "Calidad del aire",
    detail,
    type: "air",
    priority: highPm25 ? "high" : elevatedPm25 ? "medium" : "low",
  };
}

export function congestionMessage(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
): MobilityGuidanceMessage {
  const level = congestionLevel(selectedRoute.congestion_score ?? 0);
  const details: Record<RouteType, Record<string, string>> = {
    fastest: {
      Bajo: "Esta ruta presenta poca congestión y prioriza llegar antes, por lo que el viaje debería ser más directo y constante.",
      Medio: "Esta ruta puede presentar algunas detenciones, pero sigue priorizando llegar antes frente a otras alternativas.",
      Alto: "Esta ruta presenta alta congestión. Aun así, puede seguir siendo la alternativa más rápida, aunque el viaje podría tener más detenciones.",
    },
    least_congested: {
      Bajo: "Esta ruta presenta poca congestión y favorece una circulación más constante durante el viaje.",
      Medio: "Esta ruta busca evitar los sectores más congestionados, aunque todavía puede presentar algunas detenciones.",
      Alto: "Aunque esta alternativa prioriza una circulación más fluida, el horario seleccionado todavía presenta congestión alta en parte del trayecto.",
    },
    healthiest: {
      Bajo: "Esta ruta presenta poca congestión, lo que favorece un trayecto más cómodo junto con mejores condiciones del entorno.",
      Medio: "Esta ruta puede presentar algunas detenciones, pero prioriza mejores condiciones urbanas y ambientales durante el trayecto.",
      Alto: "Esta ruta presenta congestión alta en el horario seleccionado. Aun así, prioriza mejores condiciones del entorno frente a otras alternativas.",
    },
  };

  return {
    id: "base-congestion",
    title: "Congestión",
    detail: details[selectedRouteType][level],
    type: "congestion",
    priority: level === "Alto" ? "high" : level === "Medio" ? "medium" : "low",
  };
}

export function timeMessage(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
): MobilityGuidanceMessage {
  const minutes = selectedRoute.duration_min.toFixed(0);
  const details: Record<RouteType, string> = {
    fastest: `Esta ruta demora aproximadamente ${minutes} minutos y prioriza llegar antes. Puede ahorrar tiempo, aunque podría sacrificar menor exposición ambiental o mejores condiciones urbanas.`,
    least_congested: `Esta ruta demora aproximadamente ${minutes} minutos y prioriza una circulación más constante, evitando sectores con mayor congestión. Puede ser más cómoda, aunque no necesariamente es la más saludable.`,
    healthiest: `Esta ruta demora aproximadamente ${minutes} minutos. Puede que existan alternativas más rápidas, pero esta prioriza mejores condiciones del entorno durante el trayecto.`,
  };

  return {
    id: "base-travel-time",
    title: "Tiempo y prioridad",
    detail: details[selectedRouteType],
    type: "time",
    priority: "low",
  };
}

export function healthyEnvironmentMessage(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
  allRoutes: PlanRouteResponse["routes"],
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
): MobilityGuidanceMessage | null {
  if (selectedRouteType === "healthiest") {
    const evidence = routeWellbeingEvidence(selectedRoute);
    return {
      id: "healthy-route-environment",
      title: "Entorno saludable",
      detail: evidence ?? "No se encontro una alternativa con aporte urbano favorable; se priorizo menor exposicion vehicular, evitando mas exposicion directa al trafico y a tramos congestionados.",
      type: "route_attribute",
      priority: evidence ? "low" : "medium",
    };
  }

  const healthiest = allRoutes.find((route) => route.key === "healthiest");
  const healthierDelta = durationDelta(healthiest, selectedRoute);
  const selectedPm25 = routePm25(selectedRoute, weather);
  const healthiestPm25 = healthiest ? routePm25(healthiest, weather) : null;
  const categoryRank = { Baja: 0, Media: 1, Alta: 2 };
  const hasSimilarTime = healthierDelta !== null && healthierDelta <= RECOMMENDED_ROUTE_MAX_EXTRA_MIN;
  const hasLowerExposure = Boolean(
    healthiest &&
      selectedRoute.pm25_exposure?.available &&
      healthiest.pm25_exposure?.available &&
      (categoryRank[healthiest.pm25_exposure.category] < categoryRank[selectedRoute.pm25_exposure.category] ||
        healthiest.pm25_exposure.average_pm25 + 2 < selectedRoute.pm25_exposure.average_pm25),
  );
  const selectedAirIsPoor = isHighPm25(selectedRoute, weather);
  const healthiestAirIsBetter = Boolean(
    healthiest &&
      ((selectedPm25 !== null && healthiestPm25 !== null && healthiestPm25 + 2 < selectedPm25) ||
        hasLowerExposure),
  );

  if (!healthiest || (!hasSimilarTime && !healthiestAirIsBetter)) {
    return null;
  }

  const detail = selectedAirIsPoor && healthiestAirIsBetter
    ? "Si buscas reducir la exposición ambiental, considera la ruta saludable. Esta alternativa prioriza mejores condiciones del entorno durante el viaje."
    : selectedRouteType === "fastest"
      ? "Existe una alternativa saludable con un tiempo de viaje similar. Si puedes elegirla, esa ruta ofrece mejores condiciones urbanas y ambientales para el trayecto."
      : "La alternativa saludable tiene un tiempo de viaje similar y puede ofrecer un entorno más favorable, con mayor presencia de áreas verdes, cuerpos de agua, ciclovías o puntos de reciclaje.";

  return {
    id: selectedAirIsPoor && healthiestAirIsBetter
      ? "recommend-healthiest-lower-exposure"
      : "recommend-healthiest-close-option",
    title: "Entorno saludable",
    detail,
    type: "recommendation",
    priority: selectedAirIsPoor && healthiestAirIsBetter ? "high" : "medium",
    action: {
      label: "Seleccionar ruta saludable",
      targetRouteId: "healthiest",
    },
  };
}

export function weatherMessage(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
): MobilityGuidanceMessage | null {
  void selectedRouteType;
  void selectedRoute;
  const rain = Boolean(weather?.has_rain);

  if (!rain) {
    return null;
  }

  const heavyRain = weather?.rain_label === "Lluvia fuerte";
  return {
    id: heavyRain ? "weather-heavy-rain" : "weather-rain",
    title: "Lluvia",
    detail: heavyRain
      ? "Se observa lluvia fuerte. Aunque puede mejorar temporalmente la calidad del aire, aumenta el riesgo de calles resbaladizas, baja visibilidad y demoras."
      : "Se observa lluvia para el horario seleccionado. Puede mejorar temporalmente la calidad del aire, pero también puede reducir la visibilidad y dejar calles resbaladizas. Considera más tiempo para el viaje.",
    type: "weather",
    priority: heavyRain ? "high" : "medium",
  };
}

export function recommendationMessage(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
  allRoutes: PlanRouteResponse["routes"],
): MobilityGuidanceMessage | null {
  return healthyEnvironmentMessage(selectedRouteType, selectedRoute, allRoutes, null);
}

export function buildRouteInsightMessages(
  selectedRouteType: RouteType,
  selectedRoute: PlanRouteResponse["routes"][number],
  allRoutes: PlanRouteResponse["routes"],
  weather: EnvironmentalImpactResponse["summary"]["weather"] | null | undefined,
): MobilityGuidanceMessage[] {
  const optionalMessages = [
    weatherMessage(selectedRouteType, selectedRoute, weather),
    healthyEnvironmentMessage(selectedRouteType, selectedRoute, allRoutes, weather),
  ].filter((message): message is MobilityGuidanceMessage => message !== null);

  return [
    airQualityMessage(selectedRouteType, selectedRoute, weather),
    congestionMessage(selectedRouteType, selectedRoute),
    timeMessage(selectedRouteType, selectedRoute),
    ...optionalMessages,
  ];
}

function congestionLevel(score: number): string {
  if (score < 15) {
    return "Bajo";
  }
  if (score < 40) {
    return "Medio";
  }
  return "Alto";
}

function routeCongestionCoverageLabel(route: PlanRouteResponse["routes"][number]): string {
  const coverage = route.congestion_coverage;
  if (!coverage) {
    return "sin %";
  }
  if (coverage.high_pct > 0) {
    return `${coverage.high_pct.toFixed(1)}% en rojo`;
  }
  if (coverage.medium_pct > 0) {
    return `${coverage.medium_pct.toFixed(1)}% en naranja`;
  }
  return "0.0% en rojo";
}

function routeCongestedPercent(route: PlanRouteResponse["routes"][number]): string {
  const coverage = route.congestion_coverage;
  if (!coverage) {
    return "Sin dato";
  }
  const pct =
    Number.isFinite(coverage.congested_pct)
      ? coverage.congested_pct
      : (coverage.high_pct ?? 0) + (coverage.medium_pct ?? 0) + (coverage.low_pct ?? 0);
  return `${Math.max(0, pct).toFixed(1)}%`;
}

function routeExposurePreview(route: PlanRouteResponse["routes"][number]): string {
  if (route.pm25_exposure?.available) {
    return `PM2.5 ${route.pm25_exposure.average_pm25.toFixed(1)}`;
  }
  if (route.healthy_route_score !== null && route.healthy_route_score !== undefined) {
    return `Score ambiental ${route.healthy_route_score.toFixed(0)}/100`;
  }
  return "Exposicion sin dato";
}

export function PlanPage() {
  const mapStyleUrl = (import.meta.env.VITE_MAP_STYLE_URL ?? DEFAULT_MAP_STYLE_URL).trim();
  const mapboxToken = (import.meta.env.VITE_MAPBOX_TOKEN ?? "").trim();
  const mapEnabled = Boolean(mapStyleUrl);
  const geocodingEnabled = Boolean(mapboxToken);

  const [readiness, setReadiness] = useState<ReadinessStatus | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(() => !hasSeenOnboarding());
  const [onboardingStep, setOnboardingStep] = useState(0);
  const [planner, setPlanner] = useState<PlannerState>({
    origin: makeSelection("origin-default", "Plaza Peru, Concepcion", { lat: -36.8271, lon: -73.0496 }),
    destination: makeSelection("destination-default", "Mall del Centro, Concepcion", { lat: -36.826, lon: -73.0504 }),
    day_of_week: "Wednesday",
    departure_hour: 8,
    travel_style: "balanced",
    avoid_congestion: true,
  });
  const [queries, setQueries] = useState<Record<PinKey, string>>({
    origin: "Plaza Peru, Concepcion",
    destination: "Mall del Centro, Concepcion",
  });
  const [suggestions, setSuggestions] = useState<Record<PinKey, PlaceResult[]>>({
    origin: [],
    destination: [],
  });
  const [recentPlaces, setRecentPlaces] = useState<PlaceResult[]>(() => readRecentPlaces());
  const [activePin, setActivePin] = useState<PinKey>("origin");
  const [plan, setPlan] = useState<PlanRouteResponse | null>(null);
  const [cycleways, setCycleways] = useState<CyclewayFeature[]>([]);
  const [wellbeingFeatures, setWellbeingFeatures] = useState<UrbanWellbeingFeature[]>([]);
  const [inspectMode, setInspectMode] = useState(false);
  const [showEnvironmentalLayer, setShowEnvironmentalLayer] = useState(true);
  const [showCycleways, setShowCycleways] = useState(false);
  const [wellbeingVisibility, setWellbeingVisibility] = useState<WellbeingVisibility>(DEFAULT_WELLBEING_VISIBILITY);
  const [congestionCoverage, setCongestionCoverage] = useState<CongestionDateCoverage | null>(null);
  const [congestionCoverageReady, setCongestionCoverageReady] = useState(false);
  const [congestionCoverageError, setCongestionCoverageError] = useState<string | null>(null);
  const [selectedCongestionDate, setSelectedCongestionDate] = useState(DEFAULT_HISTORY_DATE);
  const [congestionMonth, setCongestionMonth] = useState("2025-03");
  const [pm25Date, setPm25Date] = useState(DEFAULT_HISTORY_DATE);
  const [pm25Hour, setPm25Hour] = useState(8);
  const [historicalQueryHour, setHistoricalQueryHour] = useState(8);
  const [pm25Snapshot, setPm25Snapshot] = useState<Pm25SnapshotResponse | null>(null);
  const [pm25Error, setPm25Error] = useState<string | null>(null);
  const [environmentalImpact, setEnvironmentalImpact] = useState<EnvironmentalImpactResponse | null>(null);
  const [environmentalImpactError, setEnvironmentalImpactError] = useState<string | null>(null);
  const [environmentalImpactLoading, setEnvironmentalImpactLoading] = useState(false);
  const pm25SnapshotCacheRef = useRef(new Map<string, Pm25SnapshotResponse>());
  const pm25SnapshotRequestIdRef = useRef(0);
  const environmentalImpactCacheRef = useRef(new Map<string, EnvironmentalImpactResponse>());
  const environmentalImpactRequestIdRef = useRef(0);
  const selectionRequestIdRef = useRef<Record<PinKey, number>>({ origin: 0, destination: 0 });
  const routeChoicePanelRef = useRef<HTMLElement | null>(null);
  const routeGuidancePanelRef = useRef<HTMLElement | null>(null);
  const deferredPlan = useDeferredValue(plan);
  const [selectedRouteType, setSelectedRouteType] = useState<RouteType | null>(null);
  const [hoveredRouteType, setHoveredRouteType] = useState<RouteType | null>(null);
  const [mapPinnedRouteType, setMapPinnedRouteType] = useState<RouteType | null>(null);
  const [journeyStarted, setJourneyStarted] = useState(false);
  const [journeyNoticeVisible, setJourneyNoticeVisible] = useState(false);
  const [routeGuidanceClosed, setRouteGuidanceClosed] = useState(false);
  const [routeGuidanceCollapsed, setRouteGuidanceCollapsed] = useState(false);
  const [routeChoicePanelHeight, setRouteChoicePanelHeight] = useState(128);
  const [routeGuidancePanelRect, setRouteGuidancePanelRect] = useState<RouteGuidancePanelRect>({
    top: 0,
    left: 0,
    width: 0,
    height: 0,
  });
  const [openMetricInfo, setOpenMetricInfo] = useState<RouteMetricKey | null>(null);
  const [calendarInfoOpen, setCalendarInfoOpen] = useState(false);
  const [openEnvironmentInfo, setOpenEnvironmentInfo] = useState<EnvironmentConditionKey | null>(null);
  const [expandedMetricInfo, setExpandedMetricInfo] = useState<RouteMetricKey | null>(null);
  const [visiblePins, setVisiblePins] = useState<Record<PinKey, boolean>>({
    origin: false,
    destination: false,
  });
  const bootstrapRequestedRef = useRef(false);
  const [busy, setBusy] = useState({ refresh: false, planning: false, geolocate: false });
  const [error, setError] = useState<string | null>(null);
  const selectedRoute = selectedRouteType
    ? deferredPlan?.routes_by_type?.[selectedRouteType] ??
      deferredPlan?.routes.find((route) => route.key === selectedRouteType) ??
      null
    : null;
  const pinnedMapRoute =
    mapPinnedRouteType && deferredPlan
      ? deferredPlan.routes_by_type?.[mapPinnedRouteType] ??
        deferredPlan.routes.find((route) => route.key === mapPinnedRouteType) ??
        null
      : null;
  const mapRoutes = journeyStarted && selectedRoute ? [selectedRoute] : pinnedMapRoute ? [pinnedMapRoute] : (deferredPlan?.routes ?? []);
  const mapHighlightedRouteKey = journeyStarted ? selectedRouteType : mapPinnedRouteType ?? hoveredRouteType;
  const journeyMessage = selectedRouteType ? journeyStartMessage(selectedRouteType) : null;
  const onboardingSlide = ONBOARDING_SLIDES[onboardingStep];
  const wellbeingCounts = wellbeingFeatures.reduce<Record<UrbanWellbeingCategory, number>>(
    (counts, feature) => {
      const category = feature.properties.category ?? "green_space";
      counts[category] += 1;
      return counts;
    },
    { green_space: 0, blue_space: 0, tree_cover: 0, public_space: 0, sustainability: 0, cycleway: 0 },
  );
  const visibleOptionalLayers =
    Object.values(wellbeingVisibility).filter(Boolean).length + Number(showCycleways);

  function closeOnboarding() {
    rememberOnboardingSeen();
    setShowOnboarding(false);
    setOnboardingStep(0);
  }

  function toggleWellbeingCategory(category: UrbanWellbeingCategory) {
    setWellbeingVisibility((current) => ({ ...current, [category]: !current[category] }));
  }

  function openOnboarding() {
    setOnboardingStep(0);
    setShowOnboarding(true);
  }

  function goToNextOnboardingSlide() {
    if (onboardingStep >= ONBOARDING_SLIDES.length - 1) {
      closeOnboarding();
      return;
    }
    setOnboardingStep((current) => current + 1);
  }

  useEffect(() => {
    if (selectedRouteType) {
      setRouteGuidanceClosed(false);
      setRouteGuidanceCollapsed(false);
      setOpenMetricInfo(null);
      setExpandedMetricInfo(null);
    }
  }, [selectedRouteType]);

  async function refreshBootState(forceWarmup: boolean) {
    setBusy((current) => ({ ...current, refresh: true }));
    setError(null);
    try {
      if (forceWarmup) {
        await startBootstrap();
      }
      const ready = await getReadiness();
      setReadiness(ready);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo revisar el estado del backend.");
    } finally {
      setBusy((current) => ({ ...current, refresh: false }));
    }
  }

  useEffect(() => {
    refreshBootState(false);
  }, []);

  useEffect(() => {
    let cancelled = false;
    getUrbanWellbeing()
      .then((collection) => {
        if (!cancelled) {
          setWellbeingFeatures(collection.features ?? []);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setWellbeingFeatures([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getCycleways()
      .then((collection) => {
        if (!cancelled) {
          setCycleways(collection.features ?? []);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCycleways([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setCongestionCoverageError(null);
    getCongestionDates()
      .then((coverage) => {
        if (cancelled) {
          return;
        }
        setCongestionCoverage(coverage);
        setCongestionCoverageReady(true);
        const initialDate = coverage.available_dates[0] ?? selectedCongestionDate;
        setSelectedCongestionDate(initialDate);
        setCongestionMonth(monthKeyFromIso(initialDate));
        setPm25Date(initialDate);
        if (coverage.available_dates.includes(initialDate)) {
          handleRouteContextChange({ day_of_week: dayOfWeekFromIso(initialDate) });
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setCongestionCoverage(null);
          setCongestionCoverageReady(true);
          setCongestionCoverageError(
            err instanceof Error ? err.message : "No se pudo cargar el calendario de congestion.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setHistoricalQueryHour(pm25Hour);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [pm25Hour]);

  useEffect(() => {
    if (!congestionCoverageReady) {
      return;
    }
    let cancelled = false;
    const requestId = pm25SnapshotRequestIdRef.current + 1;
    const cacheKey = `${pm25Date}|${historicalQueryHour}`;
    const cached = pm25SnapshotCacheRef.current.get(cacheKey);
    pm25SnapshotRequestIdRef.current = requestId;
    setPm25Error(null);
    if (cached) {
      setPm25Snapshot(cached);
      return () => {
        cancelled = true;
      };
    }
    setPm25Snapshot(null);
    getPm25Snapshot(pm25Date, historicalQueryHour)
      .then((snapshot) => {
        pm25SnapshotCacheRef.current.set(cacheKey, snapshot);
        if (!cancelled && requestId === pm25SnapshotRequestIdRef.current) {
          setPm25Snapshot(snapshot);
        }
      })
      .catch((err) => {
        if (!cancelled && requestId === pm25SnapshotRequestIdRef.current) {
          setPm25Snapshot(null);
          setPm25Error(err instanceof Error ? err.message : "No se pudo cargar PM2.5 historico.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [congestionCoverageReady, historicalQueryHour, pm25Date]);

  useEffect(() => {
    if (!congestionCoverageReady) {
      return;
    }
    let cancelled = false;
    const requestId = environmentalImpactRequestIdRef.current + 1;
    const cacheKey = `${selectedCongestionDate}|${historicalQueryHour}`;
    const cached = environmentalImpactCacheRef.current.get(cacheKey);
    environmentalImpactRequestIdRef.current = requestId;
    setEnvironmentalImpactError(null);
    if (cached) {
      setEnvironmentalImpact(cached);
      setEnvironmentalImpactLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setEnvironmentalImpact(null);
    setEnvironmentalImpactLoading(true);
    getEnvironmentalImpact(selectedCongestionDate, historicalQueryHour)
      .then((snapshot) => {
        environmentalImpactCacheRef.current.set(cacheKey, snapshot);
        if (!cancelled && requestId === environmentalImpactRequestIdRef.current) {
          setEnvironmentalImpact(snapshot);
          setEnvironmentalImpactLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled && requestId === environmentalImpactRequestIdRef.current) {
          setEnvironmentalImpact(null);
          setEnvironmentalImpactLoading(false);
          setEnvironmentalImpactError(
            err instanceof Error ? err.message : "No se pudo cargar la capa ambiental.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [congestionCoverageReady, historicalQueryHour, selectedCongestionDate]);

  useEffect(() => {
    if (!readiness || readiness.ready || readiness.status === "error" || busy.refresh) {
      return;
    }
    if (readiness.bootstrap.status === "idle" && !bootstrapRequestedRef.current) {
      bootstrapRequestedRef.current = true;
      refreshBootState(true);
      return;
    }
    if (readiness.bootstrap.status !== "running") {
      return;
    }
    const timer = window.setTimeout(() => {
      refreshBootState(false);
    }, 2500);
    return () => window.clearTimeout(timer);
  }, [busy.refresh, readiness]);

  useEffect(() => {
    if (!geocodingEnabled) {
      setSuggestions({ origin: [], destination: [] });
      return;
    }
    const entries = (Object.entries(queries) as Array<[PinKey, string]>).map(([pin, query]) => ({ pin, query }));
    const timer = window.setTimeout(async () => {
      await Promise.all(
        entries.map(async ({ pin, query }) => {
          const normalized = query.trim();
          if (normalized.length < 3 || normalized === planner[pin].label) {
            setSuggestions((current) => ({
              ...current,
              [pin]: normalized.length === 0 ? recentPlaces : [],
            }));
            return;
          }
          try {
            const results = await searchPlaces(normalized, 5);
            setSuggestions((current) => ({ ...current, [pin]: results }));
          } catch {
            setSuggestions((current) => ({ ...current, [pin]: [] }));
          }
        }),
      );
    }, 280);
    return () => window.clearTimeout(timer);
  }, [geocodingEnabled, planner, queries, recentPlaces]);

  function commitSelection(pin: PinKey, selection: PlaceSelection) {
    setPlanner((current) => ({ ...current, [pin]: selection }));
    setQueries((current) => ({ ...current, [pin]: selection.label }));
    setSuggestions((current) => ({ ...current, [pin]: [] }));
    setVisiblePins((current) => ({ ...current, [pin]: true }));
    setPlan(null);
    setSelectedRouteType(null);
    setHoveredRouteType(null);
    setMapPinnedRouteType(null);
    setJourneyStarted(false);
    setJourneyNoticeVisible(false);
  }

  function applySelection(pin: PinKey, selection: PlaceSelection) {
    selectionRequestIdRef.current[pin] += 1;
    commitSelection(pin, selection);
  }

  async function resolveLabel(pin: PinKey, point: RoutePoint) {
    const requestId = selectionRequestIdRef.current[pin] + 1;
    selectionRequestIdRef.current[pin] = requestId;
    commitSelection(pin, makeSelection(`${pin}-manual`, buildManualLabel(point), point));
    if (!geocodingEnabled) {
      return;
    }
    try {
      const result = await reversePlace(point.lat, point.lon);
      if (result && selectionRequestIdRef.current[pin] === requestId) {
        commitSelection(pin, makeSelection(result.id, result.label, point, result.bbox));
        return;
      }
    } catch {
      // noop: fallback below
    }
  }

  function handleMapPick(pin: PinKey, point: RoutePoint) {
    void resolveLabel(pin, point);
    if (pin === "origin") {
      setActivePin("destination");
    }
  }

  function handleRouteContextChange(next: Partial<Pick<PlannerState, "day_of_week" | "departure_hour">>) {
    setPlanner((current) => ({
      ...current,
      ...next,
    }));
    setPlan(null);
    setSelectedRouteType(null);
    setHoveredRouteType(null);
    setMapPinnedRouteType(null);
    setJourneyStarted(false);
    setJourneyNoticeVisible(false);
  }

  function handleCongestionDateSelect(date: string, hasData: boolean) {
    setSelectedCongestionDate(date);
    setCongestionMonth(monthKeyFromIso(date));
    setPm25Date(date);
    if (hasData) {
      handleRouteContextChange({ day_of_week: dayOfWeekFromIso(date) });
    }
  }

  function handleHistoricalHourChange(hour: number) {
    setPm25Hour(hour);
    handleRouteContextChange({ departure_hour: hour });
  }

  function handleSuggestionSelect(pin: PinKey, place: PlaceResult) {
    const nextRecentPlaces = upsertRecentPlace(recentPlaces, place);
    setRecentPlaces(nextRecentPlaces);
    writeRecentPlaces(nextRecentPlaces);
    applySelection(pin, selectionFromPlace(place));
  }

  function handleSwap() {
    selectionRequestIdRef.current.origin += 1;
    selectionRequestIdRef.current.destination += 1;
    setPlanner((current) => ({
      ...current,
      origin: current.destination,
      destination: current.origin,
    }));
    setQueries((current) => ({
      origin: current.destination,
      destination: current.origin,
    }));
    setVisiblePins((current) => ({
      origin: current.destination,
      destination: current.origin,
    }));
    setPlan(null);
    setSelectedRouteType(null);
    setHoveredRouteType(null);
    setMapPinnedRouteType(null);
    setJourneyStarted(false);
    setJourneyNoticeVisible(false);
  }

  function handleGeolocate() {
    if (!navigator.geolocation) {
      setError("Tu navegador no expone geolocalizacion.");
      return;
    }
    setBusy((current) => ({ ...current, geolocate: true }));
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const point = {
          lat: position.coords.latitude,
          lon: position.coords.longitude,
        };
        void resolveLabel("origin", point);
        setBusy((current) => ({ ...current, geolocate: false }));
      },
      () => {
        setBusy((current) => ({ ...current, geolocate: false }));
        setError("No fue posible tomar tu ubicacion actual.");
      },
      { enableHighAccuracy: true, timeout: 7000, maximumAge: 0 },
    );
  }

  async function handlePlan() {
    setBusy((current) => ({ ...current, planning: true }));
    setError(null);
    try {
      if (!readiness?.ready) {
        await startBootstrap();
        setReadiness(await getReadiness());
      }
      const response = await planRoute({
        origin: planner.origin.point,
        destination: planner.destination.point,
        congestion_date: selectedCongestionDate,
        day_of_week: planner.day_of_week,
        departure_hour: planner.departure_hour,
        travel_style: planner.travel_style,
        avoid_congestion: planner.avoid_congestion,
        avoid_accidents: false,
      });
      startTransition(() => {
        setPlan(response);
        setSelectedRouteType(null);
        setHoveredRouteType(null);
        setMapPinnedRouteType(null);
        setJourneyStarted(false);
        setJourneyNoticeVisible(false);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo planificar el viaje.");
    } finally {
      setBusy((current) => ({ ...current, planning: false }));
    }
  }

  const routeCards = deferredPlan?.routes ?? [];
  const selectedRouteMessages = selectedRoute && selectedRouteType
    ? buildRouteInsightMessages(
        selectedRouteType,
        selectedRoute,
        routeCards,
        environmentalImpact?.summary.weather,
      )
    : [];
  const routeMetricCards = selectedRoute
    ? ([
        ["duration", `${selectedRoute.duration_min.toFixed(1)} min`],
        ["distance", `${selectedRoute.distance_km.toFixed(2)} km`],
        [
          "congestion",
          `${congestionLevel(selectedRoute.congestion_score ?? 0)} · ${routeCongestionCoverageLabel(selectedRoute)}`,
        ],
        [
          "pm25",
          selectedRoute.pm25_exposure?.available
            ? `${selectedRoute.pm25_exposure.average_pm25.toFixed(1)} ug/m3`
            : "Sin dato",
        ],
        [
          "wellbeing",
          selectedRoute.urban_wellbeing?.available ? `${selectedRoute.urban_wellbeing.score.toFixed(0)}/100` : "Sin dato",
        ],
      ] as Array<[RouteMetricKey, string]>)
    : [];
  const expandedMetric = expandedMetricInfo ? ROUTE_METRIC_INFO[expandedMetricInfo] : null;
  const environmentWeather = environmentalImpact?.summary.weather;
  const pm25Percent = localRangePercent(
    environmentWeather?.pm25,
    environmentWeather?.pm25_min,
    environmentWeather?.pm25_max,
  );
  const windPercent = localRangePercent(
    environmentWeather?.wind_speed_kmh,
    environmentWeather?.wind_speed_min_kmh,
    environmentWeather?.wind_speed_max_kmh,
  );
  const environmentImpactLevel = environmentalImpact?.summary.dominant_level ?? "none";

  function handleGuidanceAction(message: MobilityGuidanceMessage) {
    const targetRouteId = message.action?.targetRouteId;
    if (isRouteType(targetRouteId)) {
      setSelectedRouteType(targetRouteId);
      setHoveredRouteType(null);
      setMapPinnedRouteType(targetRouteId);
      setJourneyStarted(false);
      setJourneyNoticeVisible(false);
      setRouteGuidanceClosed(false);
      setRouteGuidanceCollapsed(false);
    }
    routeChoicePanelRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handleRoutePreferenceSelect(routeType: RouteType) {
    setSelectedRouteType(routeType);
    setHoveredRouteType(null);
    setMapPinnedRouteType(routeType);
    setJourneyStarted(false);
    setJourneyNoticeVisible(false);
  }

  function handleStartJourney() {
    if (!selectedRouteType) {
      return;
    }
    setMapPinnedRouteType(selectedRouteType);
    setHoveredRouteType(null);
    setJourneyStarted(true);
    setJourneyNoticeVisible(true);
    setRouteGuidanceClosed(true);
    setRouteGuidanceCollapsed(false);
  }

  function handleBackFromJourney() {
    setJourneyStarted(false);
    setJourneyNoticeVisible(false);
    setMapPinnedRouteType(null);
    setRouteGuidanceClosed(false);
  }

  function handleFinishJourney() {
    setJourneyStarted(false);
    setJourneyNoticeVisible(false);
    setHoveredRouteType(null);
    setMapPinnedRouteType(selectedRouteType);
    setRouteGuidanceClosed(false);
  }

  const availableCongestionDates = new Set(congestionCoverage?.available_dates ?? []);
  const missingCongestionDates = new Set(congestionCoverage?.missing_dates ?? []);
  const rainDates = new Set(congestionCoverage?.rain_dates ?? []);
  const congestionCalendarDays = buildCalendarDays(
    congestionMonth,
    availableCongestionDates,
    missingCongestionDates,
    rainDates,
  );
  const previousCongestionMonth = shiftMonth(congestionMonth, -1);
  const nextCongestionMonth = shiftMonth(congestionMonth, 1);
  const canGoPreviousCongestionMonth =
    !congestionCoverage?.start || previousCongestionMonth >= monthKeyFromIso(congestionCoverage.start);
  const canGoNextCongestionMonth =
    !congestionCoverage?.end || nextCongestionMonth <= monthKeyFromIso(congestionCoverage.end);
  const routeGuidanceActive = Boolean(
    selectedRoute && selectedRouteMessages.length > 0 && !routeGuidanceClosed && !journeyStarted,
  );
  const planShellStyle = {
    "--route-choice-panel-height": `${routeChoicePanelHeight}px`,
    "--route-guidance-panel-top": `${routeGuidancePanelRect.top}px`,
    "--route-guidance-panel-left": `${routeGuidancePanelRect.left}px`,
    "--route-guidance-panel-width": `${routeGuidancePanelRect.width}px`,
    "--route-guidance-panel-height": `${routeGuidancePanelRect.height}px`,
  } as CSSProperties;

  useEffect(() => {
    const node = routeChoicePanelRef.current;
    if (!node) {
      return;
    }

    const updateHeight = () => {
      setRouteChoicePanelHeight(Math.ceil(node.getBoundingClientRect().height));
    };

    updateHeight();
    window.addEventListener("resize", updateHeight);

    if (!("ResizeObserver" in window)) {
      return () => window.removeEventListener("resize", updateHeight);
    }

    const observer = new ResizeObserver(updateHeight);
    observer.observe(node);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateHeight);
    };
  }, [deferredPlan, selectedRoute, selectedRouteMessages.length]);

  useEffect(() => {
    const node = routeGuidancePanelRef.current;
    if (!routeGuidanceActive || !node) {
      setRouteGuidancePanelRect({ top: 0, left: 0, width: 0, height: 0 });
      return;
    }

    const updateRect = () => {
      const rect = node.getBoundingClientRect();
      setRouteGuidancePanelRect({
        top: Math.round(rect.top),
        left: Math.round(rect.left),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
      });
    };

    updateRect();
    window.addEventListener("resize", updateRect);

    if (!("ResizeObserver" in window)) {
      return () => window.removeEventListener("resize", updateRect);
    }

    const observer = new ResizeObserver(updateRect);
    observer.observe(node);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateRect);
    };
  }, [routeGuidanceActive, routeGuidanceCollapsed, selectedRouteMessages.length]);

  return (
    <main
      className={`plan-shell ${routeGuidanceActive ? "route-guidance-active" : ""} ${
        journeyStarted ? "journey-active" : ""
      }`}
      style={planShellStyle}
    >
      <section className="topbar topbar-product">
        <div>
          <p className="eyebrow">Movilidad clara</p>
          <h1>Planifica tu viaje con congestion historica</h1>
          <p className="lead">
            Define origen y destino, elige una preferencia y revisa el contexto urbano antes de salir.
          </p>
        </div>
        <Link className="secondary-link" to="/demo">
          Ir al modo demo
        </Link>
        <button className="secondary-link" type="button" onClick={openOnboarding}>
          Ver paso a paso
        </button>
      </section>

      {showOnboarding ? (
        <section className="onboarding-overlay" role="presentation">
          <div
            className={`onboarding-dialog ${onboardingStep === 0 ? "onboarding-dialog-intro" : ""}`}
            role="dialog"
            aria-modal="true"
            aria-labelledby={onboardingStep > 0 ? "onboarding-title" : undefined}
            aria-label={onboardingStep === 0 ? "Bienvenidos a" : undefined}
          >
            {onboardingStep > 0 ? (
              <div className="onboarding-copy">
                <div className="onboarding-step">
                  Paso {onboardingStep} de {ONBOARDING_SLIDES.length - 1}
                </div>
                <h2 id="onboarding-title">{onboardingSlide.title}</h2>
                {onboardingSlide.body ? <p>{onboardingSlide.body}</p> : null}
                <div className="onboarding-progress" aria-label="Progreso del tutorial">
                  {ONBOARDING_SLIDES.slice(1).map((slide, index) => {
                    const slideIndex = index + 1;
                    return (
                      <button
                        key={slide.title}
                        type="button"
                        className={slideIndex === onboardingStep ? "active" : ""}
                        aria-label={`Ir al paso ${index + 1}`}
                        aria-current={slideIndex === onboardingStep ? "step" : undefined}
                        onClick={() => setOnboardingStep(slideIndex)}
                      />
                    );
                  })}
                </div>
              </div>
            ) : null}
            <OnboardingPreview focus={onboardingSlide.focus} onIntroClick={goToNextOnboardingSlide} />
            <div className="onboarding-actions">
              <button className="ghost-button" type="button" onClick={closeOnboarding}>
                Entrar a la app
              </button>
              {onboardingStep > 0 ? (
                <div>
                  <button
                    className="ghost-button"
                    type="button"
                    onClick={() => setOnboardingStep((current) => Math.max(0, current - 1))}
                  >
                    Anterior
                  </button>
                  <button className="primary-button" type="button" onClick={goToNextOnboardingSlide}>
                    {onboardingStep === ONBOARDING_SLIDES.length - 1 ? "Entrar a la app" : "Siguiente"}
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}

      <section className="search-shell sticky-shell">
        <div className="search-grid">
          {(["origin", "destination"] as const).map((pin) => (
            <label className="search-field" key={pin}>
              <span>{pin === "origin" ? "Origen" : "Destino"}</span>
              <input
                type="text"
                value={queries[pin]}
                placeholder={pin === "origin" ? "Buscar punto de partida" : "Buscar destino"}
                onFocus={() => {
                  setActivePin(pin);
                  if (geocodingEnabled && queries[pin].trim().length === 0) {
                    setSuggestions((current) => ({ ...current, [pin]: recentPlaces }));
                  }
                }}
                onChange={(event) => {
                  setActivePin(pin);
                  setQueries((current) => ({ ...current, [pin]: event.target.value }));
                }}
                disabled={!geocodingEnabled}
              />
              {geocodingEnabled && suggestions[pin].length ? (
                <div className="suggestion-list">
                  {suggestions[pin].map((place) => (
                    <button
                      className="suggestion-item"
                      key={`${pin}-${place.id}`}
                      type="button"
                      onClick={() => handleSuggestionSelect(pin, place)}
                    >
                      <strong>{place.label}</strong>
                      <span>
                        {place.lat.toFixed(4)}, {place.lon.toFixed(4)}
                      </span>
                    </button>
                  ))}
                </div>
              ) : null}
            </label>
          ))}
        </div>

        <div className="search-actions">
          <button className="ghost-button" type="button" onClick={handleSwap}>
            Intercambiar
          </button>
          <button className="ghost-button" type="button" onClick={handleGeolocate} disabled={busy.geolocate}>
            {busy.geolocate ? "Ubicando..." : "Usar mi ubicacion"}
          </button>
          <button
            className="primary-button"
            type="button"
            onClick={handlePlan}
            disabled={busy.planning || !readiness?.ready}
          >
            {busy.planning ? "Planificando..." : "Planificar viaje"}
          </button>
        </div>
      </section>

      {readiness && !readiness.ready ? (
        <section className="banner info-banner" role="status">
          <strong>{readiness.status === "error" ? "Backend con problema" : "Preparando rutas"}</strong>
          <span>{readiness.message}</span>
        </section>
      ) : null}

      {error ? (
        <section className="banner error-banner" role="alert">
          <strong>No se pudo completar la accion.</strong>
          <span>{error}</span>
        </section>
      ) : null}

      <section className="planner-layout">
        <div className="map-stage">
          <PlanningMap
            enabled={mapEnabled}
            styleUrl={mapStyleUrl}
            mapboxToken={mapboxToken}
            routes={mapRoutes}
            selectedRouteKey={mapHighlightedRouteKey}
            hotspots={deferredPlan?.hotspots ?? []}
            environmentalImpact={showEnvironmentalLayer ? environmentalImpact : null}
            environmentalImpactLoading={environmentalImpactLoading}
            showImpactCard={false}
            cycleways={cycleways}
            wellbeingFeatures={wellbeingFeatures}
            inspectMode={inspectMode}
            showCycleways={showCycleways}
            wellbeingVisibility={wellbeingVisibility}
            origin={visiblePins.origin ? planner.origin.point : null}
            destination={visiblePins.destination ? planner.destination.point : null}
            activePin={activePin}
            onPickPoint={(pin, point) => {
              handleMapPick(pin, point);
            }}
            onMarkerDrag={(pin, point) => {
              setActivePin(pin);
              void resolveLabel(pin, point);
            }}
          />
          {selectedRoute && selectedRouteMessages.length > 0 && routeGuidanceClosed && !journeyStarted ? (
            <button
              className="route-guidance-reopen"
              type="button"
              onClick={() => setRouteGuidanceClosed(false)}
            >
              Ver resumen de ruta
            </button>
          ) : null}
          {selectedRoute && selectedRouteMessages.length > 0 && !routeGuidanceClosed && !journeyStarted ? (
            <aside
              ref={routeGuidancePanelRef}
              className={`route-guidance-panel ${routeGuidanceCollapsed ? "collapsed" : ""}`}
              aria-label="Resumen de condiciones del trayecto"
              aria-live="polite"
            >
              <div className="route-guidance-header">
                <div>
                  <span>Ruta seleccionada</span>
                  <h2>{routeDisplayName(selectedRoute.key)}</h2>
                  <p>Resumen de condiciones del trayecto</p>
                </div>
                <div className="route-guidance-controls">
                  <button
                    type="button"
                    aria-label={routeGuidanceCollapsed ? "Expandir resumen" : "Contraer resumen"}
                    onClick={() => setRouteGuidanceCollapsed((current) => !current)}
                  >
                    {routeGuidanceCollapsed ? "+" : "-"}
                  </button>
                  <button type="button" aria-label="Cerrar resumen" onClick={() => setRouteGuidanceClosed(true)}>
                    x
                  </button>
                </div>
              </div>
              {!routeGuidanceCollapsed ? (
                <div className="route-context-messages">
                  {selectedRouteMessages.map((message) => (
                    <article
                      key={message.id}
                      className={`route-insight-card priority-${message.priority} type-${message.type}`}
                    >
                      <span>{MESSAGE_TYPE_LABELS[message.type]}</span>
                      <strong>{message.title}</strong>
                      <p>{message.detail}</p>
                      {message.action ? (
                        <button
                          className="text-button route-guidance-action"
                          type="button"
                          onClick={() => handleGuidanceAction(message)}
                        >
                          {message.action.label}
                        </button>
                      ) : null}
                    </article>
                  ))}
                </div>
              ) : null}
            </aside>
          ) : null}
          {journeyStarted && selectedRoute && selectedRouteType ? (
            <aside className="journey-bar" aria-label="Viaje en curso" aria-live="polite">
              {journeyNoticeVisible && journeyMessage ? (
                <div className={`journey-start-message journey-${selectedRouteType}`}>
                  <div>
                    <span>{journeyMessage.title}</span>
                    <p>{journeyMessage.detail}</p>
                    {journeyMessage.reward ? (
                      <strong className="journey-reward">
                        {journeyMessage.reward} <span aria-hidden="true">★★★</span>
                      </strong>
                    ) : null}
                    {journeyMessage.localFact ? <small>{journeyMessage.localFact}</small> : null}
                  </div>
                  <button type="button" aria-label="Cerrar mensaje" onClick={() => setJourneyNoticeVisible(false)}>
                    x
                  </button>
                </div>
              ) : null}
              <div className="journey-bar-main">
                <div className="journey-metrics">
                  <span>
                    <small>Ruta</small>
                    <b>{routeDisplayName(selectedRoute.key)}</b>
                  </span>
                  <span>
                    <small>Tiempo</small>
                    <b>{selectedRoute.duration_min.toFixed(1)} min</b>
                  </span>
                  <span>
                    <small>Distancia</small>
                    <b>{selectedRoute.distance_km.toFixed(2)} km</b>
                  </span>
                  <span>
                    <small>Congestion</small>
                    <b>
                      {congestionLevel(selectedRoute.congestion_score ?? 0)} ·{" "}
                      {routeCongestionCoverageLabel(selectedRoute)}
                    </b>
                  </span>
                </div>
                <div className="journey-actions">
                  <button className="ghost-button" type="button" onClick={handleBackFromJourney}>
                    Volver atras
                  </button>
                  <button className="primary-button" type="button" onClick={handleFinishJourney}>
                    Finalizar
                  </button>
                </div>
              </div>
            </aside>
          ) : null}
        </div>

        <aside className="results-sheet">
          <section className="panel product-panel">
            <div className="section-header">
              <div>
                <div className="eyebrow">Viaje</div>
                <h2>Puntos del viaje</h2>
              </div>
            </div>

            <div className="air-history-panel">
              <div className={`route-side-metrics ${selectedRoute ? "" : "empty"}`} aria-live="polite">
                <div className="card-title-row">
                  <div>
                    <div className="eyebrow">Metricas de ruta</div>
                    <h3>{selectedRoute ? routeDisplayName(selectedRoute.key) : "Elige una ruta"}</h3>
                  </div>
                  {selectedRoute ? <span className="panel-tag">{selectedRoute.duration_min.toFixed(1)} min</span> : null}
                </div>
                {selectedRoute ? (
                  <div className="route-metric-grid">
                    {routeMetricCards.map(([metricKey, value]) => {
                      const metric = ROUTE_METRIC_INFO[metricKey];
                      const isOpen = openMetricInfo === metricKey;
                      return (
                        <article className={`route-metric-card ${isOpen ? "open" : ""}`} key={metricKey}>
                          <div className="route-metric-card-header">
                            <small>{metric.label}</small>
                            <button
                              className="metric-info-button"
                              type="button"
                              aria-expanded={isOpen}
                              aria-label={`Ver explicacion breve de ${metric.label}`}
                              onClick={() => setOpenMetricInfo((current) => (current === metricKey ? null : metricKey))}
                            >
                              i
                            </button>
                          </div>
                          <strong>{value}</strong>
                          {isOpen ? (
                            <div className="metric-short-info">
                              <p>{metric.short}</p>
                              <button
                                className="text-button metric-more-button"
                                type="button"
                                onClick={() => setExpandedMetricInfo(metricKey)}
                              >
                                Ver más
                              </button>
                            </div>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                ) : (
                  <p className="empty-route-message">
                    Planifica tu viaje y selecciona una preferencia abajo para revisar tiempo, distancia, congestion y
                    exposicion ambiental.
                  </p>
                )}
              </div>
              <div className={`environment-side-panel impact-${environmentImpactLevel}`} aria-live="polite">
                <div className="card-title-row">
                  <div>
                    <div className="eyebrow">Condiciones del entorno</div>
                    {environmentalImpactLoading && !environmentalImpact ? <h3>Cargando condiciones</h3> : null}
                  </div>
                </div>
                <div className="environment-conditions">
                  <div className="condition-row">
                    <div className="condition-title">
                      <span>
                        {ENVIRONMENT_CONDITION_INFO.pm25.label}
                        <button
                          className="metric-info-button condition-info-button"
                          type="button"
                          aria-expanded={openEnvironmentInfo === "pm25"}
                          aria-label={`Ver descripcion de ${ENVIRONMENT_CONDITION_INFO.pm25.label}`}
                          onClick={() => setOpenEnvironmentInfo((current) => (current === "pm25" ? null : "pm25"))}
                        >
                          i
                        </button>
                      </span>
                      <strong>{formatConditionValue(environmentWeather?.pm25, "ug/m3")}</strong>
                    </div>
                    <div className="condition-bar pm25-bar">
                      {pm25Percent !== null ? <span className="condition-marker" style={{ left: `${pm25Percent}%` }} /> : null}
                    </div>
                    {openEnvironmentInfo === "pm25" ? (
                      <div className="condition-info-panel">
                        <p>{ENVIRONMENT_CONDITION_INFO.pm25.short}</p>
                        <details className="condition-more">
                          <summary>Ver mas</summary>
                          <p>{ENVIRONMENT_CONDITION_INFO.pm25.technical}</p>
                          <p>{ENVIRONMENT_CONDITION_INFO.pm25.recommendation}</p>
                          <div className="condition-sources">
                            <strong>Fuentes de referencia</strong>
                            <a href="https://www.who.int/publications/i/item/9789240034228" rel="noreferrer" target="_blank">
                              OMS: guias mundiales de calidad del aire
                            </a>
                            <a href="https://www.epa.gov/pm-pollution/particulate-matter-pm-basics" rel="noreferrer" target="_blank">
                              EPA: que es PM2.5 y efectos en salud
                            </a>
                            <a href="https://www.airnow.gov/aqi/aqi-basics/" rel="noreferrer" target="_blank">
                              AirNow/EPA: categorias de calidad del aire y salud
                            </a>
                            <a href="https://sinca.mma.gob.cl/" rel="noreferrer" target="_blank">
                              SINCA Chile: observaciones oficiales de calidad del aire
                            </a>
                          </div>
                        </details>
                      </div>
                    ) : null}
                  </div>
                  <div className="condition-row">
                    <div className="condition-title">
                      <span>
                        {ENVIRONMENT_CONDITION_INFO.wind.label}
                        <button
                          className="metric-info-button condition-info-button"
                          type="button"
                          aria-expanded={openEnvironmentInfo === "wind"}
                          aria-label={`Ver descripcion de ${ENVIRONMENT_CONDITION_INFO.wind.label}`}
                          onClick={() => setOpenEnvironmentInfo((current) => (current === "wind" ? null : "wind"))}
                        >
                          i
                        </button>
                      </span>
                      <strong>
                        {formatConditionValue(environmentWeather?.wind_speed_kmh, "km/h")} -{" "}
                        {environmentWeather?.wind_label ?? "Sin dato"}
                      </strong>
                    </div>
                    <div className="condition-bar wind-bar">
                      {windPercent !== null ? <span className="condition-marker" style={{ left: `${windPercent}%` }} /> : null}
                    </div>
                    {openEnvironmentInfo === "wind" ? (
                      <div className="condition-info-panel">
                        <p>{ENVIRONMENT_CONDITION_INFO.wind.short}</p>
                        <details className="condition-more">
                          <summary>Ver mas</summary>
                          <p>{ENVIRONMENT_CONDITION_INFO.wind.technical}</p>
                          <p>{ENVIRONMENT_CONDITION_INFO.wind.recommendation}</p>
                        </details>
                      </div>
                    ) : null}
                  </div>
                  <div className="condition-row">
                    <div className="condition-rain">
                      <span>
                        {ENVIRONMENT_CONDITION_INFO.rain.label}
                        <button
                          className="metric-info-button condition-info-button"
                          type="button"
                          aria-expanded={openEnvironmentInfo === "rain"}
                          aria-label={`Ver descripcion de ${ENVIRONMENT_CONDITION_INFO.rain.label}`}
                          onClick={() => setOpenEnvironmentInfo((current) => (current === "rain" ? null : "rain"))}
                        >
                          i
                        </button>
                      </span>
                      <strong>{environmentWeather?.rain_label ?? "Sin dato"}</strong>
                    </div>
                    {openEnvironmentInfo === "rain" ? (
                      <div className="condition-info-panel">
                        <p>{ENVIRONMENT_CONDITION_INFO.rain.short}</p>
                        <details className="condition-more">
                          <summary>Ver mas</summary>
                          <p>{ENVIRONMENT_CONDITION_INFO.rain.technical}</p>
                          <p>{ENVIRONMENT_CONDITION_INFO.rain.recommendation}</p>
                        </details>
                      </div>
                    ) : null}
                  </div>
                  {environmentalImpactError ? <p className="muted">{environmentalImpactError}</p> : null}
                </div>
              </div>
              <div className={`map-tools-panel environmental-map-panel ${showEnvironmentalLayer ? "" : "collapsed"}`}>
                <div className="card-title-row">
                  <div>
                    <div className="eyebrow">Capa ambiental</div>
                    {showEnvironmentalLayer ? <h3>Consecuencias de la congestión</h3> : null}
                  </div>
                  <label className="environment-layer-switch">
                    <span>
                      {environmentalImpact?.summary.available
                        ? showEnvironmentalLayer
                          ? "Activada"
                          : "Desactivada"
                        : "Sin datos"}
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      checked={showEnvironmentalLayer}
                      disabled={!environmentalImpact?.summary.available}
                      aria-label="Mostrar u ocultar la capa ambiental"
                      onChange={(event) => {
                        const visible = event.target.checked;
                        setShowEnvironmentalLayer(visible);
                        if (!visible) setInspectMode(false);
                      }}
                    />
                    <span className="environment-switch-track" aria-hidden="true" />
                  </label>
                </div>
                {showEnvironmentalLayer ? <div className="map-guide-content panel-map-guide-content">
                  <div className="guide-key-message">
                    Representa gráficamente cómo la congestión puede generar emisiones y favorecer la concentración
                    de partículas en su entorno.
                  </div>
                  <div className="environment-read-guide">
                    <strong>Componentes de la capa</strong>
                  </div>
                  <div className="environment-map-legend" aria-label="Elementos visibles de la capa ambiental">
                    <div className="environment-map-item">
                      <span className="environment-symbol congestion-symbol" aria-hidden="true" />
                      <div>
                        <strong>Líneas de congestión</strong>
                        <p>Tramos con tráfico lento o detenido.</p>
                      </div>
                    </div>
                    <div className="environment-map-item">
                      <span className="environment-symbol cloud-symbol" aria-hidden="true" />
                      <div>
                        <strong>Nube ambiental</strong>
                        <p>Área donde se estima la concentración de emisiones y partículas.</p>
                        <div className="cloud-intensity-scale" aria-label="Intensidad estimada de la nube ambiental">
                          <span className="low"><b>Verde</b><small>Baja</small></span>
                          <span className="medium"><b>Naranjo</b><small>Media</small></span>
                          <span className="high"><b>Rojo</b><small>Alta</small></span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <button
                    className={`primary-button environmental-explore-button ${inspectMode ? "active" : ""}`}
                    type="button"
                    aria-pressed={inspectMode}
                    disabled={!environmentalImpact?.summary.available}
                    onClick={() => setInspectMode((current) => !current)}
                  >
                    {inspectMode ? "Salir de la exploración" : "Explorar la capa ambiental"}
                  </button>
                </div> : null}
              </div>
              <div className="map-tools-panel urban-layers-panel">
                <div className="card-title-row">
                  <div>
                    <div className="eyebrow">Capas urbanas</div>
                    <h3>Elementos del entorno</h3>
                  </div>
                  <span className="panel-tag">
                    {visibleOptionalLayers ? `${visibleOptionalLayers} activas` : "Capas"}
                  </span>
                </div>
                <div className="guide-key-message">
                  Activa estas capas para ver elementos del entorno en el mapa.
                  <details className="condition-more urban-layer-more">
                    <summary>Ver más</summary>
                    <p>
                      Marca una capa para mostrarla en el mapa. Por ejemplo, puedes ver parques, agua, árboles,
                      espacios públicos, puntos de reciclaje o ciclovías cerca de tu recorrido. Estas capas son solo una
                      ayuda visual para explorar el entorno; no representan calidad del aire ni nivel de exposición
                      ambiental.
                    </p>
                  </details>
                </div>
                <div className="map-layer-filter-list panel-layer-filter-list">
                  {WELLBEING_LAYER_OPTIONS.map((option) => (
                    <div className="map-layer-filter panel-layer-filter" key={option.category}>
                      <label className="layer-activation">
                        <input
                          type="checkbox"
                          checked={wellbeingVisibility[option.category]}
                          onChange={() => toggleWellbeingCategory(option.category)}
                        />
                        <span className={`layer-filter-swatch wellbeing-${option.category}`} aria-hidden="true" />
                        <strong>
                          {option.label} <small>{wellbeingCounts[option.category]}</small>
                        </strong>
                      </label>
                      <details className="layer-info-details">
                        <summary aria-label={`Ver descripción de ${option.label}`}>i</summary>
                        <p>{option.description}</p>
                      </details>
                    </div>
                  ))}
                  <div className="map-layer-filter panel-layer-filter">
                    <label className="layer-activation">
                      <input
                        type="checkbox"
                        checked={showCycleways}
                        onChange={() => setShowCycleways((current) => !current)}
                      />
                      <span className="layer-filter-swatch cycleways" aria-hidden="true" />
                      <strong>
                        Ciclovías <small>{cycleways.length}</small>
                      </strong>
                    </label>
                    <details className="layer-info-details">
                      <summary aria-label="Ver descripción de Ciclovías">i</summary>
                      <p>Líneas celestes discontinuas: infraestructura ciclista disponible, planificada o registrada.</p>
                    </details>
                  </div>
                </div>
              </div>
              <div className="history-calendar-block">
                <div className="card-title-row">
                  <div>
                    <div className="eyebrow">Elige una fecha</div>
                    <h3>Calendario y hora</h3>
                  </div>
                  <button
                    className="metric-info-button calendar-info-button"
                    type="button"
                    aria-expanded={calendarInfoOpen}
                    aria-label="Ver informacion del calendario"
                    onClick={() => setCalendarInfoOpen((current) => !current)}
                  >
                    i
                  </button>
                </div>
                <div className="calendar-title-row">
                  <div>
                    <strong>{monthLabel(congestionMonth)}</strong>
                  </div>
                </div>
                <div className="calendar-nav" aria-label="Navegacion del calendario historico">
                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => setCongestionMonth(previousCongestionMonth)}
                    disabled={!canGoPreviousCongestionMonth}
                    title="Mes anterior"
                  >
                    {"<"}
                  </button>
                  <strong>{selectedCongestionDate}</strong>
                  <button
                    className="icon-button"
                    type="button"
                    onClick={() => setCongestionMonth(nextCongestionMonth)}
                    disabled={!canGoNextCongestionMonth}
                    title="Mes siguiente"
                  >
                    {">"}
                  </button>
                </div>
                <div className="congestion-calendar" aria-label="Calendario historico PM2.5 y congestion">
                  {WEEKDAY_LABELS.map((label, index) => (
                    <span className="calendar-weekday" key={`${label}-${index}`}>
                      {label}
                    </span>
                  ))}
                  {congestionCalendarDays.map((item) => (
                    <button
                      className={[
                        "calendar-day",
                        item.inMonth ? "" : "outside-month",
                        item.hasRain ? "has-rain" : "",
                        item.hasData ? "has-data" : "",
                        item.isMissing ? "missing-data" : "",
                        item.date === selectedCongestionDate ? "selected" : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      key={item.date}
                      type="button"
                      onClick={() => handleCongestionDateSelect(item.date, item.hasData)}
                      disabled={!item.inMonth}
                      title={
                        item.hasData
                          ? `${item.date}: hay datos historicos de congestion${item.hasRain ? " y lluvia" : ""}`
                          : `${item.date}: sin datos historicos de congestion`
                      }
                    >
                      {item.day}
                    </button>
                  ))}
                </div>
                <div className="field-grid air-history-grid">
                  <label>
                    <span>Hora historica</span>
                    <input
                      type="range"
                      min={0}
                      max={23}
                      step={1}
                      value={pm25Hour}
                      onChange={(event) => handleHistoricalHourChange(Number(event.target.value))}
                    />
                    <small>{pm25Hour}:00</small>
                  </label>
                </div>
                {calendarInfoOpen ? (
                  <div className="calendar-info-panel">
                    <div className="calendar-legend">
                      <span className="calendar-dot has-data" /> Con datos
                      <span className="calendar-dot has-rain" /> Lluvia
                      <span className="calendar-dot missing-data" /> Sin datos
                    </div>
                    {congestionCoverageError ? <p className="muted">{congestionCoverageError}</p> : null}
                    <div className="congestion-line-legend">
                      <strong>Lineas: congestion historica</strong>
                      <span className="congestion-line-swatch congestion-low" /> Fluida
                      <span className="congestion-line-swatch congestion-medium" /> Moderada
                      <span className="congestion-line-swatch congestion-high" /> Alta
                    </div>
                    {selectedRoute?.pm25_exposure?.available ? (
                      <p className="muted">
                        {pm25Snapshot?.requested_at ?? "Horario seleccionado"}: PM2.5 estimado sobre la ruta seleccionada{" "}
                        {selectedRoute.pm25_exposure.average_pm25.toFixed(1)} ug/m3.
                      </p>
                    ) : null}
                    {pm25Error ? <p className="muted">{pm25Error}</p> : null}
                  </div>
                ) : null}
              </div>
            </div>

          </section>

          {deferredPlan ? (
            <section
              ref={routeChoicePanelRef}
              className={`panel product-panel route-choice-panel ${selectedRoute ? "has-selection" : ""}`}
            >
                <div className="section-header">
                  <div>
                    <h2>
                      {selectedRoute
                        ? "Puedes cambiar tu estilo de viaje"
                        : "¿Que es lo mas importante para ti en este viaje?"}
                    </h2>
                  </div>
                  {selectedRoute ? <span className="route-choice-hint">La ruta cambia inmediatamente en el mapa</span> : null}
                </div>
                {selectedRoute ? (
                  <div className="route-choice-toolbar">
                    <button className="primary-button" type="button" onClick={handleStartJourney}>
                      Iniciar viaje
                    </button>
                  </div>
                ) : null}
                <div className="route-card-list" aria-label="Preferencias de viaje">
                  {routeCards.map((route) => (
                    <button
                      key={route.key}
                      type="button"
                      className={`route-card preference-card preference-${ROUTE_PREFERENCES[route.key].tone} ${
                        selectedRouteType === route.key ? "selected" : ""
                      }`}
                      aria-pressed={selectedRouteType === route.key}
                      onMouseEnter={() => setHoveredRouteType(route.key)}
                      onMouseLeave={() => setHoveredRouteType(null)}
                      onFocus={() => setHoveredRouteType(route.key)}
                      onBlur={() => setHoveredRouteType(null)}
                      onClick={() => handleRoutePreferenceSelect(route.key)}
                    >
                      <div className="preference-card-heading">
                        <span className="preference-icon" aria-hidden="true">
                          {ROUTE_PREFERENCES[route.key].icon}
                        </span>
                        <div>
                          <span className="preference-status">
                            {selectedRouteType === route.key ? "Seleccionada" : ROUTE_PREFERENCES[route.key].tagline}
                          </span>
                          <h3>{ROUTE_PREFERENCES[route.key].title}</h3>
                        </div>
                        <span className="preference-time">{route.duration_min.toFixed(1)} min</span>
                      </div>
                      <p className="preference-description">{ROUTE_PREFERENCES[route.key].description}</p>
                      <div className="preference-preview">
                        <span>{route.distance_km.toFixed(2)} km</span>
                        <span>
                          Congestion {congestionLevel(route.congestion_score ?? 0)} · {routeCongestionCoverageLabel(route)}
                        </span>
                        <span className="route-exposure-preview">{routeExposurePreview(route)}</span>
                      </div>
                      <div className="preference-metrics">
                        <span>
                          <strong>{routeCongestedPercent(route)}</strong>
                          <small>% con congestion</small>
                        </span>
                        <span>
                          <strong>{routeExposurePreview(route)}</strong>
                          <small>Calidad aire</small>
                        </span>
                        <span>
                          <strong>{route.distance_km.toFixed(2)} km</strong>
                          <small>Distancia</small>
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
                {selectedRoute ? (
                  <div className="journey-start-panel">
                    <div>
                      <span>Ruta lista</span>
                      <strong>{routeDisplayName(selectedRoute.key)}</strong>
                      <small>
                        {selectedRoute.duration_min.toFixed(1)} min · {selectedRoute.distance_km.toFixed(2)} km ·
                        Congestion {congestionLevel(selectedRoute.congestion_score ?? 0)} ·{" "}
                        {routeCongestionCoverageLabel(selectedRoute)}
                      </small>
                    </div>
                    <button className="primary-button" type="button" onClick={handleStartJourney}>
                      Iniciar viaje
                    </button>
                  </div>
                ) : null}
            </section>
          ) : inspectMode ? (
            <section ref={routeChoicePanelRef} className="panel product-panel environmental-exploration-panel" aria-live="polite">
              <div className="environmental-exploration-content">
                <div className="environmental-exploration-copy">
                  <div className="eyebrow">Modo ambiental activo</div>
                  <h2>Explora la capa ambiental</h2>
                  <p>Selecciona un elemento del mapa para consultar su información.</p>
                </div>
                <div className="environmental-exploration-options" aria-label="Elementos disponibles en el mapa">
                  <span><i className="congestion-symbol" aria-hidden="true" />Línea de congestión</span>
                  <span><i className="cloud-symbol" aria-hidden="true" />Nube ambiental</span>
                </div>
                <button className="ghost-button" type="button" onClick={() => setInspectMode(false)}>
                  Salir del modo de exploración
                </button>
              </div>
            </section>
          ) : (
            <section ref={routeChoicePanelRef} className="panel product-panel empty-panel">
              <div className="eyebrow">Define tu recorrido</div>
              <h2>Marca el origen y el destino en el mapa</h2>
              <p>
                Haz un primer clic en el mapa para fijar el origen y un segundo clic para marcar el destino. Cuando
                ambos puntos estén listos, presiona el botón Planificar viaje para ver las rutas disponibles.
              </p>
            </section>
          )}
        </aside>
      </section>
      {expandedMetric ? (
        <div className="metric-detail-overlay" role="presentation" onClick={() => setExpandedMetricInfo(null)}>
          <section
            className="metric-detail-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="metric-detail-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="metric-detail-header">
              <div>
                <div className="eyebrow">Explicacion de metrica</div>
                <h2 id="metric-detail-title">{expandedMetric.label}</h2>
              </div>
              <button type="button" aria-label="Cerrar explicacion" onClick={() => setExpandedMetricInfo(null)}>
                x
              </button>
            </div>
            <div className="metric-detail-content">
              <section>
                <h3>Que representa</h3>
                <p>{expandedMetric.represents}</p>
              </section>
              <section>
                <h3>Como se calcula</h3>
                <ul>
                  {expandedMetric.calculation.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </section>
              <section>
                <h3>Datos que usa</h3>
                <ul>
                  {expandedMetric.variables.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </section>
              <section>
                <h3>Ten en cuenta</h3>
                <p>{expandedMetric.limitation}</p>
              </section>
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}
