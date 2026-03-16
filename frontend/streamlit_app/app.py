# -*- coding: utf-8 -*-
"""
Demo académica de ruta segura explicable para Concepción.
"""

from __future__ import annotations

import time
from typing import Dict, List

import folium
from folium.plugins import HeatMap, MiniMap
import requests
import streamlit as st
from streamlit_folium import st_folium

from frontend.streamlit_app.api_client import BackendClient
from frontend.streamlit_app.demo_content import PROFILE_ORDER, TRAVELER_PROFILES, VARIANT_META
from frontend.streamlit_app.view_models import comparison_highlights, variant_summary_cards

client = BackendClient()

REQUEST_TIMEOUT = client.timeout
DEFAULT_ORIGIN = {"lat": -36.8267, "lon": -73.0498}
DEFAULT_DESTINATION = {"lat": -36.8114, "lon": -73.0490}
DEFAULT_BOUNDS = {"lat_min": -36.95, "lat_max": -36.7, "lon_min": -73.2, "lon_max": -72.9}
DAY_CHOICES = [
    ("Monday", "Lunes"),
    ("Tuesday", "Martes"),
    ("Wednesday", "Miércoles"),
    ("Thursday", "Jueves"),
    ("Friday", "Viernes"),
    ("Saturday", "Sábado"),
    ("Sunday", "Domingo"),
]
DAY_LABELS = {code: label for code, label in DAY_CHOICES}


def set_page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(187, 247, 208, 0.55), transparent 28%),
                radial-gradient(circle at top right, rgba(191, 219, 254, 0.55), transparent 25%),
                linear-gradient(180deg, #f8faf7 0%, #f3f6ef 45%, #eef2e8 100%);
            color: #132a13;
            font-family: "Avenir Next", "Helvetica Neue", "Segoe UI", sans-serif;
        }
        .hero-card, .info-card, .warning-card, .metric-card {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(19, 42, 19, 0.08);
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 18px 40px rgba(19, 42, 19, 0.06);
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(230, 247, 235, 0.98), rgba(246, 250, 244, 0.96));
        }
        .warning-card {
            background: linear-gradient(135deg, rgba(255, 247, 237, 0.98), rgba(255, 252, 245, 0.96));
            border-color: rgba(194, 65, 12, 0.18);
        }
        .metric-title {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #4f6f52;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #163020;
        }
        .pill {
            display: inline-block;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
            padding: 0.22rem 0.6rem;
            border-radius: 999px;
            background: #edf6ee;
            border: 1px solid rgba(19, 42, 19, 0.08);
            font-size: 0.85rem;
        }
        .streamlit-folium, .streamlit-folium iframe {
            width: 100% !important;
        }
        .streamlit-folium {
            min-height: 620px !important;
            height: 620px !important;
        }
        .streamlit-folium iframe {
            min-height: 620px !important;
            height: 620px !important;
            border-radius: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def fetch_metadata(profile: str) -> Dict[str, object]:
    _ = profile
    return client.metadata()


@st.cache_data(show_spinner=False)
def fetch_hotspots(profile: str) -> List[Dict[str, float]]:
    _ = profile
    return client.hotspots().get("points", [])


@st.cache_data(show_spinner=False)
def fetch_demo_scenarios(profile: str) -> List[dict]:
    _ = profile
    return client.demo_scenarios().get("scenarios", [])


def clear_backend_cache_views() -> None:
    fetch_metadata.clear()
    fetch_hotspots.clear()
    fetch_demo_scenarios.clear()


def init_state() -> None:
    st.session_state.setdefault("app_ready", False)
    st.session_state.setdefault("dataset_status", None)
    st.session_state.setdefault("ready_status", None)
    st.session_state.setdefault("metadata", None)
    st.session_state.setdefault("hotspots", [])
    st.session_state.setdefault("demo_scenarios", [])
    st.session_state.setdefault("route_origin", DEFAULT_ORIGIN.copy())
    st.session_state.setdefault("route_destination", DEFAULT_DESTINATION.copy())
    st.session_state.setdefault("route_bounds", DEFAULT_BOUNDS.copy())
    st.session_state.setdefault("trip_day", "Wednesday")
    st.session_state.setdefault("trip_hour", 8.0)
    st.session_state.setdefault("user_profile", "usuario_demo")
    st.session_state.setdefault("avoid_congestion", True)
    st.session_state.setdefault("avoid_accidents", False)
    st.session_state.setdefault("route_result", None)
    st.session_state.setdefault("playground_results", None)
    st.session_state.setdefault("playground_profile", None)
    st.session_state.setdefault("selected_scenario_id", None)
    st.session_state.setdefault("map_assign_target", "Origen")


def _update_backend_context() -> None:
    dataset_status = client.dataset_status()
    profile = dataset_status.get("current", "concepcion")
    ready_status = client.ready()
    st.session_state["dataset_status"] = dataset_status
    st.session_state["ready_status"] = ready_status
    st.session_state["metadata"] = fetch_metadata(profile)
    st.session_state["hotspots"] = fetch_hotspots(profile)
    st.session_state["demo_scenarios"] = fetch_demo_scenarios(profile)
    bounds = (st.session_state["metadata"] or {}).get("bounds") or DEFAULT_BOUNDS.copy()
    st.session_state["route_bounds"] = bounds
    st.session_state["app_ready"] = True


def load_backend_context(force: bool = False) -> bool:
    if st.session_state.get("app_ready") and not force:
        return True
    if force:
        clear_backend_cache_views()
        st.session_state["app_ready"] = False
    progress = st.progress(0)
    status_box = st.empty()
    try:
        client.health()
    except requests.RequestException as err:
        status_box.error(f"No se pudo conectar al backend FastAPI: {err}")
        return False
    try:
        bootstrap = client.bootstrap()
    except requests.RequestException as err:
        status_box.error(f"No se pudo iniciar el warm-up del backend: {err}")
        return False

    start = time.time()
    while time.time() - start < 420:
        try:
            ready_status = client.ready()
            bootstrap = ready_status.get("bootstrap") or client.bootstrap_status()
        except requests.RequestException as err:
            status_box.error(f"Error consultando el estado del backend: {err}")
            return False
        progress.progress(max(0, min(100, int(bootstrap.get("percent", 0)))))
        status_box.info(bootstrap.get("message", "Preparando demo..."))
        if ready_status.get("ready"):
            _update_backend_context()
            progress.progress(100)
            status_box.success(
                f"Backend listo para demo en {bootstrap.get('duration_ms', 0)} ms "
                f"con perfil {st.session_state['dataset_status'].get('current_label', 'activo')}."
            )
            return True
        if bootstrap.get("status") == "error":
            status_box.error(bootstrap.get("message", "Ocurrió un error durante el warm-up."))
            return False
        time.sleep(2)
    status_box.error("El backend demoró demasiado en quedar listo para la demo.")
    return False


def apply_scenario(scenario: dict) -> None:
    st.session_state["selected_scenario_id"] = scenario.get("id")
    st.session_state["route_origin"] = scenario.get("origin", DEFAULT_ORIGIN).copy()
    st.session_state["route_destination"] = scenario.get("destination", DEFAULT_DESTINATION).copy()
    st.session_state["trip_day"] = scenario.get("day_of_week", "Wednesday")
    st.session_state["trip_hour"] = float(scenario.get("departure_hour", 8.0))
    st.session_state["user_profile"] = scenario.get("profile", "usuario_demo")
    st.session_state["route_result"] = None


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0 0 0.4rem 0;">Ruta Segura Explicable · Concepción</h1>
            <p style="margin:0; font-size:1.05rem;">
                Demo académica para comparar rutas con <strong>incidentes históricos</strong>,
                horario de viaje y <strong>perfiles simulados de viajero</strong>.
                No es una app de tiempo real: el valor está en explicar el tradeoff entre rapidez y exposición.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_controls() -> None:
    status = st.session_state.get("dataset_status") or client.dataset_status()
    st.session_state["dataset_status"] = status
    options = status.get("available", [])
    profile_keys = [item["key"] for item in options]
    labels = {item["key"]: item["label"] for item in options}
    current = status.get("current", "concepcion")
    selected = st.selectbox(
        "Alcance geográfico",
        options=profile_keys,
        index=profile_keys.index(current) if current in profile_keys else 0,
        format_func=lambda key: labels.get(key, key),
        help="Concepción es el modo principal de la demo. El modo regional queda como cobertura secundaria.",
    )
    if st.button("Aplicar perfil de datos", use_container_width=True):
        try:
            client.set_dataset(selected)
            clear_backend_cache_views()
            st.session_state["app_ready"] = False
            st.session_state["playground_results"] = None
            load_backend_context(force=True)
            st.rerun()
        except requests.RequestException as err:
            st.error(f"No se pudo cambiar el perfil: {err}")
    if current == "regional":
        st.warning("La cobertura regional se mantiene como vista secundaria y puede tener más ruido de datos.")


def render_data_quality_panel() -> None:
    ready_status = st.session_state.get("ready_status") or {}
    quality = ((ready_status.get("bootstrap") or {}).get("quality")) or {}
    if not quality:
        return
    warnings = quality.get("warnings") or []
    notes = quality.get("notes") or []
    counts = quality.get("raw_counts") or {}
    date_range = quality.get("date_range") or {}

    card_class = "warning-card" if warnings else "info-card"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="metric-title">Evidencia de datos</div>
            <div style="margin-top:0.3rem;">
                Perfil: <strong>{quality.get('dataset_profile', 'desconocido')}</strong> ·
                Incidentes analizados: <strong>{counts.get('combined', 0):,}</strong> ·
                Cobertura: <strong>{date_range.get('start', 'N/D')} a {date_range.get('end', 'N/D')}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if warnings:
        for item in warnings:
            st.warning(item)
    if notes:
        for note in notes:
            st.caption(note)


def render_overview_metrics() -> None:
    metadata = st.session_state.get("metadata") or {}
    quality = (((st.session_state.get("ready_status") or {}).get("bootstrap") or {}).get("quality")) or {}
    counts = quality.get("raw_counts") or {}
    cols = st.columns(4)
    metrics = [
        ("Eventos visibles", f"{metadata.get('total_events', 0):,}"),
        ("Vías monitoreadas", f"{metadata.get('total_vias', 0):,}"),
        ("Incidentes crudos", f"{counts.get('combined', 0):,}"),
        ("Cobertura temporal", f"{(quality.get('date_range') or {}).get('days', 0)} días"),
    ]
    for col, (title, value) in zip(cols, metrics):
        col.markdown(
            f"<div class='metric-card'><div class='metric-title'>{title}</div><div class='metric-value'>{value}</div></div>",
            unsafe_allow_html=True,
        )


def render_context_block() -> None:
    st.subheader("1. Contexto del viaje")
    scenarios = st.session_state.get("demo_scenarios") or []
    scenario_options = {scenario["title"]: scenario for scenario in scenarios}
    left, right = st.columns((1.1, 0.9), gap="large")
    with left:
        if scenario_options:
            current_scenario = st.session_state.get("selected_scenario_id")
            titles = list(scenario_options.keys())
            default_index = 0
            if current_scenario:
                for idx, item in enumerate(scenarios):
                    if item.get("id") == current_scenario:
                        default_index = idx
                        break
            selected_title = st.selectbox(
                "Escenario curado",
                options=titles,
                index=default_index,
                help="Escenarios listos para una demo de 3 minutos.",
            )
            scenario = scenario_options[selected_title]
            st.caption(scenario.get("description", ""))
            if st.button("Aplicar escenario", use_container_width=True):
                apply_scenario(scenario)
                st.rerun()
            st.info(f"Foco sugerido: {scenario.get('recommended_focus', '')}")

        profile_key = st.selectbox(
            "Perfil simulado de viajero",
            options=PROFILE_ORDER,
            index=PROFILE_ORDER.index(st.session_state.get("user_profile", "usuario_demo")),
            format_func=lambda key: TRAVELER_PROFILES[key]["label"],
            key="user_profile",
        )
        profile = TRAVELER_PROFILES[profile_key]
        st.caption(profile["description"])
        st.caption(profile["intent"])

        st.selectbox(
            "Día del viaje",
            options=[code for code, _ in DAY_CHOICES],
            format_func=lambda code: DAY_LABELS[code],
            index=[code for code, _ in DAY_CHOICES].index(st.session_state.get("trip_day", "Wednesday")),
            key="trip_day",
        )
        st.slider(
            "Hora de salida",
            min_value=0.0,
            max_value=23.0,
            value=float(st.session_state.get("trip_hour", 8.0)),
            step=1.0,
            key="trip_hour",
        )
        st.toggle("Evitar congestiones históricas", value=st.session_state.get("avoid_congestion", True), key="avoid_congestion")
        st.toggle("Evitar accidentes históricos", value=st.session_state.get("avoid_accidents", False), key="avoid_accidents")

    with right:
        origin = st.session_state["route_origin"]
        destination = st.session_state["route_destination"]
        st.markdown("**Coordenadas manuales**")
        point_cols = st.columns(2)
        with point_cols[0]:
            st.number_input("Origen lat", key="origin_lat", value=float(origin["lat"]), format="%.5f")
            st.number_input("Origen lon", key="origin_lon", value=float(origin["lon"]), format="%.5f")
        with point_cols[1]:
            st.number_input("Destino lat", key="destination_lat", value=float(destination["lat"]), format="%.5f")
            st.number_input("Destino lon", key="destination_lon", value=float(destination["lon"]), format="%.5f")
        if st.button("Aplicar coordenadas manuales", use_container_width=True):
            st.session_state["route_origin"] = {
                "lat": float(st.session_state["origin_lat"]),
                "lon": float(st.session_state["origin_lon"]),
            }
            st.session_state["route_destination"] = {
                "lat": float(st.session_state["destination_lat"]),
                "lon": float(st.session_state["destination_lon"]),
            }
            st.session_state["route_result"] = None
            st.rerun()
        st.caption("También puedes usar el mapa para asignar el próximo clic al origen o al destino.")
        render_selector_map(st.session_state["route_origin"], st.session_state["route_destination"])


def render_selector_map(origin: dict, destination: dict) -> None:
    avg_lat = (origin["lat"] + destination["lat"]) / 2
    avg_lon = (origin["lon"] + destination["lon"]) / 2
    fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="CartoDB positron", control_scale=True)
    MiniMap(toggle_display=True).add_to(fmap)
    folium.Marker([origin["lat"], origin["lon"]], tooltip="Origen", icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker([destination["lat"], destination["lon"]], tooltip="Destino", icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    assign_target = st.radio(
        "Asignar siguiente clic a",
        options=["Origen", "Destino"],
        horizontal=True,
        index=0 if st.session_state.get("map_assign_target", "Origen") == "Origen" else 1,
    )
    st.session_state["map_assign_target"] = assign_target
    click_event = st_folium(fmap, height=320, width=None, key="selector_map")
    if click_event and click_event.get("last_clicked"):
        target = "route_origin" if assign_target == "Origen" else "route_destination"
        st.session_state[target] = {
            "lat": float(click_event["last_clicked"]["lat"]),
            "lon": float(click_event["last_clicked"]["lng"]),
        }
        st.success(f"{assign_target} actualizado desde el mapa.")


def _build_polyline_points(raw_points: List[Dict[str, float]]) -> List[List[float]]:
    if not raw_points:
        return []
    return [[float(point["lat"]), float(point["lon"])] for point in raw_points if point.get("lat") is not None and point.get("lon") is not None]


def render_route_map(route_result: dict) -> None:
    st.subheader("2. Mapa comparativo")
    reference_points = _build_polyline_points((route_result.get("reference") or {}).get("geometry") or [])
    ubcf_points = _build_polyline_points((route_result.get("ubcf") or {}).get("geometry") or [])
    ibcf_points = _build_polyline_points((route_result.get("ibcf") or {}).get("geometry") or [])
    base_points = reference_points or ubcf_points or ibcf_points
    if not base_points:
        st.info("No hay geometría disponible para dibujar la ruta.")
        return
    fmap = folium.Map(location=base_points[0], zoom_start=13, tiles="CartoDB positron", control_scale=True)
    hotspots = st.session_state.get("hotspots") or []
    if hotspots:
        heat_data = [[spot["lat"], spot["lon"], spot.get("weight", 1.0)] for spot in hotspots if spot.get("lat") is not None and spot.get("lon") is not None]
        if heat_data:
            HeatMap(heat_data, radius=14, blur=12, max_zoom=12, name="Incidentes históricos").add_to(fmap)
    for variant, points, color, dash in [
        ("reference", reference_points, "#2563eb", "8"),
        ("ubcf", ubcf_points, "#16a34a", None),
        ("ibcf", ibcf_points, "#f97316", None),
    ]:
        if points:
            folium.PolyLine(
                points,
                color=color,
                weight=7,
                opacity=0.9,
                dash_array=dash,
                tooltip=VARIANT_META[variant]["label"],
            ).add_to(fmap)
    origin = st.session_state["route_origin"]
    destination = st.session_state["route_destination"]
    folium.Marker([origin["lat"], origin["lon"]], tooltip="Origen", icon=folium.Icon(color="green")).add_to(fmap)
    folium.Marker([destination["lat"], destination["lon"]], tooltip="Destino", icon=folium.Icon(color="red")).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    st_folium(fmap, height=620, width=None, use_container_width=True, key="route_map")
    st.caption("Capas: ruta base, variantes colaborativas e intensidad de incidentes históricos.")


def render_explanation_block(route_result: dict) -> None:
    st.subheader("3. Tarjetas de explicación")
    highlight_cols = st.columns(4)
    for col, item in zip(highlight_cols, comparison_highlights(route_result)):
        col.markdown(
            f"<div class='info-card'><div class='metric-title'>{item['title']}</div><div class='metric-value' style='font-size:1.2rem'>{item['label']}</div></div>",
            unsafe_allow_html=True,
        )

    cards = variant_summary_cards(route_result)
    card_cols = st.columns(len(cards))
    for col, card in zip(card_cols, cards):
        with col:
            st.markdown(f"#### {card['label']}")
            st.caption(card["story"])
            st.metric("Tiempo total", f"{card['total_minutes']:.1f} min")
            st.metric("Distancia", f"{card['distance_km']:.2f} km")
            st.metric("Riesgo", f"{card['risk_score']:.1f}/100")
            st.metric("Exposición", f"{card['matched_incidents']} segmentos")
            for reason in card["why_changed"]:
                st.write(f"- {reason}")
            penalized = card["top_penalized_segments"][:2]
            preferred = card["top_preferred_vias"][:2]
            if penalized:
                st.caption("Segmentos conflictivos")
                for item in penalized:
                    st.write(f"- {item['via']} · {item['event_type']} · impacto {item['impact_score']:.1f}")
            if preferred:
                st.caption("Vías destacadas por el perfil")
                for item in preferred:
                    st.write(f"- {item['via']} · factor {item['factor']:.2f}")


def render_evidence_block() -> None:
    st.subheader("4. Evidencia de datos y modelo")
    metadata = st.session_state.get("metadata") or {}
    quality = (((st.session_state.get("ready_status") or {}).get("bootstrap") or {}).get("quality")) or {}
    profile = TRAVELER_PROFILES[st.session_state.get("user_profile", "usuario_demo")]

    left, right = st.columns((1.1, 0.9), gap="large")
    with left:
        st.markdown("**Lo que sostiene la demo**")
        st.markdown(
            f"- Perfil activo: **{st.session_state.get('dataset_status', {}).get('current_label', 'Concepción')}**"
            f"\n- Días cubiertos: **{(quality.get('date_range') or {}).get('days', 0)}**"
            f"\n- Vías visibles: **{metadata.get('total_vias', 0):,}**"
            f"\n- Perfil simulado: **{profile['short_label']}**"
        )
        render_profile_recommendations()
    with right:
        st.markdown("**Límites conocidos**")
        st.markdown(
            "- Datos históricos de julio de 2025.\n"
            "- No hay tiempo real ni trazas reales de usuarios.\n"
            "- Los perfiles de viajero son sintéticos y se usan para comparar estrategias.\n"
            "- La cobertura regional puede incluir más ruido en etiquetas de red vial."
        )
        if quality.get("anomalous_communes"):
            st.caption("Etiquetas anómalas detectadas en la red: " + ", ".join(quality["anomalous_communes"][:5]))


def load_profile_recommendations(force: bool = False) -> dict | None:
    profile = st.session_state.get("user_profile", "usuario_demo")
    if not force and st.session_state.get("playground_results") and st.session_state.get("playground_profile") == profile:
        return st.session_state.get("playground_results")
    payload = {
        "user_id": profile,
        "known_vias": [],
        "limit": 12,
        "strategies": ["ubcf", "ibcf"],
    }
    try:
        result = client.playground(payload)
    except requests.RequestException as err:
        st.error(f"No se pudieron cargar rankings de vías para el perfil simulado: {err}")
        return None
    st.session_state["playground_results"] = result
    st.session_state["playground_profile"] = profile
    return result


def build_route_preferences(results: dict | None) -> Dict[str, List[Dict[str, float]]]:
    preferences = {"ubcf": [], "ibcf": []}
    if not results:
        return preferences
    for strategy in ["ubcf", "ibcf"]:
        recs = results.get(strategy) or []
        selected = (recs[:8] + recs[-4:]) if len(recs) > 8 else recs
        for item in selected:
            rating = float(item.get("estimated_rating", 0.0))
            preferences[strategy].append(
                {
                    "via": item["via"],
                    "weight": round(max(0.0, min(1.0, rating / 5.0)), 3),
                }
            )
    return preferences


def render_profile_recommendations() -> None:
    results = load_profile_recommendations(force=False)
    if not results:
        return
    st.markdown("**Ranking de vías por perfil simulado**")
    cols = st.columns(2)
    for col, strategy in zip(cols, ["ubcf", "ibcf"]):
        with col:
            st.caption(VARIANT_META[strategy]["label"])
            for item in (results.get(strategy) or [])[:5]:
                st.write(f"- {item['via']} · score {item['estimated_rating']:.2f}")


def generate_route() -> None:
    recommendations = load_profile_recommendations(force=True)
    prefs = build_route_preferences(recommendations)
    payload = {
        "origin": st.session_state["route_origin"],
        "destination": st.session_state["route_destination"],
        "preferences": [],
        "ubcf_preferences": prefs.get("ubcf", []),
        "ibcf_preferences": prefs.get("ibcf", []),
        "day_of_week": st.session_state.get("trip_day", "Wednesday"),
        "departure_hour": float(st.session_state.get("trip_hour", 8.0)),
        "avoid_congestion": st.session_state.get("avoid_congestion", True),
        "avoid_accidents": st.session_state.get("avoid_accidents", False),
    }
    try:
        st.session_state["route_result"] = client.optimal_route(payload, timeout=max(REQUEST_TIMEOUT, 240))
    except requests.HTTPError as err:
        detail = err.response.text if err.response is not None else str(err)
        st.error(f"El backend rechazó la simulación: {detail}")
    except requests.RequestException as err:
        st.error(f"No se pudo calcular la ruta: {err}")


def main() -> None:
    st.set_page_config(page_title="Ruta Segura Explicable", page_icon="🧭", layout="wide")
    set_page_style()
    init_state()
    render_header()

    if not load_backend_context(force=False):
        st.stop()

    top_cols = st.columns((0.75, 0.25), gap="large")
    with top_cols[0]:
        render_overview_metrics()
    with top_cols[1]:
        render_dataset_controls()
        if st.button("Refrescar evidencia", use_container_width=True):
            load_backend_context(force=True)
            st.rerun()

    render_data_quality_panel()
    render_context_block()

    if st.button("Generar comparación de rutas", type="primary", use_container_width=True):
        generate_route()

    route_result = st.session_state.get("route_result")
    if route_result:
        render_route_map(route_result)
        render_explanation_block(route_result)

    render_evidence_block()


if __name__ == "__main__":
    main()
