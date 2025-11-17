# -*- coding: utf-8 -*-
"""
Aplicación Streamlit simple: selecciona origen/destino en el Biobío, ajusta preferencias
para evitar congestiones/accidentes y genera la ruta con Dijkstra + recomendaciones.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple

import folium
from folium.plugins import BeautifyIcon, MiniMap, HeatMap
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT", "90"))
DEFAULT_ORIGIN = {"lat": -36.8270, "lon": -73.0500}
DEFAULT_DESTINATION = {"lat": -36.8200, "lon": -73.0435}
DEFAULT_LAT_RANGE = (-38.8, -35.8)
DEFAULT_LON_RANGE = (-74.8, -71.0)
DEFAULT_BOUNDS = {
    "lat_min": DEFAULT_LAT_RANGE[0],
    "lat_max": DEFAULT_LAT_RANGE[1],
    "lon_min": DEFAULT_LON_RANGE[0],
    "lon_max": DEFAULT_LON_RANGE[1],
}


def get_bounds() -> Dict[str, float]:
    bounds = st.session_state.get("bounds")
    if not bounds:
        bounds = DEFAULT_BOUNDS.copy()
        st.session_state["bounds"] = bounds
    return bounds


def get_lat_range() -> tuple[float, float]:
    bounds = get_bounds()
    return (bounds["lat_min"], bounds["lat_max"])


def get_lon_range() -> tuple[float, float]:
    bounds = get_bounds()
    return (bounds["lon_min"], bounds["lon_max"])


def is_within_bounds(lat: float, lon: float) -> bool:
    lat_min, lat_max = get_lat_range()
    lon_min, lon_max = get_lon_range()
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def wait_for_backend(max_wait: int = 300) -> None:
    if st.session_state.get("backend_ready"):
        return
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    start_time = time.time()
    poll_interval = 5
    attempt = 0
    while time.time() - start_time < max_wait:
        attempt += 1
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            resp.raise_for_status()
            status_placeholder.success("Backend disponible. Cargando datos...")
            st.session_state["backend_ready"] = True
            progress_bar.progress(100)
            return
        except requests.RequestException:
            elapsed = time.time() - start_time
            pct = min(int((elapsed / max_wait) * 100), 100)
            progress_bar.progress(pct)
            status_placeholder.info(
                f"Inicializando backend (intento {attempt}). Quedan {max(0, max_wait - int(elapsed))}s..."
            )
            time.sleep(poll_interval)
    status_placeholder.error("No se pudo validar el backend a tiempo. Revisa el servicio FastAPI.")
    st.stop()


def load_backend_data(force: bool = False) -> bool:
    if st.session_state.get("app_ready") and not force:
        return True
    if force:
        fetch_metadata.clear()
        st.session_state["metadata"] = None
    wait_for_backend()
    status_placeholder = st.empty()
    status_placeholder.info("Inicializando infraestructura en el backend...")
    progress_bar = st.progress(0)
    try:
        resp = requests.post(f"{BACKEND_URL}/system/bootstrap", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as err:
        status_placeholder.error(f"No se pudo iniciar el bootstrap: {err}")
        return False

    poll_interval = 2
    start_time = time.time()
    while True:
        try:
            status_resp = requests.get(f"{BACKEND_URL}/system/bootstrap/status", timeout=REQUEST_TIMEOUT)
            status_resp.raise_for_status()
            info = status_resp.json()
        except requests.RequestException as err:
            status_placeholder.error(f"Error consultando el estado del backend: {err}")
            return False
        percent = info.get("percent", 0)
        progress_bar.progress(max(0, min(100, percent)))
        status_placeholder.info(info.get("message", "Preparando..."))
        if info.get("status") == "completed":
            status_placeholder.success(
                f"Backend listo: {info.get('routing_nodes', 0):,} nodos cargados en "
                f"{info.get('duration_ms', 0)} ms."
            )
            break
        if info.get("status") == "error":
            status_placeholder.error(f"Ocurrió un error durante el bootstrap: {info.get('message')}")
            return False
        if time.time() - start_time > 900:
            status_placeholder.error("El bootstrap está tomando demasiado tiempo. Intenta nuevamente.")
            return False
        time.sleep(poll_interval)

    metadata = fetch_metadata()
    st.session_state["metadata"] = metadata
    st.session_state["bounds"] = metadata.get("bounds") or DEFAULT_BOUNDS.copy()
    st.session_state["hotspots"] = fetch_hotspots()
    st.session_state["app_ready"] = True
    return True


def set_page_style() -> None:
    st.markdown(
        """
        <style>
        .metric-card {
            padding: 1rem;
            border-radius: 0.6rem;
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.7;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def fetch_metadata() -> Dict[str, List[str]]:
    try:
        resp = requests.get(f"{BACKEND_URL}/metadata/options", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return {
            "event_types": ["Congestión", "Accidente"],
            "communes": [],
            "franjas": ["No definido"],
            "durations": [],
            "velocities": [],
            "vias": [],
            "total_events": 0,
            "total_vias": 0,
            "accident_ratio": 0.0,
            "bounds": DEFAULT_BOUNDS,
        }


@st.cache_data(show_spinner=False)
def fetch_hotspots(limit: int = 2000) -> List[Dict[str, float]]:
    try:
        resp = requests.get(
            f"{BACKEND_URL}/metadata/hotspots",
            params={"limit": limit},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("points", [])
    except requests.RequestException:
        return []


def call_backend(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("No se pudo conectar con el backend. Asegúrate de ejecutar FastAPI en el puerto 8000.")
    except requests.HTTPError as err:
        st.error(f"Error del backend: {err.response.text}")
    except requests.RequestException as err:
        st.error(f"Ocurrió un error al llamar al backend: {err}")
    return None


def init_state() -> None:
    st.session_state.setdefault("route_origin", DEFAULT_ORIGIN.copy())
    st.session_state.setdefault("route_destination", DEFAULT_DESTINATION.copy())
    st.session_state.setdefault("assign_target", "Origen")
    st.session_state.setdefault("origin_lat_input", st.session_state["route_origin"]["lat"])
    st.session_state.setdefault("origin_lon_input", st.session_state["route_origin"]["lon"])
    st.session_state.setdefault("dest_lat_input", st.session_state["route_destination"]["lat"])
    st.session_state.setdefault("dest_lon_input", st.session_state["route_destination"]["lon"])
    st.session_state.setdefault("last_route_result", None)
    st.session_state.setdefault("bounds", DEFAULT_BOUNDS.copy())
    st.session_state.setdefault("metadata", None)
    st.session_state.setdefault("app_ready", False)
    st.session_state.setdefault("hotspots", [])
    st.session_state.setdefault("playground_results", None)


def reset_locations() -> None:
    st.session_state["route_origin"] = DEFAULT_ORIGIN.copy()
    st.session_state["route_destination"] = DEFAULT_DESTINATION.copy()
    st.session_state["origin_lat_input"] = DEFAULT_ORIGIN["lat"]
    st.session_state["origin_lon_input"] = DEFAULT_ORIGIN["lon"]
    st.session_state["dest_lat_input"] = DEFAULT_DESTINATION["lat"]
    st.session_state["dest_lon_input"] = DEFAULT_DESTINATION["lon"]
    st.session_state["last_route_result"] = None


def render_overview(metadata: Dict[str, List[str]]) -> None:
    total_events = metadata.get("total_events", 0)
    total_vias = metadata.get("total_vias", 0)
    accident_ratio = metadata.get("accident_ratio", 0.0)
    bounds = metadata.get("bounds") or DEFAULT_BOUNDS
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"<div class='metric-card'><div class='metric-label'>Eventos procesados</div>"
        f"<div class='metric-value'>{total_events:,}</div></div>",
        unsafe_allow_html=True,
    )
    col2.markdown(
        f"<div class='metric-card'><div class='metric-label'>Vías monitoreadas</div>"
        f"<div class='metric-value'>{total_vias:,}</div></div>",
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"<div class='metric-card'><div class='metric-label'>Ratio de accidentes</div>"
        f"<div class='metric-value'>{accident_ratio*100:.1f}%</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Cobertura geográfica: lat {bounds['lat_min']:.2f}° a {bounds['lat_max']:.2f}°, "
        f"lon {bounds['lon_min']:.2f}° a {bounds['lon_max']:.2f}°."
    )


def render_guidelines() -> None:
    with st.expander("¿Cómo usar la aplicación?", expanded=False):
        st.markdown(
            "- Selecciona origen y destino directamente en el mapa o introduce coordenadas precisas.\n"
            "- Genera la ruta segura y consulta el detalle paso a paso en la tabla.\n"
            "- Usa el playground colaborativo para comparar UBCF vs IBCF y decidir qué estrategia se ajusta mejor.\n"
            "- Si el backend no responde, revisa que el servicio FastAPI esté activo en el puerto 8000."
        )


def render_sidebar_tools() -> None:
    st.sidebar.header("Herramientas rápidas")
    if st.sidebar.button("Restablecer puntos de ruta", use_container_width=True):
        reset_locations()
        st.sidebar.success("Coordenadas restauradas a los valores sugeridos.")
        st.experimental_rerun()
    st.sidebar.caption(
        "La sección principal permite generar la ruta y comparar estrategias colaborativas (UBCF vs IBCF)."
    )


def validate_coordinates(origin: dict, destination: dict) -> tuple[bool, str]:
    lat_range = get_lat_range()
    lon_range = get_lon_range()
    for label, point in (("Origen", origin), ("Destino", destination)):
        lat, lon = point["lat"], point["lon"]
        if not (lat_range[0] <= lat <= lat_range[1]) or not (lon_range[0] <= lon <= lon_range[1]):
            return (
                False,
                f"{label} fuera del polígono soportado ({lat_range[0]}° a {lat_range[1]}°, "
                f"{lon_range[0]}° a {lon_range[1]}°).",
            )
    return True, ""


def render_selector_map(origin: dict, destination: dict) -> None:
    if not origin or not destination:
        st.warning("Selecciona dos puntos válidos dentro del Biobío.")
        return
    st.caption(
        "El mapa muestra capas claras, oscuras y de relieve. Cambia capas desde el control en la esquina superior derecha."
    )
    avg_lat = (origin["lat"] + destination["lat"]) / 2
    avg_lon = (origin["lon"] + destination["lon"]) / 2
    selector_map = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=12,
        tiles="CartoDB positron",
        control_scale=True,
    )
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        name="Terreno",
        attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors",
        overlay=False,
    ).add_to(selector_map)
    folium.TileLayer("CartoDB dark_matter", name="Oscuro", overlay=False).add_to(selector_map)
    folium.LayerControl(collapsed=True).add_to(selector_map)
    MiniMap(toggle_display=True).add_to(selector_map)

    folium.Marker(
        location=[origin["lat"], origin["lon"]],
        tooltip="Origen",
        draggable=False,
        icon=BeautifyIcon(
            icon_shape="marker",
            number=1,
            border_color="#0ea5e9",
            text_color="#0ea5e9",
            background_color="#ecfeff",
        ),
    ).add_to(selector_map)
    folium.Marker(
        location=[destination["lat"], destination["lon"]],
        tooltip="Destino",
        draggable=False,
        icon=BeautifyIcon(
            icon_shape="marker",
            number=2,
            border_color="#f97316",
            text_color="#f97316",
            background_color="#fff7ed",
        ),
    ).add_to(selector_map)
    assign_target = st.radio(
        "Asignar último clic a:",
        options=["Origen", "Destino"],
        index=0 if st.session_state["assign_target"] == "Origen" else 1,
        horizontal=True,
    )
    st.session_state["assign_target"] = assign_target
    click_event = st_folium(selector_map, height=360, width=None, key="selector_map")
    if click_event and click_event.get("last_clicked"):
        lat = click_event["last_clicked"]["lat"]
        lon = click_event["last_clicked"]["lng"]
        if not is_within_bounds(lat, lon):
            st.warning(
                "El punto seleccionado está fuera de la red soportada. Elige una ubicación dentro del Biobío."
            )
            return
        target = "route_origin" if assign_target == "Origen" else "route_destination"
        st.session_state[target] = {"lat": lat, "lon": lon}
        st.success(f"{assign_target} actualizado a {lat:.5f}, {lon:.5f}")
        st.session_state["origin_lat_input"] = st.session_state["route_origin"]["lat"]
        st.session_state["origin_lon_input"] = st.session_state["route_origin"]["lon"]
        st.session_state["dest_lat_input"] = st.session_state["route_destination"]["lat"]
        st.session_state["dest_lon_input"] = st.session_state["route_destination"]["lon"]


def render_route_map(route_geometry: List[Dict[str, float]]) -> None:
    if not route_geometry:
        st.info("No hay coordenadas suficientes para dibujar la ruta.")
        return
    coords = pd.DataFrame(route_geometry).dropna()
    if coords.empty:
        st.info("No hay coordenadas suficientes para dibujar la ruta.")
        return
    fmap = folium.Map(
        location=[coords["lat"].mean(), coords["lon"].mean()],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )
    points = coords[["lat", "lon"]].values.tolist()
    if len(points) < 2:
        start = points[0]
        end = points[-1]
        origin = st.session_state.get("route_origin", DEFAULT_ORIGIN)
        destination = st.session_state.get("route_destination", DEFAULT_DESTINATION)
        if len(points) == 1 and (origin["lat"], origin["lon"]) != tuple(points[0]):
            start = [origin["lat"], origin["lon"]]
        if len(points) == 1 and (destination["lat"], destination["lon"]) != tuple(points[0]):
            end = [destination["lat"], destination["lon"]]
        points = [start, end]
        st.warning(
            "La ruta reportada contiene un único punto, se muestra un enlace directo estimado entre origen y destino."
        )
    hotspots = st.session_state.get("hotspots") or []
    if hotspots:
        heat_data = [
            [spot["lat"], spot["lon"], spot.get("weight", 1.0)]
            for spot in hotspots
            if spot.get("lat") is not None and spot.get("lon") is not None
        ]
        if heat_data:
            HeatMap(
                heat_data,
                radius=15,
                blur=12,
                max_zoom=12,
                gradient={0.2: "blue", 0.4: "cyan", 0.6: "lime", 0.8: "yellow", 1.0: "red"},
                name="Congestiones recientes",
            ).add_to(fmap)
    MiniMap(toggle_display=True, position="bottomright").add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    folium.PolyLine(
        points,
        color="#38bdf8",
        weight=8,
        opacity=0.9,
        dash_array="10",
        tooltip="Ruta optimizada",
    ).add_to(fmap)
    folium.CircleMarker(
        points[0],
        radius=7,
        color="#22c55e",
        fill=True,
        fill_opacity=0.9,
        popup="Inicio",
    ).add_to(fmap)
    folium.CircleMarker(
        points[-1],
        radius=7,
        color="#ef4444",
        fill=True,
        fill_opacity=0.9,
        popup="Destino",
    ).add_to(fmap)
    st_folium(fmap, height=430, width=None, key="route_map")
    st.caption("La línea azul marca la trayectoria optimizada considerando congestiones/accidentes.")


def render_route_summary(route_result: dict) -> None:
    steps_df = pd.DataFrame(route_result["steps"])
    geometry = route_result.get("geometry") or steps_df[["lat", "lon"]].to_dict("records")
    col1, col2 = st.columns(2)
    col1.metric("Distancia estimada", f"{route_result['distance_km']} km")
    col2.metric("Duración aproximada", f"{route_result['estimated_duration_min']} min")
    tab_map, tab_table = st.tabs(["Mapa interactivo", "Detalle del trayecto"])
    with tab_map:
        render_route_map(geometry)
    with tab_table:
        st.dataframe(
            steps_df[["via", "comuna", "cumulative_cost"]],
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = steps_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar ruta (CSV)",
            data=csv_bytes,
            file_name="ruta_segura_biobio.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Compara estrategias colaborativas en el playground inferior para descubrir vías alternativas personalizadas."
        )


def render_playground_results(results: Dict[str, List[dict]], strategies: List[str]) -> None:
    if not strategies:
        strategies = ["ubcf", "ibcf"]
    cols = st.columns(len(strategies))
    for idx, strategy in enumerate(strategies):
        human_label = strategy.upper()
        recs = results.get(strategy, [])
        with cols[idx]:
            st.markdown(f"#### {human_label}")
            if not recs:
                st.caption("Sin recomendaciones para esta estrategia.")
                continue
            for rec in recs:
                st.write(
                    f"- **{rec['via']}** · puntuación estimada {rec['estimated_rating']:.2f} "
                    f"({rec['strategy'].upper()})"
                )


def playground_section(metadata: Dict[str, List[str]]) -> None:
    st.subheader("Laboratorio colaborativo (UBCF vs IBCF)")
    vias = metadata.get("vias") or []
    default_user = st.session_state.get("play_user_id", "usuario_demo")
    default_known = [via for via in st.session_state.get("play_known_vias", []) if via in vias]
    default_limit = st.session_state.get("play_limit", 5)
    default_strategies = st.session_state.get("play_strategies", ["ubcf", "ibcf"])
    with st.form("playground_form"):
        user_id = st.text_input("Usuario objetivo", value=default_user)
        known_vias = st.multiselect(
            "Vías ya conocidas (se excluyen de la recomendación)",
            options=vias,
            default=default_known,
            help="Selecciona vías que el usuario ya conoce para forzar recomendaciones frescas.",
        )
        limit = st.slider("Cantidad de recomendaciones por estrategia", 1, 10, default_limit)
        strategies = st.multiselect(
            "Estrategias a comparar",
            options=["ubcf", "ibcf"],
            default=default_strategies or ["ubcf", "ibcf"],
        )
        submitted = st.form_submit_button("Comparar estrategias", use_container_width=True)
    if submitted:
        payload = {
            "user_id": user_id.strip() or "usuario_demo",
            "known_vias": known_vias,
            "limit": limit,
            "strategies": strategies,
        }
        with st.spinner("Calculando recomendaciones colaborativas..."):
            result = call_backend("/recommendations/playground", payload)
        if result:
            st.session_state["playground_results"] = result
            st.session_state["play_user_id"] = payload["user_id"]
            st.session_state["play_known_vias"] = known_vias
            st.session_state["play_limit"] = limit
            st.session_state["play_strategies"] = strategies
        else:
            st.session_state["playground_results"] = None
    results = st.session_state.get("playground_results")
    if results:
        active_strategies = st.session_state.get("play_strategies") or ["ubcf", "ibcf"]
        render_playground_results(results, active_strategies)
    else:
        st.info("Genera recomendaciones colaborativas para visualizar la diferencia entre UBCF e IBCF.")


def routing_section() -> None:
    origin = st.session_state["route_origin"]
    destination = st.session_state["route_destination"]
    lat_range = get_lat_range()
    lon_range = get_lon_range()
    st.subheader("1. Selecciona tu ubicación y destino")
    map_col, form_col = st.columns((1.2, 0.8), gap="large")
    with map_col:
        render_selector_map(origin, destination)
    with form_col:
        st.subheader("2. Ajusta coordenadas y genera la ruta")
        with st.form("route_form"):
            col1, col2 = st.columns(2)
            with col1:
                origin_lat = st.number_input(
                    "Latitud origen",
                    value=st.session_state["origin_lat_input"],
                    key="origin_lat_input",
                    format="%.6f",
                    min_value=float(lat_range[0]),
                    max_value=float(lat_range[1]),
                )
                origin_lon = st.number_input(
                    "Longitud origen",
                    value=st.session_state["origin_lon_input"],
                    key="origin_lon_input",
                    format="%.6f",
                    min_value=float(lon_range[0]),
                    max_value=float(lon_range[1]),
                )
            with col2:
                dest_lat = st.number_input(
                    "Latitud destino",
                    value=st.session_state["dest_lat_input"],
                    key="dest_lat_input",
                    format="%.6f",
                    min_value=float(lat_range[0]),
                    max_value=float(lat_range[1]),
                )
                dest_lon = st.number_input(
                    "Longitud destino",
                    value=st.session_state["dest_lon_input"],
                    key="dest_lon_input",
                    format="%.6f",
                    min_value=float(lon_range[0]),
                    max_value=float(lon_range[1]),
                )
            submitted = st.form_submit_button("Generar ruta segura", use_container_width=True)

    results_container = st.container()
    last_route = st.session_state.get("last_route_result")

    if submitted:
        st.session_state["route_origin"] = {"lat": origin_lat, "lon": origin_lon}
        st.session_state["route_destination"] = {"lat": dest_lat, "lon": dest_lon}
        origin = st.session_state["route_origin"]
        destination = st.session_state["route_destination"]
        valid, message = validate_coordinates(origin, destination)
        if not valid:
            st.warning(message)
            return
        with st.spinner("Calculando ruta con Dijkstra y ajustando pesos..."):
            route_payload = {"origin": origin, "destination": destination}
            route_result = call_backend("/routes/optimal", route_payload)

        if route_result and route_result.get("steps"):
            st.session_state["last_route_result"] = route_result
            with results_container:
                render_route_summary(route_result)
        else:
            st.session_state["last_route_result"] = None
            st.error("No se pudo construir la ruta. Ajusta los puntos y vuelve a intentar.")
    elif last_route and last_route.get("steps"):
        with results_container:
            render_route_summary(last_route)


def main() -> None:
    st.set_page_config(page_title="Ruta segura Biobío", page_icon="🧭", layout="wide")
    set_page_style()
    st.title("🧭 Ruta Segura Biobío")
    st.write(
        "Planifica recorridos con datos actualizados de congestiones y accidentes. "
        "La ruta que generes prioriza la seguridad ajustando los pesos del grafo vial."
    )
    init_state()
    render_sidebar_tools()
    if not st.session_state.get("app_ready"):
        st.info("Carga el backend cuando estés listo para trabajar con la red vial del Biobío.")
        if st.button("Cargar datos de la Región del Biobío", type="primary", use_container_width=True):
            with st.spinner("Consultando backend y preparando datos..."):
                if load_backend_data(force=True):
                    st.rerun()
        st.stop()
    metadata = st.session_state.get("metadata") or fetch_metadata()
    cols = st.columns([1, 3])
    if cols[0].button("Actualizar datos", use_container_width=True):
        with st.spinner("Refrescando datos del backend..."):
            if load_backend_data(force=True):
                st.rerun()
    render_overview(metadata)
    render_guidelines()
    routing_section()
    st.divider()
    playground_section(metadata)


if __name__ == "__main__":
    main()
