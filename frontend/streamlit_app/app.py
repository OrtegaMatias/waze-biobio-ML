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
import numpy as np
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
HOUR_BUCKETS = [
    (0, 6, "Madrugada (00-05h)"),
    (6, 10, "Punta AM (06-09h)"),
    (10, 16, "Horario Medio (10-15h)"),
    (16, 21, "Punta PM (16-20h)"),
    (21, 24, "Nocturno (21-23h)"),
]


def get_hour_bucket_label(hour: float) -> str:
    hour = float(hour or 0)
    for start, end, label in HOUR_BUCKETS:
        if start <= hour < end:
            return label
    return HOUR_BUCKETS[0][2]


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
        .streamlit-folium, .streamlit-folium iframe {
            width: 100% !important;
        }
        .streamlit-folium {
            min-height: 680px !important;
            height: 680px !important;
        }
        .streamlit-folium iframe {
            min-height: 680px !important;
            height: 680px !important;
            border-radius: 0.75rem;
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


def fetch_dataset_status() -> dict | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/system/dataset", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def update_dataset_profile(profile: str) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/system/dataset",
            json={"profile": profile},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as err:
        st.error(f"No se pudo actualizar el perfil de datos: {err}")
        return None


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
    st.session_state.setdefault("last_route_result", None)
    st.session_state.setdefault("bounds", DEFAULT_BOUNDS.copy())
    st.session_state.setdefault("metadata", None)
    st.session_state.setdefault("app_ready", False)
    st.session_state.setdefault("hotspots", [])
    st.session_state.setdefault("playground_results", None)
    st.session_state.setdefault("dataset_status", None)
    st.session_state.setdefault("trip_day", "Monday")
    st.session_state.setdefault("trip_hour", 8)
    st.session_state.setdefault("avoid_congestion", True)
    st.session_state.setdefault("avoid_accidents", False)


def reset_locations() -> None:
    st.session_state["route_origin"] = DEFAULT_ORIGIN.copy()
    st.session_state["route_destination"] = DEFAULT_DESTINATION.copy()
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


def render_dataset_selector() -> None:
    status = st.session_state.get("dataset_status") or fetch_dataset_status()
    if not status:
        st.warning("No se pudo consultar el perfil de datos actual.")
        return
    st.session_state["dataset_status"] = status
    options = [info["key"] for info in status.get("available", [])]
    if not options:
        st.info("No hay perfiles alternativos disponibles.")
        return
    labels = {info["key"]: info["label"] for info in status.get("available", [])}
    current = status.get("current")
    try:
        index = options.index(current)
    except ValueError:
        index = 0
    selection = st.selectbox(
        "Perfil de datos",
        options=options,
        index=index,
        format_func=lambda key: labels.get(key, key),
        help="Selecciona el conjunto de ratings que alimenta el filtrado colaborativo.",
    )
    apply_disabled = selection == current
    if st.button("Aplicar perfil", use_container_width=True, disabled=apply_disabled):
        with st.spinner("Actualizando fuente de datos en el backend..."):
            result = update_dataset_profile(selection)
        if result:
            st.session_state["dataset_status"] = result
            st.session_state["playground_results"] = None
            st.success(f"Perfil actualizado a {labels.get(selection, selection)}")


def render_sidebar_tools() -> None:
    st.sidebar.header("Herramientas rápidas")
    if st.sidebar.button("Restablecer puntos de ruta", use_container_width=True):
        reset_locations()
        st.sidebar.success("Coordenadas restauradas a los valores sugeridos.")
        if hasattr(st, "rerun"):
            st.rerun()
        else:  # compatibilidad con versiones anteriores
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
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.TileLayer("CartoDB positron", name="Clásico", overlay=False).add_to(selector_map)
    folium.TileLayer(
        tiles="https://stamen-tiles.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png",
        name="Toner claro",
        attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors",
        overlay=False,
    ).add_to(selector_map)
    folium.LayerControl(collapsed=True).add_to(selector_map)
    MiniMap(tile_layer="OpenStreetMap", toggle_display=True).add_to(selector_map)

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


def _build_polyline_points(raw_points: List[Dict[str, float]]) -> List[List[float]]:
    coords = pd.DataFrame(raw_points).dropna()
    if coords.empty:
        return []
    points = coords[["lat", "lon"]].values.tolist()
    if len(points) < 2 and points:
        origin = st.session_state.get("route_origin", DEFAULT_ORIGIN)
        destination = st.session_state.get("route_destination", DEFAULT_DESTINATION)
        start = [origin["lat"], origin["lon"]]
        end = [destination["lat"], destination["lon"]]
        points = [start, end]
    return points


def render_route_map(route_result: dict) -> None:
    reference = route_result.get("reference") or {}
    personalized = route_result.get("personalized") or {}
    reference_points = _build_polyline_points(reference.get("geometry") or [])
    personalized_points = _build_polyline_points(personalized.get("geometry") or [])
    base_points = reference_points or personalized_points
    if not base_points:
        st.info("No hay coordenadas suficientes para dibujar la ruta.")
        return
    fmap = folium.Map(
        location=[np.mean([p[0] for p in base_points]), np.mean([p[1] for p in base_points])],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )
    hotspots = st.session_state.get("hotspots") or []
    hotspots = st.session_state.get("hotspots") or []
    selected_day = st.session_state.get("trip_day")
    if hotspots and selected_day:
        current_hour = float(st.session_state.get("trip_hour", 8))
        bucket_label = get_hour_bucket_label(current_hour)
        heat_data = []
        for spot in hotspots:
            day = (spot.get("day") or "").strip()
            bucket = (spot.get("bucket") or "").strip()
            start = spot.get("hora_inicio_float")
            end = spot.get("hora_fin_float")
            matches_hour = True
            if start is not None and end is not None:
                if start <= end:
                    matches_hour = start <= current_hour <= end
                else:
                    matches_hour = current_hour >= start or current_hour <= end
            if day and day != selected_day:
                continue
            if bucket and bucket != bucket_label:
                continue
            if not matches_hour:
                continue
            if spot.get("lat") is None or spot.get("lon") is None:
                continue
            heat_data.append([spot["lat"], spot["lon"], spot.get("weight", 1.0)])
        if not heat_data:
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
                name=f"Congestiones {bucket_label}",
            ).add_to(fmap)
    MiniMap(toggle_display=True, position="bottomright").add_to(fmap)

    def add_route_layer(points: List[List[float]], color: str, name: str, dash: str | None = None, show: bool = True) -> None:
        if not points:
            return
        layer = folium.FeatureGroup(name=name, show=show)
        folium.PolyLine(
            points,
            color=color,
            weight=7,
            opacity=0.9,
            dash_array=dash,
        ).add_to(layer)
        layer.add_to(fmap)

    add_route_layer(reference_points, "#2563eb", "Ruta base (solo Dijkstra)", dash="8", show=True)
    add_route_layer(personalized_points, "#fb923c", "Ruta personalizada (con preferencias)", dash=None, show=False)

    origin = st.session_state.get("route_origin", DEFAULT_ORIGIN)
    destination = st.session_state.get("route_destination", DEFAULT_DESTINATION)
    folium.Marker(
        [origin["lat"], origin["lon"]],
        icon=folium.Icon(color="green", icon="play"),
        tooltip="Inicio",
    ).add_to(fmap)
    folium.Marker(
        [destination["lat"], destination["lon"]],
        icon=folium.Icon(color="red", icon="stop"),
        tooltip="Destino",
    ).add_to(fmap)

    folium.LayerControl(collapsed=True).add_to(fmap)
    st_folium(fmap, height=700, width=None, use_container_width=True, key="route_map")
    st.caption(
        "Activa/desactiva cada capa para comparar la ruta base de Dijkstra contra la ajustada con tus preferencias."
    )


def render_route_summary(route_result: dict) -> None:
    reference = route_result.get("reference") or {}
    personalized = route_result.get("personalized") or {}
    ref_distance = reference.get("distance_km")
    ref_duration = reference.get("estimated_duration_min")
    per_distance = personalized.get("distance_km")
    per_duration = personalized.get("estimated_duration_min")
    diff_distance = None if ref_distance is None or per_distance is None else per_distance - ref_distance
    diff_duration = None if ref_duration is None or per_duration is None else per_duration - ref_duration
    ref_delay = reference.get("extra_delay_min", 0.0)
    per_delay = personalized.get("extra_delay_min", 0.0)

    col1, col2 = st.columns(2)
    col1.metric(
        "Dijkstra base - Distancia",
        f"{ref_distance:.2f} km" if ref_distance is not None else "N/A",
    )
    col1.metric(
        "Dijkstra base - Duración",
        f"{ref_duration:.1f} min" if ref_duration is not None else "N/A",
        delta=f"+{ref_delay:.1f} min por congestión" if ref_delay else None,
    )
    col2.metric(
        "Ruta personalizada - Distancia",
        f"{per_distance:.2f} km" if per_distance is not None else "N/A",
        delta=f"{diff_distance:+.2f} km" if diff_distance is not None else None,
    )
    personalized_delta = None
    if diff_duration is not None:
        personalized_delta = f"{diff_duration:+.1f} min vs base"
    if per_delay:
        delay_text = f"+{per_delay:.1f} min por congestión"
        personalized_delta = (
            f"{personalized_delta} | {delay_text}" if personalized_delta else delay_text
        )
    col2.metric(
        "Ruta personalizada - Duración",
        f"{per_duration:.1f} min" if per_duration is not None else "N/A",
        delta=personalized_delta,
    )

    st.markdown("### Comparación en el mapa")
    render_route_map(route_result)


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


def collect_route_preferences() -> List[Dict[str, float]]:
    results = st.session_state.get("playground_results") or {}
    weights: Dict[str, List[float]] = {}
    for entries in results.values():
        for rec in entries:
            via = rec.get("via")
            rating = rec.get("estimated_rating")
            if not via or rating is None:
                continue
            weights.setdefault(via, []).append(float(rating) / 5.0)
    preferences = [
        {"via": via, "weight": round(sum(values) / len(values), 3)}
        for via, values in weights.items()
    ]
    return preferences


def _has_route_steps(result: dict | None) -> bool:
    if not result:
        return False
    variant = result.get("personalized") or result.get("reference")
    return bool(variant and variant.get("steps"))


def routing_section() -> None:
    origin = st.session_state["route_origin"]
    destination = st.session_state["route_destination"]
    st.subheader("1. Selecciona tu ubicación y destino")
    map_col, action_col = st.columns((1.3, 0.7), gap="large")
    with map_col:
        render_selector_map(origin, destination)
    with action_col:
        st.subheader("2. Genera la ruta segura")
        st.caption(
            "Usa el mapa para mover los pines de origen/destino y configura el escenario de viaje para ajustar las recomendaciones."
        )
        selected_day = st.selectbox(
            "Día del viaje",
            options=[code for code, _ in DAY_CHOICES],
            format_func=lambda code: DAY_LABELS[code],
            index=[code for code, _ in DAY_CHOICES].index(st.session_state.get("trip_day", "Monday")),
            key="trip_day",
        )
        departure_hour = st.slider(
            "Hora de salida (0-23h)",
            min_value=0,
            max_value=23,
            value=st.session_state.get("trip_hour", 8),
            key="trip_hour",
        )
        avoid_congestion = st.checkbox(
            "Evitar congestiones",
            key="avoid_congestion",
            value=st.session_state.get("avoid_congestion", True),
        )
        avoid_accidents = st.checkbox(
            "Evitar accidentes",
            key="avoid_accidents",
            value=st.session_state.get("avoid_accidents", False),
        )
        generate_clicked = st.button("Generar ruta segura", use_container_width=True)

    results_container = st.container()
    last_route = st.session_state.get("last_route_result")

    if generate_clicked:
        origin = st.session_state["route_origin"]
        destination = st.session_state["route_destination"]
        valid, message = validate_coordinates(origin, destination)
        if not valid:
            st.warning(message)
            return
        with st.spinner("Calculando ruta con Dijkstra y ajustando pesos..."):
            route_payload = {
                "origin": origin,
                "destination": destination,
                "preferences": collect_route_preferences(),
                "day_of_week": st.session_state.get("trip_day", "Monday"),
                "departure_hour": float(st.session_state.get("trip_hour", 8)),
                "avoid_congestion": avoid_congestion,
                "avoid_accidents": avoid_accidents,
            }
            route_result = call_backend("/routes/optimal", route_payload)

        if _has_route_steps(route_result):
            st.session_state["last_route_result"] = route_result
            with results_container:
                render_route_summary(route_result)
        else:
            st.session_state["last_route_result"] = None
            st.error("No se pudo construir la ruta. Ajusta los puntos y vuelve a intentar.")
    elif _has_route_steps(last_route):
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
        render_dataset_selector()
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
    render_dataset_selector()
    render_guidelines()
    routing_section()
    st.divider()
    playground_section(metadata)


if __name__ == "__main__":
    main()
