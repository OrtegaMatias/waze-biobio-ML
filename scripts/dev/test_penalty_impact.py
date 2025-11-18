#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para demostrar el impacto de las penalizaciones de congestiones/accidentes.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from algorithms.recommenders import routing, data_loader


def test_penalty_scenarios():
    """Prueba diferentes escenarios de penalizaciones"""
    print("=" * 80)
    print("TEST: Impacto de penalizaciones en rutas")
    print("=" * 80)

    # Cargar grafo
    print("\n1. Cargando grafo...")
    events = data_loader.load_raw_events()
    graph = routing.RouteGraph.from_events(events)
    print(f"   ✓ Grafo cargado: {len(graph.nodes)} nodos")

    # Puntos de prueba (Concepción centro)
    origin = (-36.8270, -73.0500)
    destination = (-36.8200, -73.0435)

    # Contexto de incidentes (lunes, 8am = hora punta)
    day = "Monday"
    hour_bucket = "Punta AM (06-09h)"

    print(f"\n2. Ruta de prueba:")
    print(f"   Origen: {origin}")
    print(f"   Destino: {destination}")
    print(f"   Contexto: {day}, {hour_bucket}")

    # Escenario 1: Sin penalizaciones
    print("\n" + "─" * 80)
    print("ESCENARIO 1: Sin penalizaciones (solo distancia)")
    print("─" * 80)

    path_no_penalties = graph.shortest_path(
        origin,
        destination,
        apply_penalties=False,
    )

    if path_no_penalties:
        distance_km = sum(step.peso for step in path_no_penalties if step.peso)
        print(f"✓ Distancia: {distance_km:.2f} km")
        print(f"✓ Pasos: {len(path_no_penalties)}")

        # Contar incidentes en la ruta
        congestions = sum(1 for step in path_no_penalties if step.tipo_evento == "Congestión")
        accidents = sum(1 for step in path_no_penalties if step.tipo_evento == "Accidente")
        print(f"⚠️  Incidentes en ruta: {congestions} congestiones, {accidents} accidentes")

    # Escenario 2: Solo congestiones
    print("\n" + "─" * 80)
    print("ESCENARIO 2: Evitar congestiones (4x-400x)")
    print("─" * 80)

    path_avoid_cong = graph.shortest_path(
        origin,
        destination,
        incident_ctx={
            "day": day,
            "hour_bucket": hour_bucket,
            "avoid_congestion": True,
            "avoid_accidents": False,
        },
        apply_penalties=True,
    )

    if path_avoid_cong:
        distance_km = sum(step.peso for step in path_avoid_cong if step.peso)
        print(f"✓ Distancia: {distance_km:.2f} km")
        print(f"✓ Pasos: {len(path_avoid_cong)}")

        # Contar incidentes en la ruta
        congestions = sum(1 for step in path_avoid_cong if step.tipo_evento == "Congestión")
        accidents = sum(1 for step in path_avoid_cong if step.tipo_evento == "Accidente")
        print(f"✓ Incidentes en ruta: {congestions} congestiones, {accidents} accidentes")

    # Escenario 3: Solo accidentes
    print("\n" + "─" * 80)
    print("ESCENARIO 3: Evitar accidentes (2x-200x)")
    print("─" * 80)

    path_avoid_acc = graph.shortest_path(
        origin,
        destination,
        incident_ctx={
            "day": day,
            "hour_bucket": hour_bucket,
            "avoid_congestion": False,
            "avoid_accidents": True,
        },
        apply_penalties=True,
    )

    if path_avoid_acc:
        distance_km = sum(step.peso for step in path_avoid_acc if step.peso)
        print(f"✓ Distancia: {distance_km:.2f} km")
        print(f"✓ Pasos: {len(path_avoid_acc)}")

        # Contar incidentes en la ruta
        congestions = sum(1 for step in path_avoid_acc if step.tipo_evento == "Congestión")
        accidents = sum(1 for step in path_avoid_acc if step.tipo_evento == "Accidente")
        print(f"✓ Incidentes en ruta: {congestions} congestiones, {accidents} accidentes")

    # Escenario 4: Ambas penalizaciones
    print("\n" + "─" * 80)
    print("ESCENARIO 4: Evitar congestiones Y accidentes")
    print("─" * 80)

    path_avoid_both = graph.shortest_path(
        origin,
        destination,
        incident_ctx={
            "day": day,
            "hour_bucket": hour_bucket,
            "avoid_congestion": True,
            "avoid_accidents": True,
        },
        apply_penalties=True,
    )

    if path_avoid_both:
        distance_km = sum(step.peso for step in path_avoid_both if step.peso)
        print(f"✓ Distancia: {distance_km:.2f} km")
        print(f"✓ Pasos: {len(path_avoid_both)}")

        # Contar incidentes en la ruta
        congestions = sum(1 for step in path_avoid_both if step.tipo_evento == "Congestión")
        accidents = sum(1 for step in path_avoid_both if step.tipo_evento == "Accidente")
        print(f"✓ Incidentes en ruta: {congestions} congestiones, {accidents} accidentes")

    # Comparación final
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print("\nSi las penalizaciones funcionan correctamente, deberías ver:")
    print("  • Escenario 1: MÁS incidentes (no evita nada)")
    print("  • Escenario 2: MENOS congestiones")
    print("  • Escenario 3: MENOS accidentes")
    print("  • Escenario 4: MENOS incidentes en total")
    print("\n✅ Las penalizaciones están funcionando si los números de incidentes disminuyen.")


if __name__ == "__main__":
    test_penalty_scenarios()
