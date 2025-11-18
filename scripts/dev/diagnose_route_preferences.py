#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para diagnosticar preferencias de ruta y verificar si UBCF/IBCF afectan la ruta.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from algorithms.recommenders.collaborative import CollaborativeFilteringRecommender
from algorithms.recommenders import routing, data_loader
import pandas as pd


def main():
    print("=" * 80)
    print("DIAGNÓSTICO: ¿Por qué UBCF e IBCF generan rutas iguales?")
    print("=" * 80)

    # Cargar grafo
    print("\n1. Cargando grafo de rutas...")
    events = data_loader.load_raw_events()
    print(f"   Eventos cargados: {len(events)}")
    graph = routing.RouteGraph.from_events(events)
    print(f"   Total nodos: {len(graph.nodes)}")

    # Crear modelos CF
    print("\n2. Creando modelos UBCF e IBCF...")
    ubcf_model = CollaborativeFilteringRecommender(strategy="ubcf")
    ibcf_model = CollaborativeFilteringRecommender(strategy="ibcf")
    ubcf_model.fit()
    ibcf_model.fit()

    # Generar recomendaciones para usuario_demo
    user_id = "usuario_demo"
    num_recs = 50

    print(f"\n3. Generando {num_recs} recomendaciones para usuario '{user_id}'...")
    ubcf_recs = ubcf_model.recommend(user_id=user_id, top_n=num_recs)
    ibcf_recs = ibcf_model.recommend(user_id=user_id, top_n=num_recs)

    print(f"   UBCF: {len(ubcf_recs)} recomendaciones")
    print(f"   IBCF: {len(ibcf_recs)} recomendaciones")

    # Calcular ruta de ejemplo (Concepción a Los Ángeles)
    origin = (-36.8270, -73.0500)  # Concepción
    destination = (-37.4700, -72.3500)  # Los Ángeles

    print(f"\n4. Calculando ruta Dijkstra baseline...")
    print(f"   Origen: {origin}")
    print(f"   Destino: {destination}")

    baseline_path = graph.shortest_path(origin, destination, apply_penalties=False)

    if not baseline_path:
        print("   ❌ No se pudo calcular ruta")
        return

    # Obtener vías únicas en la ruta baseline
    vias_in_route = set()
    for step in baseline_path:
        if step.via:
            vias_in_route.add(step.via)

    print(f"   ✅ Ruta calculada: {len(baseline_path)} pasos")
    print(f"   Vías únicas en ruta: {len(vias_in_route)}")

    # Mostrar primeras 20 vías
    print(f"\n5. Vías en la ruta baseline (primeras 20):")
    for i, via in enumerate(list(vias_in_route)[:20], 1):
        print(f"      {i:2d}. {via}")

    # Verificar cuántas vías recomendadas están en la ruta
    ubcf_vias = set(rec.via for rec in ubcf_recs)
    ibcf_vias = set(rec.via for rec in ibcf_recs)

    ubcf_in_route = vias_in_route & ubcf_vias
    ibcf_in_route = vias_in_route & ibcf_vias

    print(f"\n6. Análisis de cobertura de recomendaciones:")
    print(f"   Vías en ruta baseline: {len(vias_in_route)}")
    print(f"   Vías recomendadas por UBCF: {len(ubcf_vias)}")
    print(f"   Vías recomendadas por IBCF: {len(ibcf_vias)}")
    print(f"   ✓ UBCF cubre: {len(ubcf_in_route)}/{len(vias_in_route)} vías de la ruta ({len(ubcf_in_route)/len(vias_in_route)*100:.1f}%)")
    print(f"   ✓ IBCF cubre: {len(ibcf_in_route)}/{len(vias_in_route)} vías de la ruta ({len(ibcf_in_route)/len(vias_in_route)*100:.1f}%)")

    if ubcf_in_route:
        print(f"\n7. Vías de la ruta que UBCF recomienda:")
        for via in sorted(ubcf_in_route)[:10]:
            rating = next((r.estimated_rating for r in ubcf_recs if r.via == via), None)
            print(f"      • {via:40s} - Rating: {rating:.2f}")

    if ibcf_in_route:
        print(f"\n8. Vías de la ruta que IBCF recomienda:")
        for via in sorted(ibcf_in_route)[:10]:
            rating = next((r.estimated_rating for r in ibcf_recs if r.via == via), None)
            print(f"      • {via:40s} - Rating: {rating:.2f}")

    # Comparar ratings para vías en común
    common_vias = ubcf_in_route & ibcf_in_route
    if common_vias:
        print(f"\n9. Vías en común en la ruta (UBCF ∩ IBCF ∩ Ruta):")
        print(f"   Total: {len(common_vias)}")
        for via in sorted(common_vias)[:10]:
            ubcf_rating = next((r.estimated_rating for r in ubcf_recs if r.via == via), None)
            ibcf_rating = next((r.estimated_rating for r in ibcf_recs if r.via == via), None)
            diff = abs(ubcf_rating - ibcf_rating) if ubcf_rating and ibcf_rating else 0
            print(f"      • {via:40s} - UBCF: {ubcf_rating:.2f}, IBCF: {ibcf_rating:.2f}, Δ={diff:.2f}")

    # Conclusión
    print(f"\n" + "=" * 80)
    print("CONCLUSIÓN:")
    print("=" * 80)

    coverage_ubcf = len(ubcf_in_route)/len(vias_in_route)*100
    coverage_ibcf = len(ibcf_in_route)/len(vias_in_route)*100

    if coverage_ubcf < 10 and coverage_ibcf < 10:
        print("❌ PROBLEMA: Las recomendaciones NO cubren las vías de la ruta (<10%)")
        print("   → Las preferencias CF no afectan la ruta porque no incluyen las vías relevantes")
        print("   → Solución: Aumentar el número de recomendaciones (100-200) o cambiar estrategia")
    elif len(common_vias) > len(vias_in_route) * 0.5:
        print("⚠️  PROBLEMA: UBCF e IBCF recomiendan las MISMAS vías de la ruta (>50%)")
        print("   → Aunque los ratings son diferentes, el algoritmo selecciona rutas similares")
        print("   → Solución: Verificar que las matrices de ratings sean realmente diferentes")
    else:
        print("✅ Las recomendaciones cubren vías diferentes de la ruta")
        print("   → UBCF e IBCF deberían generar rutas distintas")


if __name__ == "__main__":
    main()
