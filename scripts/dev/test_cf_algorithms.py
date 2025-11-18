#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnóstico para verificar que UBCF e IBCF generan resultados diferentes.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from algorithms.recommenders.collaborative import CollaborativeFilteringRecommender
import pandas as pd


def main():
    print("=" * 80)
    print("DIAGNÓSTICO: Comparación UBCF vs IBCF")
    print("=" * 80)

    # Crear modelos
    print("\n1. Creando modelos UBCF e IBCF...")
    ubcf_model = CollaborativeFilteringRecommender(strategy="ubcf")
    ibcf_model = CollaborativeFilteringRecommender(strategy="ibcf")

    # Cargar datos
    print("2. Cargando ratings...")
    ubcf_model.fit()
    ibcf_model.fit()

    print(f"   Total ratings: {len(ubcf_model._ratings)}")
    print(f"   Usuarios únicos: {ubcf_model._ratings['user_id'].nunique()}")
    print(f"   Vías únicas: {ubcf_model._ratings['via'].nunique()}")

    # Estadísticas de ratings por usuario
    ratings_by_user = ubcf_model._ratings.groupby('user_id')['rating'].agg(['mean', 'std', 'count'])
    print("\n3. Estadísticas de ratings por usuario:")
    print(ratings_by_user.to_string())

    # Verificar matrices de similitud
    print("\n4. Matrices de similitud:")
    print(f"   User similarity matrix shape: {ubcf_model._user_sim.shape}")
    print(f"   Item similarity matrix shape: {ibcf_model._item_sim.shape}")

    # Mostrar primeras similitudes de usuario
    if ubcf_model._user_sim.shape[0] > 0:
        print("\n   User similarity (primeros 5x5):")
        print(ubcf_model._user_sim.iloc[:5, :5].to_string())

    # Probar con cada usuario
    print("\n5. Generando recomendaciones para cada perfil:")
    print("=" * 80)

    for user_id in ['safety_focused', 'usuario_demo', 'moderate_risk', 'risk_taker']:
        print(f"\n📊 Usuario: {user_id}")
        print("-" * 80)

        # UBCF
        ubcf_recs = ubcf_model.recommend(user_id=user_id, top_n=10)
        print(f"\n  🔵 UBCF - Top 10 recomendaciones:")
        if ubcf_recs:
            for i, rec in enumerate(ubcf_recs[:10], 1):
                print(f"     {i:2d}. {rec.via:30s} - Rating: {rec.estimated_rating:.3f}")
        else:
            print("     ⚠️  Sin recomendaciones")

        # IBCF
        ibcf_recs = ibcf_model.recommend(user_id=user_id, top_n=10)
        print(f"\n  🟠 IBCF - Top 10 recomendaciones:")
        if ibcf_recs:
            for i, rec in enumerate(ibcf_recs[:10], 1):
                print(f"     {i:2d}. {rec.via:30s} - Rating: {rec.estimated_rating:.3f}")
        else:
            print("     ⚠️  Sin recomendaciones")

        # Comparación
        if ubcf_recs and ibcf_recs:
            ubcf_vias = set(r.via for r in ubcf_recs)
            ibcf_vias = set(r.via for r in ibcf_recs)
            overlap = len(ubcf_vias & ibcf_vias)
            print(f"\n  📈 Comparación:")
            print(f"     Vías en común: {overlap}/10 ({overlap*10:.0f}%)")
            print(f"     Vías únicas UBCF: {len(ubcf_vias - ibcf_vias)}")
            print(f"     Vías únicas IBCF: {len(ibcf_vias - ubcf_vias)}")

            # Comparar ratings promedio
            ubcf_avg = sum(r.estimated_rating for r in ubcf_recs) / len(ubcf_recs)
            ibcf_avg = sum(r.estimated_rating for r in ibcf_recs) / len(ibcf_recs)
            print(f"     Rating promedio UBCF: {ubcf_avg:.3f}")
            print(f"     Rating promedio IBCF: {ibcf_avg:.3f}")
            print(f"     Diferencia: {abs(ubcf_avg - ibcf_avg):.3f}")

    print("\n" + "=" * 80)
    print("✅ Diagnóstico completado")
    print("=" * 80)


if __name__ == "__main__":
    main()
