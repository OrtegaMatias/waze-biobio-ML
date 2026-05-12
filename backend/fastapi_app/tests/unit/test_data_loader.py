import pandas as pd
import pytest

from algorithms.recommenders import data_loader


def _base_raw_df():
    return pd.DataFrame(
        {
            "segment_id": ["segA", "segA", "segA", "segB", "segB"],
            "indice_coord": [2, 0, 1, 5, 1],
            "duracion_hrs": [0.25] * 5,
            "distancia_km": [0.8] * 5,
            "velocidad_kmh": [20] * 5,
            "hora_inicio": ["08:00"] * 5,
            "hora_fin": ["08:05"] * 5,
            "comuna": ["Concepción"] * 5,
            "via": ["Barros Arana"] * 5,
            "lat": [-36.82] * 5,
            "lon": [-73.05] * 5,
            "fecha": ["2025-01-01"] * 5,
        }
    )


def test_prepare_dataframe_generates_monotonic_segment_seq():
    processed = data_loader._prepare_dataframe(_base_raw_df(), label="accidente")

    seq_seg_a = processed[processed["segment_id"] == "segA"]["segment_seq"].tolist()
    seq_seg_b = processed[processed["segment_id"] == "segB"]["segment_seq"].tolist()

    assert seq_seg_a == [0, 1, 2]
    assert seq_seg_b == [0, 1]
    assert (processed["penalty_factor"] == data_loader.ACCIDENT_PENALTY).all()


def test_prepare_dataframe_marks_reference_oneway():
    raw = _base_raw_df()
    raw["oneway"] = [True, True, True, False, False]
    processed = data_loader._prepare_dataframe(raw, label="referencia")

    assert processed["tipo_evento"].unique().tolist() == ["Referencia"]
    assert processed["oneway"].dtype == bool
    assert processed[processed["segment_id"] == "segA"]["oneway"].all()
    assert processed["penalty_factor"].eq(1.0).all()


def test_prepare_dataframe_scores_congestion_severity():
    raw = pd.DataFrame(
        {
            "segment_id": ["high", "high", "low"],
            "indice_coord": [0, 1, 0],
            "duracion_hrs": [1.0, 1.0, 0.15],
            "distancia_km": [2.5, 2.5, 0.2],
            "velocidad_kmh": [8, 8, 45],
            "hora_inicio": ["08:00", "08:05", "22:00"],
            "hora_fin": ["09:00", "09:05", "22:10"],
            "comuna": ["Concepcion", "Concepcion", "Concepcion"],
            "via": ["Ruta Alta", "Ruta Alta", "Ruta Baja"],
            "lat": [-36.82, -36.821, -36.9],
            "lon": [-73.05, -73.051, -73.2],
            "fecha": ["2025-01-01", "2025-01-01", "2025-01-01"],
        }
    )

    processed = data_loader._prepare_dataframe(raw, label="congestion")
    high_penalty = processed[processed["segment_id"] == "high"]["penalty_factor"].max()
    low_penalty = processed[processed["segment_id"] == "low"]["penalty_factor"].max()

    assert high_penalty > low_penalty
    assert high_penalty <= data_loader.CONGESTION_PENALTY_MAX
    assert low_penalty >= data_loader.CONGESTION_PENALTY_MIN


def test_load_reference_network_normalizes(tmp_path):
    csv_path = tmp_path / "roads.csv"
    ref_df = _base_raw_df().assign(oneway=True)
    ref_df.to_csv(csv_path, index=False)
    data_loader._load_reference_network.cache_clear()
    normalized = data_loader.load_reference_network(csv_path)

    assert not normalized.empty
    assert normalized["segment_seq"].tolist()[:3] == [0, 1, 2]
    assert normalized["oneway"].all()
    assert normalized["tipo_evento"].unique().tolist() == ["Referencia"]


def test_apply_penalties_updates_reference():
    events = data_loader._prepare_dataframe(_base_raw_df(), label="accidente")
    reference = data_loader._prepare_dataframe(_base_raw_df(), label="referencia")
    lookup = data_loader._build_penalty_lookup(events)
    adjusted = data_loader._apply_penalties(reference, lookup)

    assert adjusted["penalty_factor"].max() == data_loader.ACCIDENT_PENALTY
