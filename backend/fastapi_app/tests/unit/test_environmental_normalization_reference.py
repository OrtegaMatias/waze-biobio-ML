from __future__ import annotations

from scripts.dev.build_environmental_normalization_reference import build_reference


def test_environmental_reference_uses_only_fixed_2021_2024_period(tmp_path):
    pm25_path = tmp_path / "pm25.csv"
    pm25_path.write_text(
        "\n".join(
            [
                "timestamp,PM25",
                "2020-12-31 23:00:00,999",
                "2021-01-01 00:00:00,10",
                "2022-01-01 00:00:00,20",
                "2023-01-01 00:00:00,30",
                "2024-12-31 23:00:00,40",
                "2025-01-01 00:00:00,999",
            ]
        ),
        encoding="utf-8",
    )
    wind_path = tmp_path / "wind.csv"
    wind_path.write_text(
        "\n".join(
            [
                "timestamp,wind_speed_mean",
                "2020-12-31 23:00:00,99",
                "2021-01-01 00:00:00,1",
                "2022-01-01 00:00:00,2",
                "2023-01-01 00:00:00,3",
                "2024-12-31 23:00:00,4",
                "2025-01-01 00:00:00,99",
            ]
        ),
        encoding="utf-8",
    )
    congestion_path = tmp_path / "congestion.csv"
    congestion_path.write_text(
        "\n".join(
            [
                "segment_id,velocidad_kmh,duracion_min,datetime_inicio",
                "a,5,15,2025-03-13 06:00:00",
                "b,10,30,2025-04-01 08:00:00",
                "c,20,60,2025-06-01 12:00:00",
                "d,25,120,2025-08-22 20:00:00",
                "outside,1,999,2025-08-23 00:00:00",
            ]
        ),
        encoding="utf-8",
    )

    reference = build_reference(
        pm25_path=pm25_path,
        wind_path=wind_path,
        congestion_path=congestion_path,
    )

    assert reference["version"] == "environmental-normalization-v1"
    assert reference["variables"]["pm25"]["sample_size"] == 4
    assert reference["variables"]["wind_speed"]["sample_size"] == 4
    assert reference["variables"]["congestion_speed_kmh"]["sample_size"] == 4
    assert reference["variables"]["congestion_duration_min"]["sample_size"] == 4
    assert reference["variables"]["congestion_duration_min"]["p90"] < 999
    assert reference["variables"]["pm25"]["p90"] < 999
    assert reference["variables"]["wind_speed"]["p90"] < 99


def test_environmental_reference_rejects_insufficient_historical_range(tmp_path):
    pm25_path = tmp_path / "pm25.csv"
    pm25_path.write_text(
        "timestamp,PM25\n2024-01-01 00:00:00,10\n2024-01-02 00:00:00,10\n",
        encoding="utf-8",
    )
    wind_path = tmp_path / "wind.csv"
    wind_path.write_text(
        "timestamp,wind_speed_mean\n2024-01-01 00:00:00,1\n2024-01-02 00:00:00,2\n",
        encoding="utf-8",
    )
    congestion_path = tmp_path / "congestion.csv"
    congestion_path.write_text(
        "segment_id,velocidad_kmh,duracion_min,datetime_inicio\na,10,15,2025-04-01 08:00:00\n",
        encoding="utf-8",
    )

    try:
        build_reference(
            pm25_path=pm25_path,
            wind_path=wind_path,
            congestion_path=congestion_path,
        )
    except ValueError as exc:
        assert "insuficiente" in str(exc)
    else:
        raise AssertionError("Expected an insufficient P10-P90 range to be rejected")
