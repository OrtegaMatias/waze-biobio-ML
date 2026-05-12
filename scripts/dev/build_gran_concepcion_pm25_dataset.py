from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "data" / "air_quality" / "sinca_biobio_hourly_2021plus"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
STATIONS_METADATA_PATH = DATASET_DIR / "stations_metadata.csv"
OUTPUT_PROCESSED_DIR = ROOT_DIR / "data_processed"
OUTPUT_ANALYSIS_DIR = ROOT_DIR / "data_analysis"

VALID_START = pd.Timestamp("2021-01-01 00:00:00")
VALID_END = pd.Timestamp.now().floor("h")
LOW_COVERAGE_FLAG_THRESHOLD = 0.80
PM25_SELECTION_COVERAGE_THRESHOLD = 0.75
RECENT_LAG = pd.Timedelta(days=14)
PM_SPIKE_ABS_THRESHOLD = 500.0
PM_SPIKE_PEER_RATIO_THRESHOLD = 10.0

# The user explicitly asked to treat these communes as in-scope candidates.
CANDIDATE_COMMUNES = {
    "concepcion",
    "talcahuano",
    "hualpen",
    "san pedro de la paz",
    "chiguayante",
    "coronel",
    "lota",
    "penco",
    "tome",
    "hualqui",
}
PERIPHERAL_COMMUNES = {"tome", "hualqui"}
CORE_URBAN_EXCLUDED_COMMUNES = {"coronel", "hualqui", "tome"}


@dataclass
class SeriesMetrics:
    valid_timestamp_rows: int
    unique_timestamps: int
    non_null_values: int
    invalid_time_rows: int
    duplicate_timestamps: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    first_non_null: pd.Timestamp | None
    last_non_null: pd.Timestamp | None
    expected_hours: int


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", "" if value is None else str(value))
    stripped = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(stripped.strip().lower().split())


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["rows"] = pd.to_numeric(manifest["rows"], errors="coerce").fillna(0).astype(int)
    manifest["csv_abspath"] = manifest["csv_path"].map(lambda rel: ROOT_DIR / str(rel))
    return manifest


def load_stations_metadata() -> pd.DataFrame:
    stations = pd.read_csv(STATIONS_METADATA_PATH)
    stations["commune_norm"] = stations["commune"].map(normalize_text)
    stations["station_name_norm"] = stations["station_name"].map(normalize_text)
    return stations


def station_candidate_scope(row: pd.Series) -> tuple[bool, str]:
    commune_norm = row.get("commune_norm", "")
    station_name_norm = row.get("station_name_norm", "")
    if commune_norm in CANDIDATE_COMMUNES:
        return True, "commune_match"
    if "lota" in station_name_norm:
        return True, "station_name_hint"
    return False, "out_of_scope"


def inspect_series(csv_path: Path) -> SeriesMetrics:
    series = pd.read_csv(csv_path, usecols=["datetime_local", "preferred_value"])
    if series.empty:
        return SeriesMetrics(
            valid_timestamp_rows=0,
            unique_timestamps=0,
            non_null_values=0,
            invalid_time_rows=0,
            duplicate_timestamps=0,
            start=None,
            end=None,
            first_non_null=None,
            last_non_null=None,
            expected_hours=0,
        )

    timestamps = pd.to_datetime(series["datetime_local"], errors="coerce")
    values = pd.to_numeric(series["preferred_value"], errors="coerce")
    valid_time_mask = timestamps.notna() & timestamps.between(VALID_START, VALID_END, inclusive="both")
    valid_time_rows = int(valid_time_mask.sum())
    invalid_time_rows = int((~valid_time_mask).sum())

    if valid_time_rows == 0:
        return SeriesMetrics(
            valid_timestamp_rows=0,
            unique_timestamps=0,
            non_null_values=0,
            invalid_time_rows=invalid_time_rows,
            duplicate_timestamps=0,
            start=None,
            end=None,
            first_non_null=None,
            last_non_null=None,
            expected_hours=0,
        )

    valid_timestamps = timestamps.loc[valid_time_mask]
    valid_values = values.loc[valid_time_mask]
    non_null_mask = valid_values.notna()
    start = valid_timestamps.min()
    end = valid_timestamps.max()
    unique_timestamps = int(valid_timestamps.nunique())
    duplicate_timestamps = valid_time_rows - unique_timestamps
    non_null_values = int(non_null_mask.sum())
    first_non_null = valid_timestamps.loc[non_null_mask].min() if non_null_values else None
    last_non_null = valid_timestamps.loc[non_null_mask].max() if non_null_values else None
    expected_hours = int(((end - start) / pd.Timedelta(hours=1))) + 1

    return SeriesMetrics(
        valid_timestamp_rows=valid_time_rows,
        unique_timestamps=unique_timestamps,
        non_null_values=non_null_values,
        invalid_time_rows=invalid_time_rows,
        duplicate_timestamps=duplicate_timestamps,
        start=start,
        end=end,
        first_non_null=first_non_null,
        last_non_null=last_non_null,
        expected_hours=expected_hours,
    )


def period_end(period_start: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return period_start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    if freq == "MS":
        return period_start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1)
    if freq == "YS":
        return period_start + pd.offsets.YearBegin(1) - pd.Timedelta(hours=1)
    raise ValueError(f"Unsupported aggregation frequency: {freq}")


def build_station_summary(manifest: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    stations = stations.copy()
    stations[["is_candidate_gc", "candidate_scope"]] = stations.apply(
        station_candidate_scope,
        axis=1,
        result_type="expand",
    )

    candidate_ids = set(stations.loc[stations["is_candidate_gc"], "station_id"].astype(str))
    manifest = manifest[manifest["station_id"].astype(str).isin(candidate_ids)].copy()
    metadata_lookup = stations.set_index("station_id")

    inspected_rows: list[dict[str, object]] = []
    for row in manifest.to_dict("records"):
        station_id = str(row["station_id"])
        metrics = inspect_series(Path(row["csv_abspath"]))
        inspected_rows.append(
            {
                "station_id": station_id,
                "station_name": row["station_name"],
                "parameter_label": row["parameter_label"],
                "rows_manifest": int(row["rows"]),
                "valid_timestamp_rows": metrics.valid_timestamp_rows,
                "unique_timestamps": metrics.unique_timestamps,
                "non_null_values": metrics.non_null_values,
                "invalid_time_rows": metrics.invalid_time_rows,
                "duplicate_timestamps": metrics.duplicate_timestamps,
                "start": metrics.start,
                "end": metrics.end,
                "first_non_null": metrics.first_non_null,
                "last_non_null": metrics.last_non_null,
                "expected_hours": metrics.expected_hours,
                "status": row["status"],
            }
        )

    inspected = pd.DataFrame(inspected_rows)
    records: list[dict[str, object]] = []
    for station_id, station_rows in inspected.groupby("station_id", sort=True):
        meta = metadata_lookup.loc[int(station_id)] if int(station_id) in metadata_lookup.index else metadata_lookup.loc[station_id]
        valid_rows = station_rows[station_rows["valid_timestamp_rows"] > 0]
        data_rows = station_rows[station_rows["non_null_values"] > 0]
        pm25_rows = station_rows[station_rows["parameter_label"] == "PM25"]
        pm25_data = pm25_rows[pm25_rows["non_null_values"] > 0]

        fecha_inicio = data_rows["first_non_null"].min() if not data_rows.empty else pd.NaT
        fecha_fin = data_rows["last_non_null"].max() if not data_rows.empty else pd.NaT
        expected_total = int(valid_rows["expected_hours"].sum())
        non_null_total = int(station_rows["non_null_values"].sum())
        coverage = (non_null_total / expected_total) if expected_total else 0.0

        pm25_inicio = pm25_data["first_non_null"].min() if not pm25_data.empty else pd.NaT
        pm25_fin = pm25_data["last_non_null"].max() if not pm25_data.empty else pd.NaT
        pm25_expected = int(pm25_rows["expected_hours"].sum())
        pm25_non_null = int(pm25_rows["non_null_values"].sum())
        pm25_coverage = (pm25_non_null / pm25_expected) if pm25_expected else 0.0

        issues: list[str] = []
        invalid_rows_total = int(station_rows["invalid_time_rows"].sum())
        duplicate_total = int(station_rows["duplicate_timestamps"].sum())
        if invalid_rows_total:
            issues.append(f"invalid_timestamps_removed={invalid_rows_total}")
        if duplicate_total:
            issues.append(f"duplicate_timestamps_detected={duplicate_total}")
        if normalize_text(meta["commune"]) == "coronel" and "lota" in normalize_text(meta["station_name"]):
            issues.append("station_name_mentions_lota_but_metadata_commune_is_coronel")
        if normalize_text(meta["commune"]) in PERIPHERAL_COMMUNES:
            issues.append("candidate_scope_is_peripheral_to_core_gran_concepcion")

        records.append(
            {
                "station_id": station_id,
                "station_name": meta["station_name"],
                "commune": meta["commune"],
                "latitude": meta["latitude"],
                "longitude": meta["longitude"],
                "candidate_scope": meta["candidate_scope"],
                "n_series": int(len(station_rows)),
                "n_series_with_valid_data": int((station_rows["non_null_values"] > 0).sum()),
                "n_registros": non_null_total,
                "n_registros_manifest": int(station_rows["rows_manifest"].sum()),
                "fecha_inicio": fecha_inicio,
                "fecha_fin": fecha_fin,
                "porcentaje_cobertura_aprox": round(coverage * 100, 2),
                "variables_disponibles": "|".join(sorted(data_rows["parameter_label"].unique())),
                "has_pm25": bool(pm25_non_null > 0),
                "pm25_registros": pm25_non_null,
                "pm25_fecha_inicio": pm25_inicio,
                "pm25_fecha_fin": pm25_fin,
                "pm25_cobertura_aprox": round(pm25_coverage * 100, 2),
                "sin_datos": bool(data_rows.empty),
                "baja_cobertura": bool((not valid_rows.empty) and coverage < LOW_COVERAGE_FLAG_THRESHOLD),
                "datos_recientes_2026": bool(pd.notna(fecha_fin) and fecha_fin.year == 2026),
                "data_issues": "; ".join(issues),
                "selected_for_pm25_analysis": False,
                "selection_reason": "",
            }
        )

    summary = pd.DataFrame(records).sort_values(["commune", "station_name"]).reset_index(drop=True)

    latest_pm25_end = summary["pm25_fecha_fin"].dropna().max()
    if pd.notna(latest_pm25_end):
        is_recent_pm25 = summary["pm25_fecha_fin"].notna() & (summary["pm25_fecha_fin"] >= latest_pm25_end - RECENT_LAG)
    else:
        is_recent_pm25 = pd.Series(False, index=summary.index)

    summary["pm25_recent_relative"] = is_recent_pm25
    summary["selected_for_pm25_analysis"] = (
        summary["has_pm25"]
        & (~summary["sin_datos"])
        & (summary["pm25_cobertura_aprox"] >= PM25_SELECTION_COVERAGE_THRESHOLD * 100)
        & summary["pm25_recent_relative"]
    )
    summary["selection_reason"] = summary.apply(describe_selection_reason, axis=1)
    return summary


def describe_selection_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["selected_for_pm25_analysis"]:
        reasons.append("candidate_station")
        reasons.append("pm25_available")
        reasons.append("good_pm25_coverage")
        reasons.append("recent_pm25_continuity")
        return "; ".join(reasons)

    if row["sin_datos"]:
        reasons.append("excluded_no_valid_data")
    if not row["has_pm25"]:
        reasons.append("excluded_without_pm25")
    if row["has_pm25"] and row["pm25_cobertura_aprox"] < PM25_SELECTION_COVERAGE_THRESHOLD * 100:
        reasons.append("excluded_low_pm25_coverage")
    if pd.notna(row["pm25_fecha_fin"]) and not row["pm25_recent_relative"]:
        reasons.append("excluded_not_recent_relative_to_candidate_max")
    elif pd.notna(row["pm25_fecha_fin"]) and row["pm25_fecha_fin"].year < VALID_END.year:
        reasons.append("excluded_not_recent_to_current_year")
    return "; ".join(reasons)


def load_selected_parameter_hourly(
    manifest: pd.DataFrame,
    selected_ids: set[str],
    parameter_label: str,
    value_label: str,
) -> pd.DataFrame:
    parameter_manifest = manifest[
        manifest["station_id"].astype(str).isin(selected_ids) & (manifest["parameter_label"] == parameter_label)
    ].copy()
    frames: list[pd.DataFrame] = []
    for row in parameter_manifest.to_dict("records"):
        frame = pd.read_csv(
            Path(row["csv_abspath"]),
            usecols=["station_id", "station_name", "datetime_local", "preferred_value"],
        )
        frame["timestamp"] = pd.to_datetime(frame["datetime_local"], errors="coerce")
        frame[value_label] = pd.to_numeric(frame["preferred_value"], errors="coerce")
        frame = frame[frame["timestamp"].notna()]
        frame = frame[frame["timestamp"].between(VALID_START, VALID_END, inclusive="both")]
        frame = frame.drop(columns=["datetime_local", "preferred_value"])
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "station_id", "station_name", value_label])

    parameter_hourly = pd.concat(frames, ignore_index=True)
    parameter_hourly["station_id"] = parameter_hourly["station_id"].astype(str)
    parameter_hourly = parameter_hourly.sort_values(["station_id", "timestamp"]).drop_duplicates(
        subset=["station_id", "timestamp"],
        keep="last",
    )
    parameter_hourly = parameter_hourly.dropna(subset=[value_label]).reset_index(drop=True)
    parameter_hourly = parameter_hourly[["timestamp", "station_id", "station_name", value_label]]
    return parameter_hourly


def flag_suspicious_pm25_rows(hourly: pd.DataFrame, pm10_hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty or pm10_hourly.empty:
        return pd.DataFrame(
            columns=["row_id", "timestamp", "station_id", "station_name", "PM25", "PM10", "peer_median_pm25", "flag_reason"]
        )

    hourly_with_id = hourly.reset_index(names="row_id")
    candidates = hourly_with_id.merge(
        pm10_hourly[["timestamp", "station_id", "PM10"]],
        on=["timestamp", "station_id"],
        how="left",
    )
    candidates = candidates[
        (candidates["PM25"] >= PM_SPIKE_ABS_THRESHOLD) & (candidates["PM10"] >= PM_SPIKE_ABS_THRESHOLD)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(
            columns=["row_id", "timestamp", "station_id", "station_name", "PM25", "PM10", "peer_median_pm25", "flag_reason"]
        )

    flagged_rows: list[dict[str, object]] = []
    hourly_by_timestamp = {
        timestamp: group[["station_id", "PM25"]].copy()
        for timestamp, group in hourly_with_id.groupby("timestamp", sort=False)
    }
    for row in candidates.itertuples(index=False):
        peer_series = hourly_by_timestamp[row.timestamp]
        peer_values = peer_series.loc[peer_series["station_id"] != row.station_id, "PM25"].dropna()
        if peer_values.empty:
            continue
        peer_median = float(peer_values.median())
        if peer_median <= 0:
            continue
        if (row.PM25 / peer_median) >= PM_SPIKE_PEER_RATIO_THRESHOLD:
            flagged_rows.append(
                {
                    "row_id": row.row_id,
                    "timestamp": row.timestamp,
                    "station_id": row.station_id,
                    "station_name": row.station_name,
                    "PM25": row.PM25,
                    "PM10": row.PM10,
                    "peer_median_pm25": round(peer_median, 4),
                    "flag_reason": (
                        "extreme_pm25_with_high_pm10_and_large_gap_vs_network_median"
                    ),
                }
            )

    return pd.DataFrame(flagged_rows)


def load_selected_pm25_hourly(manifest: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_ids = set(summary.loc[summary["selected_for_pm25_analysis"], "station_id"].astype(str))
    hourly = load_selected_parameter_hourly(
        manifest=manifest,
        selected_ids=selected_ids,
        parameter_label="PM25",
        value_label="PM25",
    )
    pm10_hourly = load_selected_parameter_hourly(
        manifest=manifest,
        selected_ids=selected_ids,
        parameter_label="PM10",
        value_label="PM10",
    )
    qc_flags = flag_suspicious_pm25_rows(hourly=hourly, pm10_hourly=pm10_hourly)
    if not qc_flags.empty:
        hourly = hourly.reset_index(names="row_id")
        hourly = hourly[~hourly["row_id"].isin(qc_flags["row_id"])].drop(columns=["row_id"]).reset_index(drop=True)
        qc_flags = qc_flags.drop(columns=["row_id"]).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return hourly, qc_flags


def aggregate_pm25(hourly: pd.DataFrame, freq: str, label: str) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(columns=["station_id", "station_name", label, "PM25", "n_obs"])

    station_latest = (
        hourly.groupby(["station_id", "station_name"], dropna=False)["timestamp"]
        .max()
        .reset_index()
        .rename(columns={"timestamp": "station_latest_timestamp"})
    )
    grouped = (
        hourly.groupby(
            [
                "station_id",
                "station_name",
                pd.Grouper(key="timestamp", freq=freq),
            ],
            dropna=False,
        )["PM25"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "PM25", "count": "n_obs", "timestamp": label})
    )
    grouped = grouped.merge(station_latest, on=["station_id", "station_name"], how="left")
    grouped["period_end"] = period_end(grouped[label], freq)
    grouped = grouped[grouped["station_latest_timestamp"] >= grouped["period_end"]].copy()
    grouped = grouped.drop(columns=["station_latest_timestamp", "period_end"])
    grouped["PM25"] = grouped["PM25"].round(4)
    return grouped.sort_values(["station_id", label]).reset_index(drop=True)


def ensure_output_dirs() -> None:
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def serialize_summary(summary: pd.DataFrame) -> pd.DataFrame:
    serializable = summary.copy()
    for column in ["fecha_inicio", "fecha_fin", "pm25_fecha_inicio", "pm25_fecha_fin"]:
        serializable[column] = serializable[column].astype("string")
    return serializable


def annotate_summary_with_qc_flags(summary: pd.DataFrame, qc_flags: pd.DataFrame) -> pd.DataFrame:
    annotated = summary.copy()
    annotated["pm25_qc_rows_removed"] = 0
    if qc_flags.empty:
        return annotated

    removed_counts = qc_flags.groupby("station_id").size().rename("pm25_qc_rows_removed")
    annotated["pm25_qc_rows_removed"] = (
        annotated["station_id"].astype(str).map(removed_counts).fillna(0).astype(int)
    )
    flagged_mask = annotated["pm25_qc_rows_removed"] > 0
    annotated.loc[flagged_mask, "data_issues"] = annotated.loc[flagged_mask].apply(
        lambda row: "; ".join(
            item
            for item in [
                row["data_issues"],
                f"pm25_qc_extreme_spikes_removed={row['pm25_qc_rows_removed']}",
            ]
            if item
        ),
        axis=1,
    )
    return annotated


def annotate_summary_with_core_scope(summary: pd.DataFrame) -> pd.DataFrame:
    annotated = summary.copy()
    annotated["selected_for_pm25_core"] = (
        annotated["selected_for_pm25_analysis"]
        & (~annotated["commune"].map(normalize_text).isin(CORE_URBAN_EXCLUDED_COMMUNES))
    )
    return annotated


def filter_hourly_to_core(summary: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    core_ids = set(summary.loc[summary["selected_for_pm25_core"], "station_id"].astype(str))
    return hourly[hourly["station_id"].astype(str).isin(core_ids)].reset_index(drop=True)


def main() -> None:
    ensure_output_dirs()
    manifest = load_manifest()
    stations = load_stations_metadata()

    summary = build_station_summary(manifest=manifest, stations=stations)
    hourly, qc_flags = load_selected_pm25_hourly(manifest=manifest, summary=summary)
    summary = annotate_summary_with_qc_flags(summary=summary, qc_flags=qc_flags)
    summary = annotate_summary_with_core_scope(summary=summary)
    daily = aggregate_pm25(hourly, freq="D", label="date")
    monthly = aggregate_pm25(hourly, freq="MS", label="month")
    yearly = aggregate_pm25(hourly, freq="YS", label="year")
    core_hourly = filter_hourly_to_core(summary=summary, hourly=hourly)
    core_daily = aggregate_pm25(core_hourly, freq="D", label="date")
    core_monthly = aggregate_pm25(core_hourly, freq="MS", label="month")
    core_yearly = aggregate_pm25(core_hourly, freq="YS", label="year")
    core_summary = summary[summary["selected_for_pm25_core"]].copy().reset_index(drop=True)

    summary_path = OUTPUT_ANALYSIS_DIR / "station_summary_gran_concepcion.csv"
    core_summary_path = OUTPUT_ANALYSIS_DIR / "station_summary_gran_concepcion_core.csv"
    qc_flags_path = OUTPUT_ANALYSIS_DIR / "gran_concepcion_pm25_qc_flags.csv"
    hourly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_hourly_clean.csv"
    daily_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_daily.csv"
    monthly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_monthly.csv"
    yearly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_yearly.csv"
    core_hourly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_core_hourly_clean.csv"
    core_daily_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_core_daily.csv"
    core_monthly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_core_monthly.csv"
    core_yearly_path = OUTPUT_PROCESSED_DIR / "gran_concepcion_pm25_core_yearly.csv"

    serialize_summary(summary).to_csv(summary_path, index=False)
    serialize_summary(core_summary).to_csv(core_summary_path, index=False)
    qc_flags.to_csv(qc_flags_path, index=False)
    hourly.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    yearly.to_csv(yearly_path, index=False)
    core_hourly.to_csv(core_hourly_path, index=False)
    core_daily.to_csv(core_daily_path, index=False)
    core_monthly.to_csv(core_monthly_path, index=False)
    core_yearly.to_csv(core_yearly_path, index=False)

    payload = {
        "summary_rows": int(len(summary)),
        "selected_stations": summary.loc[summary["selected_for_pm25_analysis"], "station_name"].tolist(),
        "core_selected_stations": summary.loc[summary["selected_for_pm25_core"], "station_name"].tolist(),
        "qc_flagged_rows": int(len(qc_flags)),
        "hourly_rows": int(len(hourly)),
        "daily_rows": int(len(daily)),
        "monthly_rows": int(len(monthly)),
        "yearly_rows": int(len(yearly)),
        "core_hourly_rows": int(len(core_hourly)),
        "core_daily_rows": int(len(core_daily)),
        "core_monthly_rows": int(len(core_monthly)),
        "core_yearly_rows": int(len(core_yearly)),
        "summary_path": str(summary_path.relative_to(ROOT_DIR)),
        "core_summary_path": str(core_summary_path.relative_to(ROOT_DIR)),
        "qc_flags_path": str(qc_flags_path.relative_to(ROOT_DIR)),
        "hourly_path": str(hourly_path.relative_to(ROOT_DIR)),
        "daily_path": str(daily_path.relative_to(ROOT_DIR)),
        "monthly_path": str(monthly_path.relative_to(ROOT_DIR)),
        "yearly_path": str(yearly_path.relative_to(ROOT_DIR)),
        "core_hourly_path": str(core_hourly_path.relative_to(ROOT_DIR)),
        "core_daily_path": str(core_daily_path.relative_to(ROOT_DIR)),
        "core_monthly_path": str(core_monthly_path.relative_to(ROOT_DIR)),
        "core_yearly_path": str(core_yearly_path.relative_to(ROOT_DIR)),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
