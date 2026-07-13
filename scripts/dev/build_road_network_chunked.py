from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "road_network.csv"
PARTS_DIR = PROJECT_ROOT / "data" / "processed" / "road_network_parts"
MANIFEST_PATH = PARTS_DIR / "manifest.csv"
REQUIRED_COLUMNS = {
    "segment_id",
    "indice_coord",
    "lat",
    "lon",
    "comuna",
    "via",
    "distancia_km",
    "velocidad_kmh",
    "duracion_hrs",
    "fecha",
    "hora_inicio",
    "hora_fin",
}
DEDUP_COLUMNS = ["segment_id", "indice_coord", "lat", "lon", "via"]

BIOBIO_COMMUNES = [
    "Antuco",
    "Arauco",
    "Cabrero",
    "Cañete",
    "Chiguayante",
    "Concepción",
    "Contulmo",
    "Coronel",
    "Curanilahue",
    "Florida",
    "Hualpén",
    "Hualqui",
    "Laja",
    "Lebu",
    "Los Ángeles",
    "Lota",
    "Mulchén",
    "Nacimiento",
    "Negrete",
    "Penco",
    "San Pedro de la Paz",
    "San Rosendo",
    "Santa Bárbara",
    "Santa Juana",
    "Talcahuano",
    "Tirúa",
    "Tomé",
    "Tucapel",
    "Yumbel",
]


def _part_filename(idx: int, commune: str) -> str:
    return f"{idx:02d}__{commune.replace(' ', '_').replace('Ã¡','a').replace('Ã©','e').replace('Ã­','i').replace('Ã³','o').replace('Ãº','u').replace('Ã±','n')}.csv"


def _expected_part_paths() -> dict[str, Path]:
    return {
        commune: PARTS_DIR / _part_filename(idx, commune)
        for idx, commune in enumerate(BIOBIO_COMMUNES, start=1)
    }


def _expected_part_prefixes() -> dict[int, str]:
    return {
        idx: commune
        for idx, commune in enumerate(BIOBIO_COMMUNES, start=1)
    }


def _validate_columns(part_path: Path) -> list[str]:
    header = pd.read_csv(part_path, nrows=0)
    missing = sorted(REQUIRED_COLUMNS - set(header.columns))
    if missing:
        raise ValueError(f"{part_path} no contiene columnas requeridas: {missing}")
    return list(header.columns)


def assemble_existing_parts(output_path: Path, *, strict: bool = False, chunksize: int = 100_000) -> None:
    part_paths = sorted(PARTS_DIR.glob("*.csv"))
    part_paths = [path for path in part_paths if path.name != MANIFEST_PATH.name]
    if not part_paths:
        raise FileNotFoundError(f"No se encontraron partes CSV en {PARTS_DIR}")

    expected_prefixes = _expected_part_prefixes()
    existing_prefixes = {
        int(path.name.split("__", 1)[0])
        for path in part_paths
        if "__" in path.name and path.name.split("__", 1)[0].isdigit()
    }
    missing_expected = [
        f"{idx:02d}__{commune}.csv"
        for idx, commune in expected_prefixes.items()
        if idx not in existing_prefixes
    ]
    if strict and missing_expected:
        raise FileNotFoundError(
            "Faltan partes esperadas para ensamblar la red: " + ", ".join(missing_expected)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_keys: set[tuple[str, str, str, str, str]] = set()
    manifest_rows: list[dict[str, str | int]] = []
    wrote_header = False
    total_written = 0
    total_duplicates = 0

    if output_path.exists():
        output_path.unlink()

    for part_path in part_paths:
        columns = _validate_columns(part_path)
        raw_rows = 0
        written_rows = 0
        duplicate_rows = 0
        for chunk in pd.read_csv(part_path, chunksize=chunksize):
            raw_rows += len(chunk)
            key_frame = chunk[DEDUP_COLUMNS].astype(str)
            keep_mask = []
            for key in key_frame.itertuples(index=False, name=None):
                if key in seen_keys:
                    keep_mask.append(False)
                    duplicate_rows += 1
                    continue
                seen_keys.add(key)
                keep_mask.append(True)
            clean_chunk = chunk.loc[keep_mask, columns]
            if clean_chunk.empty:
                continue
            clean_chunk.to_csv(
                output_path,
                mode="a",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
            written_rows += len(clean_chunk)

        total_written += written_rows
        total_duplicates += duplicate_rows
        manifest_rows.append(
            {
                "commune": part_path.stem.split("__", 1)[-1].replace("_", " "),
                "place": "",
                "rows": raw_rows,
                "written_rows": written_rows,
                "duplicate_rows": duplicate_rows,
                "status": "assembled",
                "part_path": str(part_path.relative_to(PROJECT_ROOT)),
            }
        )
        print(f"{part_path.name}: {written_rows}/{raw_rows} filas escritas")

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "commune",
                "place",
                "rows",
                "written_rows",
                "duplicate_rows",
                "status",
                "part_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nEscribi {total_written} filas en {output_path}")
    print(f"Duplicados omitidos: {total_duplicates}")
    print(f"Manifest: {MANIFEST_PATH}")
    if missing_expected:
        print("\nPartes esperadas ausentes:")
        for name in missing_expected:
            print(f" - {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera data/processed/road_network.csv por comunas para evitar problemas de memoria."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Ruta de salida final.",
    )
    parser.add_argument(
        "--network-type",
        default="drive",
        help="Tipo de red OSMnx.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reutiliza comunas ya descargadas si existe su CSV parcial.",
    )
    parser.add_argument(
        "--assemble-existing",
        action="store_true",
        help="Ensamble reproducible desde road_network_parts/*.csv sin descargar OSM.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla si falta alguna parte esperada al ensamblar desde CSV existentes.",
    )
    args = parser.parse_args()

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    if args.assemble_existing:
        assemble_existing_parts(output_path, strict=args.strict)
        return

    from build_road_network import build_dataframe

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    manifest_rows: list[dict[str, str | int]] = []

    for idx, commune in enumerate(BIOBIO_COMMUNES, start=1):
        place = f"{commune}, Región del Biobío, Chile"
        print(f"[{idx}/{len(BIOBIO_COMMUNES)}] {place}")
        part_path = PARTS_DIR / f"{idx:02d}__{commune.replace(' ', '_').replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n')}.csv"
        if args.resume and part_path.exists():
            frame = pd.read_csv(part_path)
            frames.append(frame)
            manifest_rows.append(
                {
                    "commune": commune,
                    "place": place,
                    "rows": len(frame),
                    "status": "reused",
                    "part_path": str(part_path.relative_to(PROJECT_ROOT)),
                }
            )
            print(f"  -> reused {len(frame)} filas")
            continue
        try:
            frame = build_dataframe(
                place=place,
                dist=None,
                lat=None,
                lon=None,
                network_type=args.network_type,
            )
            frame["osm_place"] = place
            part_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(part_path, index=False)
            frames.append(frame)
            manifest_rows.append(
                {
                    "commune": commune,
                    "place": place,
                    "rows": len(frame),
                    "status": "ok",
                    "part_path": str(part_path.relative_to(PROJECT_ROOT)),
                }
            )
            print(f"  -> {len(frame)} filas")
        except Exception as exc:
            failures.append(f"{place}: {exc}")
            manifest_rows.append(
                {
                    "commune": commune,
                    "place": place,
                    "rows": 0,
                    "status": f"error: {exc}",
                    "part_path": str(part_path.relative_to(PROJECT_ROOT)),
                }
            )
            print(f"  !! ERROR: {exc}")

    if not frames:
        raise RuntimeError("No se pudo construir ninguna comuna.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["segment_id", "indice_coord", "lat", "lon", "via"],
        keep="first",
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["commune", "place", "rows", "status", "part_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nEscribí {len(combined)} filas en {output_path}")
    print(f"Manifest: {MANIFEST_PATH}")
    if failures:
        print("\nFallos:")
        for failure in failures:
            print(f" - {failure}")


if __name__ == "__main__":
    main()
