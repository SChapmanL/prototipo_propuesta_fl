"""Descarga y valida los DICOM de un manifiesto con el cliente oficial de Redivis.

No guarda credenciales en el codigo. Usa la sesion autorizada de Redivis.
"""
import argparse
import base64
import csv
import hashlib
import sys
from pathlib import Path

REDIVIS_TABLE = "aimi.chexpert_plus:5yyj:v1_0.dicom_train:1934"
LOG_COLUMNS = ["file_name", "status", "shape", "photometric", "error"]


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md5_base64(path):
    """MD5 en base64, el mismo formato que usa Redivis en el manifiesto."""
    digest = hashlib.md5(path.read_bytes()).digest()
    return base64.b64encode(digest).decode()


def validate_dicom(path, row):
    """Revisa tamano, MD5 e imagen. Devuelve (forma, interpretacion fotometrica)."""
    import pydicom

    if path.stat().st_size != int(row["size_bytes"]):
        raise ValueError("Size mismatch")
    if md5_base64(path) != row["md5_hash"]:
        raise ValueError("MD5 mismatch")

    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array
    if pixels.ndim != 2 or pixels.max() <= pixels.min():
        raise ValueError("Unexpected or constant pixel array")
    if ds.PhotometricInterpretation not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Unexpected photometric interpretation")

    return f"{pixels.shape[0]}x{pixels.shape[1]}", ds.PhotometricInterpretation


def destination_path(output_dir, file_name):
    """Ruta local del archivo. Rechaza rutas que salgan de output_dir."""
    if file_name.startswith("/") or ".." in file_name.split("/"):
        raise ValueError("Unsafe file path")
    return output_dir / file_name


def download_file(table, file_name, dest):
    """Descarga a un archivo .part y solo lo renombra si pasa la validacion."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".dcm.part")
    table.file(file_name).download(str(temporary), overwrite=True)
    return temporary


def process_row(table, row, output_dir):
    """Descarga el archivo si falta y lo valida. Devuelve una fila del log."""
    entry = {"file_name": row["file_name"], "status": "failed",
             "shape": "", "photometric": "", "error": ""}
    try:
        dest = destination_path(output_dir, row["file_name"])
        if dest.exists():
            shape, photometric = validate_dicom(dest, row)
        else:
            temporary = download_file(table, row["file_name"], dest)
            shape, photometric = validate_dicom(temporary, row)
            temporary.replace(dest)
        entry.update(status="ok", shape=shape, photometric=photometric)
    except Exception as exc:
        # Solo el tipo de error, para no registrar URLs ni credenciales.
        entry["error"] = type(exc).__name__
    return entry


def download_all(rows, output_dir):
    """Procesa todas las filas y escribe el log. Devuelve el numero de fallos."""
    import redivis

    table = redivis.table(REDIVIS_TABLE)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download_validation.csv"
    failures = 0

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            entry = process_row(table, row, output_dir)
            writer.writerow(entry)
            f.flush()
            if entry["status"] != "ok":
                failures += 1
            print(f'{i}/{len(rows)} {row["file_name"]}: {entry["status"]}')

    print(f"Failures: {failures}. Log: {log_path}")
    return failures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/pilot20.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/dicom"))
    parser.add_argument("--download", action="store_true",
                        help="Without this flag, inspect the list only")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_manifest(args.manifest)
    total_gb = sum(int(r["size_bytes"]) for r in rows) / 1e9
    print(f"{len(rows)} files, {total_gb:.3f} GB")

    if not args.download:
        print("Dry run. Add --download to download and validate. "
              "Requires your authorized Redivis account.")
        return

    failures = download_all(rows, args.output_dir)
    if failures:
        print("Review failures before assigning final splits. No automatic replacements.")
        sys.exit(1)


if __name__ == "__main__":
    main()
