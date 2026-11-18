"""Trae las tablas de metadatos de CheXpert Plus desde la API de Redivis.

Completa el primer eslabon del pipeline. El orden real es:

    fetch_metadata.py  ->  data/raw/metadata/   (tablas de metadatos, por API)
    prepare_sample.py  ->  data/manifests/      (seleccion de 800 pacientes, local)
    download_dicoms.py ->  data/raw/dicom/      (imagenes, por API)

prepare_sample.py NO usa la API. Lee tres archivos locales. Este script es el
que los consigue.

No guarda credenciales. Usa la sesion autorizada de Redivis de la usuaria.

Uso (desde la raiz del repositorio):

    # 1. Ver que tablas existen y como se llaman de verdad
    python scripts/fetch_metadata.py

    # 2. Descargar una tabla al nombre de archivo que espera prepare_sample.py
    python scripts/fetch_metadata.py --table <nombre> --out dicom_train.csv

Archivos que necesita prepare_sample.py en data/raw/metadata/:
    df_chexpert_plus_240401.csv   tabla principal de metadatos
    report_fixed.json             etiquetas (una linea JSON por registro)
    dicom_train.csv               indice de archivos DICOM

Sobre el control de SHA256: al terminar, el script compara el hash del archivo
con el registrado en configs/sample_config.json. Si coincide, la fuente es
identica a la que produjo la seleccion de 800 pacientes. Si no coincide, puede
deberse al formato de exportacion de Redivis (orden de columnas, comillas,
fin de linea) y no necesariamente a datos distintos. En ese caso hay que
comparar el contenido antes de concluir nada.
"""
import argparse
import hashlib
import json
from pathlib import Path

DATASET = "aimi.chexpert_plus:5yyj:v1_0"
CONFIG = Path("configs/sample_config.json")


def sha256(path):
    """SHA256 del archivo, leido por bloques para no cargarlo entero en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_hashes():
    """Hashes de las tres fuentes que produjeron la seleccion actual."""
    if not CONFIG.is_file():
        return {}
    return json.loads(CONFIG.read_text(encoding="utf-8")).get("source_sha256", {})


def table_rows(table):
    """Numero de filas si la API lo informa. Devuelve '?' si no aparece."""
    props = table.properties or {}
    for key in ("numRows", "num_rows", "numberOfRows"):
        if key in props:
            return props[key]
    return "?"


def list_tables():
    """Muestra las tablas del dataset con su nombre real."""
    import redivis

    tables = redivis.dataset(DATASET).list_tables()
    print(f"Tablas de {DATASET}\n")
    for table in tables:
        print(f"  {table.name:<40}  filas: {table_rows(table)}")
    print(
        "\nDescargue cada una con:"
        "\n  python scripts/fetch_metadata.py --table <nombre> --out <archivo>"
    )


def fetch(table_name, out_path, fmt, overwrite):
    """Descarga una tabla y verifica su hash. Escribe .part y renombra al final."""
    import redivis

    if out_path.exists() and not overwrite:
        raise SystemExit(
            f"{out_path} ya existe. No lo reemplazo. "
            "Use --overwrite solo si realmente quiere perder el archivo actual."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(out_path.name + ".part")
    table = redivis.dataset(DATASET).table(table_name)
    table.download(str(temporary), format=fmt, overwrite=True)
    temporary.replace(out_path)

    size_gb = out_path.stat().st_size / 1e9
    print(f"\nGuardado: {out_path}  ({size_gb:.3f} GB)")

    digest = sha256(out_path)
    expected = reference_hashes().get(out_path.name)
    if expected is None:
        print(f"SHA256: {digest}")
        print(f"No hay hash de referencia para {out_path.name} en {CONFIG}.")
    elif digest == expected:
        print("SHA256 coincide con configs/sample_config.json.")
        print("La fuente es identica a la que produjo la seleccion de 800 pacientes.")
    else:
        print(f"SHA256 obtenido:  {digest}")
        print(f"SHA256 esperado:  {expected}")
        print(
            "Los hashes NO coinciden. Puede ser el formato de exportacion y no "
            "datos distintos. Compare el contenido antes de regenerar la seleccion."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", help="Nombre de la tabla en el dataset")
    parser.add_argument("--out", help="Nombre del archivo de salida")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/metadata"))
    parser.add_argument("--format", default="csv", help="csv o json (por defecto csv)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reemplaza el archivo existente. Uselo con cuidado")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.table:
        list_tables()
        return
    if not args.out:
        raise SystemExit("Indique tambien --out con el nombre del archivo de salida.")
    fetch(args.table, args.output_dir / args.out, args.format, args.overwrite)


if __name__ == "__main__":
    main()
