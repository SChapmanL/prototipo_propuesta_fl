"""Control de la muestra descargada: compara el manifiesto con el registro de validacion.

No descarga nada y no modifica ningun archivo. Solo lee, cuenta y compara.
Muestra conteos. Nunca imprime identificadores de pacientes ni rutas de imagenes.

Comprueba:
    1. El manifiesto tiene el numero de filas esperado.
    2. Ninguna etiqueta es distinta de 0 o 1.
    3. Las clases estan balanceadas (mitad y mitad).
    4. Hay un patient_id distinto por fila.
    5. El registro tiene el numero esperado de filas con status=ok.
    6. Cada imagen del manifiesto aparece como ok en el registro.
    7. Cada dicom_path existe en disco.
    8. Las imagenes validadas conservan el balance de clases y los pacientes distintos.

Uso (desde la raiz del repositorio):
    python scripts/check_sample.py
    python scripts/check_sample.py --manifest data/manifests/pilot20.csv --expected-total 20

Devuelve codigo de salida 1 si algun control falla.
"""
import argparse
import collections
import csv
from pathlib import Path


def read_csv(path):
    """Lee un CSV con encabezado y devuelve una lista de diccionarios."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class Checks:
    """Acumula los resultados de cada control para imprimirlos juntos al final."""

    def __init__(self):
        self.results = []

    def add(self, passed, title, detail):
        self.results.append((bool(passed), title, detail))

    @property
    def failed(self):
        return [r for r in self.results if not r[0]]

    def report(self):
        width = max(len(title) for _, title, _ in self.results)
        for passed, title, detail in self.results:
            mark = "OK   " if passed else "FALLA"
            print(f"{mark}  {title.ljust(width)}  {detail}")


def describe(title, counter):
    """Imprime un conteo ordenado. Solo categorias, nunca identificadores."""
    items = ", ".join(f"{key or 'sin_detalle'}: {n}" for key, n in sorted(counter.items()))
    print(f"  {title}: {items or 'ninguno'}")


def run_checks(manifest, log, expected_total):
    checks = Checks()
    per_class = expected_total // 2

    # --- Manifiesto ---
    total = len(manifest)
    checks.add(total == expected_total, "Filas en el manifiesto",
               f"{total} (esperado {expected_total})")

    labels = collections.Counter(row["label"] for row in manifest)
    invalid = {value: n for value, n in labels.items() if value not in ("0", "1")}
    checks.add(not invalid, "Etiquetas solo 0 o 1",
               f"{sum(invalid.values())} filas invalidas {invalid or ''}".strip())
    checks.add(labels["0"] == per_class and labels["1"] == per_class, "Balance de clases",
               f"0: {labels['0']}, 1: {labels['1']} (esperado {per_class} y {per_class})")

    patients = {row["patient_id"] for row in manifest}
    checks.add(len(patients) == expected_total, "patient_id distintos",
               f"{len(patients)} (esperado {expected_total})")

    # --- Registro de validacion ---
    status = collections.Counter(row["status"] for row in log)
    checks.add(status["ok"] == expected_total, "Filas con status=ok",
               f"{status['ok']} de {len(log)} en el registro (esperado {expected_total})")

    validated = {row["file_name"] for row in log if row["status"] == "ok"}
    sin_validar = [row for row in manifest if row["file_name"] not in validated]
    checks.add(not sin_validar, "Imagenes validadas en el registro",
               f"{total - len(sin_validar)} de {total}")

    # --- Archivos en disco ---
    # dicom_path viene del manifiesto y es relativo a la raiz del repositorio.
    ausentes = [row for row in manifest if not Path(row["dicom_path"]).is_file()]
    checks.add(not ausentes, "Archivos presentes en disco",
               f"{total - len(ausentes)} de {total}")

    # --- Las validadas, por si el fallo afecta solo a una clase ---
    label_by_file = {row["file_name"]: row["label"] for row in manifest}
    patient_by_file = {row["file_name"]: row["patient_id"] for row in manifest}
    ok_files = [f for f in validated if f in label_by_file]
    ok_labels = collections.Counter(label_by_file[f] for f in ok_files)
    ok_patients = {patient_by_file[f] for f in ok_files}
    checks.add(ok_labels["0"] == per_class and ok_labels["1"] == per_class,
               "Balance de clases ya validadas",
               f"0: {ok_labels['0']}, 1: {ok_labels['1']} (esperado {per_class} y {per_class})")
    checks.add(len(ok_patients) == expected_total, "Pacientes distintos ya validados",
               f"{len(ok_patients)} (esperado {expected_total})")

    return checks


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/manifests/download_manifest.csv"))
    parser.add_argument("--log", type=Path,
                        default=Path("data/raw/dicom/download_validation.csv"))
    parser.add_argument("--expected-total", type=int, default=800,
                        help="Filas esperadas. Use 20 para pilot20")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.manifest, args.log):
        if not path.is_file():
            raise SystemExit(f"No existe {path}. Ejecute este script desde la raiz del repositorio.")

    manifest = read_csv(args.manifest)
    log = read_csv(args.log)
    checks = run_checks(manifest, log, args.expected_total)

    print(f"Manifiesto: {args.manifest}")
    print(f"Registro:   {args.log}\n")
    checks.report()

    # Informacion descriptiva. No decide el resultado.
    print("\nDescripcion de la muestra")
    describe("vista", collections.Counter(row["view"] for row in manifest))
    describe("interpretacion fotometrica (validadas)",
             collections.Counter(row["photometric"] for row in log if row["status"] == "ok"))
    describe("tipo de error (no validadas)",
             collections.Counter(row["error"] for row in log if row["status"] != "ok"))

    if checks.failed:
        print(f"\n{len(checks.failed)} control(es) fallaron. "
              "No asigne particiones finales todavia. "
              "Registre el motivo y elija reemplazos de reserve100.csv de la misma clase.")
        raise SystemExit(1)
    print("\nTodos los controles pasaron. La muestra puede congelarse.")


if __name__ == "__main__":
    main()
