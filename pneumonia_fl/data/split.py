"""Paso 2: reparte los 800 pacientes en particiones y clientes.

Reglas que cumple:

1. train/val/test 70/15/15, estratificado por etiqueta.
2. Regla reproducible por hash con semilla 42, la misma idea que la seleccion
   de scripts/prepare_sample.py. No usa generadores aleatorios de bibliotecas.
3. Cada paciente pertenece a UN solo cliente y a UNA sola particion.
4. Los 20 pacientes de pilot20 quedan en train y se descuentan de su cupo.

Resultado, seis archivos:
    data/manifests/client_1/train.csv, val.csv, test.csv
    data/manifests/client_2/train.csv, val.csv, test.csv

Y un resumen sin identificadores en data/manifests/split_summary.json.

Uso (desde la raiz del repositorio):
    python -m pneumonia_fl.data.split
"""

import collections
import csv
import hashlib
import json
from pathlib import Path

SEED = 42
CLIENTES = ("client_1", "client_2")
PARTICIONES = ("train", "val", "test")
PROPORCIONES = {"train": 0.70, "val": 0.15, "test": 0.15}

# Columnas minimas del contrato (docs/07) mas la vista, que hace falta para
# comprobar el reparto AP/PA de cada archivo.
COLUMNAS = ["image_id", "patient_id", "dicom_path", "label", "view"]


def rank(*partes):
    """Orden reproducible por hash. Mismo criterio que prepare_sample.py.

    No depende del orden de las filas ni del generador aleatorio de ninguna
    biblioteca. Con la misma semilla y los mismos pacientes, el reparto
    siempre sale igual.
    """
    return hashlib.sha256((str(SEED) + "|" + "|".join(partes)).encode()).hexdigest()


def leer_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def escribir_csv(path, filas):
    """Escribe un manifiesto de cliente.

    dicom_path se guarda RELATIVO A data_root, como exige docs/07. El
    manifiesto maestro usa otra convencion y lo guarda relativo a la raiz
    del repositorio. Por eso aqui se reescribe a partir de file_name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUMNAS, extrasaction="ignore")
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({**fila, "dicom_path": fila["file_name"]})


def reparte_particiones(filas, pilot_ids):
    """Asigna train/val/test dentro de cada clase.

    Los pacientes de pilot20 van a train antes de repartir el resto, y su
    cupo se descuenta. Asi se mantiene la proporcion 70/15/15 por clase.
    """
    por_particion = collections.defaultdict(list)

    for label in ("0", "1"):
        de_la_clase = [r for r in filas if r["label"] == label]
        total = len(de_la_clase)
        cupos = {p: round(total * PROPORCIONES[p]) for p in PARTICIONES}
        # El redondeo podria no sumar el total. train absorbe la diferencia.
        cupos["train"] += total - sum(cupos.values())

        pilotos = [r for r in de_la_clase if r["patient_id"] in pilot_ids]
        resto = [r for r in de_la_clase if r["patient_id"] not in pilot_ids]
        resto.sort(key=lambda r: rank("split", r["patient_id"]))

        por_particion["train"] += pilotos
        restantes_train = cupos["train"] - len(pilotos)
        if restantes_train < 0:
            raise ValueError("pilot20 no cabe en el cupo de train de su clase.")

        cortes = {
            "train": resto[:restantes_train],
            "val": resto[restantes_train:restantes_train + cupos["val"]],
            "test": resto[restantes_train + cupos["val"]:],
        }
        for particion, grupo in cortes.items():
            por_particion[particion] += grupo

    return por_particion


def reparte_clientes(grupo):
    """Divide un grupo en dos mitades estables por hash.

    Se aplica dentro de cada combinacion de particion y clase, para que los
    dos clientes queden con el mismo balance de clases.
    """
    ordenado = sorted(grupo, key=lambda r: rank("client", r["patient_id"]))
    mitad = len(ordenado) // 2
    return {"client_1": ordenado[:mitad], "client_2": ordenado[mitad:]}


def construye_reparto(filas, pilot_ids):
    """Devuelve {cliente: {particion: filas}}."""
    por_particion = reparte_particiones(filas, pilot_ids)
    reparto = {c: {p: [] for p in PARTICIONES} for c in CLIENTES}

    for particion, grupo in por_particion.items():
        for label in ("0", "1"):
            de_la_clase = [r for r in grupo if r["label"] == label]
            for cliente, filas_cliente in reparte_clientes(de_la_clase).items():
                reparto[cliente][particion] += filas_cliente

    return reparto


def comprueba(reparto, filas, pilot_ids):
    """Controles que deben pasar antes de escribir nada."""
    todos = [r for c in CLIENTES for p in PARTICIONES for r in reparto[c][p]]
    ids = [r["patient_id"] for r in todos]

    if len(ids) != len(filas):
        raise ValueError(f"Se repartieron {len(ids)} filas y hay {len(filas)}.")
    if len(set(ids)) != len(ids):
        raise ValueError("Hay pacientes repartidos mas de una vez.")
    if set(ids) != {r["patient_id"] for r in filas}:
        raise ValueError("El reparto no cubre exactamente los mismos pacientes.")

    # Ningun paciente compartido entre clientes ni entre particiones.
    por_cliente = {c: {r["patient_id"] for p in PARTICIONES for r in reparto[c][p]}
                   for c in CLIENTES}
    if por_cliente["client_1"] & por_cliente["client_2"]:
        raise ValueError("Hay pacientes compartidos entre los dos clientes.")

    for c in CLIENTES:
        vistos = set()
        for p in PARTICIONES:
            actuales = {r["patient_id"] for r in reparto[c][p]}
            if actuales & vistos:
                raise ValueError(f"{c}: paciente en mas de una particion.")
            vistos |= actuales

    # pilot20 debe estar entero en train.
    en_train = {r["patient_id"] for c in CLIENTES for r in reparto[c]["train"]}
    if not pilot_ids <= en_train:
        raise ValueError("Algun paciente de pilot20 quedo fuera de train.")


def resumen(reparto):
    """Conteos por cliente y particion. Sin identificadores de pacientes."""
    datos = {}
    for cliente in CLIENTES:
        datos[cliente] = {}
        for particion in PARTICIONES:
            grupo = reparto[cliente][particion]
            datos[cliente][particion] = {
                "imagenes": len(grupo),
                "pacientes": len({r["patient_id"] for r in grupo}),
                "por_etiqueta": dict(collections.Counter(r["label"] for r in grupo)),
                "por_vista": dict(collections.Counter(r["view"] for r in grupo)),
            }
    return datos


def main():
    manifest_dir = Path("data/manifests")
    filas = leer_csv(manifest_dir / "download_manifest.csv")
    pilot_ids = {r["patient_id"] for r in leer_csv(manifest_dir / "pilot20.csv")}

    reparto = construye_reparto(filas, pilot_ids)
    comprueba(reparto, filas, pilot_ids)

    for cliente in CLIENTES:
        for particion in PARTICIONES:
            escribir_csv(manifest_dir / cliente / f"{particion}.csv",
                         reparto[cliente][particion])

    datos = resumen(reparto)
    salida = {
        "seed": SEED,
        "proporciones": PROPORCIONES,
        "regla_particion": "menor SHA256 de 42|split|patient_id dentro de cada clase",
        "regla_cliente": "menor SHA256 de 42|client|patient_id dentro de cada particion y clase",
        "pilot20_en_train": True,
        "total_imagenes": len(filas),
        "reparto": datos,
    }
    (manifest_dir / "split_summary.json").write_text(
        json.dumps(salida, indent=2), encoding="utf-8")

    print(json.dumps(salida, indent=2))
    print("\nControles superados. Seis manifiestos escritos en data/manifests/.")


if __name__ == "__main__":
    main()
