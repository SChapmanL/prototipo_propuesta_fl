"""Ejecuta la federación con los datos reales, después de comprobarlo todo.

Hace tres cosas, en este orden:

1. Controles previos. Entorno, manifiestos por cliente, pacientes sin
   repetir y DICOM presentes en disco. Si algo falla, se detiene y explica
   qué hacer. No entrena nada.
2. Ejecución. Lanza "flwr run" con dos clientes, uno a la vez en CPU, y con
   rutas absolutas. Así los clientes encuentran los datos aunque Flower los
   ejecute desde otra carpeta.
3. Reporte. Escribe reporte.md y tres CSV con métricas por cliente y del
   modelo global (scripts/report_federation.py). Muestra las cifras principales.

Cada ejecución escribe en una carpeta nueva con fecha y hora. No sobrescribe
resultados anteriores, tampoco los de la prueba técnica sintética.

Uso, con el entorno activado y desde la raíz del repositorio:
    python scripts/run_federation.py --check-only
    python scripts/run_federation.py

Opciones útiles:
    --rounds 2                       prueba de integración corta (por defecto 10)
    --local-epochs 1                 cambia épocas sin editar archivos
    --data-root RUTA                 si los DICOM están en otra carpeta
"""

import argparse
import csv
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIENTS = ("client_1", "client_2")
SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = ("patient_id", "dicom_path", "label")
EXPECTED_VERSIONS = {"flwr": "1.43.0", "tensorflow": "2.21.0", "pydicom": "3.0.2"}
RESNET_WEIGHTS = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Un cliente a la vez: Ray tiene 2 CPU y cada cliente pide 2. Es el modo
# probado en docs/08 para una PC sin GPU.
FEDERATION_CONFIG = (
    "num-supernodes=2 client-resources-num-cpus=2 init-args-num-cpus=2"
)


class CheckFailed(Exception):
    """Un control previo no se cumple. El mensaje dice qué hacer."""


def ok(message):
    print(f"  [ok] {message}")


def warn(message):
    print(f"  [aviso] {message}")


def load_app_config():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return data["tool"]["flwr"]["app"]["config"]


def absolute(path_text):
    path = Path(path_text)
    return (path if path.is_absolute() else REPO / path).resolve()


def check_environment():
    """Python, paquetes y ejecutables de Flower del entorno activo."""
    print("Entorno")
    version = sys.version_info[:2]
    if version not in ((3, 11), (3, 12)):
        raise CheckFailed(
            f"Python {version[0]}.{version[1]}. Se necesita 3.11 o 3.12. "
            "Active el entorno del proyecto (.venv) y repita."
        )
    ok(f"Python {sys.version.split()[0]} en {sys.executable}")

    for package, expected in EXPECTED_VERSIONS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise CheckFailed(
                f"Falta {package}. Instale el proyecto con: "
                "python -m pip install -e ."
            ) from error
        if installed != expected:
            warn(f"{package} {installed}, el proyecto fija {expected}")
        else:
            ok(f"{package} {installed}")

    scripts_dir = Path(sys.executable).parent
    for program in ("flwr", "flower-superlink"):
        if not (shutil.which(program, path=str(scripts_dir))
                or shutil.which(program)):
            raise CheckFailed(
                f"No se encuentra {program}. Instale el proyecto con: "
                'python -m pip install -e .'
            )
    ok("flwr y flower-superlink disponibles")

    cache = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras"))
    if (cache / "models" / RESNET_WEIGHTS).is_file():
        ok("pesos de ImageNet de ResNet50 en caché")
    else:
        warn("los pesos de ImageNet no están en caché. Keras los descargará "
             "(unos 95 MB) en la primera ronda")


def read_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise CheckFailed(f"{path} no tiene las columnas {missing}.")
        return list(reader)


def check_data(manifest_root, data_root):
    """Seis manifiestos, etiquetas 0/1, pacientes sin repetir y DICOM en disco."""
    print("Datos")
    owner = {}  # patient_id -> (cliente, partición)
    missing_files = []
    for client in CLIENTS:
        for split in SPLITS:
            path = manifest_root / client / f"{split}.csv"
            if not path.is_file():
                raise CheckFailed(
                    f"No existe {path}. Genere las particiones con: "
                    "python -m pneumonia_fl.data.split"
                )
            rows = read_manifest(path)
            if not rows:
                raise CheckFailed(f"{path} está vacío.")

            labels = [row["label"] for row in rows]
            invalid = sorted(set(labels) - {"0", "1"})
            if invalid:
                raise CheckFailed(
                    f"{path} tiene etiquetas {invalid}. Solo se admiten 0 y 1. "
                    "No convierta null ni -1 en 0."
                )

            for row in rows:
                place = (client, split)
                previous = owner.setdefault(row["patient_id"], place)
                if previous != place:
                    raise CheckFailed(
                        f"Un paciente aparece en {previous} y en {place}. "
                        "La separación por paciente no se cumple."
                    )
                if not (data_root / row["dicom_path"]).is_file():
                    missing_files.append(row["dicom_path"])

            ok(f"{client}/{split}: {len(rows)} imágenes, "
               f"{labels.count('0')} de clase 0 y {labels.count('1')} de clase 1")

    if missing_files:
        raise CheckFailed(
            f"Faltan {len(missing_files)} DICOM bajo {data_root}. "
            "Revise --data-root o descargue la muestra con "
            "scripts/download_dicoms.py."
        )
    ok(f"{len(owner)} pacientes, cada uno en un solo cliente y partición")
    ok(f"todos los DICOM existen bajo {data_root}")


def write_overrides(path, overrides):
    """Escribe los valores de --run-config en un TOML.

    Un archivo evita problemas de comillas con rutas que tienen espacios o
    tildes. json.dumps produce cadenas TOML válidas.
    """
    lines = [f"{key} = {json.dumps(value)}" for key, value in overrides.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_scripts_to_path():
    """Pone la carpeta de ejecutables del entorno al inicio del PATH.

    flwr lanza flower-superlink y esta lanza flower-superexec, ambos por
    nombre. Así se encuentran aunque el entorno no esté activado.
    """
    scripts_dir = str(Path(sys.executable).parent)
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")


def ensure_superlink():
    """Arranca la SuperLink local con margen de tiempo, si no está activa.

    flwr run espera solo 15 s a que la SuperLink responda. En una PC sin GPU
    la primera carga de Python puede tardar más y flwr se rinde aunque la
    SuperLink habría arrancado. Aquí se usa la misma función de Flower 1.43
    con 180 s de espera. Después flwr run encuentra la SuperLink activa y la
    reutiliza.
    """
    from flwr.cli import local_superlink

    if local_superlink._is_local_superlink_started():
        ok("SuperLink local ya activa")
        return
    local_superlink.LOCAL_SUPERLINK_STARTUP_TIMEOUT = 180
    local_superlink._start_local_superlink(in_memory=False)
    ok("SuperLink local activa")


def run_flower(overrides_file):
    env = dict(os.environ)
    flwr = shutil.which("flwr", path=str(Path(sys.executable).parent)) or "flwr"
    command = [
        flwr, "run", ".", "--stream",
        "--run-config", str(overrides_file),
        "--federation-config", FEDERATION_CONFIG,
    ]
    print("\nEjecutando:", " ".join(command), "\n", flush=True)
    return subprocess.run(command, cwd=REPO, env=env, check=False).returncode


def show_summary(output_dir):
    """Genera el reporte formal y muestra las cifras principales."""
    if not (output_dir / "metrics.json").is_file():
        raise CheckFailed(f"La ejecución terminó sin {output_dir / 'metrics.json'}.")
    # Import tardío: carga TensorFlow, que los controles previos no necesitan.
    from report_federation import build as build_report

    report, summary = build_report(output_dir, REPO / "outputs" / "local_baselines")

    print("\nResumen")
    print("  ", summary["nota"])
    for server_round, metrics in sorted(summary["val_global_por_ronda"].items(),
                                        key=lambda item: int(item[0])):
        print(f"  ronda {server_round:>2}  AUROC val global {metrics['auroc']:.4f}")
    print(f"  ronda elegida con validación: {summary['ronda_elegida']}")
    for scope in CLIENTS:
        print(f"  test {scope}: AUROC "
              f"{summary['test'].get(f'{scope}_test_auroc', float('nan')):.4f}")
    print(f"  test global:   AUROC {summary['test_global']['auroc']:.4f}")
    print(f"\nReporte formal: {report}")
    print(f"Resultados en {output_dir}")
    print("Verificación independiente de FedAvg:")
    print(f'  python scripts/check_federation.py --output-dir "{output_dir}"')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-only", action="store_true",
                        help="solo controles previos, sin entrenar")
    parser.add_argument("--data-root", help="carpeta raíz de los DICOM")
    parser.add_argument("--manifest-root", help="carpeta con client_1/ y client_2/")
    parser.add_argument("--output-dir", help="carpeta nueva para los resultados")
    parser.add_argument("--rounds", type=int, help="num-server-rounds")
    parser.add_argument("--local-epochs", type=int, help="local-epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_app_config()

    data_root = absolute(args.data_root or config["data-root"])
    manifest_root = absolute(args.manifest_root or config["manifest-root"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = absolute(args.output_dir or f"outputs/federation_real_{stamp}")

    try:
        check_environment()
        check_data(manifest_root, data_root)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise CheckFailed(
                f"{output_dir} ya tiene archivos. Use otra carpeta para no "
                "sobrescribir resultados."
            )
        ok(f"salida en {output_dir}")
    except CheckFailed as error:
        print(f"\n  [ERROR] {error}")
        return 1

    if args.check_only:
        print("\nControles superados. Ejecute sin --check-only para entrenar.")
        return 0

    overrides = {
        "data-root": data_root.as_posix(),
        "manifest-root": manifest_root.as_posix(),
        "output-dir": output_dir.as_posix(),
    }
    if args.rounds is not None:
        overrides["num-server-rounds"] = args.rounds
    if args.local_epochs is not None:
        overrides["local-epochs"] = args.local_epochs

    output_dir.mkdir(parents=True, exist_ok=True)
    overrides_file = output_dir / "run_overrides.toml"
    write_overrides(overrides_file, overrides)

    add_scripts_to_path()
    print("\nSuperLink")
    try:
        ensure_superlink()
    except Exception as error:  # click.ClickException y errores de red
        print(f"\n  [ERROR] No arrancó la SuperLink local: {error}")
        print("  Registro: ~/.flwr/local-superlink/superlink.log")
        return 1

    code = run_flower(overrides_file)
    if code != 0:
        print(f"\n  [ERROR] flwr terminó con código {code}. Revise el registro.")
        return code
    try:
        show_summary(output_dir)
    except CheckFailed as error:
        print(f"\n  [ERROR] {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
