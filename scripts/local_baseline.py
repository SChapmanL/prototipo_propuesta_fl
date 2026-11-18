"""Paso 5: referencias locales, un modelo por cliente y sin federacion.

Entrena un modelo SOLO con el train de cada cliente y lo evalua con la
validacion de ese mismo cliente. Sirve para saber, mas adelante, si la
federacion aporta algo frente a entrenar por separado.

Dos escenarios:

  federacion   2 epocas. Es la configuracion de pyproject.toml, donde
               num-server-rounds=2 y local-epochs=1, asi que cada cliente
               ejecuta dos epocas locales en total. Es la comparacion que
               toca hacer contra la federacion actual.
  piloto       10 epocas. docs/05 propone diez rondas para el primer piloto.
               Se incluye porque dos epocas con la base congelada y tasa
               0,001 dejan el modelo en el azar, y una referencia en el azar
               no sirve para comparar nada.

NO se evalua en test. Test se reserva para el final, una vez fijadas las
decisiones con validacion.

Las metricas se guardan POR CLIENTE. No se promedian entre clientes: el
promedio de dos AUROC locales no es el AUROC del conjunto.

Uso (desde la raiz del repositorio):
    python scripts/local_baseline.py
    python scripts/local_baseline.py --escenario federacion
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pneumonia_fl.config import load_config
from pneumonia_fl.data.dicom import load_data
from pneumonia_fl.metrics import sensibilidad_especificidad
from pneumonia_fl.model import build_model
from pneumonia_fl.training import evaluate_local, train_local

CLIENTES = ("client_1", "client_2")
SALIDA = Path("outputs/local_baselines")


def escenarios(config):
    """Epocas de cada escenario. La federacion hace rondas por epocas."""
    por_federacion = int(config["num-server-rounds"]) * int(config["local-epochs"])
    return {"federacion": por_federacion, "piloto": 10}


def una_referencia(cliente, epocas, config):
    """Entrena y evalua un cliente. Devuelve el registro completo."""
    import keras
    keras.backend.clear_session()

    # Misma inicializacion que usaria el servidor, por la semilla comun.
    modelo = build_model(config)

    train_data, n_train = load_data(
        cliente, "train", config["data-root"], config["manifest-root"],
        int(config["batch-size"]))
    val_data, n_val = load_data(
        cliente, "val", config["data-root"], config["manifest-root"],
        int(config["batch-size"]))

    inicio = time.time()
    metricas_train = train_local(modelo, train_data, epochs=epocas)
    segundos = time.time() - inicio

    metricas_val = evaluate_local(modelo, val_data)
    metricas_val.update(sensibilidad_especificidad(metricas_val))

    return {
        "cliente": cliente,
        "epocas": epocas,
        "ejemplos_train": n_train,
        "ejemplos_val": n_val,
        "segundos_entrenamiento": round(segundos, 1),
        "segundos_por_epoca": round(segundos / epocas, 1),
        "train": metricas_train,
        "val": metricas_val,
    }


def imprime(registro):
    v = registro["val"]
    print(f'\n{registro["cliente"]}  ({registro["epocas"]} epocas, '
          f'{registro["segundos_entrenamiento"]:.0f} s, '
          f'{registro["ejemplos_train"]} de train)')
    print(f'    val loss          {v["loss"]:.4f}')
    print(f'    val accuracy      {v["accuracy"]:.4f}')
    print(f'    val AUROC         {v["auroc"]:.4f}')
    print(f'    val AUPRC         {v["auprc"]:.4f}')
    print(f'    sensibilidad      {v["sensibilidad"]:.4f}')
    print(f'    especificidad     {v["especificidad"]:.4f}')
    print(f'    matriz 0,5        tp {v["tp"]:.0f}  tn {v["tn"]:.0f}  '
          f'fp {v["fp"]:.0f}  fn {v["fn"]:.0f}')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escenario", choices=("federacion", "piloto", "ambos"),
                        default="ambos")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    todos = escenarios(config)
    elegidos = todos if args.escenario == "ambos" else {args.escenario: todos[args.escenario]}

    SALIDA.mkdir(parents=True, exist_ok=True)

    for nombre, epocas in elegidos.items():
        print(f"\n{'='*62}\nEscenario {nombre}: {epocas} epocas por cliente\n{'='*62}")
        registros = [una_referencia(c, epocas, config) for c in CLIENTES]
        for registro in registros:
            imprime(registro)

        salida = {
            "escenario": nombre,
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nota": ("Referencia LOCAL sin federacion. Metricas por cliente, "
                     "sin promediar entre clientes. Test no se uso."),
            "entorno": {
                "python": platform.python_version(),
                "tensorflow": __import__("tensorflow").__version__,
                "dispositivo": "CPU",
            },
            "config": dict(config),
            "epocas": epocas,
            "resultados": {r["cliente"]: r for r in registros},
        }
        ruta = SALIDA / f"{nombre}.json"
        ruta.write_text(json.dumps(salida, indent=2), encoding="utf-8")
        print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()
