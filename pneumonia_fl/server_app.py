"""Servidor Flower: coordina las rondas y agrega con FedAvg.

El servidor nunca ve imágenes. Solo maneja listas de pesos y métricas.

Una ronda:
    1. Envía los MISMOS pesos globales a los dos clientes.
    2. Cada cliente entrena con sus datos y devuelve pesos y número de ejemplos.
    3. FedAvg calcula el promedio ponderado:
           global = (n1 * pesos_1 + n2 * pesos_2) / (n1 + n2)
    4. Cada cliente evalúa el nuevo modelo global con su validación.

Al terminar las rondas:
    5. Se elige el checkpoint con mayor AUROC conjunto de validación.
    6. Cada cliente evalúa ESE checkpoint una sola vez con su test.
    7. Las métricas se guardan por cliente y para el modelo global sobre los
       dos clientes juntos (pneumonia_fl/global_metrics.py).
"""

import importlib.metadata
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pneumonia_fl.components import COMPONENTS_ARE_PROVISIONAL, build_model
from pneumonia_fl.global_metrics import pooled_from_record
from pneumonia_fl.weights import weights_fingerprint

NUM_CLIENTS = 2

app = ServerApp()


def metrics_per_client(records, weighting_key):
    """Conserva las métricas de cada cliente por separado.

    Por defecto Flower promedia las métricas de los clientes. Aquí se guardan
    con el prefijo client_1_ o client_2_, porque el promedio de métricas
    locales (por ejemplo AUROC) no equivale a una métrica sobre todos los datos.

    Esta función solo afecta a las métricas. Los pesos los promedia FedAvg.
    """
    per_client = MetricRecord({"num-clients": len(records)})
    for record in records:
        metrics = record["metrics"]
        prefix = f"client_{int(metrics['client-number'])}_"
        for name, value in metrics.items():
            if name != "client-number":
                per_client[prefix + name] = value
    return per_client


def save_global_model(model, arrays, path):
    """Carga los pesos globales en el modelo y lo guarda en formato .keras."""
    model.set_weights(arrays.to_numpy_ndarrays())
    model.save(path)


def check_model_reloads(path, arrays):
    """Recarga el archivo guardado y exige pesos idénticos a los globales."""
    reloaded = keras.models.load_model(path)
    expected = arrays.to_numpy_ndarrays()
    for saved, original in zip(reloaded.get_weights(), expected, strict=True):
        if not np.array_equal(saved, original):
            raise RuntimeError(f"El modelo recargado de {path} no coincide.")


def check_rounds(result, sent_fingerprints, num_rounds):
    """Exige que cada ronda sea una federación completa de dos clientes.

    Controla dos cosas por ronda:
    - Respondieron los dos clientes. Si uno falla, Flower agrega solo con el
      otro. Eso no se debe presentar como federación de dos participantes.
    - Los dos recibieron exactamente los pesos que el servidor envió.
    """
    for server_round in range(1, num_rounds + 1):
        metrics = result.train_metrics_clientapp.get(server_round)
        if metrics is None or metrics["num-clients"] != NUM_CLIENTS:
            raise RuntimeError(f"Ronda {server_round} incompleta: faltan clientes.")
        for number in range(1, NUM_CLIENTS + 1):
            received = metrics[f"client_{number}_received-weights-fingerprint"]
            if received != sent_fingerprints[server_round]:
                raise RuntimeError(
                    f"Ronda {server_round}: client_{number} no recibió "
                    "los pesos globales enviados."
                )


def evaluate_on_test(strategy, grid, arrays, server_round):
    """Evaluación final: cada cliente evalúa el modelo global con su test."""
    messages = strategy.configure_evaluate(
        server_round, arrays, ConfigRecord({"split": "test"}), grid
    )
    replies = grid.send_and_receive(messages)
    return strategy.aggregate_evaluate(server_round, replies)


def is_histogram(name):
    return name.endswith("hist-pos") or name.endswith("hist-neg")


def evaluate_aggregator(store):
    """Agregador de evaluación que aparta los histogramas.

    Los histogramas tienen NUM_BINS números por cliente. Si quedaran en el
    registro, Flower los imprimiría en cada ronda. Se guardan en store, en el
    orden de las llamadas: rondas 1..N con val y después test.
    """

    def aggregate(records, weighting_key):
        per_client = metrics_per_client(records, weighting_key)
        store.append({k: list(v) for k, v in per_client.items() if is_histogram(k)})
        return MetricRecord(
            {k: v for k, v in per_client.items() if not is_histogram(k)}
        )

    return aggregate


def pooled(record, histograms, split):
    """Métricas del modelo global sobre los dos clientes juntos."""
    clients = [f"client_{n}" for n in range(1, NUM_CLIENTS + 1)]
    return pooled_from_record({**dict(record), **histograms}, clients, split)


def best_round(val_pooled):
    """Ronda con mayor AUROC conjunto de validación. Empate: la más temprana."""
    def score(item):
        auroc = item[1]["auroc"]
        return -1.0 if math.isnan(auroc) else auroc

    return max(sorted(val_pooled.items()), key=score)[0]


def environment():
    return {
        "python": platform.python_version(),
        "flwr": importlib.metadata.version("flwr"),
        "tensorflow": importlib.metadata.version("tensorflow"),
        "keras": keras.__version__,
        "dispositivo": "CPU",
    }


def save_metrics(path, summary):
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Punto de entrada del servidor."""
    config = dict(context.run_config)
    num_rounds = int(config["num-server-rounds"])
    output_dir = Path(config["output-dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Configuración efectiva de esta ejecución. scripts/check_federation.py
    # la usa para repetir FedAvg con los mismos valores.
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # Inicialización global común: el servidor construye el modelo UNA vez.
    # Los clientes siempre reemplazan sus pesos por los que reciben.
    model = build_model(config)
    initial_arrays = ArrayRecord(model.get_weights())

    # Huella de los pesos que el servidor envía en cada ronda.
    sent_fingerprints = {}

    def save_round_checkpoint(server_round, arrays):
        """Flower llama a esta función antes de la ronda 1 y después de cada ronda.

        arrays son los pesos globales vigentes. Son los que se enviarán en la
        ronda siguiente, por eso su huella se registra en server_round + 1.
        Guardar cada ronda permite elegir después el checkpoint con validación.
        """
        save_global_model(
            model, arrays, output_dir / f"global_round_{server_round}.keras"
        )
        sent_fingerprints[server_round + 1] = weights_fingerprint(
            arrays.to_numpy_ndarrays()
        )

    # FedAvg pondera con la métrica "num-examples" que devuelve cada cliente.
    histograms = []  # uno por evaluación: rondas 1..N (val) y luego test
    strategy = FedAvg(
        min_train_nodes=NUM_CLIENTS,
        min_evaluate_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        train_metrics_aggr_fn=metrics_per_client,
        evaluate_metrics_aggr_fn=evaluate_aggregator(histograms),
    )
    started = time.time()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        evaluate_config=ConfigRecord({"split": "val"}),
        evaluate_fn=save_round_checkpoint,
    )
    rounds_seconds = time.time() - started
    check_rounds(result, sent_fingerprints, num_rounds)

    # Métricas del modelo global con validación de ambos clientes, por ronda.
    val_records = {int(r): m for r, m in result.evaluate_metrics_clientapp.items()}
    val_pooled = {
        r: pooled(val_records[r], histograms[r - 1], "val") for r in sorted(val_records)
    }

    # Checkpoint elegido SOLO con validación. Test se usa una vez, con ese modelo.
    chosen = best_round(val_pooled)
    chosen_model = keras.models.load_model(output_dir / f"global_round_{chosen}.keras")
    chosen_arrays = ArrayRecord(chosen_model.get_weights())
    test_metrics = evaluate_on_test(strategy, grid, chosen_arrays, num_rounds)
    test_pooled = pooled(test_metrics, histograms[-1], "test")

    final_path = output_dir / "global_final.keras"
    save_global_model(model, result.arrays, final_path)
    check_model_reloads(final_path, result.arrays)

    save_metrics(output_dir / "metrics.json", {
        "nota": (
            "PRUEBA TECNICA con datos sinteticos. No es un resultado clinico."
            if COMPONENTS_ARE_PROVISIONAL
            else "Ejecucion con los componentes reales."
        ),
        "componentes_provisionales": COMPONENTS_ARE_PROVISIONAL,
        "inicio": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "fin": datetime.now().isoformat(timespec="seconds"),
        "segundos_rondas": round(rounds_seconds, 1),
        "entorno": environment(),
        "config": config,
        "huellas_enviadas": {str(r): sent_fingerprints[r] for r in range(1, num_rounds + 1)},
        "train_por_ronda": {
            str(r): dict(m) for r, m in result.train_metrics_clientapp.items()
        },
        "val_por_ronda": {str(r): dict(m) for r, m in val_records.items()},
        "val_global_por_ronda": {str(r): m for r, m in val_pooled.items()},
        "criterio_checkpoint": "mayor AUROC conjunto de validacion",
        "ronda_elegida": chosen,
        "test": dict(test_metrics),
        "test_global": test_pooled,
        "histogramas": {
            **{f"val_ronda_{r}": histograms[r - 1] for r in sorted(val_records)},
            "test": histograms[-1],
        },
    })

    kind = "PRUEBA TECNICA SINTETICA" if COMPONENTS_ARE_PROVISIONAL else "datos reales"
    print(f"[{kind}] Ronda elegida con validación: {chosen}. Resultados en {output_dir}")
