"""Servidor Flower: coordina las rondas y agrega con FedAvg.

El servidor nunca ve imágenes. Solo maneja listas de pesos y métricas.

Una ronda:
    1. Envía los MISMOS pesos globales a los dos clientes.
    2. Cada cliente entrena con sus datos y devuelve pesos y número de ejemplos.
    3. FedAvg calcula el promedio ponderado:
           global = (n1 * pesos_1 + n2 * pesos_2) / (n1 + n2)
    4. Cada cliente evalúa el nuevo modelo global con su validación.

Al terminar las rondas, cada cliente evalúa una sola vez con test.
"""

import json
from pathlib import Path

import keras
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pneumonia_fl.components import COMPONENTS_ARE_PROVISIONAL, build_model
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


def save_metrics(path, result, test_metrics):
    """Guarda en JSON las métricas por ronda y la evaluación final con test."""
    summary = {
        "nota": (
            "PRUEBA TECNICA con datos sinteticos. No es un resultado clinico."
            if COMPONENTS_ARE_PROVISIONAL
            else "Ejecucion con los componentes reales."
        ),
        "train_por_ronda": {
            r: dict(m) for r, m in result.train_metrics_clientapp.items()
        },
        "val_por_ronda": {
            r: dict(m) for r, m in result.evaluate_metrics_clientapp.items()
        },
        "test_final": dict(test_metrics),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Punto de entrada del servidor."""
    config = dict(context.run_config)
    num_rounds = int(config["num-server-rounds"])
    output_dir = Path(config["output-dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    strategy = FedAvg(
        min_train_nodes=NUM_CLIENTS,
        min_evaluate_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        train_metrics_aggr_fn=metrics_per_client,
        evaluate_metrics_aggr_fn=metrics_per_client,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        evaluate_config=ConfigRecord({"split": "val"}),
        evaluate_fn=save_round_checkpoint,
    )
    check_rounds(result, sent_fingerprints, num_rounds)

    # Test se usa una sola vez, con el modelo global final.
    test_metrics = evaluate_on_test(strategy, grid, result.arrays, num_rounds)

    final_path = output_dir / "global_final.keras"
    save_global_model(model, result.arrays, final_path)
    check_model_reloads(final_path, result.arrays)
    save_metrics(output_dir / "metrics.json", result, test_metrics)

    kind = "PRUEBA TECNICA SINTETICA" if COMPONENTS_ARE_PROVISIONAL else "datos reales"
    print(f"[{kind}] Modelo global y métricas guardados en {output_dir}")
