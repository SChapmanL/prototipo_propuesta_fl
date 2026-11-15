"""Verificación independiente de una ejecución federada ya terminada.

No usa Flower. Lee los modelos guardados en outputs/federation y comprueba:
1. Que global_final.keras se recarga y coincide con el de la última ronda.
2. Que el modelo global de la ronda 1 es el promedio ponderado por ejemplos
   de train de los pesos que cada cliente obtiene al entrenar por separado
   desde global_round_0.keras.

Uso:  python scripts/check_federation.py
"""

import tomllib
from pathlib import Path

import keras
import numpy as np

from pneumonia_fl.components import build_model, load_data, train_local

config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
config = config["tool"]["flwr"]["app"]["config"]
output_dir = Path(config["output-dir"])


def saved_weights(name):
    return keras.models.load_model(output_dir / name).get_weights()


def max_difference(weights_a, weights_b):
    """Mayor diferencia absoluta entre dos listas de pesos. 0.0 significa iguales."""
    return max(
        float(np.max(np.abs(a - b))) for a, b in zip(weights_a, weights_b, strict=True)
    )


# 1. El modelo final se recarga y es el de la última ronda.
last_round = f"global_round_{config['num-server-rounds']}.keras"
print("diferencia final vs última ronda:", max_difference(saved_weights("global_final.keras"), saved_weights(last_round)))

# 2. FedAvg manual de la ronda 1.
local_weights, num_examples = [], []
for client_id in ("client_1", "client_2"):
    # Cada cliente parte de los MISMOS pesos globales iniciales.
    # Se repite el orden del cliente Flower: construir (fija la semilla) y
    # luego cargar los pesos globales.
    model = build_model(config)
    model.set_weights(saved_weights("global_round_0.keras"))
    train_data, n = load_data(
        client_id, "train", config["data-root"], config["manifest-root"], config["batch-size"]
    )
    train_local(model, train_data, config["local-epochs"])
    local_weights.append(model.get_weights())
    num_examples.append(n)

total = sum(num_examples)
manual_average = [
    sum(w * (n / total) for w, n in zip(layer, num_examples))
    for layer in zip(*local_weights)
]
round_1 = saved_weights("global_round_1.keras")
print("ejemplos de train:", num_examples)
print("diferencia FedAvg ponderado vs ronda 1:", max_difference(manual_average, round_1))

# Control negativo: un promedio simple NO debe coincidir.
simple_average = [sum(layer) / len(layer) for layer in zip(*local_weights)]
print("diferencia promedio simple vs ronda 1:", max_difference(simple_average, round_1))
