"""Componente PROVISIONAL para probar la federación sin datos reales.

Implementa las cuatro funciones del contrato con datos sintéticos y un modelo
pequeño. Sus resultados son pruebas técnicas de coordinación. No dicen nada
sobre la detección de neumonía.

Este archivo se retira de la ejecución real cuando Ariana entregue sus
funciones (ver pneumonia_fl/components.py).
"""

import zlib

import keras
import numpy as np
import tensorflow as tf

IMAGE_SHAPE = (224, 224, 3)

# Tamaños distintos a propósito: así se nota la ponderación de FedAvg.
# client_1 tiene el doble de ejemplos de train, por lo tanto pesa el doble.
SYNTHETIC_EXAMPLES = {
    "client_1": {"train": 48, "val": 16, "test": 16},
    "client_2": {"train": 24, "val": 8, "test": 8},
}


def load_data(client_id, split, data_root, manifest_root, batch_size):
    """Devuelve (dataset, num_examples) con imágenes y etiquetas sintéticas.

    Relación con ML clásico: equivale a obtener X e y. La diferencia es que
    tf.data entrega los datos por lotes, en lugar de una sola matriz en memoria.

    data_root y manifest_root se ignoran aquí. Existen porque el contrato los
    exige para la versión real de Ariana.
    """
    num_examples = SYNTHETIC_EXAMPLES[client_id][split]

    # Semilla estable por cliente y partición: cada ejecución genera los mismos datos.
    seed = zlib.crc32(f"{client_id}/{split}".encode())
    rng = np.random.default_rng(seed)

    # Mitad de etiquetas 0 y mitad 1, con forma (N, 1) como pide el contrato.
    labels = (np.arange(num_examples) % 2).astype("float32").reshape(-1, 1)

    # Ruido más un desplazamiento de brillo en la clase 1. Así existe una señal
    # simple que el modelo puede aprender.
    images = rng.normal(size=(num_examples, *IMAGE_SHAPE)).astype("float32")
    images += labels.reshape(-1, 1, 1, 1)

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if split == "train":
        dataset = dataset.shuffle(num_examples, seed=seed)
    return dataset.batch(batch_size), num_examples


def build_model(config):
    """Construye y compila un modelo pequeño con la misma interfaz que ResNet50.

    Entrada (224, 224, 3) y una salida sigmoide. Es una regresión logística con
    una capa oculta sobre el brillo promedio de cada canal.

    El orden de las capas define el orden de la lista de pesos. Ese orden debe
    ser idéntico en el servidor y en los clientes para poder promediar.
    """
    keras.utils.set_random_seed(int(config["seed"]))
    model = keras.Sequential(
        [
            keras.Input(shape=IMAGE_SHAPE),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    # compile() fija el optimizador, la pérdida y las métricas.
    # En ML clásico esto suele venir decidido dentro del estimador.
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=float(config["learning-rate"])),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_local(model, train_data, epochs):
    """Entrena el modelo recibido y devuelve las métricas de la última época.

    Relación con ML clásico: es el fit(). No reconstruye el modelo, por lo que
    el entrenamiento parte de los pesos globales que ya trae cargados.
    """
    history = model.fit(train_data, epochs=epochs, verbose=0)
    return {name: float(values[-1]) for name, values in history.history.items()}


def evaluate_local(model, eval_data):
    """Evalúa sin modificar pesos. Devuelve un diccionario con loss y métricas.

    Relación con ML clásico: es el score() sobre un conjunto que no se usó
    para entrenar.
    """
    results = model.evaluate(eval_data, verbose=0, return_dict=True)
    return {name: float(value) for name, value in results.items()}
