"""Métricas del modelo global sobre los datos de AMBOS clientes juntos.

Problema: el servidor no ve imágenes ni predicciones individuales. Solo
recibe números de cada cliente. Promediar el AUROC de dos clientes no da el
AUROC del conjunto. Entonces, ¿cómo se obtiene una métrica conjunta?

Solución: cada cliente envía resúmenes que SÍ se pueden sumar.

- loss y accuracy son promedios por imagen. Se combinan ponderando por el
  número de imágenes de cada cliente. Es exacto.
- tp, tn, fp y fn son conteos. Se suman. Sensibilidad, especificidad y
  precisión se calculan después, sobre los totales. Es exacto.
- AUROC y AUPRC necesitan el orden de las puntuaciones. Cada cliente envía
  un histograma: cuántas imágenes positivas y cuántas negativas caen en cada
  uno de NUM_BINS intervalos de la salida sigmoide. Los histogramas se
  suman, y del total se calcula el AUROC y el AUPRC. La aproximación es la
  resolución del intervalo (1/NUM_BINS). Keras también aproxima, con 200
  umbrales.

Limitación de privacidad: un histograma fino de pocas imágenes revela la
puntuación y la etiqueta de cada imagen, aunque no la imagen ni el paciente.
En un despliegue real se usarían intervalos más gruesos o agregación segura.

Relación con ML clásico: es lo mismo que calcular las métricas sobre la
concatenación de los conjuntos de validación de ambos clientes, pero sin
juntar los datos en un solo lugar.
"""

import numpy as np

from pneumonia_fl.metrics import sensibilidad_especificidad

NUM_BINS = 1000


def score_histograms(model, dataset):
    """Histogramas de puntuaciones por clase. Se ejecuta en el CLIENTE.

    Recorre el dataset una vez, sin modificar pesos. Devuelve dos listas de
    enteros de largo NUM_BINS: positivos (etiqueta 1) y negativos (etiqueta 0).
    """
    positives = np.zeros(NUM_BINS, dtype="int64")
    negatives = np.zeros(NUM_BINS, dtype="int64")
    for images, labels in dataset:
        scores = np.asarray(model(images, training=False)).ravel()
        labels = np.asarray(labels).ravel()
        bins = np.clip((scores * NUM_BINS).astype("int64"), 0, NUM_BINS - 1)
        positives += np.bincount(bins[labels == 1], minlength=NUM_BINS)
        negatives += np.bincount(bins[labels == 0], minlength=NUM_BINS)
    return positives.tolist(), negatives.tolist()


def auroc_from_histograms(positives, negatives):
    """Probabilidad de que un positivo puntúe más alto que un negativo.

    Los empates dentro del mismo intervalo cuentan como medio acierto.
    """
    positives = np.asarray(positives, dtype="float64")
    negatives = np.asarray(negatives, dtype="float64")
    total_pos, total_neg = positives.sum(), negatives.sum()
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    negatives_below = np.cumsum(negatives) - negatives
    wins = np.sum(positives * (negatives_below + 0.5 * negatives))
    return float(wins / (total_pos * total_neg))


def auprc_from_histograms(positives, negatives):
    """Precisión promedio: recorre los umbrales de mayor a menor puntuación."""
    positives = np.asarray(positives, dtype="float64")[::-1]
    negatives = np.asarray(negatives, dtype="float64")[::-1]
    total_pos = positives.sum()
    if total_pos == 0:
        return float("nan")
    tp = np.cumsum(positives)
    fp = np.cumsum(negatives)
    has_predictions = (tp + fp) > 0
    precision = np.where(has_predictions, tp / np.maximum(tp + fp, 1), 0.0)
    return float(np.sum(precision * positives) / total_pos)


def pooled_metrics(per_client):
    """Combina las métricas de evaluación de varios clientes.

    per_client: lista de diccionarios con las claves sin prefijo de partición:
    num-examples, loss, accuracy, tp, tn, fp, fn, hist-pos y hist-neg.
    """
    n = [float(m["num-examples"]) for m in per_client]
    total = sum(n)
    pooled = {
        "num-examples": int(total),
        "loss": sum(w * m["loss"] for w, m in zip(n, per_client)) / total,
        "accuracy": sum(w * m["accuracy"] for w, m in zip(n, per_client)) / total,
    }
    # El componente provisional no calcula la matriz de confusión.
    if all(name in m for m in per_client for name in ("tp", "tn", "fp", "fn")):
        for name in ("tp", "tn", "fp", "fn"):
            pooled[name] = sum(float(m[name]) for m in per_client)
        pooled.update(sensibilidad_especificidad(pooled))
        predicted_pos = pooled["tp"] + pooled["fp"]
        pooled["precision"] = pooled["tp"] / predicted_pos if predicted_pos else 0.0

    hist_pos = np.sum([m["hist-pos"] for m in per_client], axis=0)
    hist_neg = np.sum([m["hist-neg"] for m in per_client], axis=0)
    pooled["auroc"] = auroc_from_histograms(hist_pos, hist_neg)
    pooled["auprc"] = auprc_from_histograms(hist_pos, hist_neg)
    return pooled


def client_metrics(record, client, split):
    """Extrae del registro del servidor las métricas de un cliente sin prefijos.

    El servidor guarda claves como client_1_val_auroc. Esto devuelve
    {"auroc": ..., "num-examples": ...} para client_1 y val.
    """
    prefix = f"{client}_"
    result = {}
    for key, value in record.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if name.startswith(f"{split}_"):
            name = name[len(split) + 1:]
        result[name] = value
    return result


def pooled_from_record(record, clients, split):
    """Métricas conjuntas a partir del registro por cliente de una ronda."""
    return pooled_metrics([client_metrics(record, c, split) for c in clients])
