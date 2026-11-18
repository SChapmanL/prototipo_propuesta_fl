"""Metricas de la clasificacion binaria.

Estan en un solo sitio para que el modelo, el entrenamiento local y la
evaluacion usen exactamente las mismas. Todas devuelven numeros, que es lo
unico que Flower transporta como metrica.

AUROC y AUPRC no dependen del umbral. La matriz de confusion si, y se
calcula en el umbral 0,5. Ese umbral es el de arranque y solo se deberia
mover con datos de validacion.
"""

import keras

UMBRAL = 0.5


def metricas():
    """Lista nueva de metricas. Keras las modifica, por eso no se reutilizan."""
    return [
        keras.metrics.BinaryAccuracy(name="accuracy", threshold=UMBRAL),
        keras.metrics.AUC(name="auroc"),
        keras.metrics.AUC(name="auprc", curve="PR"),
        # Con estos cuatro se obtienen sensibilidad y especificidad despues.
        # docs/07 permite enviarlos por Flower como cuatro numeros sueltos.
        keras.metrics.TruePositives(name="tp", thresholds=UMBRAL),
        keras.metrics.TrueNegatives(name="tn", thresholds=UMBRAL),
        keras.metrics.FalsePositives(name="fp", thresholds=UMBRAL),
        keras.metrics.FalseNegatives(name="fn", thresholds=UMBRAL),
    ]


def sensibilidad_especificidad(resultados):
    """Deriva sensibilidad y especificidad de la matriz de confusion.

    Se calcula fuera del modelo para no promediar razones entre clientes.
    Sumar tp, tn, fp y fn de varios clientes si es correcto. Promediar sus
    sensibilidades no lo es.
    """
    tp, tn = resultados["tp"], resultados["tn"]
    fp, fn = resultados["fp"], resultados["fn"]
    return {
        "sensibilidad": tp / (tp + fn) if (tp + fn) else 0.0,
        "especificidad": tn / (tn + fp) if (tn + fp) else 0.0,
    }
