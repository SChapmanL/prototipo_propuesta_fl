"""Paso 4: entrenamiento y evaluacion locales.

Las dos funciones trabajan sobre un modelo que YA tiene sus pesos. En la
federacion esos pesos son los globales que envio el servidor. Ninguna de las
dos reconstruye el modelo ni lo reinicializa.

Relacion con ML clasico: train_local es el fit y evaluate_local es el score
sobre datos que no se usaron para entrenar.
"""


def _a_float(resultados):
    """Flower solo transporta numeros. Keras devuelve tensores."""
    return {nombre: float(valor) for nombre, valor in resultados.items()}


def train_local(model, train_data, epochs):
    """Entrena el modelo recibido y devuelve las metricas de la ultima epoca.

    No toca los pesos de entrada mas alla de lo que hace el entrenamiento,
    asi que el modelo parte de los pesos globales que ya trae cargados.
    """
    historia = model.fit(train_data, epochs=epochs, verbose=0)
    return _a_float({nombre: valores[-1]
                     for nombre, valores in historia.history.items()})


def evaluate_local(model, eval_data):
    """Evalua sin modificar pesos. Devuelve loss y las metricas del modelo."""
    return _a_float(model.evaluate(eval_data, verbose=0, return_dict=True))
