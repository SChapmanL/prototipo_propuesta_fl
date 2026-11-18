"""Paso 4: la ResNet50 del prototipo.

Una sola definicion del modelo. Los dos clientes y el servidor llaman a esta
misma funcion, asi que la arquitectura y el ORDEN de los pesos son identicos.
Ese orden es lo que permite promediar con FedAvg.

Relacion con ML clasico: build_model equivale a crear el estimador y fijar
sus hiperparametros, antes de ver un solo dato.
"""

import keras

from pneumonia_fl.metrics import metricas

IMAGE_SHAPE = (224, 224, 3)


def build_model(config):
    """Contrato de docs/07. Devuelve un modelo Keras ya compilado.

    Decisiones, todas tomadas de docs/05:

    - ResNet50 con pesos de ImageNet y sin su clasificador original.
    - Base congelada. Solo aprende la cabeza, que tiene 2 049 parametros.
      Con la base congelada, BatchNormalization trabaja en modo inferencia y
      sus medias y varianzas no cambian, asi que promediarlas es inocuo.
    - Promedio global y una salida sigmoide, con forma (lote, 1).
    - SGD sin momentum. Sin momentum no hay estado del optimizador que
      perder cuando el cliente reconstruye el modelo en cada ronda.

    set_random_seed se llama primero para que la cabeza se inicialice igual
    en todas las llamadas. Sin eso, cada cliente partiria de pesos distintos.
    """
    keras.utils.set_random_seed(int(config["seed"]))

    base = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=IMAGE_SHAPE)
    base.trainable = False

    modelo = keras.Sequential([
        keras.Input(shape=IMAGE_SHAPE),
        base,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    modelo.compile(
        optimizer=keras.optimizers.SGD(learning_rate=float(config["learning-rate"])),
        loss="binary_crossentropy",
        metrics=metricas(),
    )
    return modelo
