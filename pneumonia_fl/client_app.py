"""Cliente Flower: adaptador entre el servidor y el entrenamiento local.

Existe UNA sola implementación. Flower la ejecuta una vez por cada cliente
simulado y le entrega un identificador distinto. El código es el mismo y solo
cambian los datos.

Este archivo no sabe nada de DICOM ni de ResNet50. Solo llama a las cuatro
funciones del contrato (pneumonia_fl/components.py).

Flujo de un mensaje:
    servidor --(pesos globales)--> cliente --(pesos locales + métricas)--> servidor
"""

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pneumonia_fl.components import (
    build_model,
    evaluate_local,
    load_data,
    train_local,
)
from pneumonia_fl.weights import weights_fingerprint

app = ClientApp()


def client_number(context):
    """Número de cliente (1 o 2). La simulación de Flower numera desde 0."""
    return int(context.node_config["partition-id"]) + 1


def model_with_global_weights(msg, context):
    """Construye el modelo y reemplaza sus pesos por los globales recibidos.

    Devuelve también la huella de los pesos recibidos. El servidor la usa para
    comprobar que ambos clientes partieron del mismo punto.
    """
    keras.backend.clear_session()  # libera modelos de rondas anteriores
    model = build_model(dict(context.run_config))
    global_weights = msg.content["arrays"].to_numpy_ndarrays()
    model.set_weights(global_weights)
    return model, weights_fingerprint(global_weights)


def load_split(context, split):
    """Carga train, val o test de este cliente con la configuración común."""
    config = context.run_config
    return load_data(
        client_id=f"client_{client_number(context)}",
        split=split,
        data_root=config["data-root"],
        manifest_root=config["manifest-root"],
        batch_size=int(config["batch-size"]),
    )


def reply(msg, context, num_examples, metrics, model=None):
    """Arma la respuesta al servidor.

    num-examples es el peso de este cliente en FedAvg. client-number permite
    reportar las métricas por cliente. Si se pasa un modelo, se adjuntan sus
    pesos (solo ocurre al entrenar).
    """
    metrics = {
        **metrics,
        "num-examples": num_examples,
        "client-number": client_number(context),
    }
    content = {"metrics": MetricRecord(metrics)}
    if model is not None:
        content["arrays"] = ArrayRecord(model.get_weights())
    return Message(content=RecordDict(content), reply_to=msg)


@app.train()
def train(msg: Message, context: Context):
    """Una ronda local: recibir pesos globales, entrenar y devolver pesos."""
    model, fingerprint = model_with_global_weights(msg, context)
    train_data, num_examples = load_split(context, "train")

    local_metrics = train_local(
        model, train_data, epochs=int(context.run_config["local-epochs"])
    )

    metrics = {f"train_{name}": value for name, value in local_metrics.items()}
    metrics["received-weights-fingerprint"] = fingerprint
    return reply(msg, context, num_examples, metrics, model=model)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evalúa el modelo global con datos locales. No devuelve pesos.

    El servidor indica la partición en el mensaje: "val" durante las rondas y
    "test" una sola vez al final.
    """
    split = msg.content["config"]["split"]
    model, _ = model_with_global_weights(msg, context)
    eval_data, num_examples = load_split(context, split)

    local_metrics = evaluate_local(model, eval_data)

    metrics = {f"{split}_{name}": value for name, value in local_metrics.items()}
    return reply(msg, context, num_examples, metrics)
