"""Controles del Paso 3: comprueba lo que load_data entrega al modelo.

Recorre una particion entera y verifica forma, tipo, valores finitos,
etiquetas y numero de ejemplos. Guarda ademas una figura de control con las
imagenes tal como las recibe ResNet50, devueltas a 0-255 para poder mirarlas.

La figura va a outputs/, que esta excluido de Git. No la subas.

Uso (desde la raiz del repositorio):
    python scripts/check_load_data.py
    python scripts/check_load_data.py --client client_2 --split train
"""
import argparse
import sys
from pathlib import Path

# Permite ejecutar como "python scripts/check_load_data.py" ademas de
# "python -m scripts.check_load_data". Al lanzar un script suelto, Python
# solo pone scripts/ en la ruta de busqueda, no la raiz del repositorio.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pneumonia_fl.data.dicom import CANALES, IMAGE_SIZE, load_data

# Medias que resta keras.applications.resnet.preprocess_input, en orden BGR.
MEDIAS_BGR = np.array([103.939, 116.779, 123.68], dtype="float32")


def deshace_preprocess(lote):
    """Devuelve el lote a 0-255 RGB, solo para poder visualizarlo."""
    return (lote.numpy() + MEDIAS_BGR)[..., ::-1]


def recorre(dataset, num_examples, batch_size):
    """Recorre el dataset completo y acumula los controles."""
    vistos = 0
    lotes = 0
    etiquetas = []
    todo_finito = True
    formas_ok = True
    minimo, maximo = np.inf, -np.inf

    for imagenes, etiqueta in dataset:
        lotes += 1
        vistos += int(imagenes.shape[0])
        etiquetas += etiqueta.numpy().ravel().tolist()
        todo_finito &= bool(tf.reduce_all(tf.math.is_finite(imagenes)))
        formas_ok &= (imagenes.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, CANALES)
                      and etiqueta.shape[1:] == (1,))
        minimo = min(minimo, float(tf.reduce_min(imagenes)))
        maximo = max(maximo, float(tf.reduce_max(imagenes)))

    etiquetas = np.array(etiquetas)
    controles = [
        (vistos == num_examples,
         "Imagenes recorridas", f"{vistos} (num_examples {num_examples})"),
        (lotes == -(-num_examples // batch_size),
         "Numero de lotes", f"{lotes} (esperado {-(-num_examples // batch_size)})"),
        (formas_ok,
         "Formas", f"imagen ({IMAGE_SIZE}, {IMAGE_SIZE}, {CANALES}), etiqueta (1,)"),
        (todo_finito, "Valores finitos", "sin NaN ni infinitos"),
        (set(np.unique(etiquetas)) <= {0.0, 1.0},
         "Etiquetas 0.0 o 1.0", f"distintas: {sorted(set(etiquetas.tolist()))}"),
        (etiquetas.sum() * 2 == len(etiquetas),
         "Balance de clases", f"0: {int((etiquetas == 0).sum())}, "
                              f"1: {int((etiquetas == 1).sum())}"),
    ]
    return controles, (minimo, maximo)


def figura(dataset, ruta, titulo, por_clase=4):
    """Dibuja por_clase imagenes de cada clase, tal como entran al modelo.

    Recorre hasta reunir las dos clases. Hace falta porque val y test no se
    mezclan y el manifiesto viene ordenado por etiqueta, asi que los primeros
    lotes serian todos de la misma clase.
    """
    elegidas = {0: [], 1: []}
    for imagenes, etiquetas in dataset.unbatch().batch(16):
        visibles = deshace_preprocess(imagenes)
        for imagen, etiqueta in zip(visibles, etiquetas.numpy().ravel()):
            grupo = elegidas[int(etiqueta)]
            if len(grupo) < por_clase:
                grupo.append(imagen)
        if all(len(g) >= por_clase for g in elegidas.values()):
            break

    muestras = [(0, i) for i in elegidas[0]] + [(1, i) for i in elegidas[1]]
    fig, ejes = plt.subplots(2, por_clase, figsize=(3.5 * por_clase, 7.5))
    for (clase, imagen), eje in zip(muestras, ejes.ravel()):
        eje.imshow(np.clip(imagen[:, :, 0], 0, 255), cmap="gray", vmin=0, vmax=255)
        eje.set_title(f"clase {clase}", fontsize=11)
        eje.axis("off")
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=80)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--client", default="client_1")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, num_examples = load_data(
        args.client, args.split, "data/raw/dicom", "data/manifests", args.batch_size)

    controles, (minimo, maximo) = recorre(dataset, num_examples, args.batch_size)

    print(f"{args.client} / {args.split}\n")
    ancho = max(len(t) for _, t, _ in controles)
    for pasa, titulo, detalle in controles:
        print(f'{"OK   " if pasa else "FALLA"}  {titulo.ljust(ancho)}  {detalle}')
    print(f"\nRango tras preprocess_input: [{minimo:.1f}, {maximo:.1f}]")
    print("Es el esperado al restar las medias de ImageNet a una imagen de 0-255.")

    Path("outputs").mkdir(exist_ok=True)
    ruta = Path("outputs") / f"control_load_data_{args.client}_{args.split}.png"
    figura(dataset, ruta,
           f"Imagenes tal como las recibe ResNet50 ({args.client} / {args.split}).\n"
           "Devueltas a 0-255 solo para verlas. 224x224, tres canales iguales.")
    print(f"\nFigura: {ruta}")

    if any(not pasa for pasa, _, _ in controles):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
