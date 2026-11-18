"""Paso 3: lectura de DICOM y carga de datos para el modelo local.

Politica de intensidad, decidida a partir de medir los 800 archivos reales:

1. Aplicar RescaleSlope y RescaleIntercept. No es opcional: 7 de los 800
   archivos traen slope 0,684 e intercept 200,0 en lugar de la identidad.
2. Recortar al rango p1-p99 de ESA imagen y escalar a 0-255.
3. Invertir MONOCHROME1 al final, como 255 - x.
4. Redimensionar a 224x224 con antialias y repetir en 3 canales.
5. Aplicar resnet.preprocess_input una sola vez, aqui y en ningun otro lugar.

Por que el recorte por percentiles y no otra cosa:

- BitsStored no es uniforme: 12 bits en el 73 % de los archivos, 14 en el
  18 % y 15 en el 9 %. Dividir por 2**bits-1 deja brillos distintos segun el
  equipo.
- Hay saturacion. En las imagenes de 12 bits, 17 de cada 20 llegan al maximo
  teorico, asi que un min-max simple deja que unos pocos pixeles extremos se
  lleven el rango y aplasten el contraste del pulmon.
- apply_voi_lut seria lo clinicamente ortodoxo, pero solo el 21 % de los
  archivos tiene WindowCenter y WindowWidth, y no hay ninguna VOILUTSequence.
  Usarlo supondria dos reglas distintas sobre el mismo conjunto, y como esa
  presencia depende del equipo, podria introducir una senal espuria.

Medido sobre 30 imagenes, el contraste aprovechado (desviacion tipica) es
57,7 con percentiles, 27,7 con min-max y 21,5 dividiendo por 2**bits-1.

El 1 y el 99 son una eleccion de arranque. Se pueden ajustar, pero deben
quedar fijos antes de entrenar y ser los mismos en los dos clientes.

Este modulo no depende de Flower y se puede ejecutar solo:
    python -m pneumonia_fl.data.dicom
"""

import csv
import zlib
from pathlib import Path

import keras
import numpy as np
import pydicom
import tensorflow as tf

IMAGE_SIZE = 224
CANALES = 3
PERCENTIL_BAJO = 1
PERCENTIL_ALTO = 99


def leer_dicom(ruta):
    """Devuelve la imagen como float32 (alto, ancho) en el rango 0-255.

    Funcion pura de numpy y pydicom. No usa TensorFlow, asi que se puede
    probar sola. Implementa los pasos 1 a 3 de la politica.
    """
    ds = pydicom.dcmread(str(ruta))

    # 1. Valor real del pixel segun la cabecera.
    pixeles = ds.pixel_array.astype(np.float32)
    pixeles = pixeles * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    # 2. Recorte por percentiles y escalado a 0-255.
    bajo, alto = np.percentile(pixeles, [PERCENTIL_BAJO, PERCENTIL_ALTO])
    if alto <= bajo:  # imagen constante, no deberia ocurrir tras la validacion
        return np.zeros_like(pixeles)
    pixeles = np.clip((pixeles - bajo) / (alto - bajo), 0.0, 1.0) * 255.0

    # 3. MONOCHROME1 guarda los valores invertidos: su minimo se ve blanco.
    # Invertir aqui, despues de escalar, da el mismo resultado que invertir
    # antes y evita tener que elegir que maximo usar.
    if ds.PhotometricInterpretation == "MONOCHROME1":
        pixeles = 255.0 - pixeles

    return pixeles.astype(np.float32)


def _decodifica(ruta_tensor):
    """Puente entre tf.data y leer_dicom. pydicom no es una operacion de TF."""
    return leer_dicom(ruta_tensor.numpy().decode("utf-8"))


def _a_tensor(ruta, etiqueta):
    """Pasos 4 de la politica: tamano fijo y tres canales."""
    imagen = tf.py_function(_decodifica, [ruta], tf.float32)
    imagen = tf.ensure_shape(imagen, [None, None])
    imagen = imagen[..., tf.newaxis]
    # antialias importa: se reduce de miles de pixeles a 224.
    imagen = tf.image.resize(imagen, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
    imagen = tf.repeat(imagen, CANALES, axis=-1)
    imagen = tf.ensure_shape(imagen, [IMAGE_SIZE, IMAGE_SIZE, CANALES])
    return imagen, etiqueta


def _lee_manifiesto(client_id, split, manifest_root):
    ruta = Path(manifest_root) / client_id / f"{split}.csv"
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe {ruta}. Genere las particiones con "
            "python -m pneumonia_fl.data.split"
        )
    with ruta.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_data(client_id, split, data_root, manifest_root, batch_size):
    """Contrato de docs/07. Devuelve (dataset, num_examples).

    num_examples cuenta IMAGENES, no lotes. Es el peso del cliente en FedAvg.
    Solo se mezcla train. La mezcla usa una semilla estable derivada del
    cliente y la particion, asi el orden es reproducible entre ejecuciones.

    dicom_path del manifiesto se resuelve respecto de data_root.
    """
    filas = _lee_manifiesto(client_id, split, manifest_root)
    num_examples = len(filas)

    raiz = Path(data_root)
    rutas = [str(raiz / fila["dicom_path"]) for fila in filas]
    # Forma (N, 1) float32, igual que la salida sigmoide del modelo.
    etiquetas = np.array([[float(fila["label"])] for fila in filas], dtype="float32")

    dataset = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    dataset = dataset.map(_a_tensor, num_parallel_calls=tf.data.AUTOTUNE)
    # Guarda en memoria las imagenes ya decodificadas. Decodificar un DICOM
    # grande es lo mas caro de todo el proceso y no cambia entre epocas.
    dataset = dataset.cache()

    if split == "train":
        semilla = zlib.crc32(f"{client_id}/{split}".encode())
        dataset = dataset.shuffle(num_examples, seed=semilla,
                                  reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    # Paso 5. Una sola vez, sobre el lote ya formado. El cliente Flower no
    # vuelve a normalizar.
    dataset = dataset.map(
        lambda x, y: (keras.applications.resnet.preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE), num_examples


def main():
    """Comprobacion rapida sobre una particion pequena."""
    dataset, num_examples = load_data(
        "client_1", "val", "data/raw/dicom", "data/manifests", batch_size=8)
    imagenes, etiquetas = next(iter(dataset))
    print(f"num_examples (imagenes) : {num_examples}")
    print(f"forma del lote          : {imagenes.shape}  {imagenes.dtype}")
    print(f"forma de las etiquetas  : {etiquetas.shape}  {etiquetas.dtype}")
    print(f"valores finitos         : {bool(tf.reduce_all(tf.math.is_finite(imagenes)))}")
    print(f"rango tras preprocess   : [{float(tf.reduce_min(imagenes)):.1f}, "
          f"{float(tf.reduce_max(imagenes)):.1f}]")
    print(f"etiquetas distintas     : {sorted(set(etiquetas.numpy().ravel().tolist()))}")


if __name__ == "__main__":
    main()
