"""Control visual del Paso 1: comprueba que los DICOM no salen invertidos.

Genera dos figuras en outputs/:

    control_visual_muestra.png    8 imagenes, 4 por clase, con los dos tipos
                                  de interpretacion fotometrica
    control_visual_inversion.png  la misma imagen sin corregir y corregida,
                                  para ver que la correccion es la correcta

IMPORTANTE. El escalado de intensidad que usa este script es SOLO para mirar
las imagenes. No es la politica de preprocesamiento del modelo. Esa se define
en el Paso 3, dentro de pneumonia_fl/data/, e incluye la decision de ventana
o normalizacion, que todavia no esta tomada (ver docs/05).

Cuando exista el lector definitivo del Paso 3, este script debe importarlo en
lugar de repetir la lectura.

Las figuras no llevan identificadores de pacientes. outputs/ esta excluido
de Git y las figuras no se deben subir.

Uso (desde la raiz del repositorio):
    python scripts/preview_sample.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana, solo archivos
import matplotlib.pyplot as plt
import numpy as np
import pydicom

MANIFEST = Path("data/manifests/download_manifest.csv")
LOG = Path("data/raw/dicom/download_validation.csv")
OUTPUT_DIR = Path("outputs")
POR_CLASE = 4


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def leer_dicom(path):
    """Devuelve (pixeles_corregidos, pixeles_crudos, interpretacion).

    MONOCHROME1 guarda los valores invertidos: el minimo se ve blanco. Para
    verlo como una radiografia normal hay que invertirlo.
    """
    ds = pydicom.dcmread(path)
    crudo = ds.pixel_array.astype(np.float32)
    photometric = ds.PhotometricInterpretation
    corregido = crudo.max() - crudo if photometric == "MONOCHROME1" else crudo
    return corregido, crudo, photometric


def para_mostrar(pixeles):
    """Escala a 0-255 solo para dibujar. No es el preprocesamiento del modelo."""
    minimo, maximo = float(pixeles.min()), float(pixeles.max())
    if maximo <= minimo:
        return np.zeros_like(pixeles)
    return (pixeles - minimo) / (maximo - minimo) * 255.0


def elegir(filas, photometric_por_archivo):
    """Elige POR_CLASE imagenes de cada clase, mezclando MONOCHROME1 y 2."""
    elegidas = []
    for label in ("0", "1"):
        del_label = [r for r in filas if r["label"] == label]
        mono1 = [r for r in del_label if photometric_por_archivo.get(r["file_name"]) == "MONOCHROME1"]
        mono2 = [r for r in del_label if photometric_por_archivo.get(r["file_name"]) == "MONOCHROME2"]
        # La mitad de cada tipo cuando existan los dos.
        mitad = POR_CLASE // 2
        grupo = mono1[:mitad] + mono2[:mitad]
        # Si falta alguno de los dos tipos, se completa con lo que haya.
        if len(grupo) < POR_CLASE:
            resto = [r for r in del_label if r not in grupo]
            grupo += resto[: POR_CLASE - len(grupo)]
        elegidas.append(grupo)
    return elegidas


def figura_muestra(por_clase, ruta):
    """Rejilla 2x4: una fila por clase."""
    fig, ejes = plt.subplots(2, POR_CLASE, figsize=(4 * POR_CLASE, 9))
    for fila, grupo in enumerate(por_clase):
        for columna, registro in enumerate(grupo):
            corregido, _, photometric = leer_dicom(Path(registro["dicom_path"]))
            eje = ejes[fila][columna]
            eje.imshow(para_mostrar(corregido), cmap="gray", vmin=0, vmax=255)
            eje.set_title(
                f'clase {registro["label"]}  ({photometric})\n'
                f'{corregido.shape[0]}x{corregido.shape[1]}  vista {registro["view"]}',
                fontsize=10,
            )
            eje.axis("off")
    fig.suptitle(
        "Control visual de la muestra. Los pulmones deben verse oscuros y "
        "los huesos claros.\nEscalado solo para visualizacion, no es el "
        "preprocesamiento del modelo.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=80)
    plt.close(fig)


def figura_inversion(por_clase, photometric_por_archivo, ruta):
    """Compara sin corregir y corregida, para un MONOCHROME1 y un MONOCHROME2."""
    todas = por_clase[0] + por_clase[1]
    ejemplos = []
    for tipo in ("MONOCHROME1", "MONOCHROME2"):
        for registro in todas:
            if photometric_por_archivo.get(registro["file_name"]) == tipo:
                ejemplos.append((tipo, registro))
                break

    fig, ejes = plt.subplots(len(ejemplos), 2, figsize=(9, 4.5 * len(ejemplos)), squeeze=False)
    for fila, (tipo, registro) in enumerate(ejemplos):
        corregido, crudo, _ = leer_dicom(Path(registro["dicom_path"]))
        for columna, (pixeles, titulo) in enumerate(
            ((crudo, "como viene en el archivo"), (corregido, "despues de corregir"))
        ):
            eje = ejes[fila][columna]
            eje.imshow(para_mostrar(pixeles), cmap="gray", vmin=0, vmax=255)
            eje.set_title(f"{tipo}\n{titulo}", fontsize=11)
            eje.axis("off")
    fig.suptitle(
        "Comprobacion de la inversion. En MONOCHROME1 las dos columnas deben "
        "diferir.\nEn MONOCHROME2 deben ser identicas, porque no se invierte.",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=80)
    plt.close(fig)


def main():
    for path in (MANIFEST, LOG):
        if not path.is_file():
            raise SystemExit(f"No existe {path}. Ejecute desde la raiz del repositorio.")

    filas = read_csv(MANIFEST)
    photometric_por_archivo = {
        r["file_name"]: r["photometric"] for r in read_csv(LOG) if r["status"] == "ok"
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    por_clase = elegir(filas, photometric_por_archivo)

    # Resumen numerico, sin identificadores.
    print("Imagenes elegidas para el control visual\n")
    for grupo in por_clase:
        for registro in grupo:
            corregido, crudo, photometric = leer_dicom(Path(registro["dicom_path"]))
            invertida = "si" if photometric == "MONOCHROME1" else "no"
            print(
                f'  clase {registro["label"]}  {photometric:<12} '
                f'{corregido.shape[0]}x{corregido.shape[1]:<6} '
                f'crudo [{crudo.min():.0f}, {crudo.max():.0f}]  '
                f'invertida: {invertida}'
            )

    muestra = OUTPUT_DIR / "control_visual_muestra.png"
    inversion = OUTPUT_DIR / "control_visual_inversion.png"
    figura_muestra(por_clase, muestra)
    figura_inversion(por_clase, photometric_por_archivo, inversion)
    print(f"\nFiguras guardadas:\n  {muestra}\n  {inversion}")


if __name__ == "__main__":
    main()
