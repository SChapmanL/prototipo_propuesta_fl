"""Diagnostico: curva de entrenamiento y validacion por epocas.

Entrena un cliente en bloques y mide, despues de cada bloque, como van la
perdida y el AUROC en train y en validacion. Sirve para distinguir dos
problemas que se confunden con facilidad:

  - Si la metrica de TRAIN no mejora, el modelo no consigue ajustar ni sus
    propios datos. Es un problema de optimizacion o de capacidad.
  - Si la de train mejora y la de VALIDACION no, el modelo memoriza. Es
    sobreajuste, y la palanca es mas datos o mas regularizacion.

No usa test.

Uso (desde la raiz del repositorio):
    python scripts/curva_epocas.py
    python scripts/curva_epocas.py --client client_2 --bloques 3 --por-bloque 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pneumonia_fl.config import load_config
from pneumonia_fl.data.dicom import load_data
from pneumonia_fl.model import build_model
from pneumonia_fl.training import evaluate_local, train_local


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--client", default="client_1")
    parser.add_argument("--bloques", type=int, default=6)
    parser.add_argument("--por-bloque", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    train_data, n_train = load_data(
        args.client, "train", config["data-root"], config["manifest-root"],
        int(config["batch-size"]))
    val_data, n_val = load_data(
        args.client, "val", config["data-root"], config["manifest-root"],
        int(config["batch-size"]))

    print(f"{args.client}: {n_train} de train, {n_val} de val")
    print(f"tasa {config['learning-rate']}, base congelada\n")
    print(f'{"epocas":>7} {"train loss":>11} {"train AUROC":>12} '
          f'{"val loss":>9} {"val AUROC":>10}')

    modelo = build_model(config)
    acumuladas = 0
    for _ in range(args.bloques):
        m_train = train_local(modelo, train_data, epochs=args.por_bloque)
        acumuladas += args.por_bloque
        m_val = evaluate_local(modelo, val_data)
        print(f'{acumuladas:>7} {m_train["loss"]:>11.4f} {m_train["auroc"]:>12.4f} '
              f'{m_val["loss"]:>9.4f} {m_val["auroc"]:>10.4f}')

    print("\n0,693 es la perdida del azar para dos clases equilibradas.")
    print("Train que mejora y validacion que no, es sobreajuste.")


if __name__ == "__main__":
    main()
