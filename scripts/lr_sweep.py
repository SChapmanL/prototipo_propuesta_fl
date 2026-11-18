"""Diagnostico: por que las referencias locales se quedan en el azar.

Entrena la misma cabeza con varias tasas de aprendizaje sobre el train de un
cliente y mide AUROC en su validacion. El dataset se construye UNA vez y se
reutiliza, asi solo se paga una decodificacion.

No decide nada por si solo. Sirve para fundamentar si la tasa 0,001 que
propone docs/05 es el cuello de botella o si el problema es otro.

No usa test.

Uso (desde la raiz del repositorio):
    python scripts/lr_sweep.py
    python scripts/lr_sweep.py --client client_2 --epocas 20
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pneumonia_fl.config import load_config
from pneumonia_fl.data.dicom import load_data
from pneumonia_fl.model import build_model
from pneumonia_fl.training import evaluate_local, train_local

TASAS = (0.001, 0.01, 0.1, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--client", default="client_1")
    parser.add_argument("--epocas", type=int, default=10)
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

    print(f"{args.client}: {n_train} de train, {n_val} de val, "
          f"{args.epocas} epocas por tasa\n")
    print(f'{"tasa":>8} {"train loss":>11} {"val loss":>9} {"val AUROC":>10} '
          f'{"val AUPRC":>10} {"segundos":>9}')

    for tasa in TASAS:
        import keras
        keras.backend.clear_session()
        modelo = build_model({**config, "learning-rate": tasa})
        inicio = time.time()
        m_train = train_local(modelo, train_data, epochs=args.epocas)
        m_val = evaluate_local(modelo, val_data)
        print(f'{tasa:>8} {m_train["loss"]:>11.4f} {m_val["loss"]:>9.4f} '
              f'{m_val["auroc"]:>10.4f} {m_val["auprc"]:>10.4f} '
              f'{time.time()-inicio:>9.0f}')

    print("\nAUROC 0,5 es el azar. Una tasa util debe superarlo de forma clara.")


if __name__ == "__main__":
    main()
