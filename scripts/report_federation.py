"""Reporte formal de una ejecución federada terminada.

Lee metrics.json de la carpeta de la ejecución y escribe en ella:

    reporte.md                     informe legible para la tesis
    metricas_val_por_ronda.csv     validación por ronda: client_1, client_2 y global
    metricas_train_por_ronda.csv   entrenamiento local por ronda y cliente
    metricas_test.csv              test del checkpoint elegido: clientes y global

"global" significa el modelo global evaluado con los datos de los dos
clientes juntos. Se calcula con resúmenes sumables, sin juntar imágenes
(ver pneumonia_fl/global_metrics.py).

Si existe una referencia local con el mismo número de épocas en
outputs/local_baselines/, el reporte agrega la comparación.

Los archivos no contienen identificadores de pacientes. Igual quedan en
outputs/, que está excluido de Git.

Uso:
    python scripts/report_federation.py --output-dir outputs/federation_real_...
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pneumonia_fl.global_metrics import client_metrics  # noqa: E402
from pneumonia_fl.metrics import sensibilidad_especificidad  # noqa: E402

CLIENTS = ("client_1", "client_2")
SCOPES = (*CLIENTS, "global")
COLUMNS = ["num-examples", "loss", "accuracy", "auroc", "auprc",
           "sensibilidad", "especificidad", "precision", "tp", "tn", "fp", "fn"]


# --- Lectura ---------------------------------------------------------------

def with_ratios(metrics):
    """Agrega sensibilidad, especificidad y precisión si hay matriz de confusión."""
    metrics = dict(metrics)
    if all(k in metrics for k in ("tp", "tn", "fp", "fn")):
        metrics.update(sensibilidad_especificidad(metrics))
        predicted = metrics["tp"] + metrics["fp"]
        metrics["precision"] = metrics["tp"] / predicted if predicted else 0.0
    return metrics


def scopes_for(record, pooled, split):
    """Métricas de client_1, client_2 y global para una evaluación."""
    rows = {c: with_ratios(client_metrics(record, c, split)) for c in CLIENTS}
    rows["global"] = pooled
    return rows


def find_baseline(baselines_dir, epochs):
    """Referencia local con el mismo número de épocas por cliente, si existe.

    Si hay varias, usa la de fecha más reciente. Así una copia de respaldo
    de una referencia anterior no reemplaza a la nueva.
    """
    if not baselines_dir.is_dir():
        return None
    matches = []
    for path in sorted(baselines_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("epocas", -1)) == epochs:
            matches.append((data.get("fecha", ""), path, data))
    if not matches:
        return None
    _, path, data = max(matches, key=lambda item: item[0])
    return path, data


# --- Formato ---------------------------------------------------------------

def number(value, digits=4):
    """Número con coma decimal, como el resto de la documentación."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/d"
    if isinstance(value, float) and value.is_integer() and digits == 0:
        return str(int(value))
    return f"{value:.{digits}f}".replace(".", ",")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def metric_cells(metrics):
    return [metrics.get(name, "") for name in COLUMNS]


# --- Reporte ---------------------------------------------------------------

def build(output_dir, baselines_dir):
    summary = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    config = summary["config"]
    rounds = sorted(int(r) for r in summary["val_por_ronda"])
    chosen = int(summary["ronda_elegida"])
    epochs_total = int(config["num-server-rounds"]) * int(config["local-epochs"])

    val = {r: scopes_for(summary["val_por_ronda"][str(r)],
                         summary["val_global_por_ronda"][str(r)], "val")
           for r in rounds}
    train = {r: {c: client_metrics(summary["train_por_ronda"][str(r)], c, "train")
                 for c in CLIENTS} for r in rounds}
    test = scopes_for(summary["test"], summary["test_global"], "test")

    # CSV para análisis posterior.
    write_csv(output_dir / "metricas_val_por_ronda.csv", ["ronda", "alcance", *COLUMNS],
              [[r, s, *metric_cells(val[r][s])] for r in rounds for s in SCOPES])
    write_csv(output_dir / "metricas_train_por_ronda.csv",
              ["ronda", "cliente", *COLUMNS],
              [[r, c, *metric_cells(with_ratios(train[r][c]))]
               for r in rounds for c in CLIENTS])
    write_csv(output_dir / "metricas_test.csv",
              ["ronda_elegida", "alcance", *COLUMNS],
              [[chosen, s, *metric_cells(test[s])] for s in SCOPES])

    md = []
    title_kind = ("Prueba técnica con datos sintéticos"
                  if summary["componentes_provisionales"] else "Datos reales")
    md.append(f"# Reporte de ejecución federada. {title_kind}\n")
    md.append(
        "Prototipo de detección de neumonía con CheXpert Plus, ResNet50 y Flower. "
        "Dos clientes simulados y un servidor con FedAvg. "
        "**Las métricas son técnicas. No constituyen validación clínica.**\n"
    )

    # 1. Identificación
    env = summary["entorno"]
    md.append("## 1. Identificación\n")
    md.append(table(["Campo", "Valor"], [
        ["Inicio", summary["inicio"]],
        ["Fin", summary["fin"]],
        ["Duración de las rondas", f"{number(summary['segundos_rondas'] / 60, 1)} min"],
        ["Componentes", "provisionales sintéticos" if summary["componentes_provisionales"]
         else "reales: data/dicom.py, model.py, training.py"],
        ["Entorno", f"Python {env['python']}, Flower {env['flwr']}, "
                    f"TensorFlow {env['tensorflow']}, Keras {env['keras']}, {env['dispositivo']}"],
        ["Carpeta", output_dir.name],
    ]))

    # 2. Configuración
    md.append("\n## 2. Configuración\n")
    md.append(table(["Parámetro", "Valor"], [
        ["Rondas de agregación", config["num-server-rounds"]],
        ["Épocas locales por ronda", config["local-epochs"]],
        ["Épocas totales por cliente", epochs_total],
        ["Tamaño de lote", config["batch-size"]],
        ["Tasa de aprendizaje (SGD sin momentum)", number(float(config["learning-rate"]), 4)],
        ["Semilla", config["seed"]],
        ["Agregación", "FedAvg ponderado por imágenes de train"],
        ["Criterio de checkpoint", summary["criterio_checkpoint"]],
    ]))

    # 3. Datos
    first = rounds[0]
    md.append("\n## 3. Datos por cliente\n")
    md.append(table(["Cliente", "Train", "Val", "Test"], [
        [c, int(train[first][c]["num-examples"]), int(val[first][c]["num-examples"]),
         int(test[c]["num-examples"])] for c in CLIENTS
    ]))
    md.append("\nCada paciente pertenece a un solo cliente y a una sola partición.")

    # 4. Controles de integridad
    md.append("\n## 4. Controles de integridad por ronda\n")
    rows = []
    for r in rounds:
        sent = summary["huellas_enviadas"][str(r)]
        same = all(train[r][c]["received-weights-fingerprint"] == sent for c in CLIENTS)
        n = [train[r][c]["num-examples"] for c in CLIENTS]
        weights = " / ".join(number(x / sum(n), 3) for x in n)
        rows.append([r, f"{len(CLIENTS)} de {len(CLIENTS)}",
                     "sí" if same else "**NO**", weights])
    md.append(table(["Ronda", "Clientes", "Mismos pesos globales recibidos",
                     "Peso FedAvg client_1 / client_2"], rows))
    md.append("\nEl servidor detiene la ejecución si falta un cliente o si las huellas "
              "no coinciden. Que este reporte exista implica que ambos controles pasaron.")

    # 5. Validación por ronda
    md.append("\n## 5. Validación del modelo global por ronda\n")
    rows = []
    for r in rounds:
        mark = f"**{r}** (elegida)" if r == chosen else str(r)
        rows.append([mark,
                     number(val[r]["client_1"]["auroc"]), number(val[r]["client_2"]["auroc"]),
                     number(val[r]["global"]["auroc"]), number(val[r]["global"]["auprc"]),
                     number(val[r]["global"]["loss"])])
    md.append(table(["Ronda", "AUROC client_1", "AUROC client_2", "AUROC global",
                     "AUPRC global", "Loss global"], rows))
    md.append("\nCada fila evalúa el modelo global de esa ronda con la validación de cada "
              "cliente. La columna global usa la validación de los dos clientes juntos.")

    # 6. Test
    md.append(f"\n## 6. Test con el checkpoint de la ronda {chosen}\n")
    md.append(table(
        ["Alcance", "Imágenes", "Loss", "Accuracy", "AUROC", "AUPRC",
         "Sensibilidad", "Especificidad", "Precisión", "TP", "TN", "FP", "FN"],
        [[s, int(test[s]["num-examples"]), number(test[s]["loss"]),
          number(test[s]["accuracy"]), number(test[s]["auroc"]), number(test[s]["auprc"]),
          number(test[s].get("sensibilidad")), number(test[s].get("especificidad")),
          number(test[s].get("precision")),
          *(number(test[s].get(k), 0) for k in ("tp", "tn", "fp", "fn"))]
         for s in SCOPES]))
    md.append("\nTest se usó una sola vez, después de elegir el checkpoint con validación. "
              "Sensibilidad, especificidad, precisión y la matriz usan el umbral 0,5.")

    # 7. Comparación con referencia local
    md.append("\n## 7. Comparación con la referencia local\n")
    baseline = find_baseline(baselines_dir, epochs_total)
    if baseline is None:
        md.append(f"No se encontró en `{baselines_dir.name}/` una referencia local de "
                  f"{epochs_total} épocas. Ejecutar `python scripts/local_baseline.py` "
                  "y volver a generar este reporte.")
    else:
        path, data = baseline
        last = rounds[-1]
        rows = []
        for c in CLIENTS:
            local = data["resultados"][c]["val"]["auroc"]
            rows.append([c, number(local), number(val[chosen][c]["auroc"]),
                         number(val[last][c]["auroc"]),
                         number(val[last][c]["auroc"] - local)])
        md.append(f"Referencia: `{path.name}` del {data.get('fecha', 'n/d')}, "
                  f"{data['epocas']} épocas por cliente, cada cliente entrenado solo "
                  "con sus datos. Debe ser posterior a la corrección de la semilla de "
                  "la cabeza del 22 de noviembre de 2026 para partir de la misma "
                  "inicialización.\n")
        md.append(table(["Cliente", "AUROC val local", f"AUROC val federado ronda {chosen}",
                         f"AUROC val federado ronda {last}", "Diferencia última ronda"],
                        rows))
        md.append("\nLa comparación justa es la última ronda, porque tiene las mismas épocas "
                  "que la referencia. La ronda elegida es optimista: se eligió con esa "
                  "misma validación.")

    # 8. Entrenamiento local
    md.append("\n## 8. Entrenamiento local por ronda\n")
    md.append(table(["Ronda", "Loss client_1", "AUROC client_1", "Loss client_2",
                     "AUROC client_2"],
                    [[r, number(train[r]["client_1"]["loss"]),
                      number(train[r]["client_1"].get("auroc")),
                      number(train[r]["client_2"]["loss"]),
                      number(train[r]["client_2"].get("auroc"))] for r in rounds]))
    md.append("\nSon métricas del modelo LOCAL de cada cliente durante su época, antes de "
              "agregar. No son métricas del modelo global.")

    # 9. Método y limitaciones
    md.append("\n## 9. Método de las métricas globales y limitaciones\n")
    md.append("\n".join([
        "- Las métricas globales no promedian métricas de clientes. Loss y accuracy se "
        "ponderan por imágenes. TP, TN, FP y FN se suman. AUROC y AUPRC se calculan sobre "
        "la suma de histogramas de puntuaciones de 1 000 intervalos por clase.",
        "- Cada cliente envía histogramas, no imágenes ni predicciones individuales. Con "
        "pocas imágenes un histograma fino puede revelar la puntuación y la etiqueta de "
        "cada imagen. En un despliegue real se usarían intervalos más gruesos o "
        "agregación segura.",
        "- Simulación en una sola PC. Los dos clientes provienen de CheXpert Plus y tienen "
        "particiones balanceadas, es decir datos IID. No demuestra generalización entre "
        "hospitales.",
        "- Las etiquetas provienen de informes radiológicos. La clase 0 no significa "
        "radiografía sana.",
        "- Sin agregación segura, privacidad diferencial ni TLS.",
    ]))

    md.append("\n\n## Archivos de esta ejecución\n")
    md.append("`metrics.json`, `metricas_val_por_ronda.csv`, `metricas_train_por_ronda.csv`, "
              "`metricas_test.csv`, `global_round_N.keras`, `global_final.keras` y "
              "`run_config.json`.")

    report = output_dir / "reporte.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baselines-dir", default="outputs/local_baselines")
    args = parser.parse_args()
    report, _ = build(Path(args.output_dir).resolve(), Path(args.baselines_dir).resolve())
    print(f"Reporte escrito en {report}")


if __name__ == "__main__":
    main()
