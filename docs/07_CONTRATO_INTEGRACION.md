# Contrato de integración entre modelo local y federación

Estado: propuesta de Sebastián del 15 de noviembre de 2026. Ariana debe revisarla y confirmarla.

Este contrato define cuatro funciones. Ariana las implementa con datos reales y ResNet50. Sebastián las invoca desde Flower. Mientras tanto, `pneumonia_fl/provisional.py` las implementa con datos sintéticos y un modelo pequeño.

## Punto único de conexión

`pneumonia_fl/components.py` es el único archivo que decide qué implementación se usa. El cliente y el servidor Flower importan las cuatro funciones solo desde ese archivo.

## Firmas

```python
def load_data(client_id: str, split: str, data_root: str,
              manifest_root: str, batch_size: int) -> tuple[tf.data.Dataset, int]

def build_model(config: dict) -> keras.Model

def train_local(model: keras.Model, train_data: tf.data.Dataset,
                epochs: int) -> dict[str, float]

def evaluate_local(model: keras.Model, eval_data: tf.data.Dataset) -> dict[str, float]
```

## Entradas, salidas y reglas

| Función | Entradas | Salida | Reglas |
|---|---|---|---|
| `load_data` | `client_id`: `"client_1"` o `"client_2"`. `split`: `"train"`, `"val"` o `"test"`. `data_root` y `manifest_root`: rutas locales. `batch_size`: entero | `(dataset, num_examples)` | El dataset ya viene agrupado por lotes. `num_examples` cuenta imágenes, no lotes. El preprocesamiento de ResNet se aplica aquí y en ningún otro lugar |
| `build_model` | `config`: diccionario con la configuración de `pyproject.toml` | Modelo Keras compilado | Misma arquitectura y mismo orden de pesos en cada llamada. Usa `config["learning-rate"]` y `config["seed"]` |
| `train_local` | Modelo con los pesos globales ya cargados, datos de train y épocas | Diccionario con `loss` y otras métricas, todas `float` | Entrena el modelo recibido. No lo reconstruye ni reinicia sus pesos |
| `evaluate_local` | Modelo y datos de val o test | Diccionario con `loss` y otras métricas, todas `float` | No modifica los pesos |

## Tensores

| Elemento | Forma por lote | Tipo |
|---|---|---|
| Imágenes | `(batch, 224, 224, 3)` | `float32` |
| Etiquetas | `(batch, 1)` | `float32`, valores 0.0 o 1.0 |

La salida del modelo tiene forma `(batch, 1)` con activación sigmoide. La etiqueta tiene la misma forma para evitar errores de difusión de dimensiones en la pérdida.

## Qué hace la federación con estas funciones

1. El servidor llama una vez a `build_model(config)`. Esos pesos son la inicialización global común.
2. En cada ronda, cada cliente llama a `build_model(config)`, reemplaza los pesos por los globales con `model.set_weights(...)`, llama a `load_data(..., "train", ...)` y a `train_local(...)`.
3. El cliente devuelve `model.get_weights()` y el `num_examples` de train. FedAvg pondera con ese número.
4. Después de agregar, cada cliente evalúa el modelo global con `load_data(..., "val", ...)` y `evaluate_local(...)`.
5. Al terminar todas las rondas, cada cliente evalúa una vez el modelo global final con `split="test"`.

## Consecuencias que Ariana debe conocer

- `model.get_weights()` incluye los pesos no entrenables. Las medias y varianzas de BatchNormalization también se promedian. Con la base congelada no cambian.
- El cliente reconstruye el modelo en cada ronda. El estado del optimizador no se conserva entre rondas. Con SGD sin momentum no hay estado que perder.
- Las métricas viajan por Flower como números. Los valores del diccionario deben ser `int` o `float`. Una matriz de confusión se puede enviar como cuatro números separados.
- La configuración se define una sola vez, en `[tool.flwr.app.config]` de `pyproject.toml`.

## Rutas de los manifiestos

`manifest_root/client_1/train.csv`, `val.csv` y `test.csv`, y los equivalentes de `client_2`. Columnas mínimas: `image_id`, `patient_id`, `dicom_path` y `label`. `dicom_path` se resuelve respecto de `data_root`.

## Pendiente de acuerdo

- Nombres finales de las métricas de `evaluate_local`.
- Ubicación del paquete. Este trabajo usa `pneumonia_fl/` en la raíz, igual que el ejemplo oficial de Flower, y así está probado. `LEER_PRIMERO.md` sugería `src/pneumonia_fl/`.
