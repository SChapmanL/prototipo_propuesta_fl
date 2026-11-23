# Estado verificado

Este documento registra lo ejecutado. Claude debe volver a inspeccionar los archivos y el entorno de la usuaria antes de asumir disponibilidad local.

## Paso 1 cerrado el 18 de noviembre de 2026

La muestra física de 800 DICOM está descargada y validada en la PC de Ariana. Lo anterior a esta sección corresponde a sesiones previas y se conserva como historial.

### Entorno

Python 3.12.10 en `.venv`, creado desde cero. Se eligió 3.12 y no 3.13 porque `pyproject.toml` declara `requires-python = ">=3.11,<3.13"`. Un solo entorno cubre descargas y TensorFlow. La separación `.venv` y `.venv-fl` que describe `docs/08` correspondía a la PC de Sebastián.

| Paquete | Versión |
|---|---|
| tensorflow | 2.21.0 |
| keras | 3.15.1 |
| pydicom | 3.0.2 |
| redivis | 0.22.2 |
| numpy | 2.5.5 |
| matplotlib | 3.11.2 |

Sin GPU, como estaba previsto. Los pesos de ImageNet de ResNet50 quedaron en caché local.

### Metadatos obtenidos por API

Faltaba un eslabón en el repositorio. `prepare_sample.py` lee tres archivos locales y no usa la API, y ningún script los conseguía. Se añadió `scripts/fetch_metadata.py`.

Nombres reales de las tablas en `aimi.chexpert_plus:5yyj:v1_0`:

| Archivo | Tabla | Tipo | SHA256 |
|---|---|---|---|
| `df_chexpert_plus_240401.csv` | `df_chexpert_plus_240401` | tabla de datos | coincide |
| `dicom_train.csv` | `DICOM_train` | tabla de datos | coincide |
| `report_fixed.json` | `CheXpert Labels` | archivo adjunto | coincide |

Los tres SHA256 coinciden con `configs/sample_config.json`. Las fuentes son idénticas a las que produjeron la selección original. `CheXpert Labels` contiene además `findings_fixed.json` e `impression_fixed.json`, que no se usan.

### Selección regenerada con igualdad exacta

`prepare_sample.py` ejecutado sobre esas fuentes reprodujo los seis archivos. Cada cifra coincide con `docs/03`: 800 pacientes, 400 por clase, AP 648 y PA 152, Male 458 y Female 342, 8 556 elegibles antes de elegir por paciente, reservas de 2 803 y 3 526, y 10 065 641 856 bytes. El cruce por vista y etiqueta también coincide: clase 0 con AP 317 y PA 83, clase 1 con AP 331 y PA 69.

### Descarga y validación de los 800 DICOM

| Medida | Resultado |
|---|---|
| Archivos con `status=ok` | 800 de 800 |
| Fallos | 0 |
| Sustituciones desde la reserva | ninguna necesaria |
| Tamaño en disco | 10,066 GB |
| Duración | 57 min, 3,0 MB/s de media |

El descargador verificó tamaño, MD5, lectura de píxeles, rango no constante e interpretación fotométrica en cada archivo. Antes se descargó y validó `pilot20` por separado, 20 de 20 sin fallos en 2 min 11 s. Su registro se conservó en `data/raw/dicom/download_validation_pilot20.csv`, porque cada ejecución sobrescribe `download_validation.csv`.

`scripts/check_sample.py`, escrito en esta sesión, pasó sus nueve controles sobre las 800: filas, etiquetas solo 0 o 1, balance 400/400, 800 pacientes distintos, 800 con `status=ok`, correspondencia entre manifiesto y registro, presencia en disco, y balance y pacientes entre las ya validadas.

Distribución fotométrica de la muestra completa: 657 MONOCHROME1 y 143 MONOCHROME2.

### Control visual

`scripts/preview_sample.py` generó dos figuras en `outputs/`, excluido de Git. Se revisaron 8 imágenes, 4 por clase, con los dos tipos fotométricos y vistas AP y PA. Las ocho se ven como radiografías normales, con pulmones oscuros y huesos claros. Ninguna sale invertida.

La comparación lado a lado confirma la regla: en MONOCHROME1 la versión sin corregir aparece en negativo y la corregida es la correcta, y en MONOCHROME2 ambas son idénticas porque no se invierte.

Hallazgo para el Paso 3: la profundidad de bits no es uniforme. Entre solo ocho archivos se observaron máximos crudos de 4095, 8568, 16383 y 29217. Confirma la advertencia de `docs/05` de no reducir uint16 con un cast directo a uint8. El escalado de `preview_sample.py` es solo para visualizar y no es la política de preprocesamiento, que sigue sin definirse.

### Lo que esto no demuestra

Los controles son técnicos. No hay validación clínica, la etiqueta sigue viniendo de informes y la clase 0 no significa radiografía sana. El generador de `report_fixed.json` sigue sin resolverse.

## Fuentes inspeccionadas

- Tabla principal: 223 462 filas, sin duplicados en path_to_image.
- report_fixed.json: JSON por líneas, 223 462 registros, sin claves de imagen duplicadas.
- Unión completa 1:1 entre ambas fuentes mediante path_to_image.
- dicom_train.csv: 210 740 filas. Campos file_id, file_name, size, added_at y md5_hash.
- CSV principal: 223 228 filas train y 234 valid.

Distribución de Pneumonia en todo report_fixed.json:

| Valor | Registros |
|---|---:|
| null | 189 734 |
| -1 | 23 045 |
| 0 | 4 720 |
| 1 | 5 963 |

## Pruebas de DICOM

Se recibieron primero cuatro archivos de patient00001 a patient00004. Tres tenían etiqueta null y uno -1. Fueron legibles, pero no elegibles con la política actual.

Después se verificaron los cuatro candidatos correctos:

| Nombre adjunto | Ruta original en DICOM_train | Edad | Vista | Etiqueta |
|---|---|---:|---|---:|
| view1_frontal (1)(1).dcm | patient00016/study1/view1_frontal.dcm | 54 | PA | 0 |
| view2_frontal.dcm | patient00087/study3/view2_frontal.dcm | 53 | PA | 0 |
| view1_frontal (2)(1).dcm | patient00031/study3/view1_frontal.dcm | 88 | AP | 1 |
| view1_frontal (3)(1).dcm | patient00115/study2/view1_frontal.dcm | 64 | AP | 1 |

En los cuatro casos se comprobaron tamaño, MD5, identidad por índice y decodificación completa con pydicom. Transfer syntax: 1.2.840.10008.1.2.1, sin compresión de píxeles. Píxeles uint16. patient00016 usa MONOCHROME2. Los otros tres usan MONOCHROME1.

No se completó la inspección visual del preprocesamiento ni una validación clínica. No se debe inferir que todos los archivos del dataset tendrán la misma codificación.

## Artefactos producidos

- Lista de 800 pacientes, balance 400/400.
- Lista pilot20 con 10 por clase, contenida en los 800.
- Reserva de 100 pacientes adicionales, 50 por clase, sin solapamiento con los 800.
- Exclusiones, configuración, resumen y dos scripts.
- Selección reproducida por segunda ejecución con igualdad exacta de los seis archivos.
- Función de comprobación del descargador probada con los cuatro DICOM correctos.
- Modo sin descarga del descargador ejecutado correctamente.

## Pasos 2 a 5 ejecutados el 18 de noviembre de 2026

Detalle completo, con cifras y fundamentos, en `docs/10_MODELO_LOCAL.md`.

- **Particiones.** Seis manifiestos por cliente, 70/15/15 por paciente con hash y semilla 42. 280 de train, 60 de validación y 60 de prueba por cliente, con balance 50/50 en todos. Sin pacientes compartidos. Reproducible byte a byte.
- **Preprocesamiento.** Política de intensidad decidida midiendo los 800 archivos: rescalado DICOM, recorte p1-p99 por imagen, inversión de MONOCHROME1 al final, 224x224 y `preprocess_input` una sola vez. Controles superados en los dos clientes.
- **Modelo.** ResNet50 con base congelada, 2 049 parámetros entrenables. Dos llamadas a `build_model` dan pesos idénticos en los 320 tensores, que es lo que FedAvg necesita.
- **Referencias locales.** Ejecutadas por cliente, sin usar prueba.

## Integración federada del 22 de noviembre de 2026

Ejecutado en la PC de Sebastián. Detalle y comandos en `docs/11_EJECUCION_FEDERADA_REAL.md`.

- **Componentes reales conectados.** `components.py` usa `data/dicom.py`, `model.py` y `training.py`.
- **Prueba de principio a fin con pilot20.** 20 DICOM, manifiestos temporales fuera del repositorio. Ejecuciones de 2 y 3 rondas: 2 de 2 clientes en cada ronda, huellas iguales, FedAvg recalculado con diferencia 0.0 y modelo final recargado idéntico.
- **Métricas del modelo global.** Se calculan sobre los dos clientes juntos con resúmenes sumables. AUROC y AUPRC por histogramas, con error medido menor a 0,0002.
- **Checkpoint con validación.** El servidor elige la ronda con mayor AUROC global de validación y evalúa test una vez con ella.
- **Reporte formal.** `reporte.md` y tres CSV por ejecución.
- **Configuración.** 10 rondas con 1 época local, el piloto de `docs/05`.

**Hallazgo: la inicialización de la cabeza no era reproducible dentro de Flower.** Keras tomaba la semilla de la capa Dense del generador `random` de Python. En el servidor Flower otros hilos usan ese generador mientras ResNet50 carga sus pesos. Dos ejecuciones federadas idénticas empezaron con pesos distintos. Se reprodujo fuera de Flower con un hilo que usa `random` en paralelo. FedAvg no estaba afectado, porque ambos clientes siempre reciben los pesos del servidor. Sí lo estaba la reproducibilidad entre ejecuciones. Corrección en `model.py`: `GlorotUniform(seed=config["seed"])` en la cabeza, el mismo inicializador que Keras usa por defecto. Comprobado estable con y sin hilos en paralelo. Después de la corrección, dos ejecuciones federadas idénticas de 2 rondas dieron pesos iguales en las rondas 0, 1 y 2, con diferencia máxima 0.0, y las mismas huellas.

Consecuencia: la cabeza ahora empieza con otros valores aleatorios. Las referencias locales de `outputs/local_baselines/` se calcularon con la inicialización anterior. Para una comparación exacta conviene repetir `piloto` antes de la federación.

## Federación real con los 800 DICOM, 23 de noviembre de 2026

Ejecutado en la PC de Ariana, CPU. Python 3.12.10, Flower 1.43.0, TensorFlow 2.21.0, Keras 3.15.1. Carpeta `outputs/federation_real_20261123-005414/`, excluida de Git.

**Referencia local repetida.** `local_baseline.py --escenario piloto`, 10 épocas, unos 5 minutos. La versión anterior se conserva como `piloto_init_anterior.json`. `federacion.json` no se tocó.

| Cliente | AUROC val local, nueva inicialización |
|---|---|
| client_1 | 0,5528 |
| client_2 | 0,6633 |

**Controles.**

- `run_federation.py --check-only`: todos superados. 280, 60 y 60 imágenes por cliente, balance 50/50, 800 pacientes sin repetir y todos los DICOM en disco.
- 10 de 10 rondas completas, 2 de 2 clientes en cada una, 0 fallos.
- Mismas huellas de pesos globales recibidas por ambos clientes en cada ronda.
- `check_federation.py`: diferencia final frente a última ronda 0.0. FedAvg recalculado frente a ronda 1: 0.0. Control negativo no aplicable porque ambos clientes tienen 280 imágenes.
- Duración de las 10 rondas: 18,3 minutos.

**Validación del modelo global.**

| Ronda | AUROC client_1 | AUROC client_2 | AUROC global |
|---|---|---|---|
| 1 | 0,5822 | 0,6833 | 0,6326 |
| 5, elegida | 0,5994 | 0,6872 | 0,6454 |
| 10 | 0,5967 | 0,6806 | 0,6390 |

**Test con el checkpoint de la ronda 5, usado una sola vez.**

| Alcance | AUROC | AUPRC | Accuracy |
|---|---|---|---|
| client_1 | 0,4439 | 0,4870 | 0,4333 |
| client_2 | 0,6367 | 0,6019 | 0,6000 |
| global | 0,5437 | 0,5503 | 0,5167 |

**Lectura.** La integración funciona de principio a fin con datos reales. La ronda 10 supera a la referencia local en validación por 0,044 en client_1 y 0,017 en client_2. Con 60 imágenes por cliente esa diferencia está dentro del ruido. El test global de 0,54 confirma que el modelo sigue cerca del azar. No es evidencia a favor ni en contra del aprendizaje federado. Las métricas son técnicas, no validación clínica.

## Trabajo pendiente

La federación de 10 rondas con los 800 DICOM está ejecutada. Las métricas de `outputs/federation/` siguen siendo de la prueba técnica sintética de Sebastián.

Siguientes decisiones del equipo: ampliar la muestra y preparar el primer escenario Non-IID.

**El modelo no aprende todavía.** Las referencias locales dan AUROC de validación entre 0,49 y 0,59, es decir el azar. La causa está medida y es sobreajuste: con 60 épocas el AUROC de entrenamiento sube a 0,84 mientras el de validación se queda en 0,54. Con 280 imágenes de entrenamiento por cliente no hay con qué generalizar. La palanca propuesta es ampliar la muestra, decisión que corresponde al equipo.

