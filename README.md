# Prototipo federado de neumonía con CheXpert Plus

Propuesta de Investigación / Pre-Tesis, Ingeniería de Sistemas, Universidad de Lima.

Detecta neumonía en radiografías de tórax con ResNet50 y coordina dos clientes simulados con Flower y FedAvg. Cada cliente entrena con sus propios datos. Durante las rondas se intercambian parámetros y métricas, nunca imágenes.

Stack: Python 3.12, TensorFlow/Keras 2.21, ResNet50, Flower 1.43. Una PC sin GPU con 32 GB de RAM.

## Por dónde empezar

1. `AGENTS.md` y `LEER_PRIMERO.md`, que fijan las reglas de trabajo y el reparto.
2. `docs/02_ESTADO_VERIFICADO.md` para saber qué está comprobado de verdad.
3. `docs/10_MODELO_LOCAL.md` para la parte local, y `docs/08_FEDERACION_FLOWER.md` para la federación.
4. `docs/07_CONTRATO_INTEGRACION.md` para la frontera entre ambas partes.

## Estado al 18 de noviembre de 2026

| Etapa | Estado |
|---|---|
| Selección de 800 pacientes | Hecha y reproducible byte a byte |
| Descarga y validación de los 800 DICOM | **800 de 800 válidos, 0 fallos** |
| Particiones por paciente y cliente | Hechas, sin pacientes compartidos |
| Preprocesamiento DICOM y `load_data` | Implementado y con controles superados |
| Modelo, entrenamiento y evaluación local | Implementados y ejecutados |
| Referencias locales por cliente | Ejecutadas |
| Federación con componentes reales | **Conectada** el 22 de noviembre. Falta ejecutarla con los 800 DICOM. Ver `docs/11_EJECUCION_FEDERADA_REAL.md` |
| Evaluación final con prueba | Automatizada: checkpoint elegido con validación y test una vez. Falta ejecutarla con datos reales |

**El modelo todavía no aprende.** Las referencias locales dan AUROC de validación entre 0,49 y 0,59, que es el azar. La causa está medida y es sobreajuste: con 60 épocas el AUROC de entrenamiento sube a 0,84 mientras el de validación se queda en 0,54. Con 280 imágenes de entrenamiento por cliente no hay con qué generalizar. El detalle y la recomendación están en `docs/10_MODELO_LOCAL.md`.

## Qué contiene el repositorio

El repositorio tiene **código, documentación y configuración**. No contiene radiografías, informes, metadatos originales, manifiestos con identificadores ni credenciales. `data/` y `outputs/` están excluidos de Git y hay que regenerarlos en cada máquina.

### Módulos

| Archivo | Responsable | Qué hace |
|---|---|---|
| `pneumonia_fl/config.py` | Ariana | Único lector de `[tool.flwr.app.config]` de `pyproject.toml` |
| `pneumonia_fl/data/split.py` | Ariana | Reparte los 800 pacientes en particiones y clientes |
| `pneumonia_fl/data/dicom.py` | Ariana | Lee DICOM, aplica la política de intensidad y expone `load_data` |
| `pneumonia_fl/model.py` | Ariana | `build_model`: ResNet50 compilada |
| `pneumonia_fl/training.py` | Ariana | `train_local` y `evaluate_local` |
| `pneumonia_fl/metrics.py` | Ariana | Accuracy, AUROC, AUPRC y matriz de confusión |
| `pneumonia_fl/components.py` | Sebastián | Punto único que elige qué implementación del contrato se usa |
| `pneumonia_fl/client_app.py` | Sebastián | Adaptador Flower de cada cliente |
| `pneumonia_fl/server_app.py` | Sebastián | Servidor, rondas, FedAvg y guardado global |
| `pneumonia_fl/provisional.py` | Sebastián | Componente sintético para probar la coordinación |
| `pneumonia_fl/weights.py` | Sebastián | Huella numérica de una lista de pesos |
| `pneumonia_fl/global_metrics.py` | Sebastián | Métricas del modelo global sobre ambos clientes, sin juntar datos |

### Scripts

| Script | Qué hace |
|---|---|
| `fetch_metadata.py` | Trae las tablas de metadatos desde la API de Redivis |
| `prepare_sample.py` | Genera la selección de 800 pacientes |
| `download_dicoms.py` | Descarga y valida los DICOM |
| `check_sample.py` | Nueve controles sobre la muestra descargada |
| `preview_sample.py` | Control visual: comprueba que no salen invertidas |
| `check_load_data.py` | Controles del pipeline que alimenta al modelo |
| `local_baseline.py` | Referencias locales por cliente |
| `lr_sweep.py` | Diagnóstico de la tasa de aprendizaje |
| `curva_epocas.py` | Diagnóstico de sobreajuste por épocas |
| `run_federation.py` | Controles previos, ejecución federada con datos reales y reporte |
| `report_federation.py` | Reporte formal (`reporte.md` y CSV) de una ejecución terminada |
| `check_federation.py` | Verifica FedAvg sin usar Flower |

## Puesta en marcha

Requiere Python 3.12. No sirve 3.13, porque `pyproject.toml` declara `requires-python = ">=3.11,<3.13"`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install redivis pydicom numpy matplotlib "tensorflow==2.21.0"
```

Autenticarse en Redivis una sola vez. Abre el navegador y guarda las credenciales en `~/.redivis`:

```powershell
.\.venv\Scripts\python.exe -c "import redivis; redivis.authenticate()"
```

### Reconstruir los datos

Hacen falta unos 11 GB libres. La muestra ocupa 10,07 GB.

```powershell
# 1. Metadatos, por API
.\.venv\Scripts\python.exe scripts/fetch_metadata.py --table df_chexpert_plus_240401 --out df_chexpert_plus_240401.csv
.\.venv\Scripts\python.exe scripts/fetch_metadata.py --table DICOM_train --out dicom_train.csv
.\.venv\Scripts\python.exe scripts/fetch_metadata.py --table "CheXpert Labels" --file report_fixed.json --out report_fixed.json

# 2. Selección de 800 pacientes
.\.venv\Scripts\python.exe scripts/prepare_sample.py --input-dir data/raw/metadata --output-dir data/manifests

# 3. Imágenes. Unos 57 minutos
.\.venv\Scripts\python.exe scripts/download_dicoms.py --manifest data/manifests/download_manifest.csv --download

# 4. Controles
.\.venv\Scripts\python.exe scripts/check_sample.py
.\.venv\Scripts\python.exe scripts/preview_sample.py

# 5. Particiones por cliente
.\.venv\Scripts\python.exe -m pneumonia_fl.data.split
```

### Entrenar en local, sin Flower

```powershell
.\.venv\Scripts\python.exe scripts/check_load_data.py
.\.venv\Scripts\python.exe scripts/local_baseline.py
```

### Ejecutar la federación

Necesita `pip install -e .`, que añade Flower. El script comprueba los datos, lanza dos clientes en modo secuencial con rutas absolutas y escribe en una carpeta nueva.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/run_federation.py --check-only
python scripts/run_federation.py
```

Guía paso a paso y lectura de resultados en `docs/11_EJECUCION_FEDERADA_REAL.md`. Trampas ya diagnosticadas en `docs/08_FEDERACION_FLOWER.md`.

## Reglas de datos que no se negocian

- Etiquetas `Pneumonia` 0 y 1. Los valores `-1` y `null` se excluyen y nunca se convierten en 0.
- Una imagen frontal por paciente, adultos. Ningún paciente se repite entre clientes ni entre particiones.
- La clase 0 no significa radiografía sana. Validar un DICOM no es validación clínica.
- No subir imágenes, informes, metadatos originales, manifiestos con identificadores ni credenciales.
- Test se reserva para la evaluación final. Los checkpoints se eligen con validación.

## Alcance y límites

Los dos clientes son particiones de una sola fuente. No representan dos hospitales ni demuestran generalización multicéntrica. Ejecutar todo en una PC da separación lógica, no aislamiento institucional.

FedAvg intercambia pesos ponderados por el número de muestras locales de entrenamiento. No es un intercambio de gradientes.

La etiqueta viene de informes, no de confirmación clínica por imagen. El generador de `report_fixed.json` sigue sin resolverse. Los filtros actuales no excluyen COVID.

Los resultados guardados en `outputs/federation/` provienen del componente provisional con datos sintéticos. Prueban coordinación, no detección de neumonía.
