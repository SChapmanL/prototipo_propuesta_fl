# Ejecución federada con datos reales

Guía para ejecutar la federación en la PC donde están los 800 DICOM. El código ya está conectado. No hay que editar ningún archivo.

## Flujo completo

`python scripts/run_federation.py` hace esto, en orden:

1. **Controles previos.** Versiones, seis manifiestos, etiquetas 0/1, pacientes sin repetir y DICOM en disco. Si algo falla, se detiene sin entrenar.
2. **SuperLink.** La arranca con 180 s de espera y lanza `flwr run` con 2 clientes, uno a la vez.
3. **Inicialización.** El servidor construye ResNet50 una vez y guarda `global_round_0.keras`.
4. **10 rondas.** En cada una:
   - El servidor envía los mismos pesos globales a `client_1` y `client_2`.
   - Cada cliente carga esos pesos, entrena 1 época con sus 280 imágenes de train y devuelve pesos, número de imágenes y métricas.
   - El servidor aplica FedAvg ponderado por imágenes de train.
   - Cada cliente evalúa el nuevo modelo global con sus 60 imágenes de validación. Devuelve métricas y un histograma de puntuaciones.
   - El servidor guarda `global_round_N.keras`.
5. **Controles del servidor.** 2 de 2 clientes en cada ronda y la misma huella de pesos recibida por ambos.
6. **Elección del checkpoint.** La ronda con mayor AUROC global de validación. Si hay empate, la más temprana.
7. **Test.** Cada cliente evalúa ese checkpoint una sola vez con sus 60 imágenes de test.
8. **Reporte.** `reporte.md` y tres CSV con métricas por cliente y del modelo global.

## Métricas del modelo global

El servidor no ve imágenes ni predicciones individuales. Las métricas globales se calculan con resúmenes que sí se pueden sumar (`pneumonia_fl/global_metrics.py`):

| Métrica | Cómo se combina |
|---|---|
| Loss y accuracy | Promedio ponderado por imágenes. Exacto |
| TP, TN, FP, FN | Suma. Exacto |
| Sensibilidad, especificidad, precisión | Calculadas sobre las sumas. Exacto |
| AUROC y AUPRC | Suma de histogramas de 1 000 intervalos por clase. Error medido menor a 0,0002 frente al cálculo exacto |

No se promedian los AUROC de los clientes, porque ese promedio no es el AUROC del conjunto.

## Configuración

| Parámetro | Valor | Motivo |
|---|---|---|
| Rondas | 10 | Primer piloto de `docs/05` |
| Épocas locales por ronda | 1 | `docs/05` |
| Épocas totales por cliente | 10 | Igual que la referencia `piloto` de `docs/10` |
| Lote | 8 | `docs/05` |
| Tasa | 0,001, SGD sin momentum | `docs/05` y `docs/10` |

La prueba de integración corta se repite con `--rounds 2`.

## Qué cambió en el código

| Archivo | Cambio |
|---|---|
| `pneumonia_fl/components.py` | Usa `data/dicom.py`, `model.py` y `training.py`. `COMPONENTS_ARE_PROVISIONAL = False` |
| `pneumonia_fl/global_metrics.py` | Nuevo. Histogramas en el cliente y métricas globales en el servidor |
| `pneumonia_fl/client_app.py` | Envía histogramas al evaluar. Falla con un mensaje claro si hay más de dos clientes |
| `pneumonia_fl/server_app.py` | Métricas globales por ronda, elección del checkpoint con validación, test sobre ese checkpoint, entorno y duración |
| `scripts/run_federation.py` | Nuevo. Controles previos, SuperLink, ejecución y reporte |
| `scripts/report_federation.py` | Nuevo. Escribe `reporte.md` y los CSV. También se puede ejecutar solo |
| `scripts/check_federation.py` | Acepta `--output-dir` y usa la configuración de esa ejecución |
| `pyproject.toml` | 10 rondas. Declara `pydicom==3.0.2` |

`provisional.py` se conserva. Para volver a la prueba sintética, ver el comentario de `components.py`.

## Requisitos

- El entorno `.venv` con Python 3.12 y los paquetes de `docs/02`.
- Los 800 DICOM en `data/raw/dicom/`.
- Los seis manifiestos en `data/manifests/client_1/` y `data/manifests/client_2/`. Si faltan, se generan con `python -m pneumonia_fl.data.split`.
- `outputs/local_baselines/piloto.json` para la comparación, generado DESPUÉS de la corrección de la semilla del 22 de noviembre. El de la PC de Ariana es anterior. Ver el paso 2.
- Unos 1,5 GB libres para los modelos. Cada `.keras` de ResNet50 pesa cerca de 95 MB y se guardan 12.

## Pasos

Todos los comandos van en PowerShell, desde la raíz del repositorio.

**1. Activar el entorno e instalar el proyecto.** Esto instala `flwr` y `flower-superlink`. Solo hace falta la primera vez.

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -e .
```

**2. Repetir la referencia local de 10 épocas.** `model.py` ahora fija la semilla de la cabeza (ver `docs/02`). La referencia anterior partió de otra inicialización. Primero se guarda una copia, porque el script sobrescribe `piloto.json`. Tarda unos 4 minutos.

```powershell
Copy-Item outputs\local_baselines\piloto.json outputs\local_baselines\piloto_init_anterior.json
```

```powershell
python scripts/local_baseline.py --escenario piloto
```

No usar `--escenario federacion` ni `ambos`. Sobrescribirían `federacion.json`, la referencia de 2 épocas.

**3. Ejecutar solo los controles.** No entrena.

```powershell
python scripts/run_federation.py --check-only
```

Resultado esperado: todas las líneas con `[ok]` y al final `Controles superados`. Debe mostrar 280, 60 y 60 imágenes por cliente. Si aparece `[ERROR]`, el mensaje dice qué corregir.

**4. Ejecutar la federación.**

```powershell
python scripts/run_federation.py
```

Resultado esperado: el AUROC global de validación de cada ronda, la ronda elegida, el test por cliente y global, y la ruta de `reporte.md`. Los archivos quedan en `outputs/federation_real_<fecha>-<hora>/`.

**5. Verificar FedAvg por separado.** Copiar la ruta que imprime el paso 4.

```powershell
python scripts/check_federation.py --output-dir "outputs/federation_real_<fecha>-<hora>"
```

Resultado esperado: diferencia final 0.0 y diferencia de FedAvg 0.0 o muy cercana. Con ResNet50 en CPU puede haber un redondeo mínimo. Los dos clientes tienen 280 imágenes de train, así que el control con el promedio simple no se aplica y el script lo indica. Ese control solo distingue los dos promedios cuando los tamaños son distintos, como en la prueba sintética de 48 y 24.

**6. Regenerar el reporte (opcional).** Por ejemplo, después de crear otra referencia local:

```powershell
python scripts/report_federation.py --output-dir "outputs/federation_real_<fecha>-<hora>"
```

## Archivos de salida

| Archivo | Contenido |
|---|---|
| `reporte.md` | Reporte formal: identificación, configuración, datos, controles, validación por ronda, test, comparación local, entrenamiento local y limitaciones |
| `metricas_val_por_ronda.csv` | Validación por ronda para `client_1`, `client_2` y `global` |
| `metricas_train_por_ronda.csv` | Entrenamiento local por ronda y cliente |
| `metricas_test.csv` | Test del checkpoint elegido para `client_1`, `client_2` y `global` |
| `metrics.json` | Todo lo anterior en bruto, más histogramas, huellas y entorno |
| `run_overrides.toml` | Rutas absolutas y valores cambiados por línea de comandos |
| `run_config.json` | Configuración completa que usó el servidor |
| `global_round_N.keras` | Modelo global después de cada ronda. La 0 es la inicialización común |
| `global_final.keras` | Modelo global de la última ronda, recargado y comprobado |

`outputs/` está excluido de Git. No subir estos archivos.

## Cómo leer los resultados

- Se espera un AUROC cercano a 0,5, igual que en las referencias locales de `docs/10`. Eso confirma que la integración funciona. No dice nada a favor ni en contra del aprendizaje federado.
- La comparación justa con la referencia local es la última ronda, con las mismas 10 épocas. La ronda elegida es optimista porque se eligió con esa misma validación.
- Test se usa una sola vez. No ajustar decisiones con esas cifras.
- Las métricas son técnicas. No son validación clínica.

## Pruebas de integración del 22 de noviembre de 2026

Ejecutadas en la PC de Sebastián con los 20 DICOM de `pilot20`. Los manifiestos se crearon fuera del repositorio solo para estas pruebas: 6 de train, 2 de val y 2 de test por cliente, balanceados.

| Control | Resultado |
|---|---|
| `run_federation.py --check-only` | Todos los controles superados |
| Clientes por ronda | 2 de 2 en todas las rondas, 0 fallos |
| Huellas de pesos recibidos | Iguales en ambos clientes en cada ronda |
| Reproducibilidad, dos ejecuciones idénticas de 2 rondas | Pesos de las rondas 0, 1 y 2 con diferencia 0.0. Requirió fijar la semilla de la cabeza (ver `docs/02`) |
| Ronda elegida y test, ejecución de 3 rondas | Elección por validación y test una vez. Reporte y CSV generados |
| `metrics.json` | Marcado como "Ejecucion con los componentes reales" |
| `check_federation.py`, final vs última ronda | 0.0 |
| `check_federation.py`, FedAvg vs ronda 1 | 0.0 |
| Histogramas frente a cálculo exacto, 400 puntuaciones simuladas | AUROC 0,59746 frente a 0,59734. AUPRC 0,58300 frente a 0,58296 |

Las métricas de estas pruebas no tienen valor. Con 2 imágenes de validación por cliente, un AUROC de 1,0 es casualidad. Las pruebas solo demuestran que los componentes reales, Flower y el reporte funcionan juntos.

## Por qué el script y no `flwr run .` directo

- `flwr run` espera solo 15 s a que arranque la SuperLink local. En la prueba tardó más y `flwr` se rindió dos veces. El script la arranca antes con la misma función de Flower y 180 s de espera.
- Flower deja un `flower-superlink` en segundo plano y lo reutiliza. Los clientes pueden ejecutarse desde otra carpeta. Por eso el script pasa rutas absolutas.
- Sin `num-supernodes=2`, Flower puede crear más clientes de los que existen.
- Cada ejecución va a una carpeta nueva y no sobrescribe resultados anteriores.

## Si algo falla

| Síntoma | Causa probable | Acción |
|---|---|---|
| `No se encuentra flwr` | El proyecto no está instalado en el entorno | Paso 1 |
| `No existe ...client_1/train.csv` | Faltan las particiones | `python -m pneumonia_fl.data.split` |
| `Faltan N DICOM` | La muestra no está en `data/raw/dicom/` | Revisar la carpeta o usar `--data-root` |
| `Nodo 3: solo existen client_1 y client_2` | Se ejecutó `flwr run` sin `num-supernodes=2` | Usar el script |
| `Ronda N incompleta` | Un cliente falló | Revisar el registro de ese cliente más arriba |
| `No arrancó la SuperLink local` | Puerto 39091 ocupado o fallo de Flower | Revisar `~/.flwr/local-superlink/superlink.log` |
| `(raylet) Windows fatal exception: access violation` | Mensaje de Ray en Windows al crear los procesos | Apareció en la prueba y la ejecución terminó bien. Solo importa si después falta un cliente |
| Muchas advertencias `tf.function retracing` | El cliente reconstruye el modelo en cada ronda | Es esperado. No afecta el resultado |
| La sección 7 del reporte dice que no hay referencia | Falta `outputs/local_baselines/piloto.json` | Paso 2 |
