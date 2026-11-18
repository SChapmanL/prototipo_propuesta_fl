# Modelo y entrenamiento local (parte de Ariana)

Estado al 18 de noviembre de 2026. Pasos 1 a 5 implementados y ejecutados con los 800 DICOM reales. El codigo funciona sin Flower.

## Que hay y donde

| Archivo | Funcion | Que hace |
|---|---|---|
| `pneumonia_fl/config.py` | `load_config` | Unico lector de `[tool.flwr.app.config]` de `pyproject.toml` |
| `pneumonia_fl/data/split.py` | `main` | Reparte los 800 pacientes en particiones y clientes |
| `pneumonia_fl/data/dicom.py` | `leer_dicom` | Lee un DICOM y aplica la politica de intensidad |
| `pneumonia_fl/data/dicom.py` | `load_data` | Contrato de `docs/07`. Devuelve `(dataset, num_examples)` |
| `pneumonia_fl/model.py` | `build_model` | ResNet50 compilada, misma arquitectura y orden de pesos |
| `pneumonia_fl/training.py` | `train_local` | Entrena el modelo recibido sin reconstruirlo |
| `pneumonia_fl/training.py` | `evaluate_local` | Evalua sin tocar pesos |
| `pneumonia_fl/metrics.py` | `metricas` | Accuracy, AUROC, AUPRC y matriz de confusion |
| `scripts/check_load_data.py` | script | Controles y figura del Paso 3 |
| `scripts/local_baseline.py` | script | Referencias locales del Paso 5 |
| `scripts/lr_sweep.py` | script | Diagnostico de la tasa de aprendizaje |
| `scripts/curva_epocas.py` | script | Diagnostico de sobreajuste por epocas |

## Particiones

70/15/15 por paciente, estratificado por etiqueta, con regla por hash SHA256 y semilla 42. El mismo criterio que la seleccion de `prepare_sample.py`, de modo que no depende del orden de las filas ni de ningun generador aleatorio.

| Cliente | train | val | test |
|---|---:|---:|---:|
| `client_1` | 280 (140/140) | 60 (30/30) | 60 (30/30) |
| `client_2` | 280 (140/140) | 60 (30/30) | 60 (30/30) |

Los 20 pacientes de `pilot20` quedan en train, descontados de su cupo. Por el hash cayeron 10 en cada cliente.

Comprobado de forma independiente: 800 filas, 800 pacientes distintos, cero duplicados, cero solapes entre los 15 pares de archivos, cobertura exacta del manifiesto maestro y ninguna etiqueta alterada. Dos ejecuciones dan los seis archivos identicos.

El reparto AP/PA no es identico entre clientes, porque la vista no se controlo al muestrear. En train, `client_1` tiene 213 AP y 67 PA, y `client_2` tiene 234 AP y 46 PA. Conviene tenerlo presente al interpretar diferencias entre clientes.

## Politica de intensidad

Decidida midiendo los 800 archivos, no por teoria.

1. Aplicar `RescaleSlope` y `RescaleIntercept`.
2. Recortar al rango p1-p99 de esa imagen y escalar a 0-255.
3. Invertir MONOCHROME1 al final, como `255 - x`.
4. Redimensionar a 224x224 con antialias y repetir en 3 canales.
5. `resnet.preprocess_input` una sola vez.

Lo que se midio para decidirlo:

| Hecho | Medida |
|---|---|
| `BitsStored` no es uniforme | 12 bits 73 %, 14 bits 18 %, 15 bits 9 % |
| `RescaleSlope` no siempre es la identidad | 7 de 800 traen slope 0,684 e intercept 200,0 |
| Hay saturacion | 17 de cada 20 imagenes de 12 bits llegan al maximo teorico |
| `WindowCenter` y `WindowWidth` | solo en el 21 % de los archivos |
| `VOILUTSequence` | en ninguno |

Contraste aprovechado, medido como desviacion tipica sobre 30 imagenes:

| Politica | Contraste |
|---|---:|
| Recorte p1-p99 | 57,7 |
| min-max por imagen | 27,7 |
| Dividir por `2**bits-1` | 21,5 |

Por eso se descarta `apply_voi_lut`, que solo seria aplicable al 21 % y supondria dos reglas distintas sobre el mismo conjunto. Como esa presencia depende del equipo, podria introducir una senal espuria.

El 1 y el 99 son una eleccion de arranque. Se pueden ajustar, pero deben quedar fijos antes de entrenar y ser los mismos en los dos clientes.

Invertir al final da exactamente el mismo resultado que invertir antes de escalar, con diferencia maxima de 2,8e-14, y evita tener que elegir que maximo usar.

## Controles del Paso 3

Ejecutados sobre `client_1` y `client_2` en validacion:

```
OK     Imagenes recorridas  60 (num_examples 60)
OK     Numero de lotes      8 (esperado 8)
OK     Formas               imagen (224, 224, 3), etiqueta (1,)
OK     Valores finitos      sin NaN ni infinitos
OK     Etiquetas 0.0 o 1.0  distintas: [0.0, 1.0]
OK     Balance de clases    0: 30, 1: 30
Rango tras preprocess_input: [-123.7, 151.1]
```

El rango es el esperado al restar las medias de ImageNet a una imagen de 0-255.

## Modelo

ResNet50 con pesos de ImageNet, base congelada, promedio global y una salida sigmoide. SGD sin momentum.

| Medida | Valor |
|---|---:|
| Tensores de peso | 320 |
| Parametros entrenables | 2 049 |
| Parametros congelados | 23 587 712 |

Dos llamadas a `build_model` devuelven pesos identicos en los 320 tensores. Es imprescindible para FedAvg: sin eso cada cliente partiria de una inicializacion distinta.

## Tiempos medidos en CPU

| Operacion | Tiempo |
|---|---:|
| Decodificar un DICOM | 0,337 s |
| Primera epoca de 280 imagenes, incluyendo decodificacion | 41 s |
| Epocas siguientes, ya en memoria | 7 s |

El dataset se guarda en memoria tras decodificar. Las 800 imagenes a 224x224x3 ocupan unos 480 MB, que caben de sobra en 32 GB.

## Referencias locales

Un modelo por cliente, entrenado solo con su train y evaluado solo con su validacion. Test no se uso.

| Escenario | Cliente | val AUROC | val AUPRC | Segundos |
|---|---|---:|---:|---:|
| federacion (2 epocas) | `client_1` | 0,5350 | 0,5351 | 36 |
| federacion (2 epocas) | `client_2` | 0,4878 | 0,5122 | 60 |
| piloto (10 epocas) | `client_1` | 0,5906 | 0,6452 | 101 |
| piloto (10 epocas) | `client_2` | 0,5233 | 0,4897 | 116 |

**Las cuatro estan en el azar.** El codigo funciona de extremo a extremo, la perdida es finita y no hay fallos, pero esta configuracion no aprende a detectar neumonia.

## Diagnostico: el problema es sobreajuste

El dato decisivo es la curva por epocas sobre `client_1`, con la base congelada y tasa 0,001:

| Epocas | train loss | train AUROC | val loss | val AUROC |
|---:|---:|---:|---:|---:|
| 10 | 0,6859 | 0,5646 | 0,6947 | 0,5678 |
| 20 | 0,6401 | 0,6851 | 0,6887 | **0,5761** |
| 30 | 0,5999 | 0,7569 | 0,7077 | 0,5456 |
| 40 | 0,5742 | 0,8008 | 0,7073 | 0,5667 |
| 50 | 0,5498 | 0,8320 | 0,7153 | 0,5483 |
| 60 | 0,5411 | 0,8384 | 0,7163 | 0,5444 |

El modelo **si** consigue ajustar sus datos de entrenamiento: el AUROC de train sube de 0,56 a 0,84. Mientras tanto el de validacion se queda plano en torno a 0,55 y la perdida de validacion empeora de forma sostenida. Eso es sobreajuste de libro.

Dicho de otro modo, las caracteristicas de ResNet50 congelada **si** contienen informacion suficiente para separar 280 imagenes concretas. Lo que no hay es suficiente para generalizar a pacientes nuevos.

El mejor AUROC de validacion aparece hacia la epoca 20 y vale 0,5761. Sigue siendo practicamente el azar.

### Lo que se descarto como causa

**La tasa de aprendizaje no es el cuello de botella.** Barrido sobre `client_1`, 10 epocas por tasa:

| Tasa | train loss | val AUROC |
|---:|---:|---:|
| 0,001 | 0,6859 | 0,5678 |
| 0,01 | 1,2113 | 0,5161 |
| 0,1 | 10,7587 | 0,5333 |
| 1,0 | 98,7280 | 0,5000 |

0,001 es la mejor de las cuatro y subirla empeora.

**Descongelar no ayudo.** Con el ultimo bloque `conv5` descongelado, BatchNormalization congelada y tasa 1e-4, 10 epocas dieron val AUROC 0,5439 frente a 0,5678 con la base congelada. Los parametros entrenables pasaron de 2 049 a 14 955 521 y el tiempo por epoca de 7 a 12 segundos. Mas capacidad sobre los mismos 280 ejemplos agrava el sobreajuste en lugar de corregirlo.

### Que se recomienda, por orden

1. **Ampliar la muestra.** Es la palanca que ataca la causa medida. El conjunto elegible tenia 2 803 pacientes negativos y 3 526 positivos antes de recortar a 800, asi que hay margen para multiplicar el train por cliente sin cambiar ni una regla de seleccion. `prepare_sample.py` ya sabe hacerlo con el mismo criterio.
2. **Elegir el checkpoint con validacion.** La curva tiene su mejor punto hacia la epoca 20 y empeora despues. El servidor ya guarda un checkpoint por ronda, de modo que la infraestructura para esto existe.
3. **Aumento de datos y regularizacion.** Giros pequenos, desplazamientos y cambios de contraste en el componente de datos. Es barato y ataca el sobreajuste directamente.
4. **Revisar la etiqueta.** `docs/06` deja abierto el generador de `report_fixed.json`. Una etiqueta ruidosa pone un techo que ningun modelo supera.

Ninguna de las cuatro se ha aplicado. Ampliar la muestra o cambiar la arquitectura afecta a `docs/05` y al contrato, asi que es una decision del equipo y no de una sola parte.

### Lo que esto no significa

Que las metricas esten en el azar no dice nada sobre el aprendizaje federado. Dice que esta configuracion concreta, con este volumen de datos, no generaliza. La federacion con los mismos componentes dara resultados igual de pobres, y esa comparacion seguira siendo valida: si la referencia local y la federacion estan ambas en el azar, la federacion no aporta todavia y asi hay que reportarlo.

Tampoco es una medida clinica. Sigue sin haber validacion clinica, la etiqueta viene de informes y la clase 0 no significa radiografia sana.

## Cambio al contrato que Sebastian debe conocer

`docs/07` dice que `dicom_path` se resuelve respecto de `data_root`. El manifiesto maestro `download_manifest.csv` no cumple eso: guarda la ruta relativa a la raiz del repositorio, con el prefijo `data/raw/dicom/` ya incluido. Unirla a `data_root` daria una ruta duplicada.

Los seis manifiestos de cliente que genera `split.py` **si** cumplen el contrato: guardan `dicom_path` relativo a `data_root`. `load_data` resuelve `Path(data_root) / dicom_path`.

Las columnas de esos seis archivos son `image_id`, `patient_id`, `dicom_path`, `label` y `view`. La vista se incluye porque hace falta para comprobar el reparto AP/PA.

El resto del contrato se cumple sin cambios: `num_examples` cuenta imagenes, solo se mezcla train, `preprocess_input` se aplica una sola vez dentro del componente de datos, las imagenes salen `(lote, 224, 224, 3)` float32 y las etiquetas `(lote, 1)` float32.
