# Plan consolidado del prototipo federado
Fecha: 13 de noviembre de 2026

## 1. Objetivo y alcance

Construir un prototipo reproducible de clasificación binaria de neumonía con CheXpert Plus, TensorFlow/Keras, Flower, dos clientes simulados y un servidor que agrega sus modelos mediante FedAvg.

**Meta de hoy:** una muestra con archivos DICOM descargados y legibles, etiquetas trazables y particiones reproducibles por paciente y cliente. Entrenar la federación es el siguiente hito.

**Estado real:** esta guía consolida el diseño. No se han inspeccionado los archivos de su descarga ni se ha creado todavía la muestra real.

| Decisión | Estado |
|---|---|
| CheXpert Plus como única fuente del prototipo | Confirmado en esta conversación |
| Flower + TensorFlow/Keras | Confirmado |
| Dos clientes y un servidor coordinador | Confirmado |
| ResNet y FedAvg | Confirmado |
| ResNet50 con pesos ImageNet | Configuración propuesta para facilitar implementación oficial en Keras |
| Adultos, radiografías frontales y una imagen por paciente | Filtros propuestos para el piloto |
| Pneumonia 1/0, excluir inciertos y vacíos | Política inicial propuesta, verificar contra la documentación del archivo |
| 800 pacientes/imágenes, 400 positivos y 400 negativos | Objetivo operativo condicionado al inventario |
| 70 % entrenamiento, 15 % validación, 15 % prueba | Propuesta inicial |
| Distribuciones de clases similares entre clientes | Primer escenario de depuración |
| Non-IID controlado | Posterior al primer funcionamiento completo |

Se simulan dos instituciones utilizando particiones de una misma fuente. No se demuestra generalización entre hospitales reales. Las etiquetas automáticas extraídas de informes son la referencia experimental, no una confirmación clínica individual.

## 2. Stack y responsabilidades

| Herramienta | Función |
|---|---|
| Python | Scripts, preparación de datos y aplicación |
| pandas + NumPy | Unir tablas, filtrar, contar y crear manifiestos |
| pydicom | Inspeccionar y decodificar DICOM, con decodificadores adicionales si la compresión lo requiere |
| TensorFlow/Keras | Cargar lotes, construir ResNet, entrenar y evaluar |
| ResNet50 preentrenada | Extraer características y alimentar un clasificador binario |
| Flower | Ejecutar clientes y servidor, transmitir pesos y coordinar rondas |
| FedAvg | Agregar pesos locales ponderados por muestras de entrenamiento |
| scikit-learn | Particiones reproducibles y métricas |
| Matplotlib | Inspección visual y gráficos de control |
| Git + GitHub | Versionar código y documentación, distribuir tareas y revisar cambios |
| Entorno virtual + archivo de dependencias fijadas | Reproducir el mismo entorno en ambas máquinas |

No necesitamos añadir ahora una web, una API propia, una base de datos ni contenedores. Si utilizan Windows con GPU NVIDIA, revisar la instalación de TensorFlow mediante WSL2. Probar una combinación compatible de Python, Flower, TensorFlow y Keras y registrar sus versiones.

## 3. Mapa completo del pipeline

| Etapa | Inputs | Funcionamiento y algoritmos | Outputs | Cuándo |
|---|---|---|---|---|
| A. Inventario | Metadatos, etiquetas y listado de archivos de CheXpert Plus | Identificar versión y fuente de etiquetas, claves de paciente/estudio/imagen y correspondencia de rutas | Inventario y diccionario de columnas | Hoy |
| B. Cohorte elegible | Inventario validado | Filtrar adultos y frontales, definir etiquetas 1/0, excluir incertidumbre o ausencia de etiqueta objetivo, controlar duplicados | Lista de candidatos y registro de exclusiones | Hoy |
| C. Selección y descarga | Candidatos | Selección reproducible por paciente y clase, objetivo de 800 imágenes, descargar solo sus DICOM y comprobar lectura | Muestra física y manifiesto preliminar | Hoy |
| D. Asignación | Muestra comprobada | Partición estratificada por etiqueta con pacientes exclusivos entre clientes y train/val/test | Seis manifiestos y manifiesto maestro | Hoy |
| E. Preprocesamiento | DICOM de cada cliente | Interpretar píxeles y metadatos, ajustar tamaño, replicar grises a tres canales y aplicar preprocesamiento de ResNet | Tensores 224 × 224 × 3 mediante tf.data | Siguiente hito |
| F. Prueba local | Datos de entrenamiento de un cliente | Entrenar el clasificador, comprobar pérdida finita y capacidad de ajustar un subconjunto diminuto | Evidencia de que lectura, etiquetas y modelo funcionan | Antes de federar |
| G. Federación | Dos particiones y un modelo inicial común | Flower coordina entrenamiento local y FedAvg agrega los pesos en cada ronda | Modelo global, registros y checkpoints | Después |
| H. Evaluación | Modelo global, validación y prueba locales | Validar durante desarrollo, elegir modelo con validación y usar prueba al final | Métricas por cliente, matrices de confusión y reporte | Después |
| I. Non-IID | Pipeline estable y particiones conservadas | Cambiar proporciones de clases en entrenamiento, manteniendo validación/prueba fijas | Comparación exploratoria entre escenarios | Posterior |

La asignación queda congelada en archivos. No volver a repartir pacientes al iniciar cada cliente o cada ronda.

## 4. Preparación de la muestra de hoy

### 4.1 Identificar lo que descargaron

Localizar:
- Tabla de metadatos de imágenes y pacientes.
- Tabla de etiquetas, que puede estar en CSV o JSONL.
- Índice o estructura que permita localizar los DICOM.
- Identificador de la partición original.

CheXpert Plus ofrece varias fuentes de etiquetas, incluyendo CheXpert y variantes CheXbert. Registrar exactamente el archivo, el generador y, cuando corresponda, la sección del informe utilizada. Usar la misma fuente y política para ambos clientes.

No es necesario volver a ejecutar un etiquetador. Usaremos las etiquetas suministradas. El archivo de etiquetas puede referirse a rutas JPG aunque las imágenes se descarguen como DICOM. Resolver la correspondencia con el inventario y verificar que cada unión sea inequívoca. No unir por posición de fila ni asumir que basta con cambiar una extensión.

### 4.2 Definir candidatos

| Criterio | Regla del piloto |
|---|---|
| Edad | 18 años o más, excluyendo edad ausente si no puede confirmarse |
| Vista | Frontal AP o PA |
| Etiqueta positiva | Pneumonia = 1 |
| Etiqueta negativa | Pneumonia = 0 |
| Incierto | Excluir Pneumonia = -1 |
| No mencionado/vacío | Excluir inicialmente, no transformarlo automáticamente en 0 |
| Otras etiquetas | Conservar otras patologías y sus incertidumbres, salvo una regla documentada adicional |
| Imagen/paciente | Una imagen elegible por paciente para simplificar |
| Fuente de particiones | Construir el piloto a partir del entrenamiento original, sin incorporar validación/prueba oficiales |

Primero validar los códigos reales del archivo. Si vienen como texto, mapearlos explícitamente. Un valor inesperado debe generar un error o quedar registrado como exclusión, no convertirse silenciosamente en negativo.

No exigir No Finding = 1 a los negativos. La clase 0 significa ausencia explícita de neumonía según el etiquetador, no necesariamente una radiografía sana.

Si un paciente tiene varias imágenes elegibles, seleccionar una con una regla reproducible antes de equilibrar las clases. No escoger manualmente las imágenes que parezcan más fáciles.

### 4.3 Contar antes de fijar el tamaño

Generar conteos tras cada filtro, distinguiendo imágenes, estudios y pacientes. Registrar las distribuciones de edad, sexo, vista AP/PA y etiqueta.

Objetivo propuesto: **800 imágenes de 800 pacientes distintos, con 400 positivos y 400 negativos**. Esta cifra facilita la prueba técnica y no es un cálculo de potencia estadística.

Si no hay suficientes negativos o la descarga demora demasiado, reducir el tamaño y conservar las reglas. No completar cupos repitiendo pacientes ni reclasificando vacíos.

El balance 50/50 sirve para depurar. La precisión y la AUPRC obtenidas en esta muestra no representan automáticamente el rendimiento a la prevalencia natural.

### 4.4 Descargar y verificar

Descargar únicamente los DICOM elegidos. Verificar existencia, tamaño no nulo y decodificación de píxeles. Revisar visualmente unas 20 imágenes, incluyendo ambas clases y vistas disponibles, para detectar inversiones, imágenes vacías o fallos de procesamiento. Esta inspección técnica no confirma el diagnóstico.

Si hay fallos, registrarlos y reemplazar candidatos de manera reproducible antes de congelar la muestra. No descartar silenciosamente archivos durante entrenamiento.

### 4.5 Dividir de manera reproducible

Usar semilla 42 como configuración inicial, guardándola en el proyecto.

Con 800 pacientes y clases equilibradas:
1. Separar entrenamiento, validación y prueba en 70/15/15, por paciente y etiqueta.
2. Repartir cada partición entre los dos clientes, conservando el balance.
3. Verificar exclusividad de pacientes entre todos los grupos.
4. Guardar y congelar la asignación.

| Cliente | Train | Validación | Prueba | Total |
|---|---:|---:|---:|---:|
| Cliente 1 | 280: 140 positivos + 140 negativos | 60: 30 + 30 | 60: 30 + 30 | 400 |
| Cliente 2 | 280: 140 positivos + 140 negativos | 60: 30 + 30 | 60: 30 + 30 | 400 |
| Total | 560 | 120 | 120 | 800 |

Revisar que AP/PA y otras características no queden concentradas accidentalmente en un solo cliente. Proporciones de clases iguales no garantizan IID completo.

La prueba de comunicación de 40 imágenes debe salir exclusivamente del entrenamiento ya asignado, con 20 imágenes por cliente. No reutilizar validación o prueba para esa comprobación.

### 4.6 Contrato de datos

Crear un manifiesto maestro con una fila por imagen:

| Campo | Contenido |
|---|---|
| image_id | Identificador estable y único de imagen |
| patient_id | Identificador de paciente |
| study_id | Identificador de estudio, si está disponible |
| dicom_path | Ruta relativa al directorio local de datos |
| age | Edad utilizada en el filtro |
| view | AP o PA |
| label_raw | Valor original antes del mapeo |
| label | 0 o 1 |
| label_source | Archivo y procedencia de la etiqueta |
| original_split | Partición original de CheXpert Plus |
| client_id | client_1 o client_2 |
| split | train, val o test |

Conservar información adicional útil, como sexo, sin introducirla como entrada al modelo en este piloto. Las rutas absolutas personales se resuelven con una variable local como DATA_ROOT.

**Entregables de hoy:**
- DICOM seleccionados y comprobados.
- manifest.csv con toda la asignación.
- client_1/train.csv, val.csv y test.csv.
- client_2/train.csv, val.csv y test.csv.
- exclusions.csv con motivos de descarte.
- dataset_summary.json con conteos y distribuciones.
- sample_config.yaml con criterios, fuente de etiquetas, semilla y proporciones.
- Informe breve de control de calidad.

**Condición de cierre:** ninguna etiqueta final vacía, solo 0/1, todos los DICOM legibles, ninguna duplicación de paciente, ambas clases presentes en cada partición y todos los conteos consistentes. La generación de tensores y el entrenamiento pueden empezar después.

## 5. Plan de trabajo para hoy

Tiempos estimados de trabajo activo. La descarga depende del volumen y la conexión.

| Paso | Responsable propuesto | Tiempo | Resultado y dependencia |
|---|---|---|---|
| Identificar fuentes, claves y códigos | Ambos | 20–30 min | Acordar contrato de datos |
| Filtrar etiquetas, edad/vista y pacientes | Ariana | 45–60 min | Candidatos con conteos y exclusiones |
| Probar acceso y lectura de 10–20 DICOM | Sebastián | 30–45 min | Confirmar rutas, compresión y herramientas. Puede avanzar mientras se filtra |
| Elegir muestra y descargar archivos | Ambos | Variable | Requiere candidatos y rutas verificadas |
| Validar archivos y cerrar reemplazos | Sebastián | 30–45 min más lectura | Muestra física comprobada |
| Generar particiones y manifiestos | Ariana | 20–30 min | Requiere muestra definitiva |
| Revisar controles y congelar versión | Ambos | 20–30 min | dataset-v0.1 listo para el siguiente hito |

La repartición es una sugerencia, no una asignación previa confirmada. El punto de integración es el manifiesto maestro. Si la descarga limita el cierre, reducir la muestra antes de comprometer los controles.

## 6. Cómo funcionará el entrenamiento posterior

### Configuración inicial propuesta

- ResNet50 con pesos ImageNet, sin clasificador original.
- Entrada de 224 × 224 × 3.
- Promedio global y una neurona sigmoide.
- Pérdida de entropía cruzada binaria.
- Base congelada inicialmente, entrenando el clasificador para validar la integración.
- SGD sin momentum, tasa inicial de 0.001 como valor de prueba, no optimizado.
- Batch de 8 y una época local por ronda.
- Dos rondas para prueba de funcionamiento, diez para el primer piloto.
- Ambos clientes participan en todas las rondas.

El procesamiento DICOM debe seguir una política común y documentada. Para ResNet50 de Keras, el tensor previo a preprocess_input se maneja en rango 0–255. Evitar añadir una división entre 255 que cambie la escala esperada. Cachear transformaciones deterministas localmente puede reducir lecturas repetidas.

### Una ronda

1. El servidor envía los mismos pesos globales a ambos clientes, incluyendo los del clasificador.
2. Cada cliente carga esos pesos y entrena solo con su train local.
3. Cada cliente devuelve pesos actualizados y el número de muestras de entrenamiento utilizadas.
4. El servidor verifica que existen dos respuestas válidas y calcula FedAvg.
5. El servidor distribuye el nuevo modelo global para validación local y la siguiente ronda.

FedAvg para dos clientes:
w_global = [n1 / (n1 + n2)] × w_cliente1 + [n2 / (n1 + n2)] × w_cliente2.

n1 y n2 son muestras de entrenamiento, no batches ni muestras de validación. En el piloto equilibrado con 280 muestras por cliente, cada contribución pesa 0.5.

No se envían imágenes al servidor durante el ciclo. En una computadora compartida, esto es separación lógica de datos y ejecución, no una demostración de aislamiento de seguridad.

Las capas BatchNormalization y el estado del optimizador deben tener un tratamiento consistente. La base congelada y SGD sin momentum simplifican la primera versión. Cualquier ajuste fino posterior se aplica igual en ambos clientes y se registra como otra configuración.

### Evaluación

Usar validación durante el desarrollo y reservar prueba hasta fijar el modelo y la configuración. Evaluar el mismo modelo global en ambos clientes. Reportar pérdida, AUROC, AUPRC, sensibilidad, especificidad y matriz de confusión por cliente. El umbral binario inicial puede ser 0.5, cualquier ajuste se realiza con validación.

Se pueden sumar matrices de confusión con el mismo umbral para obtener métricas conjuntas basadas en conteos. El promedio de AUROC/AUPRC locales no equivale a calcular esas métricas sobre todas las predicciones agrupadas.

Para el primer escenario Non-IID, modificar solo el reparto de entrenamiento, por ejemplo 80/20 y 20/80 de positivos/negativos, conservando los conjuntos de evaluación. Mantener ambas clases en ambos clientes.

---

## 7. Estructura de GitHub, separada del pipeline

Nombre sugerido: pneumonia-fl-prototype. Es un diseño de directorios, no un repositorio ya creado.

Usar un único paquete y una única implementación del modelo. Ambos clientes ejecutan el mismo código con distinto client_id y manifiesto.

| Ruta relativa | Propósito | Git |
|---|---|---|
| README.md | Objetivo, requisitos, pasos y estado real | Sí |
| pyproject.toml | Paquete, dependencias y configuración Flower de la versión elegida | Sí |
| requirements-lock.txt | Versiones verificadas del entorno | Sí |
| .gitignore | Exclusiones de datos, secretos y artefactos pesados | Sí |
| .env.example | Nombres de variables como DATA_ROOT, sin valores privados | Sí |
| configs/sample.yaml | Fuente, filtros, semilla y tamaño del piloto | Sí |
| configs/experiment.yaml | Modelo, batch, épocas y rondas | Sí |
| src/pneumonia_fl/data/prepare.py | Uniones, etiquetas, filtros y selección | Sí |
| src/pneumonia_fl/data/split.py | Asignación por paciente y cliente | Sí |
| src/pneumonia_fl/data/validate.py | Controles de integridad y reportes | Sí |
| src/pneumonia_fl/data/dicom.py | Lectura y transformaciones DICOM | Sí |
| src/pneumonia_fl/data/loader.py | Construcción de tf.data a partir de manifiestos | Sí |
| src/pneumonia_fl/model.py | Única función de construcción de ResNet y clasificador | Sí |
| src/pneumonia_fl/client_app.py | Entrenamiento y evaluación locales con Flower | Sí |
| src/pneumonia_fl/server_app.py | Inicialización, FedAvg y guardado global | Sí |
| src/pneumonia_fl/metrics.py | Métricas y agregación correcta de conteos | Sí |
| scripts/prepare_dataset.py | Entrada para generar muestra y manifiestos | Sí |
| scripts/check_dataset.py | Entrada para verificar la muestra | Sí |
| scripts/smoke_local.py | Prueba local pequeña previa a federar | Sí |
| tests/test_data_contract.py | Validar esquema y política con datos sintéticos | Sí |
| tests/test_patient_isolation.py | Comprobar ausencia de pacientes compartidos | Sí |
| tests/fixtures/ | Datos sintéticos, sin radiografías reales | Sí |
| docs/pipeline.md | Diseño y alcance | Sí |
| docs/data_dictionary.md | Campos y significado de las etiquetas | Sí |
| docs/decisions.md | Decisiones, cambios y justificación | Sí |
| notebooks/ | Exploración breve, sin datos o salidas sensibles | Sí, revisado |
| data/raw/ | Metadatos, etiquetas originales y DICOM | No |
| data/manifests/manifest.csv | Muestra y asignación completas | No |
| data/manifests/client_1/ | train.csv, val.csv, test.csv | No |
| data/manifests/client_2/ | train.csv, val.csv, test.csv | No |
| data/processed/ | Caché local de transformaciones | No |
| outputs/ | Checkpoints, logs y reportes de cada ejecución | No, salvo resúmenes seleccionados y revisados |

Cada paquete Python debe incluir sus archivos __init__.py. Los scripts importan funciones de src, evitando duplicar lógica. No crear dos copias independientes de model.py para los clientes.

La tabla describe un directorio local compartido para preparar el piloto. Si más adelante se ejecuta en máquinas separadas, cada cliente recibe únicamente los datos y manifiestos que le corresponden.

### Trabajo entre ambos

- Rama propuesta de Ariana: feat/data-preparation, con preparación, etiquetas, manifiestos y controles.
- Rama propuesta de Sebastián: feat/dicom-flower, con lector DICOM, carga local y luego cliente/servidor.
- Revisar cambios mediante pull requests pequeños antes de integrarlos a main.
- Acordar juntos el esquema del manifiesto y la interfaz de model.py.
- Registrar cambios de muestra como una nueva versión, conservando configuración y trazabilidad.
- Mantener DICOM, informes, etiquetas originales y manifiestos con identificadores fuera de GitHub. Cada integrante accede a los datos conforme a sus permisos de uso.
- Compartir código, configuración y datos sintéticos de ejemplo. No incluir enlaces de descarga con credenciales.

## 8. Fuentes técnicas

Fuentes consultadas en esta conversación, utilizadas para sustentar las decisiones técnicas. Las cantidades y tiempos del piloto son propuestas propias.

1. [CheXpert Plus, Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-plus)
2. [Paper CheXpert Plus, etiquetas y composición](https://arxiv.org/html/2405.19538v1)
3. [Repositorio oficial de CheXpert Plus](https://github.com/Stanford-AIMI/chexpert-plus)
4. [Flower, ejemplo oficial con TensorFlow](https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html)
5. [Flower, FedAvg](https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html)
6. [TensorFlow, ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
7. [TensorFlow, preprocesamiento ResNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet/preprocess_input)
8. [Keras, transferencia y ajuste fino](https://keras.io/guides/transfer_learning/)
9. [pydicom, datos de píxeles](https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html)
10. [TensorFlow, instalación](https://www.tensorflow.org/install/pip)

