# Prompt para la sesión de Ariana

Copia el bloque completo en una sesión nueva de Claude Code abierta en la raíz del repositorio.

```text
Hola Claude. Soy Ariana. En este proyecto me encargo de los datos, del preprocesamiento DICOM, de ResNet50 y del entrenamiento local. Sebastián se encarga de la federación con Flower. Ayúdame a avanzar con mi parte, paso a paso.

## Antes de cambiar nada

1. Lee AGENTS.md, LEER_PRIMERO.md y los documentos de docs/ en el orden que indica LEER_PRIMERO.md. Lee también docs/07_CONTRATO_INTEGRACION.md y docs/08_FEDERACION_FLOWER.md.
2. Revisa git status, la rama actual y los archivos reales. Distingue lo implementado, lo ejecutado, lo comprobado y lo pendiente.
3. Comprueba qué existe en mi PC: data/manifests/ (download_manifest.csv, pilot20.csv, reserve100.csv), data/raw/dicom/ y el espacio libre en disco. Estos archivos no están en Git. Si faltan, dímelo y detente. No los regeneres sin preguntarme.
4. Crea la rama feat/model-local desde main si no existe. Haz commits pequeños.

## Reglas que no cambian

- Stack: TensorFlow/Keras + ResNet50. Mi código debe funcionar sin Flower.
- Etiquetas Pneumonia 0/1. Excluye -1 y null. Nunca los conviertas en 0.
- Una imagen frontal por paciente. Ningún paciente se repite entre clientes ni entre particiones.
- La clase 0 no significa radiografía sana. Validar un DICOM no es validación clínica.
- No subas DICOM, informes, metadatos originales, manifiestos con identificadores ni credenciales a GitHub.
- Nunca me pidas contraseñas ni tokens en el chat. Yo me autentico en Redivis en mi propia terminal.
- Preserva los archivos originales. Si la muestra cambia, guárdala como una versión nueva.
- Usa CPU. La PC no tiene GPU.
- Explícame cada paso en español, con frases cortas. Estoy aprendiendo TensorFlow.

## Paso 1. Descargar y validar los 800 DICOM

Objetivo: tener 800 DICOM válidos, 400 de clase 0 y 400 de clase 1, de 800 pacientes distintos.

1. Verifica que hay al menos 15 GB libres. La muestra ocupa unos 10,07 GB.
2. Guarda una copia del registro actual del piloto (data/raw/dicom/download_validation.csv) con otro nombre. El script sobrescribe ese registro en cada ejecución.
3. Ejecuta primero sin --download para ver la cantidad y el tamaño:
   .\.venv\Scripts\python.exe scripts/download_dicoms.py --manifest data/manifests/download_manifest.csv
4. Después descarga y valida. Yo ejecuto el comando y me autentico:
   .\.venv\Scripts\python.exe scripts/download_dicoms.py --manifest data/manifests/download_manifest.csv --download
   El script verifica tamaño, MD5, lectura de píxeles, rango no constante y MONOCHROME1/2. Los archivos que ya existen se validan sin descargarse otra vez. Si se interrumpe, vuelve a ejecutarlo.
5. Escribe un script de control que lea el manifiesto y el registro y compruebe:
   - 800 filas con status=ok.
   - 800 patient_id distintos.
   - 400 etiquetas 0 y 400 etiquetas 1.
   - Ningún valor de label distinto de 0 o 1.
   - Cada dicom_path existe en disco.
   El script debe mostrar conteos, no identificadores de pacientes.
6. Si hay fallos, no los reemplaces automáticamente. Registra el motivo. Propón un reemplazo de reserve100.csv de la misma clase, en el orden documentado en docs/03_DATOS_Y_SELECCION.md, y espera mi confirmación. Guarda la muestra corregida como una versión nueva y documenta cada sustitución.
7. Revisa visualmente unas 8 imágenes (4 de cada clase, con MONOCHROME1 y MONOCHROME2 si existen). Confirma que no salen invertidas.

Criterio de cierre: 800 ok, 800 pacientes, 400/400, registro de fallos y sustituciones. Actualiza docs/02_ESTADO_VERIFICADO.md con los resultados reales.

## Paso 2. Particiones por paciente

1. Divide los pacientes en train/val/test 70/15/15, estratificado por etiqueta. Usa una regla reproducible por hash con seed 42, como la selección actual.
2. Reparte cada partición entre client_1 y client_2. Cada paciente pertenece a un solo cliente y a una sola partición.
3. Los 20 pacientes de pilot20 deben quedar en train. Descuéntalos de sus cupos por clase.
4. Escribe seis manifiestos: data/manifests/client_1/train.csv, val.csv, test.csv y los equivalentes para client_2. Columnas mínimas: image_id, patient_id, dicom_path, label.
5. Agrega controles: sin pacientes compartidos, suma total de 800, conteos por clase y por vista AP/PA en cada archivo. Guarda un resumen sin identificadores.

Propuesta de archivo: pneumonia_fl/data/split.py (ajústalo a la estructura real).

## Paso 3. Preprocesamiento DICOM y load_data

1. Lee el DICOM con pydicom. Invierte MONOCHROME1. Convierte a float32. Redimensiona a 224x224 y repite en 3 canales.
2. Lleva la intensidad al rango 0 a 255 y aplica tf.keras.applications.resnet.preprocess_input una sola vez, aquí y en ningún otro lugar.
3. Implementa load_data(client_id, split, data_root, manifest_root, batch_size) -> (dataset, num_examples) según docs/07_CONTRATO_INTEGRACION.md. num_examples cuenta imágenes, no lotes. Mezcla solo train.
4. Controles: forma (batch, 224, 224, 3), dtype float32, valores finitos, etiquetas 0/1. Guarda una figura de control visual en outputs/ (no la subas a Git).

Propuesta de archivo: pneumonia_fl/data/dicom.py.

## Paso 4. Modelo, entrenamiento y evaluación local

1. build_model(config): ResNet50 con pesos de ImageNet, base congelada, cabeza binaria con sigmoide, SGD sin momentum, learning-rate y seed tomados del config. Llama a keras.utils.set_random_seed(config["seed"]). Mismo orden de pesos en cada llamada.
2. train_local(model, train_data, epochs): entrena el modelo recibido sin reinicializarlo. Devuelve un diccionario con valores float.
3. evaluate_local(model, eval_data): devuelve loss, accuracy, AUROC y AUPRC como float.
4. Prueba primero con los 20 DICOM de pilot20 (solo train) para verificar que la pérdida es finita. Mide el tiempo por época en CPU y anótalo.
5. Lee la configuración desde [tool.flwr.app.config] de pyproject.toml. No copies valores en otro lugar.

Propuesta de archivos: pneumonia_fl/model.py, training.py y metrics.py. No modifiques client_app.py, server_app.py ni components.py. Sebastián conectará tus funciones en components.py.

## Paso 5. Referencias locales

1. Entrena un modelo solo con el train de client_1 y otro solo con el train de client_2. Usa la misma configuración que la federación.
2. Evalúa cada modelo con la validación de su cliente.
3. Guarda las métricas, la configuración y el tiempo de cada modelo en outputs/local_baselines/.
4. No evalúes en test en esta etapa.

## Al terminar cada paso

- Muéstrame los comandos ejecutados y los resultados reales.
- Actualiza la documentación y anota los problemas pendientes.
- Avísame qué debo comunicarle a Sebastián, en especial cualquier cambio al contrato.
- Empieza ahora por la revisión inicial y el Paso 1. No avances al Paso 2 sin que yo confirme el cierre del Paso 1.
```
