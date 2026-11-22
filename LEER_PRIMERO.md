# Continuidad del prototipo CheXpert Plus

Actualización del reparto de trabajo: 15 de noviembre de 2026. Estado y orden de lectura actualizados el 22 de noviembre de 2026.

**Si vienes a continuar la federación, empieza por `docs/04_SIGUIENTE_PASO.md` y `docs/11_EJECUCION_FEDERADA_REAL.md`.**

## Cómo usar este repositorio

Abre el repositorio como proyecto local en Claude Code. Inspecciona la rama, el estado de Git y los archivos existentes antes de cambiar código. Conserva los cambios locales de ambos integrantes. Si recibes un ZIP, descomprímelo primero.

Orden de lectura:

1. AGENTS.md.
2. Este archivo completo.
3. docs/01_CONTEXTO_Y_DECISIONES.md.
4. docs/02_ESTADO_VERIFICADO.md.
5. docs/03_DATOS_Y_SELECCION.md.
6. docs/04_SIGUIENTE_PASO.md.
7. docs/05_REPOSITORIO_Y_PIPELINE.md.
8. docs/06_LIMITACIONES_Y_FUENTES.md.
9. docs/07_CONTRATO_INTEGRACION.md, la frontera entre el modelo local y la federación.
10. docs/08_FEDERACION_FLOWER.md, la federación y su prueba sintética.
11. docs/10_MODELO_LOCAL.md, el modelo local y las referencias por cliente.
12. docs/11_EJECUCION_FEDERADA_REAL.md, la ejecución con datos reales y el reporte formal.

`docs/09_PROMPT_ARIANA.md` es el texto para iniciar la sesión de Ariana. No hace falta leerlo en otras sesiones.

El plan histórico está en docs/archive/Plan_prototipo_federado_CheXpert.md. El reparto acordado en este archivo reemplaza las sugerencias anteriores de asignación de tareas. La autorización para avanzar en paralelo descrita aquí reemplaza la indicación anterior de no iniciar ningún entrenamiento. Las políticas de datos y los controles de validación siguen vigentes.

Este reparto describe responsabilidades y entregables, no trabajo terminado. Para conocer el progreso, contrasta la documentación con el código y los resultados reales.

## Objetivo y stack

Construir un prototipo que detecte neumonía con CheXpert Plus, TensorFlow/Keras y ResNet50. Flower coordina dos clientes simulados y un servidor mediante FedAvg. Cada cliente entrena con sus propios datos. Durante las rondas se intercambian parámetros del modelo y métricas, sin transferir radiografías.

El equipo confirmado es una PC sin GPU y con 32 GB de RAM. Empezar con CPU. Los clientes pueden entrenar secuencialmente, pero ambos deben partir de los mismos pesos globales de cada ronda.

ResNet50 es aprendizaje profundo. La parte de Ariana se denomina “modelo y entrenamiento local”, aunque antes la llamáramos “ML clásico”. La parte de Sebastián es “coordinación federada”. Flower utiliza el modelo y el entrenamiento de Ariana.

## Estado conocido y datos locales

Estado al 22 de noviembre de 2026. El detalle verificado está en `docs/02_ESTADO_VERIFICADO.md`.

- Los 800 DICOM están descargados y validados en la PC de Ariana, 800 de 800 sin fallos.
- Las particiones por paciente, el preprocesamiento, ResNet50 y las referencias locales están hechos. El modelo local no supera el azar todavía.
- La federación usa los componentes reales desde el 22 de noviembre. Se probó de principio a fin con los 20 DICOM de pilot20.
- Falta ejecutar la federación de 10 rondas con los 800 DICOM. Es el siguiente paso.

Comprueba el estado local antes de afirmar que algo está listo. Los datos solo existen en la PC donde se descargaron.

El repositorio contiene documentación, scripts y configuración. Las listas reales de selección pertenecen a data/manifests/ y están excluidas de Git. No supongas que vienen incluidas al clonar.

No incluye los CSV/JSON originales, DICOM ni credenciales. Para regenerar la selección se necesitan las fuentes que el equipo ya descargó:

- df_chexpert_plus_240401.csv.
- report_fixed.json.
- dicom_train.csv.

Las fuentes originales se ubican localmente en data/raw/metadata/. Los DICOM se guardan en data/raw/dicom/. No subas imágenes, informes, metadatos originales, manifiestos con identificadores ni credenciales a GitHub. Comparte código, configuración y ejemplos sintéticos.

## Reparto acordado de todo el pipeline

| Etapa | Responsable principal | Trabajo | Entregable y condición de cierre |
|---|---|---|---|
| 1. Contrato y entorno | Ambos | Acordar interfaces, rutas y versiones compatibles de Python, TensorFlow/Keras y Flower | Contrato documentado e imports comprobados |
| 2. Selección y etiquetas | Ariana | Mantener criterios de adultos, frontales, una imagen por paciente y Pneumonia 0/1. Excluir -1 y null | Selección reproducible y conteos de exclusiones |
| 3. Descarga y validación DICOM | Ariana, con apoyo de Sebastián si hace falta | Reutilizar scripts, comprobar descarga, integridad, decodificación y reemplazos | Muestra física validada y registro de fallos |
| 4. Particiones | Ariana | Congelar train/val/test por cliente sin pacientes compartidos | Seis manifiestos, resumen y controles de exclusividad |
| 5. Preprocesamiento y carga | Ariana | Implementar lectura DICOM, transformaciones de intensidad y tf.data | Una implementación común, control visual y tensores finitos |
| 6. Modelo | Ariana | Construir y compilar una única ResNet50 con salida binaria | Función reutilizable y configuración explícita |
| 7. Entrenamiento y evaluación local | Ariana | Implementar entrenamiento, métricas y guardado/recarga. Preparar referencias locales para comparar después | Funciones probadas con datos locales y resultados identificados |
| 8. Adaptador Flower de cada cliente | Sebastián | Recibir parámetros, cargar el modelo de Ariana, invocar entrenamiento/evaluación y devolver resultados | Una implementación parametrizada por client_id |
| 9. Servidor y FedAvg | Sebastián | Inicialización global común, participación de ambos clientes, rondas y agregación ponderada | Dos rondas completas verificadas y checkpoint global recargable |
| 10. Integración con datos reales | Ambos | Sustituir componentes provisionales por funciones de Ariana y ejecutar el flujo completo | Ejecución reproducible sin duplicar modelo ni preprocesamiento |
| 11. Evaluación final | Ambos | Sebastián coordina el modelo global y Ariana aporta la evaluación local con test reservado | Métricas por cliente, configuración y checkpoint elegido con validación |
| 12. Non-IID y comparación | Ambos | Ariana prepara escenarios de entrenamiento y referencias locales. Sebastián ejecuta la misma federación para cada escenario | Comparación con validación/prueba fijas y diferencias documentadas |

La responsabilidad principal implica mantener el componente. Cualquier ayuda que cambie archivos del otro integrante debe coordinarse para evitar implementaciones incompatibles.

## Contrato para “enchufar” ambas partes

Definan primero este contrato. Las firmas siguientes son una propuesta inicial, no funciones existentes. Si el repositorio ya tiene equivalentes, reutilícenlos y documenten sus firmas reales.

| Función propuesta | Responsable | Entrada | Salida y regla |
|---|---|---|---|
| load_data(client_id, split, data_root, manifest_root, batch_size) | Ariana | Cliente, train/val/test y rutas locales | (dataset, num_examples). Dataset tf.data con imágenes y etiquetas. Conteo de ejemplos, no de batches |
| build_model(config) | Ariana | Configuración común | Modelo Keras compilado con arquitectura y orden de pesos idénticos |
| train_local(model, train_data, epochs) | Ariana | Modelo con pesos recibidos y datos locales | Historial o métricas. Actualiza el modelo recibido y no lo reinicializa |
| evaluate_local(model, eval_data) | Ariana | Modelo y datos de val o test | Diccionario con loss y métricas documentadas |

El adaptador de Sebastián obtiene los pesos actualizados del modelo después de train_local. Recibe el número de ejemplos desde load_data. Convierte los resultados al formato requerido por la API de Flower elegida, conservando solo valores admitidos para el intercambio.

Antes de implementar, fijen además:

- Esquema mínimo del manifiesto: image_id, patient_id, dicom_path y label. client_id y split pueden estar en el manifiesto maestro o quedar determinados por el archivo usado.
- Rutas de integración propuestas: data/manifests/client_1/train.csv, val.csv y test.csv, y sus equivalentes para client_2. Resolver dicom_path respecto de data_root.
- Etiquetas binarias 0/1. La clase negativa no implica una radiografía sana.
- Tensor por imagen: float32 de forma (224, 224, 3), agrupado por lotes. Documentar también la forma y el tipo de las etiquetas.
- Un solo lugar para aplicar resnet.preprocess_input, dentro del componente de datos de Ariana. El cliente Flower no vuelve a normalizar. Respetar el rango previo 0–255 y evitar dividir otra vez entre 255.
- Política común para capas congeladas, BatchNormalization y estado del optimizador. Como arranque propuesto, base congelada y SGD sin momentum.
- Una inicialización global común, incluida la cabeza clasificadora. Sebastián distribuye los mismos pesos a ambos clientes. El constructor compartido no garantiza por sí solo pesos iniciales iguales.
- Configuración compartida de arquitectura, semilla, batch y épocas. No mantener valores distintos copiados en cliente y servidor.

Sebastián necesita las funciones y el contrato de Ariana. No necesita esperar un modelo previamente entrenado para desarrollar la coordinación.

## Cómo avanzar en paralelo

### Ariana

Avanza desde la descarga del piloto y su validación hasta datos, modelo y entrenamiento local. No cambies la selección existente sin necesidad. Entrena con datos reales solo después de validar los archivos utilizados, las etiquetas y la separación de pacientes. Una prueba pequeña debe utilizar únicamente datos de entrenamiento.

Su código de modelo, carga y entrenamiento debe poder ejecutarse sin Flower. Esto permite entender y comprobar primero el aprendizaje local.

### Sebastián

Puede desarrollar Flower desde ahora, sin esperar la descarga completa ni el modelo definitivo. Usa un componente provisional mínimo en TensorFlow/Keras y datos sintéticos locales que respeten las interfaces acordadas. Mantén ese ejemplo separado e identificado como prueba de integración.

No implementes otra ResNet50 ni otro preprocesamiento DICOM como solución definitiva. Al integrar, importa las funciones de Ariana y retira la dependencia del componente provisional en la ejecución real. No sobrescribas los módulos de Ariana.

Las pruebas sintéticas comprueban coordinación, intercambio de parámetros y guardado. No constituyen resultados de detección de neumonía.

### Ambos

Trabajen en ramas separadas y revisen cambios pequeños. Ramas sugeridas: feat/model-local para Ariana y feat/federation para Sebastián. No renombres ramas existentes solo para adoptar estos nombres.

Rutas futuras sugeridas, pendientes de creación:

| Área | Responsable |
|---|---|
| src/pneumonia_fl/data/, model.py, training.py y metrics.py | Ariana |
| src/pneumonia_fl/client_app.py y server_app.py | Sebastián |
| Dependencias, configuración de experimentos y documentación del contrato | Ambos |

La estructura debe adaptarse al código real. No crear módulos vacíos ni reorganizar todo el proyecto para reproducir esta tabla.

## Flujo de integración y verificación

1. Comprobar las interfaces con el componente provisional.
2. Ejecutar entrenamiento local con las funciones y datos validados de Ariana.
3. Integrar esas funciones en el cliente Flower, sin copiar su lógica.
4. Ejecutar dos rondas pequeñas con participación efectiva de los dos clientes.
5. Confirmar que cada cliente recibe los mismos pesos al inicio de cada ronda.
6. Comprobar FedAvg ponderado por los ejemplos reales de entrenamiento. No usar batches ni muestras de validación como ponderación.
7. Guardar y recargar el modelo global y registrar métricas de validación por ronda.
8. Tras comprobar el flujo, ejecutar el piloto completo con la configuración acordada.
9. Elegir checkpoint con validación y usar test al final, una vez fijadas las decisiones.

Al 22 de noviembre, los pasos 1 a 7 están comprobados. `scripts/run_federation.py` automatiza los pasos 8 y 9: ejecuta 10 rondas, elige el checkpoint con mayor AUROC conjunto de validación, evalúa test una vez y escribe un reporte formal.

Si un cliente falla, registrar la ronda como incompleta y no presentarla como federación exitosa de dos participantes. Ejecutar clientes secuencialmente no significa entrenar el cliente 2 desde los pesos recién actualizados por el cliente 1.

Reportar métricas por cliente. No presentar el promedio de AUROC/AUPRC locales como una métrica sobre predicciones agrupadas. Una simulación con una sola fuente no demuestra generalización entre hospitales ni aislamiento de seguridad institucional.

El primer escenario Non-IID cambia solo la asignación de entrenamiento y conserva validación/prueba. Las referencias locales deben usar la misma arquitectura y condiciones comparables antes de atribuir beneficios al aprendizaje federado.

## Estilo de implementación y aprendizaje

Aplicar KISS y DRY mediante abstracción y descomposición:

- Separar datos, modelo, entrenamiento local y coordinación federada con funciones pequeñas y nombres claros.
- Reutilizar un solo modelo, lector y entrenamiento. Ambos clientes ejecutan el mismo código con distintos datos.
- Evitar jerarquías innecesarias, capas genéricas de configuración y manejo de errores para casos hipotéticos.
- Mantener controles útiles de etiquetas, pacientes, formas de tensores, pérdida finita y participación de clientes.
- Explicar en español el propósito de cada función y su relación con ML clásico.
- Documentar instalación, comandos probados, flujo de una ronda y artefactos generados.
- Consultar documentación oficial para las versiones elegidas y no mezclar APIs distintas de Flower.
- Distinguir siempre lo implementado, lo ejecutado, lo comprobado y lo pendiente.

El código debe ser funcional y educativo para estudiantes que aún están aprendiendo TensorFlow y Flower.

## Mensaje para iniciar la sesión de Claude

Lee AGENTS.md, LEER_PRIMERO.md y los documentos de docs/ antes de cambiar código. Respeta el reparto de este archivo: Ariana desarrolla datos, preprocesamiento DICOM, ResNet50, entrenamiento y evaluación local en TensorFlow/Keras, mientras Sebastián desarrolla clientes, servidor, rondas, FedAvg y guardado global con Flower. Si trabajas en la parte de Sebastián, implementa solo la coordinación federada y su adaptador, reutilizando las interfaces de Ariana. Primero documenta el contrato de integración y comprueba el código existente. Puedes avanzar con un componente provisional y datos sintéticos sin esperar los datos reales, identificando sus resultados como pruebas técnicas. Aplica KISS y DRY, descompón responsabilidades y explica cada etapa en español para alguien que no conoce TensorFlow ni Flower. Conserva el stack, las políticas de datos y los cambios existentes. Verifica dos clientes y dos rondas, guarda el modelo global y documenta comandos y resultados reales. Al integrar los componentes de Ariana, evita duplicar el modelo o el preprocesamiento y reserva test para la evaluación final.

