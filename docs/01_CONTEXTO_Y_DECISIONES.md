# Contexto y decisiones

## Investigación

Integrantes: Ariana Belén Blanco Quintana y Sebastián Ricardo Chapman López. Curso: Propuesta de Investigación, sección 856, Universidad de Lima. Asesor técnico: profesor Cesar Huarachi.

Título de trabajo: Evaluación del aprendizaje federado para la detección de neumonía en radiografías de tórax bajo escenarios de distribución heterogénea de datos.

La investigación estudia detección de neumonía bajo heterogeneidad de datos. El aprendizaje federado es el medio para entrenar sin compartir imágenes durante las rondas. Este prototipo es una prueba técnica inicial. No resuelve todavía toda la evaluación de la tesis.

El profesor recomendó empezar con un dataset, una arquitectura ResNet y FedAvg. Se consideraron VinDr-CXR y RSNA. Se eligió CheXpert Plus para este prototipo porque se dispone de acceso a sus DICOM y etiquetas. No retomar la búsqueda de datasets para este siguiente paso.

## Decisiones actuales

| Elemento | Decisión |
|---|---|
| Fuente | CheXpert Plus |
| Población del piloto | Edad >=18, sin límite superior |
| Vista | Frontal AP o PA |
| Unidad | Una imagen elegible por paciente |
| Tarea | Clasificación binaria de Pneumonia |
| Etiquetas | report_fixed.json suministrado |
| Política | Conservar 1 y 0. Excluir -1 y null |
| Muestra candidata | 800 pacientes, 400 por clase |
| Reproducibilidad | Semilla 42 y orden por SHA256 |
| Framework | TensorFlow/Keras |
| Modelo | ResNet50 preentrenada en ImageNet |
| Federación | Flower, dos clientes y un servidor, FedAvg |
| Hardware | PC sin GPU, procesador fuerte y 32 GB de RAM |
| Primera ejecución | CPU, clientes secuenciales |

El rango de edad activa fue una intención anterior. El piloto operativo actual usa todos los adultos y contiene pacientes de 88 años. No afirmar que la muestra está limitada a edad laboral.

## Alcance de la simulación

Los dos clientes son particiones de una sola fuente. No representan dos hospitales reales. No se demuestra generalización geográfica ni entre hospitales con este piloto. La privacidad es una motivación, pero ejecutar todo en una sola PC solo proporciona separación lógica.

FedAvg agrega pesos, ponderados por el número de muestras locales de entrenamiento. No describir este código como intercambio exclusivo de gradientes. No existe aún implementación Flower.

## Políticas y clientes

Se acordó una sola política de etiquetas. Dos clientes no implican dos datasets de origen ni dos políticas. Si se comparan políticas después, serán variantes experimentales con manifiestos distintos, manteniendo las imágenes originales y una evaluación comparable.

