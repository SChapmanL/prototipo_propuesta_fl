# Repositorio y pipeline

## Estado del repositorio

Se propuso una estructura. No se creó ni se confirmó un repositorio remoto durante la sesión. Esta carpeta es un paquete de continuidad. Si Claude tiene acceso a una carpeta real de proyecto, debe inspeccionarla antes de copiar o mover archivos.

## Estructura inicial entregada

| Ruta | Propósito | Git |
|---|---|---|
| AGENTS.md | Instrucciones de trabajo | Sí |
| LEER_PRIMERO.md | Entrada al contexto | Sí |
| README.md | Inicio del proyecto | Sí |
| .gitignore | Exclusiones | Sí |
| scripts/prepare_sample.py | Selección de muestra | Sí |
| scripts/download_dicoms.py | Descarga y verificación | Sí |
| configs/sample_config.json | Reglas y hashes de las fuentes | Sí |
| docs/ | Decisiones, estado y plan | Sí |
| data/manifests/ | Listas con identificadores | No |
| data/raw/metadata/ | Fuentes originales, por copiar desde PC | No |
| data/raw/dicom/ | DICOM locales, pendientes | No |
| outputs/ | Resultados y registros futuros | No |

No ejecutar git add sobre el paquete sin revisar primero .gitignore y git status. No subir el ZIP, metadatos, informes, manifiestos ni DICOM. No colocar un archivo de licencia que aparente conceder derechos sobre el dataset.

## Estructura futura propuesta

Un paquete src/pneumonia_fl/ con data/prepare.py, data/split.py, data/validate.py, data/dicom.py, data/loader.py, model.py, client_app.py, server_app.py y metrics.py. Añadir pyproject.toml y dependencias fijadas tras verificar compatibilidad. Una sola implementación de modelo y cliente parametrizada por client_id. No duplicar código para cada cliente.

El nombre original propuesto es pneumonia-fl-prototype. Su adopción no está confirmada.

## Etapas y condiciones

1. Inventario: completado con las tres fuentes aportadas.
2. Selección: lista candidata de 800 preparada y reproducible.
3. Descarga y control: siguiente hito, empezar con pilot20.
4. Particiones: pacientes exclusivos entre clientes y train/val/test, después de verificar la muestra.
5. Preprocesamiento: DICOM a tensor 224x224x3. Documentar interpretación de píxeles, inversión, transformaciones de intensidad y redimensionamiento. No reducir uint16 con un simple cast a uint8.
6. Prueba local: un cliente y subconjunto diminuto de entrenamiento.
7. Federación: dos clientes, servidor Flower, FedAvg.
8. Evaluación: mismo modelo global en ambos clientes. Validación para decisiones y prueba al final.
9. Non-IID: cambiar proporciones de clases de entrenamiento, manteniendo evaluación fija, una vez estable el flujo inicial.

## Configuración del modelo aún propuesta

ResNet50 ImageNet, sin clasificador original, base congelada, promedio global, salida sigmoide, entropía cruzada binaria. SGD sin momentum, tasa 0.001 inicial, batch 8, una época local por ronda. Dos rondas de integración y diez para primer piloto. No existe medición de duración.

Para Keras ResNet50, usar su preprocess_input compatible con entradas previas en rango 0-255. No agregar una división entre 255 sin comprobar la escala esperada. No hay política final de ventana/normalización DICOM implementada.

## FedAvg secuencial en CPU

En cada ronda, guardar los pesos globales iniciales. Entrenar cliente 1 desde esos pesos. Entrenar cliente 2 desde los mismos pesos iniciales, no desde los del cliente 1. Agregar solo después de obtener ambas respuestas. Ponderar por muestras locales de entrenamiento. No confundir ejecución secuencial con aprendizaje secuencial sobre un único modelo compartido.

El servidor no necesita una GPU para promediar pesos. La base congelada reduce el coste de entrenamiento, pero ResNet sigue ejecutando el cálculo hacia adelante. Guardar características es una optimización futura, no parte del código actual.

## Evaluación propuesta

Pérdida, AUROC, AUPRC, sensibilidad, especificidad y matriz de confusión por cliente. Umbral inicial 0.5, ajustes solo con validación. No promediar AUROC/AUPRC locales y presentarlos como métricas agrupadas. El balance 50/50 del piloto no representa la prevalencia real.

