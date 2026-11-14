# Limitaciones y fuentes

## Pendientes concretos

1. Verificar el generador de report_fixed.json. El portal usa el nombre CheXpert Labels y el repositorio oficial documenta archivos similares bajo CheXbert. El archivo y su hash están identificados. El generador no está resuelto.
2. La etiqueta proviene de informes. No es confirmación clínica individual ni anotación manual de cada imagen.
3. Los filtros actuales no verifican una cohorte no COVID. No afirmar esa exclusión.
4. La cobertura del índice DICOM es menor que la de metadatos. Exigir correspondencia por ruta antes de seleccionar.
5. Confirmar visualmente el preprocesamiento. La decodificación ya probada no basta para cerrar ese control.
6. Las dependencias de descarga no están fijadas a versiones. Registrar las versiones del entorno que funcione. No decir que existe un lockfile probado.
7. El descargador no se probó contra la API autenticada. Sí se probaron el modo sin descarga y la comprobación local con cuatro DICOM.
8. No hay evidencia de que los 800 archivos ya estén en la PC de la usuaria. Preguntar por el resultado de ejecución o inspeccionarlo si hay acceso local real.
9. La muestra y las particiones se deben congelar después de sustituciones y verificación. Preservar la trazabilidad de cada sustitución.
10. Mantener pilot20 en entrenamiento al asignar las particiones, porque se utiliza para depuración.

## Qué no hacer al retomar

- No descargar los chunks grandes al azar.
- No volver a pedir los índices si las listas actuales bastan para descargar.
- No etiquetar null como negativo ni sustituir Pneumonia por Consolidation o Lung Opacity.
- No dividir una misma persona entre clientes.
- No presentar simulación con una fuente como validación multicéntrica real.
- No iniciar una web, base de datos o API de producto antes de validar el pipeline mínimo.
- No intentar usar la sesión del navegador remoto como si fuera la sesión local de la usuaria.

## Fuentes oficiales de referencia

Las APIs pueden cambiar. Verificar la documentación de las versiones que se instalen.

- CheXpert Plus: https://aimi.stanford.edu/datasets/chexpert-plus
- Portal: https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x/tables?v=next
- Repositorio: https://github.com/Stanford-AIMI/chexpert-plus
- Paper: https://arxiv.org/abs/2405.19538
- Redivis File.download: https://docs.redivis.com/api/client-libraries/redivis-python/reference/file/file.download
- Flower y TensorFlow: https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html
- Flower simulaciones: https://flower.ai/docs/framework/how-to-run-simulations.html
- Keras transferencia: https://keras.io/guides/transfer_learning/
- TensorFlow ResNet50: https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
- pydicom: https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html

## Colab

Se consideró como opción para acelerar el entrenamiento. No se decidió migrar. La disponibilidad de GPU y la duración de sesión no están garantizadas. Una simulación de dos clientes en una sola máquina no requiere dos notebooks ni conexión de trabajadores externos. Revisar las condiciones actuales de Colab antes de elegir ese entorno.

