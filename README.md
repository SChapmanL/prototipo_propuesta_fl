# Prototipo federado de neumonía con CheXpert Plus

Inicio: leer LEER_PRIMERO.md y docs/04_SIGUIENTE_PASO.md.

Estado: selección de 800 pacientes preparada. Descarga y validación de la muestra pendientes. Cuatro DICOM independientes de prueba con etiquetas 0/1 se decodificaron correctamente.

Stack previsto: Python, TensorFlow/Keras, ResNet50, Flower y FedAvg. Dos clientes simulados en una PC sin GPU, con 32 GB de RAM.

No se ha entrenado un modelo ni creado la federación. Este paquete no incluye radiografías ni credenciales.

Primera orden, después de crear el entorno:

```powershell
.\.venv\Scripts\python.exe scripts/download_dicoms.py --manifest data/manifests/pilot20.csv
```

No subir data/, outputs/, archivos originales ni este ZIP a GitHub.

