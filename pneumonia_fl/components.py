"""Punto único de conexión entre la federación y el modelo local.

El cliente y el servidor Flower importan las cuatro funciones del contrato
SOLO desde este archivo. Para cambiar de implementación se cambia este
archivo y ningún otro.

Contrato completo: docs/07_CONTRATO_INTEGRACION.md
"""

# --- Implementación activa: REAL (DICOM de CheXpert Plus y ResNet50) ---
from pneumonia_fl.data.dicom import load_data
from pneumonia_fl.model import build_model
from pneumonia_fl.training import evaluate_local, train_local

# False: los resultados guardados se marcan como ejecución con datos reales.
COMPONENTS_ARE_PROVISIONAL = False

# --- Prueba técnica sintética: reemplazar el bloque anterior por ---
# from pneumonia_fl.provisional import (
#     build_model,
#     evaluate_local,
#     load_data,
#     train_local,
# )
# COMPONENTS_ARE_PROVISIONAL = True

__all__ = [
    "build_model",
    "evaluate_local",
    "load_data",
    "train_local",
    "COMPONENTS_ARE_PROVISIONAL",
]
