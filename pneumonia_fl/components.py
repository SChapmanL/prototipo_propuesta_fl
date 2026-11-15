"""Punto único de conexión entre la federación y el modelo local.

El cliente y el servidor Flower importan las cuatro funciones del contrato
SOLO desde este archivo. Para pasar del componente provisional a los
componentes reales de Ariana se cambia este archivo y ningún otro.

Contrato completo: docs/07_CONTRATO_INTEGRACION.md
"""

# --- Implementación activa: PROVISIONAL (datos sintéticos, modelo pequeño) ---
from pneumonia_fl.provisional import (
    build_model,
    evaluate_local,
    load_data,
    train_local,
)

# Marca los resultados guardados como prueba técnica y no como resultado clínico.
COMPONENTS_ARE_PROVISIONAL = True

# --- Integración con Ariana: reemplazar el bloque anterior por algo como ---
# from pneumonia_fl.data.loader import load_data
# from pneumonia_fl.model import build_model
# from pneumonia_fl.training import train_local, evaluate_local
# COMPONENTS_ARE_PROVISIONAL = False

__all__ = [
    "build_model",
    "evaluate_local",
    "load_data",
    "train_local",
    "COMPONENTS_ARE_PROVISIONAL",
]
