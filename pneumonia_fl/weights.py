"""Utilidad compartida por cliente y servidor para comparar listas de pesos."""

import hashlib


def weights_fingerprint(weights):
    """Resume una lista de arreglos NumPy en un entero (huella).

    Dos listas con los mismos valores producen la misma huella. El servidor la
    calcula sobre los pesos que envía y cada cliente sobre los que recibe. Si
    coinciden, ambos clientes partieron de los mismos pesos globales.

    Se devuelve un entero porque Flower solo transporta números como métricas.
    """
    digest = hashlib.sha256()
    for array in weights:
        digest.update(array.tobytes())
    return int(digest.hexdigest()[:12], 16)
