"""Lectura de la configuracion comun.

Hay un solo lugar donde viven los valores: [tool.flwr.app.config] de
pyproject.toml. Cliente y servidor de Flower la reciben en context.run_config.
El codigo local, que debe funcionar sin Flower, la lee con este modulo.

No copiar estos valores a ningun otro archivo.
"""

import tomllib
from pathlib import Path


def load_config(path="pyproject.toml"):
    """Devuelve el diccionario de [tool.flwr.app.config]."""
    datos = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return datos["tool"]["flwr"]["app"]["config"]
