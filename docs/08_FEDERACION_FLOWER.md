# Federación con Flower: uso y conexión con la parte de Ariana

> **Aviso del 22 de noviembre de 2026.** Este documento describe la prueba sintética. Desde esa fecha `components.py` usa los componentes reales y `pyproject.toml` tiene 10 rondas. Para ejecutar con datos reales usar `docs/11_EJECUCION_FEDERADA_REAL.md` y `scripts/run_federation.py`, no los comandos `flwr run` de abajo.

Estado al 15 de noviembre de 2026: dos rondas con dos clientes ejecutadas y verificadas en esta PC, con datos sintéticos. Los resultados son una **prueba técnica** de coordinación. No son resultados de detección de neumonía.

## Versiones elegidas

| Componente | Versión | Motivo |
|---|---|---|
| Python | 3.12 | TensorFlow 2.21 no tiene instalador para 3.14. Ray en Windows admite 3.11 y 3.12 |
| TensorFlow | 2.21.0 | Última versión en PyPI. Incluye Keras 3 |
| Flower | 1.43.0 con el extra `simulation` | Última versión en PyPI. El extra instala Ray 2.55.1 |

El entorno `.venv` existente (Python 3.14, descargas de Redivis) no se modifica. La federación usa un entorno aparte, `.venv-fl`.

## Instalación

Requiere unos 5 GB libres en disco: cerca de 2 GB para `.venv-fl` y 1.5 GB más para la copia del entorno que Flower crea en `C:\Users\<usuario>\.flwr\runtime-envs\` en cada ejecución.

```powershell
uv venv .venv-fl --python 3.12
uv pip install --python .venv-fl\Scripts\python.exe -e .
```

## Ejecución de dos rondas

Activa primero el entorno. Es obligatorio: `flwr` lanza otro programa, `flower-superlink`, y solo lo encuentra con el entorno activado.

```powershell
.\.venv-fl\Scripts\Activate.ps1
flwr run . --stream
```

Un cliente a la vez en CPU (modo recomendado para esta PC, probado):

```powershell
flwr run . --stream --federation-config "num-supernodes=2 client-resources-num-cpus=2 init-args-num-cpus=2"
```

Verificación independiente del resultado, sin Flower:

```powershell
python scripts/check_federation.py
```

Las dos primeras diferencias deben ser 0.0 y la tercera debe ser mayor que 0.

Para cambiar un valor sin editar archivos:

```powershell
flwr run . --stream --run-config "num-server-rounds=3 learning-rate=0.01"
```

## Resultados medidos (prueba técnica sintética)

| Control | Resultado |
|---|---|
| Clientes por ronda | 2 de 2 en ambas rondas, 0 fallos |
| Huella de pesos recibidos, ronda 1 | 164755960601456 en ambos clientes |
| Huella de pesos recibidos, ronda 2 | 91154846158094 en ambos clientes |
| Ejemplos de train reportados | 48 y 24 |
| Diferencia entre ronda 1 y FedAvg ponderado calculado a mano | 0.0 |
| Diferencia entre ronda 1 y promedio simple | 0.00016 (no coincide, como se espera) |
| Diferencia entre `global_final.keras` recargado y ronda 2 | 0.0 |
| Duración de las dos rondas en modo secuencial | 91 s. El comando completo tardó 4 min 55 s |

La pérdida de validación bajó de 0.6920 a 0.6907 en `client_1`. El cambio es mínimo porque la tasa de aprendizaje es 0.001 y hay una sola época. La exactitud cercana a 0.5 no tiene significado.

## Artefactos en `outputs/federation/`

| Archivo | Contenido |
|---|---|
| `global_round_0.keras` | Inicialización global común |
| `global_round_1.keras`, `global_round_2.keras` | Modelo global después de cada ronda |
| `global_final.keras` | Modelo global final. El servidor lo recarga y compara sus pesos |
| `metrics.json` | Métricas por cliente y por ronda, más test final. Incluye la nota de prueba técnica |

`outputs/` está excluido de Git.

## Funciones principales

| Archivo | Función | Qué hace |
|---|---|---|
| `components.py` | (imports) | Decide qué implementación del contrato se usa |
| `provisional.py` | `load_data`, `build_model`, `train_local`, `evaluate_local` | Versión sintética del contrato |
| `client_app.py` | `model_with_global_weights` | Construye el modelo y le carga los pesos globales recibidos |
| `client_app.py` | `train` | Entrena con train local. Devuelve pesos, número de ejemplos y métricas |
| `client_app.py` | `evaluate` | Evalúa el modelo global con val o test local |
| `server_app.py` | `main` | Crea la inicialización común, lanza FedAvg, guarda resultados |
| `server_app.py` | `metrics_per_client` | Mantiene las métricas separadas por cliente |
| `server_app.py` | `check_rounds` | Falla si una ronda no tuvo dos clientes o si recibieron pesos distintos |
| `server_app.py` | `evaluate_on_test` | Evaluación única con test al final |
| `weights.py` | `weights_fingerprint` | Huella numérica de una lista de pesos |
| `scripts/check_federation.py` | (script) | Recalcula FedAvg a mano y lo compara con el modelo guardado |

## Pasos para conectar la parte de Ariana

**Hechos el 22 de noviembre de 2026.** `components.py` ya usa los componentes reales. La ejecución con datos reales se describe en `docs/11_EJECUCION_FEDERADA_REAL.md`. Los resultados sintéticos de esta página siguen valiendo como prueba técnica de coordinación. Para repetirlos hay que volver a activar `provisional.py` en `components.py`.

Lista original, conservada como historial:

1. Ariana confirma o corrige `docs/07_CONTRATO_INTEGRACION.md`.
2. Ariana crea sus módulos dentro de `pneumonia_fl/` con las cuatro funciones del contrato.
3. En `pneumonia_fl/components.py`, reemplazar el import de `pneumonia_fl.provisional` por los imports de los módulos de Ariana. Cambiar `COMPONENTS_ARE_PROVISIONAL` a `False`.
4. Agregar a `pyproject.toml` las dependencias que use Ariana, por ejemplo `pydicom`.
5. Agregar a `[tool.flwr.app.config]` los valores que necesite `build_model(config)`.
6. Ejecutar `flwr run . --stream` con dos rondas y revisar `metrics.json`.

`client_app.py` y `server_app.py` no cambian.

## Notas de operación

- Flower deja un proceso local `flower-superlink` en segundo plano después de `flwr run`. Las ejecuciones siguientes lo reutilizan.
- Flower instala una copia de la app en `C:\Users\<usuario>\.flwr\apps\`. La salida sí se escribe en `outputs/federation/` del proyecto.
- `pyproject.toml` no declara `fab-format-version`. El formato 1 exige un archivo `LICENSE` y el equipo no ha elegido licencia. Flower muestra un aviso por esto y continúa.
