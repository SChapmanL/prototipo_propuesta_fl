# Bitácora de la federación (parte de Sebastián)

Escribe esta bitácora Claude, en primera persona, mientras programa la coordinación federada. Cada entrada separa lo comprobado de lo que solo es hipótesis.

Estado general al 15 de noviembre de 2026: ejecuté dos rondas con dos clientes y verifiqué el resultado (etapas 9 a 13). Todo usa datos sintéticos. Son pruebas técnicas, no resultados de detección de neumonía.

---

## Etapa 1. Lectura del proyecto y elección de versiones

Estoy preparando el entorno. Antes de escribir código necesitaba saber qué versiones de Python, TensorFlow y Flower funcionan juntas en esta PC.

Revisé el `.venv` existente con `pip list`. Usa Python 3.14 y contiene las librerías de descarga (`redivis`, `pydicom`). No contiene TensorFlow ni Flower.

Consulté PyPI para cada paquete. Encontré tres restricciones:

- `tensorflow 2.21.0` publica instaladores para Python 3.10 a 3.13. No existe uno para 3.14.
- `flwr 1.43.0` necesita `ray==2.55.1` para simular clientes. En Windows, Flower solo declara esa dependencia para Python 3.11 y 3.12.
- `flwr` exige `protobuf<7` y `tensorflow` exige `protobuf>=6.31.1`. Ambos aceptan la serie 6.

Decisión: Python 3.12, TensorFlow 2.21.0 y Flower 1.43.0. Comprobé que `uv` ya tiene Python 3.12 instalado y que resuelve las 78 dependencias sin conflicto.

No quise tocar el `.venv` actual porque sirve para las descargas. Creé un entorno aparte, `.venv-fl`, y lo agregué a `.gitignore`.

**Aprendizaje.** Las tres piezas avanzan a ritmos distintos. La versión de Python la decide la pieza más lenta, que aquí es TensorFlow junto con Ray en Windows.

## Etapa 2. Instalación: el disco está lleno

Ejecuté `uv pip install "tensorflow==2.21.0" "flwr[simulation]==1.43.0"` sobre `.venv-fl`.

El primer intento terminó con `memory allocation of 128552 bytes failed`. Mi hipótesis fue que demasiadas descargas en paralelo agotaban la memoria. Repetí con `UV_CONCURRENT_DOWNLOADS=2`.

El segundo intento dio un mensaje distinto y más claro: `Espacio en disco insuficiente. (os error 112)` al extraer `ray`. Lo comprobé con `df -h /c`: el disco C: tiene 476 GB y quedaban 158 MB libres. Es la única unidad de la PC.

Causa comprobada del segundo fallo: disco lleno. La causa del primer fallo no está comprobada. Es probable que sea la misma, pero no lo verifiqué.

Ejecuté `uv cache prune` para borrar los restos de mis descargas. Recuperé 33 MB. La caché completa de `uv` mide 461 MB, así que vaciarla tampoco alcanza. TensorFlow y Ray instalados ocupan cerca de 2 GB.

No borré nada más. Decidir qué archivos personales eliminar le corresponde a Sebastián.

En ese momento el problema quedó pendiente. Escribí las etapas 3 a 8 sin poder ejecutar nada. Sebastián liberó espacio después y la instalación continúa en la etapa 9.

## Etapa 3. Elección de una sola API de Flower

Estoy decidiendo cómo se escribe un cliente y un servidor. Flower ha tenido varias APIs. La documentación de la versión 1.43 usa la API de mensajes:

- `ClientApp` con funciones decoradas `@app.train()` y `@app.evaluate()`.
- `ServerApp` con una función `@app.main()`.
- `FedAvg` de `flwr.serverapp.strategy`, que se lanza con `strategy.start(...)`.
- Se ejecuta con `flwr run .` y se configura en `pyproject.toml`.

Uso solo esta API. No aparecen `NumPyClient`, `start_simulation` ni `flwr.server.strategy`, que pertenecen a APIs anteriores.

Como no pude instalar Flower, no pude inspeccionar el paquete local. Leí el código fuente oficial en GitHub (`flwrlabs/flower`, rama `main`): el ejemplo `examples/quickstart-tensorflow` y los archivos `fedavg.py`, `strategy.py` y `strategy_utils.py`. El `pyproject.toml` de ese ejemplo declara `flwr-version-target = "1.43.0"`.

Límite de esta comprobación: leí la rama `main`, no un tag `v1.43.0`. El repositorio no lista un tag con ese nombre. Podría haber diferencias pequeñas con el paquete de PyPI.

**Aprendizaje.** En esta API todo viaja en un `Message`. Dentro lleva un `ArrayRecord` (los pesos), un `MetricRecord` (números) y un `ConfigRecord` (opciones). Los pesos de Keras son una lista de arreglos NumPy, y `ArrayRecord(model.get_weights())` los empaqueta.

## Etapa 4. El contrato con Ariana

Estoy definiendo la frontera entre mi parte y la de Ariana. Escribí `docs/07_CONTRATO_INTEGRACION.md` con cuatro firmas: `load_data`, `build_model`, `train_local` y `evaluate_local`. Partí de la propuesta de `LEER_PRIMERO.md` y precisé lo que la federación necesita:

- `client_id` es el texto `"client_1"` o `"client_2"`, igual que las carpetas de manifiestos.
- Las etiquetas tienen forma `(batch, 1)`, igual que la salida sigmoide.
- `train_local` y `evaluate_local` devuelven diccionarios de `float`, porque Flower solo transporta números.

La decisión principal fue crear `pneumonia_fl/components.py`. Es el único archivo que elige la implementación. Cliente y servidor importan desde ahí. Para integrar a Ariana se cambian esos imports y nada más.

Dejé el paquete en `pneumonia_fl/` en la raíz y no en `src/pneumonia_fl/`. El ejemplo oficial usa la raíz, y no pude probar si `flwr run` encuentra un paquete dentro de `src/`. Queda como punto por acordar.

## Etapa 5. Componente provisional

Estoy construyendo un sustituto pequeño de la parte de Ariana, en `pneumonia_fl/provisional.py`.

- `load_data` genera imágenes de ruido con forma `(224, 224, 3)`. La clase 1 tiene más brillo. Es una señal artificial y fácil.
- `build_model` crea un modelo de tres capas: promedio global, una capa densa de 8 neuronas y una salida sigmoide.
- `train_local` llama a `model.fit` y `evaluate_local` llama a `model.evaluate`.

Di tamaños distintos a los clientes: 48 ejemplos de train para `client_1` y 24 para `client_2`. Así FedAvg debe dar al primero el doble de peso, y la ponderación se puede observar.

**Relación con ML clásico.** `load_data` entrega X e y. `build_model` equivale a crear el estimador. `train_local` es `fit` y `evaluate_local` es `score`. La diferencia es que en Keras los parámetros aprendidos se pueden leer y reemplazar con `get_weights()` y `set_weights()`. FedAvg depende de eso.

## Etapa 6. Cliente Flower

Estoy construyendo el adaptador del cliente en `pneumonia_fl/client_app.py`. Hay una sola implementación. Flower le da a cada cliente simulado un `partition-id` distinto (0 o 1), y `client_number` lo convierte en 1 o 2.

La función `model_with_global_weights` construye el modelo y ejecuta `model.set_weights(...)` con los pesos del mensaje. Esto garantiza el requisito central: aunque los clientes se ejecuten uno después del otro, cada uno parte de los pesos del mensaje del servidor, no de los pesos que dejó el cliente anterior.

`train` devuelve tres cosas: los pesos nuevos, `num-examples` y las métricas. El valor de `num-examples` viene de `load_data`, por lo tanto cuenta imágenes de train y no lotes.

`evaluate` lee del mensaje qué partición usar. El servidor pide `"val"` en las rondas y `"test"` una vez al final.

Para no repetir código, `train` y `evaluate` comparten `model_with_global_weights`, `load_split` y `reply`.

## Etapa 7. Servidor y FedAvg

Estoy construyendo el servidor en `pneumonia_fl/server_app.py`. La función `main` hace esto:

1. Llama una vez a `build_model(config)`. Esos pesos son la inicialización global común.
2. Crea `FedAvg` y llama a `strategy.start(...)` con dos rondas.
3. Comprueba las rondas con `check_rounds`.
4. Pide la evaluación con test mediante `evaluate_on_test`.
5. Guarda `global_final.keras`, lo recarga y compara los pesos con `check_model_reloads`.

Al leer `fedavg.py` encontré tres comportamientos que cambiaron mi diseño:

**Flower promedia las métricas de los clientes.** El proyecto pide métricas por cliente. Escribí `metrics_per_client`, que las guarda con prefijo `client_1_` y `client_2_`. Se la paso a `FedAvg` en `train_metrics_aggr_fn` y `evaluate_metrics_aggr_fn`. No afecta al promedio de pesos, que se calcula en otra función (`aggregate_arrayrecords`).

**Si un cliente falla, Flower agrega con el que queda.** No detiene la ronda. Por eso `check_rounds` exige dos respuestas por ronda y lanza un error si falta una.

**`strategy.start` solo devuelve los pesos finales.** Para guardar cada ronda usé el parámetro `evaluate_fn`. Flower llama a esa función antes de la ronda 1 y después de cada ronda, con los pesos globales vigentes. Mi función `save_round_checkpoint` los guarda como `global_round_N.keras`.

Para comprobar que ambos clientes reciben los mismos pesos escribí `weights_fingerprint` en `pneumonia_fl/weights.py`. Calcula un número a partir de los bytes de los pesos. El servidor lo calcula sobre lo que envía, cada cliente sobre lo que recibe, y `check_rounds` compara los tres valores.

**Relación con ML clásico.** FedAvg no es un algoritmo de entrenamiento nuevo. Es un promedio ponderado de parámetros: `global = (48 * pesos_1 + 24 * pesos_2) / 72`. El entrenamiento sigue ocurriendo dentro de cada cliente con `fit`.

## Etapa 8. Verificación antes de poder instalar

Lo único que ejecuté en ese momento fue `python -m py_compile pneumonia_fl/*.py` con Python 3.12. Terminó sin errores. Esto solo demuestra que la sintaxis es válida.

Los imports, las rondas, las huellas y el guardado quedaron sin verificar hasta la etapa 9.

## Etapa 9. Instalación y primera prueba sin Flower

Con 11 GB libres repetí la instalación con `uv pip install -e .`. Terminó bien: `tensorflow 2.21.0`, `keras 3.15.1`, `flwr 1.43.0`, `ray 2.55.1` y `protobuf 6.33.6`.

Antes de lanzar Flower probé el componente provisional solo. Importé los módulos y llamé a `build_model`, `load_data`, `train_local` y `evaluate_local` desde una línea de Python. Funcionó al primer intento y devolvió 48 ejemplos y una pérdida de 0.6929.

**Aprendizaje.** Conviene probar el modelo sin Flower primero. Si algo falla después, sé que el fallo está en la federación y no en Keras. Es la misma razón por la que el código de Ariana debe poder ejecutarse sin Flower.

## Etapa 10. Primera ejecución con `flwr run`: dos fallos de configuración

Ejecuté `flwr run . --stream`. Falló dos veces antes de entrenar. Ninguno de los fallos estaba en mi código Python.

**Primer fallo.** Flower rechazó `pyproject.toml`:

```
fab-format-version = 1 requires [project].license to be { file = "LICENSE" }
```

Yo había copiado `fab-format-version = 1` del ejemplo oficial. Busqué el mensaje dentro del paquete instalado y lo encontré en `flwr/supercore/fab_format_version.py`. Ahí leí que el valor por defecto es 0 y que solo el formato 1 exige un archivo de licencia. Quité esa línea. No agregué un `LICENSE` porque elegir licencia es una decisión del equipo. Flower ahora muestra un aviso y continúa.

**Segundo fallo.**

```
Unable to launch `flower-superlink` for local simulation: [WinError 2]
```

Yo llamaba a `flwr.exe` por su ruta, sin activar el entorno. `flwr run` lanza otro programa, `flower-superlink`, y lo busca en el PATH. Agregué `.venv-fl/Scripts` al PATH, que es lo que hace `Activate.ps1`, y el error desapareció. Por eso la guía indica activar el entorno antes de ejecutar.

## Etapa 11. Dos rondas completas

La tercera ejecución terminó. Lo que observé en el registro:

- `aggregate_train: Received 2 results and 0 failures` en las dos rondas.
- Ronda 1: ambos clientes reportaron la huella `164755960601456`.
- Ronda 2: ambos reportaron `91154846158094`. La huella cambió entre rondas, por lo tanto los pesos globales sí se actualizaron.
- `check_rounds` no lanzó error. Eso significa que las huellas de los clientes coinciden con las que el servidor calculó al enviar.
- `num-examples` fue 48 y 24 en train, 16 y 8 en validación.
- El servidor imprimió la ruta absoluta de salida: `outputs/federation` dentro del proyecto. Esto resuelve la duda que tenía sobre la ruta relativa.
- Se guardaron `global_round_0.keras`, `global_round_1.keras`, `global_round_2.keras`, `global_final.keras` y `metrics.json`.

La pérdida de validación bajó muy poco (de 0.6920 a 0.6907 en `client_1`). Es coherente con una tasa de aprendizaje de 0.001 y una época. No lo interpreto como aprendizaje útil.

En esa ejecución Ray usó cuatro procesos cliente en paralelo. Al inicio del registro apareció un mensaje de Ray: `Windows fatal exception: access violation`. La ejecución terminó bien a pesar del mensaje. No investigué su causa.

## Etapa 12. Verificación independiente de FedAvg

Las huellas prueban que los clientes parten de los mismos pesos. No prueban que el promedio esté ponderado. Para eso escribí `scripts/check_federation.py`, que no usa Flower:

1. Carga `global_round_0.keras`.
2. Entrena cada cliente por separado desde esos pesos.
3. Calcula a mano `(48 * pesos_1 + 24 * pesos_2) / 72`.
4. Compara el resultado con `global_round_1.keras`.

El primer intento dio `False`. Tenía dos hipótesis: FedAvg no ponderaba como yo creía, o mi script no reproducía el entrenamiento del cliente.

Revisé la segunda. Mi script cargaba el modelo con `keras.models.load_model`. El cliente Flower hace otra cosa: llama a `build_model(config)`, que ejecuta `keras.utils.set_random_seed(42)`, y después carga los pesos. Sin fijar la semilla, el orden de mezcla de los datos era distinto y los pesos entrenados también.

Cambié el script para repetir el orden del cliente: `build_model(config)` y luego `set_weights(...)`. También cambié la salida para mostrar la diferencia numérica en lugar de verdadero o falso. Resultado:

```
diferencia final vs última ronda: 0.0
diferencia FedAvg ponderado vs ronda 1: 0.0
diferencia promedio simple vs ronda 1: 0.0001634359359741211
```

Causa comprobada: la semilla. El modelo global de la ronda 1 es exactamente el promedio ponderado por ejemplos de train, y no es el promedio simple. El modelo final recargado es idéntico al de la ronda 2.

**Aprendizaje.** En TensorFlow, el resultado de un entrenamiento depende de la semilla global aunque los pesos iniciales sean iguales, porque la semilla también decide el orden de los lotes. Para reproducir un entrenamiento hay que repetir la semilla y el orden de las llamadas.

## Etapa 13. Ejecución secuencial en CPU

Ejecuté el modo de un cliente a la vez:

```
flwr run . --stream --federation-config "num-supernodes=2 client-resources-num-cpus=2 init-args-num-cpus=2"
```

Funcionó al primer intento. En el registro aparece un solo proceso `ClientAppActor`, que atendió a los dos clientes por turnos. Las huellas fueron las mismas que en la ejecución paralela, y las pérdidas de entrenamiento también. Esto confirma que ejecutar en secuencia no cambia el resultado: el segundo cliente no hereda los pesos del primero.

Las dos rondas tardaron 91 segundos. El comando completo tardó 4 minutos 55 segundos. En esta ejecución no apareció el mensaje `access violation`.

Hallazgo sobre el disco: Flower crea una copia del entorno de 1.5 GB en `C:\Users\<usuario>\.flwr\runtime-envs\` para cada ejecución. Después de la segunda ejecución, la copia de la primera quedó reducida a 8 MB. Hipótesis: gran parte del tiempo extra del comando corresponde a crear esa copia. No lo medí.

## Problemas pendientes

1. **Contrato.** Ariana todavía no lo ha revisado.
2. **Mensaje `access violation` de Ray.** Apareció en la ejecución paralela y no en la secuencial. Causa no investigada. Hipótesis: está relacionado con el soporte experimental de Ray en Windows. Recomiendo usar el modo secuencial.
3. **Espacio en disco.** Cada ejecución necesita 1.5 GB temporales en `.flwr`. Quedan unos 7 GB libres en C:. ResNet50 y los DICOM necesitarán más.
4. **Tiempo de preparación.** Casi 5 minutos por comando para 91 segundos de rondas. Falta medir qué parte es la copia del entorno.
5. **Licencia.** `pyproject.toml` usa el formato 0 de Flower porque no hay archivo `LICENSE`.
6. **Proceso en segundo plano.** `flwr run` deja activo un `flower-superlink` local. No lo detuve.
7. **Ubicación del paquete.** `pneumonia_fl/` en la raíz funciona. No probé `src/pneumonia_fl/`.
8. **Tiempos con ResNet50.** No hay medición. El modelo provisional no sirve para estimarlos.
