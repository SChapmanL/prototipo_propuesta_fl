# Prompt para la sesión de Ariana

Actualizado el 22 de noviembre de 2026. El prompt anterior, de su parte local, está en `docs/archive/09_PROMPT_ARIANA_modelo_local.md`. Esa parte ya está hecha.

Copia el bloque completo en una sesión nueva de Claude Code abierta en la raíz del repositorio.

```text
Hola Claude. Soy Ariana. Mi parte local (datos, DICOM, ResNet50, entrenamiento y referencias locales) está terminada. Sebastián conectó mis funciones a la federación con Flower. Ahora quiero ejecutar la federación con los 800 DICOM reales en mi PC y obtener el reporte formal.

## Antes de ejecutar

1. Lee AGENTS.md, LEER_PRIMERO.md, docs/04_SIGUIENTE_PASO.md y docs/11_EJECUCION_FEDERADA_REAL.md.
2. Revisa git status y la rama. Si hay cambios locales míos, no los pierdas.
3. Comprueba que existen data/raw/dicom/ y los seis manifiestos de data/manifests/client_1/ y client_2/. No están en Git. Si faltan, dímelo y detente. No los regeneres sin preguntarme.
4. Comprueba que existe outputs/local_baselines/piloto.json. Se debe repetir, porque model.py ahora fija la semilla de la cabeza (docs/02). Guarda antes una copia como piloto_init_anterior.json.
5. Comprueba el espacio libre. Una ejecución de 10 rondas guarda unos 1,1 GB de modelos.

## Ejecución

Sigue docs/11_EJECUCION_FEDERADA_REAL.md paso a paso. Yo ejecuto los comandos largos en mi terminal. Explícame qué hace cada uno y qué resultado esperar.

1. python scripts/local_baseline.py --escenario piloto   (solo piloto, nunca federacion ni ambos)
2. python scripts/run_federation.py --check-only
3. python scripts/run_federation.py
4. python scripts/check_federation.py --output-dir <carpeta que imprime el paso 3>

## Después

1. Lee conmigo reporte.md de la carpeta de resultados. Explícame cada sección.
2. Registra en docs/02_ESTADO_VERIFICADO.md las cifras principales y la duración, sin identificadores de pacientes.
3. No subas outputs/, data/ ni ningún archivo con identificadores a GitHub.

## Reglas que no cambian

- No modifiques client_app.py, server_app.py ni components.py sin hablar con Sebastián.
- Test se usa una sola vez. No ajustes nada mirando las cifras de test.
- Un AUROC cercano a 0,5 es el resultado esperado. Confirma la integración, no el valor clínico.
- Nunca me pidas contraseñas ni tokens en el chat.
- Responde en español, con frases cortas.
```
