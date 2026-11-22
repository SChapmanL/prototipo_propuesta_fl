# Siguiente paso: ejecución federada con datos reales

Actualizado el 22 de noviembre de 2026. La versión anterior, sobre la descarga de `pilot20`, está en `docs/archive/04_SIGUIENTE_PASO_descarga_pilot20.md`. Ese paso se cerró el 18 de noviembre.

## Dónde estamos

| Etapa del reparto (`LEER_PRIMERO.md`) | Estado |
|---|---|
| 1 a 7. Datos, DICOM, particiones, modelo y referencias locales | Hechas por Ariana. Ver `docs/02` y `docs/10` |
| 8 y 9. Cliente, servidor y FedAvg | Hechos por Sebastián y verificados con datos sintéticos. Ver `docs/08` |
| 10. Integración con datos reales | **Código conectado y probado con pilot20. Falta ejecutarlo con los 800 DICOM** |
| 11. Evaluación final | El código ya elige el checkpoint con validación y usa test una vez. Falta ejecutarlo |
| 12. Non-IID y comparación | Pendiente |

## Objetivo inmediato

Ejecutar la federación de 10 rondas con los 800 DICOM en la PC de Ariana y obtener el reporte formal.

## Quién y dónde

Ariana, en su PC. Ahí están `data/raw/dicom/` con los 800 DICOM y los seis manifiestos de `data/manifests/client_1/` y `client_2/`. Ninguno de esos archivos está en Git.

## Pasos

La guía completa, con resultados esperados y solución de problemas, está en `docs/11_EJECUCION_FEDERADA_REAL.md`. Resumen:

```powershell
git pull
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item outputs\local_baselines\piloto.json outputs\local_baselines\piloto_init_anterior.json
python scripts/local_baseline.py --escenario piloto
python scripts/run_federation.py --check-only
python scripts/run_federation.py
```

**Por qué repetir la referencia `piloto`.** El 22 de noviembre se fijó la semilla de la cabeza en `model.py` (ver `docs/02`). La referencia anterior usó otra inicialización. Repetirla da una comparación exacta: misma inicialización y mismas 10 épocas. La copia conserva la versión anterior, porque el script sobrescribe `piloto.json`. Tarda unos 4 minutos. El reporte busca la referencia por número de épocas y usa `piloto.json`.

No ejecutar `--escenario federacion` ni `ambos`. Sobrescribirían `federacion.json`, que guarda la referencia de 2 épocas, con 10 épocas, porque `pyproject.toml` ahora tiene 10 rondas.

## Criterio de cierre

- `run_federation.py --check-only` termina sin errores.
- La ejecución termina con 10 de 10 rondas completas y 2 de 2 clientes en cada una.
- Existe `outputs/federation_real_<fecha>-<hora>/reporte.md`. Su sección 7 compara con `piloto.json`.
- `scripts/check_federation.py --output-dir ...` da diferencia final 0.0 y FedAvg 0.0 o casi 0.
- Se registran los resultados en `docs/02_ESTADO_VERIFICADO.md`, sin identificadores de pacientes.

## Qué resultado esperar

Un AUROC cercano a 0,5, como las referencias locales de `docs/10`. Eso confirma que la integración funciona. El modelo sigue sobreajustando con 280 imágenes de train por cliente. Ampliar la muestra es una decisión del equipo y es el paso siguiente a este.

## Después

1. Decidir con el equipo si se amplía la muestra.
2. Preparar el primer escenario Non-IID. Solo cambia la asignación de train y conserva validación y prueba.
