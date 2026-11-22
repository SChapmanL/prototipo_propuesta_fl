# Siguiente paso: descarga de 20 imágenes

## Objetivo inmediato

Comprobar el entorno local, el acceso de Redivis y la descarga automatizada de 20 imágenes. No iniciar entrenamiento ni volver a seleccionar pacientes.

## Preparación en Windows

Abrir una terminal PowerShell en la raíz de la carpeta extraída. Confirmar antes que Python está instalado. Los comandos siguientes no se han ejecutado en la PC de la usuaria.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install redivis pydicom numpy
.\.venv\Scripts\python.exe scripts/download_dicoms.py --manifest data/manifests/pilot20.csv
```

La última orden solo muestra la cantidad de archivos y su tamaño. No autentica ni descarga.

## Descargar y verificar

```powershell
.\.venv\Scripts\python.exe scripts/download_dicoms.py --manifest data/manifests/pilot20.csv --download
```

Revisar la documentación oficial del cliente para el flujo de autenticación que corresponda a la versión instalada. Usar la cuenta que ya tiene acceso al dataset. No reutilizar sesiones del navegador remoto ni pedir tokens en el chat. No inventar instrucciones de autenticación sin comprobarlas.

La descarga usa redivis.file(file_id).download(...), conserva la ruta paciente/estudio y verifica tamaño, MD5, lectura de píxeles, rango no constante y PhotometricInterpretation. Los archivos existentes se validan antes de omitir su descarga. Los nuevos archivos se descargan como .part y se renombran tras pasar los controles.

Limitación de reanudación: si la descarga de un archivo se interrumpe, la siguiente ejecución descarga de nuevo ese archivo completo. No hay reanudación por rango de bytes.

La salida está en data/raw/dicom/. El registro está en data/raw/dicom/download_validation.csv. Cada ejecución sustituye ese registro. Conservar una copia antes de otra ejecución si se necesita historial. El script registra solo el tipo de excepción para evitar divulgar URLs o credenciales, por lo que un error puede requerir diagnóstico local adicional.

## Criterio de cierre

- 20 filas con status=ok.
- Imágenes de 20 pacientes distintos.
- 10 etiquetas 0 y 10 etiquetas 1.
- Inspección visual del procesamiento, con manejo correcto de MONOCHROME1/2.
- No confundir controles técnicos con confirmación del diagnóstico.

## Después

```powershell
.\.venv\Scripts\python.exe scripts/download_dicoms.py --manifest data/manifests/download_manifest.csv --download
```

Esta orden requiere unos 10,07 GB para los DICOM, más espacio adicional. Revisar el espacio libre antes de ejecutarla. No sustituir fallos automáticamente. Registrar el motivo y elegir reservas de la misma clase con la regla documentada.

Después de validar la muestra, crear particiones 70/15/15 por paciente y repartirlas entre dos clientes. Las 20 imágenes de pilot20 se usan para depurar y deben quedar en entrenamiento, descontándolas de sus cupos por clase. No volver a usar datos de prueba para depuración.

## Reproducir la selección solo si hace falta

```powershell
.\.venv\Scripts\python.exe scripts/prepare_sample.py --input-dir data/raw/metadata --output-dir data/manifests_reproducidos
```

Comparar con la selección entregada. No sobrescribir la versión actual sin necesidad.

