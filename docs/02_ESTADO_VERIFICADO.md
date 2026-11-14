# Estado verificado

Este documento registra lo ejecutado en la sesión anterior. Claude debe volver a inspeccionar los archivos y el entorno de la usuaria antes de asumir disponibilidad local.

## Fuentes inspeccionadas

- Tabla principal: 223 462 filas, sin duplicados en path_to_image.
- report_fixed.json: JSON por líneas, 223 462 registros, sin claves de imagen duplicadas.
- Unión completa 1:1 entre ambas fuentes mediante path_to_image.
- dicom_train.csv: 210 740 filas. Campos file_id, file_name, size, added_at y md5_hash.
- CSV principal: 223 228 filas train y 234 valid.

Distribución de Pneumonia en todo report_fixed.json:

| Valor | Registros |
|---|---:|
| null | 189 734 |
| -1 | 23 045 |
| 0 | 4 720 |
| 1 | 5 963 |

## Pruebas de DICOM

Se recibieron primero cuatro archivos de patient00001 a patient00004. Tres tenían etiqueta null y uno -1. Fueron legibles, pero no elegibles con la política actual.

Después se verificaron los cuatro candidatos correctos:

| Nombre adjunto | Ruta original en DICOM_train | Edad | Vista | Etiqueta |
|---|---|---:|---|---:|
| view1_frontal (1)(1).dcm | patient00016/study1/view1_frontal.dcm | 54 | PA | 0 |
| view2_frontal.dcm | patient00087/study3/view2_frontal.dcm | 53 | PA | 0 |
| view1_frontal (2)(1).dcm | patient00031/study3/view1_frontal.dcm | 88 | AP | 1 |
| view1_frontal (3)(1).dcm | patient00115/study2/view1_frontal.dcm | 64 | AP | 1 |

En los cuatro casos se comprobaron tamaño, MD5, identidad por índice y decodificación completa con pydicom. Transfer syntax: 1.2.840.10008.1.2.1, sin compresión de píxeles. Píxeles uint16. patient00016 usa MONOCHROME2. Los otros tres usan MONOCHROME1.

No se completó la inspección visual del preprocesamiento ni una validación clínica. No se debe inferir que todos los archivos del dataset tendrán la misma codificación.

## Artefactos producidos

- Lista de 800 pacientes, balance 400/400.
- Lista pilot20 con 10 por clase, contenida en los 800.
- Reserva de 100 pacientes adicionales, 50 por clase, sin solapamiento con los 800.
- Exclusiones, configuración, resumen y dos scripts.
- Selección reproducida por segunda ejecución con igualdad exacta de los seis archivos.
- Función de comprobación del descargador probada con los cuatro DICOM correctos.
- Modo sin descarga del descargador ejecutado correctamente.

## Trabajo pendiente

No se ejecutó la descarga autenticada del script. No se descargaron los 800 DICOM. No hay particiones finales, entorno local de la usuaria comprobado, repositorio Git confirmado, versiones fijadas, preprocesamiento final, modelo, entrenamiento, servidor ni métricas.

