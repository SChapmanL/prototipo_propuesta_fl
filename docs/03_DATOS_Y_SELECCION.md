# Datos y selección reproducible

## Uniones

La tabla principal ya proporciona path_to_image y path_to_dcm. Unir etiquetas por path_to_image. Para el índice DICOM_train, retirar únicamente el prefijo train/ de path_to_dcm y comparar con file_name. No unir por posición ni usar solo el nombre base view1_frontal.dcm.

Ejemplo:
- path_to_image: train/patient00016/study1/view1_frontal.jpg.
- path_to_dcm: train/patient00016/study1/view1_frontal.dcm.
- file_name: patient00016/study1/view1_frontal.dcm.

El índice y la tabla principal tienen coberturas distintas. La selección actual exige correspondencia en ambos. No descargar chunks ZIP de 156 GB para esta muestra.

## Campos de la tabla principal

path_to_image, path_to_dcm, frontal_lateral, ap_pa, deid_patient_id, patient_report_date_order, report, section_narrative, section_clinical_history, section_history, section_comparison, section_technique, section_procedure_comments, section_findings, section_impression, section_end_of_impression, section_summary, section_accession_number, age, sex, race, ethnicity, interpreter_needed, insurance_type, recent_bmi, deceased, split.

Los informes y datos demográficos no son entradas del clasificador de este piloto. Sexo se conserva para describir la muestra.

## Selección exacta implementada

1. Conservar split=train.
2. Conservar edad numérica finita >=18.
3. Conservar frontal_lateral=Frontal y ap_pa en AP/PA.
4. Exigir etiqueta presente en el archivo y valor 0/1.
5. Exigir DICOM registrado en el índice.
6. Agrupar por deid_patient_id.
7. Elegir una imagen elegible por paciente con el menor hash SHA256 de 42|image|patient_id|image_id.
8. Por clase, ordenar pacientes por SHA256 de 42|patient|patient_id.
9. Tomar los primeros 400 por clase y los siguientes 50 como reserva.

La selección por hash no depende del orden de filas ni del generador aleatorio de una biblioteca. No utiliza preferencia clínica ni antigüedad del estudio. Pacientes con estudios de ambas clases reciben la etiqueta de la imagen seleccionada. Nunca se incluye el mismo paciente dos veces.

## Resumen de la muestra candidata

| Medida | Valor |
|---|---:|
| Pacientes/imágenes | 800 |
| Positivos | 400 |
| Negativos | 400 |
| AP | 648 |
| PA | 152 |
| Male | 458 |
| Female | 342 |
| Tamaño total según índice | 10 065 641 856 bytes |
| Pilot20 | 280 503 074 bytes |

| Etiqueta | AP | PA |
|---|---:|---:|
| 0 | 317 | 83 |
| 1 | 331 | 69 |

No se controló la vista AP/PA durante el muestreo. Comprobar su distribución al crear clientes. Balance de clase no garantiza IID.

La versión actual del script encontró 8 556 imágenes elegibles con correspondencia en el índice antes de escoger una por paciente. Tras esa elección quedan 2 803 pacientes negativos y 3 526 positivos. Estos conteos del script y dataset_summary.json tienen prioridad sobre conteos exploratorios previos.

## Archivos y contrato

Cada manifiesto incluye image_id, patient_id, study_id, file_id, file_name, dicom_path, size_bytes, md5_hash, age, sex, view, label_raw, label, label_source, label_generator, original_split y status.

label_generator=not_verified y status=pending_dicom_validation son intencionales. Actualizar el estado después de verificar los archivos. No falsificar el generador a partir del nombre de carpeta.

exclusions.csv guarda el primer motivo de descarte de cada imagen descartada por filtros o por seleccionar otra imagen del mismo paciente. No incluye todos los pacientes aptos que quedaron fuera del cupo. No es un inventario completo de cada candidato no seleccionado.

Los hashes SHA256 de las tres fuentes están en configs/sample_config.json y data/manifests/sample_config.json. Son copias idénticas al entregar el paquete. Mantener configs/sample_config.json como configuración de referencia al integrar el proyecto.

