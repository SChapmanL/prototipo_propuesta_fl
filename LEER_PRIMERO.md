# Continuidad del prototipo CheXpert Plus

## Cómo usar este paquete

Abre esta carpeta como proyecto local en Claude Code o adjunta los Markdown y scripts a una nueva conversación de Claude. Si la interfaz no permite leer un ZIP, descomprímelo primero. No basta con adjuntar un ZIP si la herramienta no puede abrirlo.

Orden de lectura:
1. AGENTS.md.
2. docs/01_CONTEXTO_Y_DECISIONES.md.
3. docs/02_ESTADO_VERIFICADO.md.
4. docs/03_DATOS_Y_SELECCION.md.
5. docs/04_SIGUIENTE_PASO.md.
6. docs/05_REPOSITORIO_Y_PIPELINE.md.
7. docs/06_LIMITACIONES_Y_FUENTES.md.

El plan histórico está en docs/archive/Plan_prototipo_federado_CheXpert.md. Los documentos de continuidad describen el estado posterior a ese plan. Si hay diferencias, conserva el estado verificado más reciente. No presentes propuestas del plan como trabajo terminado.

## Estado en una frase

Se preparó una lista reproducible de 800 pacientes de CheXpert Plus. Cuatro DICOM de prueba con etiquetas 0/1 se decodificaron correctamente. La descarga de las 800 imágenes, su validación, las particiones, el preprocesamiento visual y el entrenamiento siguen pendientes.

## Qué incluye

Documentación, dos scripts Python, seis archivos de selección y configuración, y una copia del plan histórico. Los scripts están colocados en scripts/. Las listas locales están en data/manifests/. El archivo .gitignore evita subirlas a GitHub.

## Qué no incluye

No incluye los CSV/JSON originales, DICOM ni credenciales. Para la descarga inicial bastan las listas incluidas. Para regenerar la selección se necesitan las tres fuentes originales que la usuaria ya descargó:
- df_chexpert_plus_240401.csv.
- report_fixed.json.
- dicom_train.csv.

Colócalas en data/raw/metadata/. No subas esos datos a GitHub. Los archivos DICOM de prueba tampoco son necesarios para usar las listas de descarga.

## Mensaje para iniciar la sesión de Claude

Lee LEER_PRIMERO.md, AGENTS.md y los documentos de docs/ antes de proponer cambios. Continúa como asesor técnico de tesis y copiloto de desarrollo. Tenemos una selección de 800 pacientes preparada, pero su descarga DICOM sigue pendiente. Ayúdame a crear el entorno local y ejecutar primero la descarga de pilot20.csv. Revisa el script y el flujo oficial de autenticación de Redivis. No cambies la fuente de etiquetas, la política, la muestra ni el stack sin explicar el efecto y acordarlo conmigo. No inicies entrenamiento todavía. Distingue lo ejecutado, lo comprobado y lo propuesto. Si puedes ejecutar comandos en mi equipo, inspecciona primero la carpeta real. No supongas que este paquete ya es un repositorio Git creado.

