# Datos sin procesar (Raw Data)

Este directorio contiene los archivos CSV sin procesar que alimentan el sistema Waze Biobío ML.

## Archivos requeridos

### ACCIDENTES.csv (~19 MB)
Contiene información sobre accidentes de tránsito en la región del Biobío.

**Columnas esperadas:**
- `lat`: Latitud del accidente (float)
- `lon`: Longitud del accidente (float)
- `via`: Nombre de la vía donde ocurrió el accidente (string)
- `comuna`: Comuna donde ocurrió el accidente (string)
- `hora_inicio`: Hora de inicio del incidente (formato HH:MM)
- `hora_fin`: Hora de fin del incidente (formato HH:MM)
- `dia_semana`: Día de la semana (Monday, Tuesday, etc.)
- `duracion_horas`: Duración del incidente en horas (float)

### CONGESTIONES.csv (~19 MB)
Contiene información sobre congestiones de tránsito en la región del Biobío.

**Columnas esperadas:**
- `lat`: Latitud de la congestión (float)
- `lon`: Longitud de la congestión (float)
- `via`: Nombre de la vía donde ocurrió la congestión (string)
- `comuna`: Comuna donde ocurrió la congestión (string)
- `hora_inicio`: Hora de inicio de la congestión (formato HH:MM)
- `hora_fin`: Hora de fin de la congestión (formato HH:MM)
- `dia_semana`: Día de la semana (Monday, Tuesday, etc.)
- `velocidad_kmh`: Velocidad promedio durante la congestión (float)
- `duracion_horas`: Duración de la congestión en horas (float)

## Cómo obtener los datos

### Opción 1: Git LFS (si está configurado)
Si el repositorio usa Git LFS para archivos grandes:
```bash
git lfs pull
```

### Opción 2: Script de descarga
Ejecuta el script de descarga (si hay URLs públicas configuradas):
```bash
bash scripts/download_data.sh
```

### Opción 3: Colocación manual
Coloca tus propios archivos CSV en este directorio siguiendo el formato especificado arriba.

## Nota sobre el control de versiones

Los archivos de este directorio están incluidos en `.gitignore` debido a su tamaño:
```
data/raw/ACCIDENTES.csv
data/raw/CONGESTIONES.csv
```

Esto evita problemas al hacer push/pull del repositorio.

## Datos sintéticos

**Nota:** Los archivos `user_ratings.csv` utilizados en este proyecto son **datos sintéticos** generados para propósitos de demostración y pruebas. No representan preferencias reales de usuarios.

Para generar nuevos ratings sintéticos:
```bash
python scripts/dev/build_user_ratings.py --mode regional
python scripts/dev/build_user_ratings.py --mode concepcion
```
