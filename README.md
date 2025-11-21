# Waze Biobío ML

Sistema estilo Waze para la Región del Biobío (Chile) que combina un motor de recomendaciones con ruteo óptimo para evitar congestiones y accidentes según las preferencias del usuario.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**⚠️ Nota Importante**: Los datos de `user_ratings.csv` son **sintéticos** y generados para propósitos de demostración y pruebas. No representan preferencias reales de usuarios.

## Características principales

✨ **Motor de recomendaciones colaborativas**
- Filtrado colaborativo basado en usuarios (UBCF) y en ítems (IBCF)
- Comparación lado a lado de estrategias en playground interactivo
- Recomendaciones personalizadas de rutas según preferencias históricas

🗺️ **Ruteo inteligente**
- Algoritmo Dijkstra con penalizaciones dinámicas
- Evita congestiones y accidentes en tiempo real
- Múltiples variantes de ruta (base, UBCF, IBCF)

📊 **Análisis geoespacial**
- Integración con OpenStreetMap (OSMnx)
- Radio de efecto configurable para incidentes (60m por defecto)
- Mapas de calor de congestiones

🔧 **Configuración flexible**
- Archivo YAML para todos los parámetros
- Perfiles de datos intercambiables (regional / Concepción)
- Sistema de caché inteligente

🐳 **Listo para Docker**
- Despliegue con un solo comando
- Hot-reload para desarrollo
- Health checks integrados

🛡️ **Validación robusta**
- Validación de coordenadas (región Biobío)
- Manejo de errores descriptivos
- Excepciones personalizadas

## Arquitectura

- **Datos (`data/`)**: archivos CSV crudos (`raw/`) y normalizados (`processed/`) que alimentan los modelos.  
- **Algoritmos (`algorithms/recommenders/`)**: lógica de preparación de datos, filtrado colaborativo UBCF/IBCF y grafo de rutas basado en Dijkstra.  
- **Backend (`backend/fastapi_app/`)**: API FastAPI que expone metadatos, `/recommendations/collaborative`, `/recommendations/playground` y `/routes/optimal`.  
- **Frontend (`frontend/streamlit_app/`)**: interfaz Streamlit tipo “playground” que permite probar UBCF vs IBCF, seleccionar origen/destino y visualizar rutas/alertas.  
- **Scripts (`scripts/dev/`)**: utilidades para regenerar la red vial desde OpenStreetMap.

```
waze-biobio-ML/
├── algorithms/
│   └── recommenders/
│       ├── collaborative.py
│       ├── data_loader.py
│       └── routing.py
├── backend/
│   └── fastapi_app/
│       ├── app/            # main.py, servicios y esquemas pydantic
│       └── tests/          # pruebas unitarias
├── data/
│   ├── raw/                # ACCIDENTES.csv, CONGESTIONES.csv
│   └── processed/          # road_network.csv, user_ratings.csv, user_ratings_concepcion.csv
├── frontend/
│   └── streamlit_app/app.py
├── scripts/
│   └── dev/build_road_network.py
├── requirements.txt
└── README.md
```

## Puesta en marcha

### Opción 1: Docker (Recomendado) 🐳

La forma más rápida de ejecutar el proyecto:

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/waze-biobio-ML.git
cd waze-biobio-ML

# 2. Configurar el sistema (copiar config de ejemplo)
cp config.example.yaml config.yaml

# 3. Obtener datos (ver sección "Datos" más abajo)
bash scripts/download_data.sh

# 4. Levantar servicios con Docker Compose
docker-compose up --build
```

La aplicación estará disponible en:
- **Frontend (Streamlit)**: http://localhost:8501
- **Backend (FastAPI)**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs

### Opción 2: Instalación local

1. **Requisitos previos**
   - Python 3.11+
   - pip
   - virtualenv (recomendado)

2. **Instalar dependencias**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configurar el sistema**
   ```bash
   # Copiar archivo de configuración de ejemplo
   cp config.example.yaml config.yaml

   # Editar config.yaml según tus necesidades (opcional)
   nano config.yaml
   ```

4. **Obtener datos**
   ```bash
   # Ver opciones en scripts/download_data.sh
   bash scripts/download_data.sh

   # O colocar manualmente ACCIDENTES.csv y CONGESTIONES.csv en data/raw/
   ```

5. **Ejecutar el backend**
   ```bash
   uvicorn backend.fastapi_app.app.main:app --reload
   ```

6. **Levantar el frontend** (en otra terminal)
   ```bash
   streamlit run frontend/streamlit_app/app.py
   ```

La app detecta al backend, dispara el bootstrap (`/system/bootstrap`) para construir el grafo y luego consulta los endpoints `/metadata/*`, `/recommendations/*` y `/routes/optimal`.

## Playground colaborativo

- `/recommendations/collaborative`: retorna recomendaciones para una estrategia puntual (`ubcf` o `ibcf`).
- `/recommendations/playground`: ejecuta ambas estrategias (o el subconjunto solicitado) y entrega los resultados lado a lado para comparar puntuaciones.
- En `frontend/streamlit_app/app.py` la sección “Laboratorio colaborativo” permite elegir `user_id`, vías conocidas, límite y estrategias para visualizar las diferencias entre UBCF e IBCF, y las vías mejor rankeadas se envían como preferencias para ajustar los pesos de Dijkstra.
- La visualización de rutas muestra siempre la trayectoria base (Dijkstra puro, sin penalizaciones) y la ruta personalizada (con preferencias + incidentes); puedes activar/desactivar cada capa en el mapa para comparar.
- Antes de generar la ruta el usuario define día, hora estimada y si desea evitar congestiones/accidentes; esa información ajusta las penalizaciones del grafo y suma el tiempo de los incidentes que aún no se puedan evitar.

## Perfiles de datos

- El backend expone `/system/dataset` para consultar o cambiar el perfil activo (`regional` o `concepcion`).
- El selector en Streamlit actualiza el perfil sin reiniciar el backend y borra la cache del playground.

## Configuración

El sistema utiliza `config.yaml` para todos los parámetros configurables:

### Parámetros principales

```yaml
# Ruteo
routing:
  penalty_radius_m: 60          # Radio de efecto de incidentes
  accident_penalty: 1.75        # Multiplicador de penalización para accidentes
  congestion_penalty: 1.35      # Multiplicador para congestiones
  default_speed_kmh: 40         # Velocidad por defecto

# Recomendaciones
recommendations:
  min_similar_users: 3          # Mínimo de usuarios similares (UBCF)
  min_similar_items: 3          # Mínimo de ítems similares (IBCF)
  min_similarity: 0.1           # Similitud mínima (coseno)
  default_limit: 10             # Número de recomendaciones

# Backend
backend:
  host: 0.0.0.0
  port: 8000
  log_level: INFO               # DEBUG, INFO, WARNING, ERROR
```

**Nota**: Copia `config.example.yaml` a `config.yaml` y ajusta según tus necesidades. El archivo `config.yaml` está en `.gitignore` para evitar subir configuraciones locales.

## Cache persistente

- `data/cache/` almacena artefactos derivados (`raw_events`, `segment_summary`, `transactions` y `route_graph`).
- Cada archivo se etiqueta con `data_version()`: si los CSV base no cambian, FastAPI carga todo desde disco y el bootstrap es prácticamente inmediato.

## Datos y pipelines

### Obtención de datos

Los archivos grandes (`ACCIDENTES.csv`, `CONGESTIONES.csv`) no están incluidos en el repositorio. Para obtenerlos:

```bash
# Opción 1: Script de descarga (si hay URLs configuradas)
bash scripts/download_data.sh

# Opción 2: Git LFS (si está configurado)
git lfs pull

# Opción 3: Colocar manualmente en data/raw/
# Ver data/raw/README.md para formato esperado
```

### Pipeline de datos

- `algorithms/recommenders/data_loader.py` unifica eventos de accidentes/congestiones con la red OSM, genera tokens categóricos y aplica penalizaciones geoespaciales para alimentar tanto el recomendador como el grafo.
- Usa los CSV de `data/raw` y `data/processed`. Para actualizar la red vial ejecuta:
  ```bash
  python scripts/dev/build_road_network.py --place "Región del Biobío, Chile"
  ```

### Datos sintéticos

Los `user_ratings.csv` son **datos sintéticos generados para demostración**. Para regenerarlos:

```bash
python scripts/dev/build_user_ratings.py --mode regional
python scripts/dev/build_user_ratings.py --mode concepcion
```

## Manejo de errores y validación

El sistema incluye validación robusta y manejo de errores:

### Validación de coordenadas
- Verifica que las coordenadas estén dentro de la región del Biobío
- Límites aproximados: latitud [-39, -35], longitud [-74, -71]
- Respuestas HTTP 400 con mensajes descriptivos

### Excepciones personalizadas

```python
NoRouteFoundException      # No existe ruta entre dos puntos
InvalidCoordinatesException # Coordenadas fuera de rango
UserNotFoundException       # Usuario no encontrado
InvalidStrategyException    # Estrategia CF inválida
DataNotLoadedException      # Datos no cargados (ejecutar /system/bootstrap)
```

### Respuestas de error

Todas las respuestas de error incluyen:
```json
{
  "detail": "Mensaje descriptivo del error",
  "error_type": "NombreDelError"
}
```

### Ejemplo

```bash
# Solicitar ruta fuera de la región
curl -X POST http://localhost:8000/routes/optimal \
  -H "Content-Type: application/json" \
  -d '{
    "origin": {"lat": -33.4, "lon": -70.6},
    "destination": {"lat": -36.8, "lon": -73.0}
  }'

# Respuesta
{
  "detail": "Coordenadas inválidas: (-33.4, -70.6). El origen está fuera de la región del Biobío",
  "error_type": "InvalidCoordinatesException"
}
```

## Pruebas

```bash
pytest backend/fastapi_app/tests
```
