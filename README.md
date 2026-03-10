# waze-biobio-ML

Sistema tipo Waze para la Región del Biobío que combina:

- recomendaciones colaborativas de vías (`UBCF` e `IBCF`)
- cálculo de rutas con Dijkstra
- penalizaciones por congestiones y accidentes
- una API en FastAPI y una interfaz interactiva en Streamlit

## Qué incluye

| Componente | Ubicación | Función |
| --- | --- | --- |
| Datos | `data/` | CSV crudos, datasets procesados y cache persistente |
| Algoritmos | `algorithms/recommenders/` | Carga de datos, filtrado colaborativo y ruteo |
| Backend | `backend/fastapi_app/` | API FastAPI para metadatos, recomendaciones y rutas |
| Frontend | `frontend/streamlit_app/` | Playground visual para probar recomendaciones y trayectos |
| Scripts | `scripts/dev/` | Utilidades para regenerar red vial y ratings |

## Inicio rápido con Docker Compose

Esta es la forma recomendada para levantar el proyecto.

### Requisitos

- Docker Desktop con `docker compose`

### Levantar la aplicación

```bash
docker compose up --build
```

### Abrir servicios

- Frontend: [http://localhost:8501](http://localhost:8501)
- Backend: [http://localhost:8000](http://localhost:8000)
- Healthcheck backend: [http://localhost:8000/health](http://localhost:8000/health)
- Documentación OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

### Qué hace esta configuración

- construye dos servicios: `backend` y `frontend`
- monta `./data` dentro del contenedor para no meter más de 1 GB de cache en la imagen
- conserva `data/cache/` en tu máquina, así el bootstrap siguiente es mucho más rápido
- hace que Streamlit espere a que FastAPI esté sano antes de arrancar

### Comandos útiles

```bash
docker compose up --build
docker compose down
docker compose logs -f backend
docker compose logs -f frontend
docker compose exec backend pytest backend/fastapi_app/tests
```

## Ejecución local sin Docker

Si prefieres correrlo con tu entorno Python:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.fastapi_app.app.main:app --reload
```

En otra terminal:

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app/app.py
```

## Flujo funcional

1. Streamlit valida que el backend esté disponible.
2. El frontend dispara `/system/bootstrap`.
3. FastAPI carga datos, reconstruye o reutiliza cache y deja listo el grafo.
4. La UI consulta metadatos, recomendaciones y rutas óptimas según el contexto del viaje.

## Endpoints principales

### Sistema

- `GET /health`: estado básico del backend
- `POST /system/bootstrap`: inicia la preparación de datos y del grafo
- `GET /system/bootstrap/status`: devuelve el avance del bootstrap
- `GET /system/dataset`: muestra el perfil activo
- `POST /system/dataset`: cambia entre `regional` y `concepcion`

### Recomendaciones

- `POST /recommendations/collaborative`: recomendaciones para una estrategia puntual
- `POST /recommendations/playground`: comparación lado a lado entre `UBCF` e `IBCF`

### Rutas

- `GET /metadata/options`: filtros y opciones disponibles
- `GET /metadata/hotspots`: puntos de congestión usados en la UI
- `POST /routes/optimal`: ruta base y ruta personalizada con penalizaciones

## Estructura del repositorio

```text
waze-biobio-ML/
├── algorithms/
│   └── recommenders/
├── backend/
│   └── fastapi_app/
│       ├── app/
│       └── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── frontend/
│   └── streamlit_app/
├── scripts/
│   └── dev/
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── README.md
```

## Datos y cache

- `data/raw/`: insumos base como `ACCIDENTES.csv` y `CONGESTIONES.csv`
- `data/processed/`: red vial y ratings por perfil
- `data/cache/`: artefactos generados como `raw_events.pkl`, `segment_summary.pkl` y `route_graph.pkl`

La cache se invalida automáticamente cuando cambian los CSV de entrada relevantes.

## Scripts de desarrollo

### Regenerar red vial

```bash
python scripts/dev/build_road_network.py --place "Región del Biobío, Chile"
```

### Regenerar ratings sintéticos

```bash
python scripts/dev/build_user_ratings.py --mode regional
python scripts/dev/build_user_ratings.py --mode concepcion
```

## Pruebas

```bash
pytest backend/fastapi_app/tests
```
