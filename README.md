# waze-biobio-ML

Sistema estilo Waze para la Región del Biobío (Chile) que combina un motor de recomendaciones con ruteo óptimo para evitar congestiones y accidentes según las preferencias del usuario.

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

1. **Instalar dependencias**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Ejecutar el backend**
   ```bash
   uvicorn backend.fastapi_app.app.main:app --reload
   ```
3. **Levantar el frontend**
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

## Cache persistente

- `data/cache/` almacena artefactos derivados (`raw_events`, `segment_summary`, `transactions` y `route_graph`).
- Cada archivo se etiqueta con `data_version()`: si los CSV base no cambian, FastAPI carga todo desde disco y el bootstrap es prácticamente inmediato.

## Datos y pipelines

- `algorithms/recommenders/data_loader.py` unifica eventos de accidentes/congestiones con la red OSM, genera tokens categóricos y aplica penalizaciones geoespaciales para alimentar tanto el recomendador como el grafo.
- Usa los CSV de `data/raw` y `data/processed`. Para actualizar la red vial ejecuta:
  ```bash
  python scripts/dev/build_road_network.py --place "Región del Biobío, Chile"
  ```
- Para regenerar ratings sintéticos coherentes con el `road_network` (regional o solo Concepción):
  ```bash
  python scripts/dev/build_user_ratings.py --mode regional
  python scripts/dev/build_user_ratings.py --mode concepcion
  ```

## Pruebas

```bash
pytest backend/fastapi_app/tests
```
