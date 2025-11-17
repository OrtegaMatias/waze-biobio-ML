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
│   └── processed/          # road_network.csv, user_ratings.csv
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
- En `frontend/streamlit_app/app.py` la sección “Laboratorio colaborativo” permite elegir `user_id`, vías conocidas, límite y estrategias para visualizar las diferencias entre UBCF e IBCF.

## Datos y pipelines

- `algorithms/recommenders/data_loader.py` unifica eventos de accidentes/congestiones con la red OSM, genera tokens categóricos y aplica penalizaciones geoespaciales para alimentar tanto el recomendador como el grafo.
- Usa los CSV de `data/raw` y `data/processed`. Para actualizar la red vial ejecuta:
  ```bash
  python scripts/dev/build_road_network.py --place "Región del Biobío, Chile"
  ```

## Pruebas

```bash
pytest backend/fastapi_app/tests
```
