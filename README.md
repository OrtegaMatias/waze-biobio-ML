# waze-biobio-ML

Demo académica de **ruta segura explicable** para **Concepción y entorno**.

La app no intenta reemplazar a Waze ni trabajar con datos en vivo. Su valor está en mostrar, de forma clara, cómo cambian las rutas cuando combinamos:

- incidentes históricos
- horario de viaje
- perfiles simulados de viajero
- comparación entre ruta base, UBCF e IBCF

## Qué muestra la demo

La interfaz principal quedó organizada en 4 bloques:

1. **Contexto del viaje**
   Escenario curado o coordenadas manuales, día, hora y perfil simulado.
2. **Mapa comparativo**
   Ruta base, UBCF, IBCF y capa de incidentes históricos.
3. **Tarjetas de explicación**
   Tiempo total, riesgo, exposición y razones de cambio por variante.
4. **Evidencia de datos y modelo**
   Calidad de datos, ranking de vías por perfil y límites conocidos.

## Idea central del producto

La demo está pensada para presentación académica, tesis o portafolio técnico:

- **Concepción** es el alcance principal porque da una experiencia más consistente.
- Los perfiles de viajero son **sintéticos**, no usuarios reales.
- La app usa **incidentes históricos** y simulación explicable.
- El modo **regional** sigue disponible, pero como cobertura secundaria.

## Inicio rápido con Docker Compose

Requisito:

- Docker Desktop con `docker compose`

Levantar todo:

```bash
docker compose up --build
```

La imagen backend precalcula el cache de `concepcion` y Docker lo conserva en el volumen `backend_cache`.
Eso hace que el primer arranque útil en Compose sea corto incluso si el host no tiene `data/cache` listo.

Servicios:

- Frontend Streamlit: [http://localhost:8501](http://localhost:8501)
- Frontend React v1: [http://localhost:3000](http://localhost:3000)
- Backend: [http://localhost:8000](http://localhost:8000)
- Liveness: [http://localhost:8000/health](http://localhost:8000/health)
- Readiness: [http://localhost:8000/readyz](http://localhost:8000/readyz)
- OpenAPI: [http://localhost:8000/docs](http://localhost:8000/docs)

Matriz de frontends:

- `Streamlit`: fallback operativo y demo completa actual
- `React v1`: flujo principal de `escenario + comparación`

Persistencia de cache en Docker:

- `backend_cache`: guarda el grafo y artefactos del backend dentro de Docker
- `./data/raw` y `./data/processed`: siguen montados desde el repo como fuente de datos
- `./data/cache` local queda para ejecución fuera de Docker y benchmarks locales

## Flujo recomendado

1. Abre el frontend.
2. Espera a que el backend complete el `warm-up`.
3. Usa un escenario curado de Concepción.
4. Genera la comparación de rutas.
5. Explica el tradeoff entre:
   tiempo total, riesgo, exposición histórica y vías favorecidas por cada estrategia.

Si React no está listo o quieres la demo completa actual, usa Streamlit como respaldo.

## Endpoints clave

### Estado y demo

- `GET /health`: liveness del backend
- `GET /readyz`: readiness real para demo
- `POST /system/bootstrap`: fuerza warm-up del perfil activo
- `GET /system/bootstrap/status`: estado del warm-up
- `GET /system/dataset`: perfil activo de datos
- `POST /system/dataset`: cambia entre `concepcion` y `regional`
- `GET /system/demo-scenarios`: escenarios curados para la demo

### Recomendaciones

- `POST /recommendations/collaborative`
- `POST /recommendations/playground`

### Rutas

- `GET /metadata/options`
- `GET /metadata/hotspots`
- `POST /routes/optimal`

`/routes/optimal` ahora devuelve además:

- `risk_score`
- `incident_exposure`
- `why_changed`
- `top_penalized_segments`
- `top_preferred_vias`
- `comparison`

## Estructura relevante

```text
waze-biobio-ML/
├── algorithms/recommenders/
├── backend/fastapi_app/
├── frontend/react_app/
├── frontend/streamlit_app/
├── data/
├── docs/demo-script.md
├── compose.yaml
└── README.md
```

## Datos y límites conocidos

- La cobertura histórica visible hoy corresponde a **julio de 2025**.
- Los perfiles colaborativos son **simulados**.
- No existe integración con datos en tiempo real.
- Si `ACCIDENTES.csv` y `CONGESTIONES.csv` siguen siendo idénticos, la app los trata como evidencia de calidad limitada y evita vender esa distinción como insight fuerte.
- El modo regional puede arrastrar etiquetas de red menos limpias que el foco en Concepción.

## Ejecución local sin Docker

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

React v1 en desarrollo:

```bash
cd frontend/react_app
npm install
npm run dev
```

## Pruebas

```bash
pytest backend/fastapi_app/tests
```

```bash
cd frontend/react_app
npm test
npm run build
```

Benchmark manual del flujo principal:

```bash
python scripts/dev/benchmark_demo_api.py --wait-ready
```
