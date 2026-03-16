FROM python:3.11-slim-bookworm AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.backend.txt ./

RUN pip install --upgrade pip && pip install -r requirements.backend.txt

COPY algorithms ./algorithms
COPY backend ./backend
COPY data/raw ./data/raw
COPY data/processed ./data/processed
COPY scripts ./scripts
COPY README.md ./

RUN mkdir -p /app/data/cache && \
    python -c "from backend.fastapi_app.app.core import dataset; from backend.fastapi_app.app.services.routing_service import RoutingService; dataset.set_profile('concepcion'); RoutingService().build()"

EXPOSE 8000

CMD ["uvicorn", "backend.fastapi_app.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM node:20-bookworm-slim AS frontend-build

WORKDIR /app/frontend/react_app

ARG VITE_BACKEND_URL=http://localhost:8000
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

COPY frontend/react_app/package*.json ./

RUN npm ci

COPY frontend/react_app ./

RUN npm run build

FROM node:20-bookworm-slim AS frontend

WORKDIR /app/frontend/react_app

RUN npm install --global serve

COPY --from=frontend-build /app/frontend/react_app/dist ./dist

EXPOSE 3000

CMD ["serve", "-s", "dist", "-l", "3000"]
