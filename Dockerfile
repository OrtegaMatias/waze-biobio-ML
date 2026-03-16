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

FROM python:3.11-slim-bookworm AS frontend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    BACKEND_URL=http://backend:8000 \
    BACKEND_TIMEOUT=180

WORKDIR /app

COPY requirements.frontend.txt ./

RUN pip install --upgrade pip && pip install -r requirements.frontend.txt

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]

FROM node:20-bookworm-slim AS react

WORKDIR /app/frontend/react_app

COPY frontend/react_app/package*.json ./

RUN npm install

COPY frontend/react_app ./

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "3000"]
