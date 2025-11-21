#!/bin/bash

# Script para descargar archivos de datos grandes para el proyecto Waze Biobío ML
# Estos archivos no se incluyen en el repositorio debido a su tamaño

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_RAW_DIR="$PROJECT_ROOT/data/raw"

echo "=========================================="
echo "  Waze Biobío ML - Data Download Script"
echo "=========================================="
echo ""

# Create data/raw directory if it doesn't exist
mkdir -p "$DATA_RAW_DIR"

echo "Directorio de datos: $DATA_RAW_DIR"
echo ""

# Function to download file
download_file() {
    local url=$1
    local filename=$2
    local output_path="$DATA_RAW_DIR/$filename"

    if [ -f "$output_path" ]; then
        echo "✓ $filename ya existe. Omitiendo..."
        return 0
    fi

    echo "Descargando $filename..."

    # Check if wget or curl is available
    if command -v wget &> /dev/null; then
        wget -O "$output_path" "$url" || {
            echo "✗ Error descargando $filename"
            return 1
        }
    elif command -v curl &> /dev/null; then
        curl -L -o "$output_path" "$url" || {
            echo "✗ Error descargando $filename"
            return 1
        }
    else
        echo "✗ Error: Se requiere 'wget' o 'curl' para descargar archivos"
        return 1
    fi

    echo "✓ $filename descargado exitosamente"
    return 0
}

# Check if files already exist
if [ -f "$DATA_RAW_DIR/ACCIDENTES.csv" ] && [ -f "$DATA_RAW_DIR/CONGESTIONES.csv" ]; then
    echo "✓ Todos los archivos de datos ya están presentes."
    echo ""
    echo "Archivos encontrados:"
    ls -lh "$DATA_RAW_DIR"/*.csv 2>/dev/null || true
    echo ""
    echo "Para reemplazar los archivos existentes, elimínalos primero:"
    echo "  rm $DATA_RAW_DIR/ACCIDENTES.csv"
    echo "  rm $DATA_RAW_DIR/CONGESTIONES.csv"
    exit 0
fi

echo "NOTA IMPORTANTE:"
echo "================"
echo ""
echo "Los archivos de datos grandes (ACCIDENTES.csv y CONGESTIONES.csv)"
echo "no están disponibles para descarga pública automática."
echo ""
echo "Opciones para obtener los datos:"
echo ""
echo "1. Si tienes acceso al repositorio original, clona el historial completo:"
echo "   git lfs pull"
echo ""
echo "2. Contacta al mantenedor del proyecto para obtener acceso a los archivos"
echo ""
echo "3. Genera datos sintéticos para pruebas (próximamente)"
echo ""
echo "4. Coloca manualmente tus propios archivos CSV en:"
echo "   $DATA_RAW_DIR/"
echo ""
echo "Formato esperado de los archivos:"
echo "  - ACCIDENTES.csv: columnas [lat, lon, via, comuna, hora_inicio, hora_fin, ...]"
echo "  - CONGESTIONES.csv: columnas [lat, lon, via, comuna, hora_inicio, hora_fin, velocidad_kmh, ...]"
echo ""

# Uncomment and modify these lines if you have actual download URLs
# echo "Descargando archivos de datos..."
# download_file "https://example.com/data/ACCIDENTES.csv" "ACCIDENTES.csv"
# download_file "https://example.com/data/CONGESTIONES.csv" "CONGESTIONES.csv"

echo "=========================================="
echo "  Script completado"
echo "=========================================="
