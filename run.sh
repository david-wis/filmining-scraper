#!/bin/bash
# Script para ejecutar la aplicación TMDB Movie Collector

# Verificar si el entorno virtual existe
if [ ! -d "venv" ]; then
    echo "Error: El entorno virtual 'venv' no existe."
    echo "Ejecuta el setup primero."
    exit 1
fi

# Activar entorno virtual
source venv/bin/activate

# Configurar PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Ejecutar la aplicación con los argumentos proporcionados
python src/main.py "$@"
