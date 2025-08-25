#!/bin/bash
# Script para activar el entorno virtual

if [ ! -d "venv" ]; then
    echo "Error: El entorno virtual 'venv' no existe."
    echo "Ejecuta el setup primero."
    exit 1
fi

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Entorno virtual activado."
echo "Python version: $(python --version)"
echo "Para desactivar, ejecuta: deactivate"
