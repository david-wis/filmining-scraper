#!/usr/bin/env python3
"""
Script simple para inicializar la base de datos.
Ejecuta desde el directorio raíz del proyecto.
"""

import sys
import os

# Añadir el directorio raíz al path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logger
from src.database.connection import init_database
from loguru import logger

def main():
    """Inicializa la base de datos."""
    
    # Configurar logging
    setup_logger()
    
    logger.info("=== Inicializando Base de Datos ===")
    
    try:
        # Inicializar base de datos
        if init_database():
            logger.info("✅ Base de datos inicializada correctamente")
            logger.info("📊 Tablas creadas:")
            logger.info("   - movies")
            logger.info("   - genres") 
            logger.info("   - movie_genres")
            logger.info("   - credits")
            logger.info("   - keywords")
            logger.info("   - reviews")
            logger.info("   - import_progress")
            return True
        else:
            logger.error("❌ Error inicializando la base de datos")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error inesperado: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
