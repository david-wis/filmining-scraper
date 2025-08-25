#!/usr/bin/env python3
"""
TMDB Movie Data Collector - Punto de entrada principal

Este script coordina la recolección masiva de datos de películas desde la API de TMDB
y los almacena en una base de datos PostgreSQL para análisis de datos.
"""

import sys
import argparse
from loguru import logger

from src.utils.logger import setup_logger
from src.config import Config
from src.database.connection import init_database
from src.collectors.genre_collector import GenreCollector
from src.collectors.movie_collector import MovieCollector

def main():
    """Función principal que coordina todo el proceso de recolección."""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='TMDB Movie Data Collector')
    parser.add_argument('--genres-only', action='store_true', 
                       help='Recolectar solo géneros')
    parser.add_argument('--movies-only', action='store_true', 
                       help='Recolectar solo películas')
    parser.add_argument('--popular', action='store_true', 
                       help='Recolectar películas populares')
    parser.add_argument('--top-rated', action='store_true', 
                       help='Recolectar películas mejor valoradas')
    parser.add_argument('--now-playing', action='store_true', 
                       help='Recolectar películas en cartelera')
    parser.add_argument('--upcoming', action='store_true', 
                       help='Recolectar películas próximas a estrenarse')
    parser.add_argument('--max-pages', type=int, default=None,
                       help='Número máximo de páginas a recolectar')
    parser.add_argument('--init-db', action='store_true',
                       help='Inicializar base de datos (crear tablas)')
    
    args = parser.parse_args()
    
    # Configurar logging
    setup_logger()
    
    logger.info("=== TMDB Movie Data Collector ===")
    
    # Validar configuración
    if not Config.validate():
        logger.error("Configuración inválida. Revisa el archivo .env")
        logger.error("Asegúrate de configurar:")
        logger.error("- TMDB_API_KEY: Tu API key de TMDB")
        logger.error("- DB_USER y DB_PASSWORD: Credenciales de la base de datos")
        sys.exit(1)
    
    # Mostrar configuración
    Config.print_config()
    
    try:
        # Inicializar base de datos si se solicita
        if args.init_db:
            logger.info("Inicializando base de datos...")
            if init_database():
                logger.info("Base de datos inicializada correctamente")
            else:
                logger.error("Error inicializando la base de datos")
                sys.exit(1)
        
        # Recolectar géneros
        if args.genres_only or (not args.movies_only and Config.COLLECT_GENRES):
            logger.info("Iniciando recolección de géneros...")
            genre_collector = GenreCollector()
            if genre_collector.collect_genres():
                logger.info("✅ Recolección de géneros completada exitosamente")
            else:
                logger.error("❌ Error en la recolección de géneros")
                if args.genres_only:
                    sys.exit(1)
        
        # Recolectar películas
        if not args.genres_only and (args.movies_only or Config.COLLECT_MOVIES):
            logger.info("Iniciando recolección de películas...")
            movie_collector = MovieCollector()
            
            success = True
            
            # Determinar qué tipos de películas recolectar
            if args.popular or (not any([args.top_rated, args.now_playing, args.upcoming])):
                logger.info("Recolectando películas populares...")
                if not movie_collector.collect_popular_movies(args.max_pages):
                    success = False
            
            if args.top_rated:
                logger.info("Recolectando películas mejor valoradas...")
                if not movie_collector.collect_top_rated_movies(args.max_pages):
                    success = False
            
            if args.now_playing:
                logger.info("Recolectando películas en cartelera...")
                if not movie_collector.collect_now_playing_movies(args.max_pages):
                    success = False
            
            if args.upcoming:
                logger.info("Recolectando películas próximas a estrenarse...")
                if not movie_collector.collect_upcoming_movies(args.max_pages):
                    success = False
            
            if success:
                logger.info("✅ Recolección de películas completada exitosamente")
            else:
                logger.error("❌ Error en la recolección de películas")
                sys.exit(1)
        
        logger.info("🎉 Proceso de recolección completado exitosamente!")
        
    except KeyboardInterrupt:
        logger.warning("Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
