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
from src.utils.progress_manager import ProgressManager
from src.config import Config
from src.database.connection import init_database
from src.collectors.genre_collector import GenreCollector
from src.collectors.movie_collector import MovieCollector

def main():
    """Función principal que coordina todo el proceso de recolección."""
    
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='TMDB Movie Data Collector con soporte para importación progresiva')
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
    
    # Nuevas opciones para importación progresiva
    parser.add_argument('--resume', action='store_true',
                       help='Reanudar importación interrumpida')
    parser.add_argument('--resume-id', type=int, default=None,
                       help='ID específico de importación a reanudar')
    parser.add_argument('--status', action='store_true',
                       help='Mostrar estado de importaciones')
    parser.add_argument('--list-imports', action='store_true',
                       help='Listar todas las importaciones')
    parser.add_argument('--cleanup-failed', action='store_true',
                       help='Limpiar importaciones fallidas antiguas')
    parser.add_argument('--pause', action='store_true',
                       help='Pausar importación en curso')
    parser.add_argument('--pause-id', type=int, default=None,
                       help='ID específico de importación a pausar')
    
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
    
    # Inicializar gestor de progreso
    progress_manager = ProgressManager()
    
    try:
        # Manejar opciones de gestión de importaciones
        if args.status:
            _show_import_status(progress_manager)
            return
        
        if args.list_imports:
            _list_imports(progress_manager)
            return
        
        if args.cleanup_failed:
            _cleanup_failed_imports(progress_manager)
            return
        
        if args.pause:
            _pause_import(progress_manager, args.pause_id)
            return
        
        if args.resume:
            _resume_import(progress_manager, args.resume_id, args)
            return
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
    finally:
        progress_manager.close()

def _show_import_status(progress_manager: ProgressManager):
    """Muestra el estado de las importaciones."""
    logger.info("=== Estado de Importaciones ===")
    
    # Importaciones en curso
    running_imports = progress_manager.list_imports(status='running')
    if running_imports:
        logger.info(f"Importaciones en curso: {len(running_imports)}")
        for imp in running_imports:
            summary = progress_manager.get_progress_summary()
            logger.info(f"  ID {imp.id}: {imp.import_type} - Página {imp.current_page}/{imp.total_pages} - {imp.status}")
    else:
        logger.info("No hay importaciones en curso")
    
    # Importaciones recientes
    recent_imports = progress_manager.list_imports()
    if recent_imports:
        logger.info(f"\nImportaciones recientes:")
        for imp in recent_imports[:5]:  # Mostrar solo las 5 más recientes
            status_emoji = "🟢" if imp.status == "completed" else "🔴" if imp.status == "failed" else "🟡"
            logger.info(f"  {status_emoji} ID {imp.id}: {imp.import_type} - {imp.status} - {imp.started_at.strftime('%Y-%m-%d %H:%M')}")

def _list_imports(progress_manager: ProgressManager):
    """Lista todas las importaciones."""
    logger.info("=== Lista de Importaciones ===")
    
    imports = progress_manager.list_imports()
    if not imports:
        logger.info("No hay importaciones registradas")
        return
    
    for imp in imports:
        status_emoji = "🟢" if imp.status == "completed" else "🔴" if imp.status == "failed" else "🟡" if imp.status == "running" else "⏸️"
        logger.info(f"{status_emoji} ID {imp.id}: {imp.import_type} ({imp.endpoint})")
        logger.info(f"    Estado: {imp.status}")
        logger.info(f"    Progreso: {imp.current_page}/{imp.total_pages} páginas")
        logger.info(f"    Películas: {imp.movies_processed} procesadas ({imp.movies_new} nuevas, {imp.movies_updated} actualizadas)")
        logger.info(f"    Errores: {imp.errors_count}")
        logger.info(f"    Iniciado: {imp.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if imp.completed_at:
            logger.info(f"    Completado: {imp.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

def _cleanup_failed_imports(progress_manager: ProgressManager):
    """Limpia importaciones fallidas antiguas."""
    logger.info("Limpiando importaciones fallidas antiguas...")
    
    cleaned_count = progress_manager.cleanup_failed_imports()
    logger.info(f"✅ Limpiadas {cleaned_count} importaciones fallidas antiguas")

def _pause_import(progress_manager: ProgressManager, import_id: int = None):
    """Pausa una importación específica o la actual."""
    if import_id:
        progress = progress_manager.get_import_status(import_id)
        if not progress:
            logger.error(f"No se encontró importación con ID {import_id}")
            return
        
        if progress.status != 'running':
            logger.error(f"La importación {import_id} no está en curso (estado: {progress.status})")
            return
        
        progress_manager.pause_current_import()
        logger.info(f"✅ Importación {import_id} pausada")
    else:
        logger.error("Debes especificar un ID de importación con --pause-id")
        logger.info("Usa --list-imports para ver las importaciones disponibles")

def _resume_import(progress_manager: ProgressManager, import_id: int = None, args=None):
    """Reanuda una importación específica."""
    if import_id:
        progress = progress_manager.get_import_status(import_id)
        if not progress:
            logger.error(f"No se encontró importación con ID {import_id}")
            return
        
        if progress.status not in ['paused', 'failed']:
            logger.error(f"La importación {import_id} no puede ser reanudada (estado: {progress.status})")
            return
        
        logger.info(f"Reanudando importación {import_id}: {progress.import_type}")
        
        # Crear collector y reanudar
        movie_collector = MovieCollector()
        
        # Determinar qué método usar basado en el tipo de importación
        if progress.import_type == "popular":
            success = movie_collector.collect_popular_movies(progress.total_pages)
        elif progress.import_type == "top_rated":
            success = movie_collector.collect_top_rated_movies(progress.total_pages)
        elif progress.import_type == "now_playing":
            success = movie_collector.collect_now_playing_movies(progress.total_pages)
        elif progress.import_type == "upcoming":
            success = movie_collector.collect_upcoming_movies(progress.total_pages)
        else:
            logger.error(f"Tipo de importación no soportado: {progress.import_type}")
            return
        
        if success:
            logger.info("✅ Importación reanudada y completada exitosamente")
        else:
            logger.error("❌ Error reanudando la importación")
    else:
        logger.error("Debes especificar un ID de importación con --resume-id")
        logger.info("Usa --list-imports para ver las importaciones disponibles")

if __name__ == "__main__":
    main()
