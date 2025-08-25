import json
from typing import List, Dict, Any
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.utils.api_client import TMDBAPIClient
from src.database.connection import get_db_session
from src.database.models import Genre

class GenreCollector:
    """Recolector de géneros cinematográficos desde TMDB API."""
    
    def __init__(self):
        self.api_client = TMDBAPIClient()
        self.logger = logger.bind(name="GenreCollector")
    
    def collect_genres(self) -> bool:
        """Recolecta todos los géneros disponibles desde TMDB."""
        self.logger.info("Iniciando recolección de géneros...")
        
        try:
            # Obtener géneros desde la API
            genres_data = self.api_client.get_genres()
            
            if not genres_data or 'genres' not in genres_data:
                self.logger.error("No se pudieron obtener los géneros desde la API")
                return False
            
            genres = genres_data['genres']
            self.logger.info(f"Obtenidos {len(genres)} géneros desde TMDB")
            
            # Guardar géneros en la base de datos
            saved_count = self._save_genres(genres)
            
            self.logger.info(f"Géneros recolectados exitosamente: {saved_count} géneros guardados")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recolectando géneros: {str(e)}")
            return False
    
    def _save_genres(self, genres: List[Dict[str, Any]]) -> int:
        """Guarda los géneros en la base de datos."""
        session = get_db_session()
        saved_count = 0
        
        try:
            for genre_data in genres:
                try:
                    # Verificar si el género ya existe
                    existing_genre = session.query(Genre).filter_by(tmdb_id=genre_data['id']).first()
                    
                    if existing_genre:
                        # Actualizar género existente
                        existing_genre.name = genre_data['name']
                        self.logger.debug(f"Género actualizado: {genre_data['name']}")
                    else:
                        # Crear nuevo género
                        new_genre = Genre(
                            tmdb_id=genre_data['id'],
                            name=genre_data['name']
                        )
                        session.add(new_genre)
                        self.logger.debug(f"Nuevo género agregado: {genre_data['name']}")
                    
                    saved_count += 1
                    
                except IntegrityError as e:
                    self.logger.warning(f"Error de integridad al guardar género {genre_data.get('name', 'Unknown')}: {str(e)}")
                    session.rollback()
                    continue
                except Exception as e:
                    self.logger.error(f"Error guardando género {genre_data.get('name', 'Unknown')}: {str(e)}")
                    session.rollback()
                    continue
            
            session.commit()
            self.logger.info(f"Géneros guardados exitosamente en la base de datos")
            
        except Exception as e:
            self.logger.error(f"Error en transacción de géneros: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
        
        return saved_count
    
    def get_genres_from_db(self) -> List[Genre]:
        """Obtiene todos los géneros desde la base de datos."""
        session = get_db_session()
        try:
            genres = session.query(Genre).all()
            return genres
        finally:
            session.close()
    
    def get_genre_by_tmdb_id(self, tmdb_id: int) -> Genre:
        """Obtiene un género específico por su TMDB ID."""
        session = get_db_session()
        try:
            genre = session.query(Genre).filter_by(tmdb_id=tmdb_id).first()
            return genre
        finally:
            session.close()

def main():
    """Función principal para ejecutar el recolector de géneros."""
    from src.utils.logger import setup_logger
    from src.config import Config
    
    # Configurar logging
    setup_logger()
    
    # Validar configuración
    if not Config.validate():
        logger.error("Configuración inválida. Revisa el archivo .env")
        return
    
    # Mostrar configuración
    Config.print_config()
    
    # Inicializar recolector
    collector = GenreCollector()
    
    # Recolectar géneros
    success = collector.collect_genres()
    
    if success:
        logger.info("Recolección de géneros completada exitosamente")
    else:
        logger.error("Error en la recolección de géneros")

if __name__ == "__main__":
    main()
