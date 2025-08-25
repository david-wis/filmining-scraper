import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.utils.api_client import TMDBAPIClient
from src.database.connection import get_db_session
from src.database.models import Movie, Genre, Credit, Keyword
from src.config import Config

class MovieCollector:
    """Recolector de películas desde TMDB API."""
    
    def __init__(self):
        self.api_client = TMDBAPIClient()
        self.logger = logger.bind(name="MovieCollector")
    
    def collect_popular_movies(self, max_pages: int = None) -> bool:
        """Recolecta películas populares."""
        return self._collect_movies_from_endpoint("popular", max_pages)
    
    def collect_top_rated_movies(self, max_pages: int = None) -> bool:
        """Recolecta películas mejor valoradas."""
        return self._collect_movies_from_endpoint("top_rated", max_pages)
    
    def collect_now_playing_movies(self, max_pages: int = None) -> bool:
        """Recolecta películas en cartelera."""
        return self._collect_movies_from_endpoint("now_playing", max_pages)
    
    def collect_upcoming_movies(self, max_pages: int = None) -> bool:
        """Recolecta películas próximas a estrenarse."""
        return self._collect_movies_from_endpoint("upcoming", max_pages)
    
    def _collect_movies_from_endpoint(self, endpoint: str, max_pages: int = None) -> bool:
        """Recolecta películas desde un endpoint específico."""
        max_pages = max_pages or Config.MAX_PAGES
        self.logger.info(f"Iniciando recolección de películas desde endpoint: {endpoint}")
        
        total_movies = 0
        page = 1
        
        try:
            while page <= max_pages:
                self.logger.info(f"Procesando página {page} de {endpoint}")
                
                # Obtener películas de la página actual
                if endpoint == "popular":
                    movies_data = self.api_client.get_popular_movies(page)
                elif endpoint == "top_rated":
                    movies_data = self.api_client.get_top_rated_movies(page)
                elif endpoint == "now_playing":
                    movies_data = self.api_client.get_now_playing_movies(page)
                elif endpoint == "upcoming":
                    movies_data = self.api_client.get_upcoming_movies(page)
                else:
                    self.logger.error(f"Endpoint no soportado: {endpoint}")
                    return False
                
                if not movies_data or 'results' not in movies_data:
                    self.logger.warning(f"No se obtuvieron datos para la página {page}")
                    break
                
                movies = movies_data['results']
                if not movies:
                    self.logger.info(f"No hay más películas en la página {page}")
                    break
                
                # Procesar películas en lotes
                batch_count = self._process_movie_batch(movies)
                total_movies += batch_count
                
                self.logger.info(f"Página {page} procesada: {batch_count} películas")
                
                # Verificar si hay más páginas
                if page >= movies_data.get('total_pages', 1):
                    break
                
                page += 1
            
            self.logger.info(f"Recolección completada: {total_movies} películas procesadas desde {endpoint}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recolectando películas desde {endpoint}: {str(e)}")
            return False
    
    def _process_movie_batch(self, movies: List[Dict[str, Any]]) -> int:
        """Procesa un lote de películas."""
        processed_count = 0
        
        for movie_data in tqdm(movies, desc="Procesando películas"):
            try:
                # Obtener detalles completos de la película
                movie_details = self.api_client.get_movie_details(movie_data['id'])
                
                if movie_details:
                    # Guardar película con todos sus datos relacionados
                    success = self._save_movie_with_details(movie_details)
                    if success:
                        processed_count += 1
                else:
                    self.logger.warning(f"No se pudieron obtener detalles para la película {movie_data.get('title', 'Unknown')}")
                    
            except Exception as e:
                self.logger.error(f"Error procesando película {movie_data.get('title', 'Unknown')}: {str(e)}")
                continue
        
        return processed_count
    
    def _save_movie_with_details(self, movie_data: Dict[str, Any]) -> bool:
        """Guarda una película con todos sus detalles relacionados."""
        session = get_db_session()
        
        try:
            # Verificar si la película ya existe
            existing_movie = session.query(Movie).filter_by(tmdb_id=movie_data['id']).first()
            
            if existing_movie:
                # Actualizar película existente
                self._update_movie(existing_movie, movie_data)
                movie = existing_movie
                self.logger.debug(f"Película actualizada: {movie_data.get('title', 'Unknown')}")
            else:
                # Crear nueva película
                movie = self._create_movie(movie_data)
                session.add(movie)
                # Hacer commit para obtener el ID de la película
                session.commit()
                self.logger.debug(f"Nueva película agregada: {movie_data.get('title', 'Unknown')}")
            
            # Guardar géneros si están habilitados
            if Config.COLLECT_GENRES and 'genres' in movie_data:
                self._save_movie_genres(session, movie, movie_data['genres'])
            
            # Guardar créditos si están habilitados
            if Config.COLLECT_CREDITS and 'credits' in movie_data:
                self._save_movie_credits(session, movie, movie_data['credits'])
            
            # Guardar palabras clave si están habilitadas
            if Config.COLLECT_KEYWORDS and 'keywords' in movie_data:
                self._save_movie_keywords(session, movie, movie_data['keywords'])
            
            session.commit()
            return True
            
        except IntegrityError as e:
            self.logger.warning(f"Error de integridad al guardar película {movie_data.get('title', 'Unknown')}: {str(e)}")
            session.rollback()
            return False
        except Exception as e:
            self.logger.error(f"Error guardando película {movie_data.get('title', 'Unknown')}: {str(e)}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def _create_movie(self, movie_data: Dict[str, Any]) -> Movie:
        """Crea un nuevo objeto Movie desde los datos de la API."""
        return Movie(
            tmdb_id=movie_data['id'],
            title=movie_data.get('title', ''),
            original_title=movie_data.get('original_title', ''),
            overview=movie_data.get('overview', ''),
            tagline=movie_data.get('tagline', ''),
            release_date=self._parse_date(movie_data.get('release_date')),
            runtime=movie_data.get('runtime'),
            budget=movie_data.get('budget'),
            revenue=movie_data.get('revenue'),
            popularity=movie_data.get('popularity'),
            vote_average=movie_data.get('vote_average'),
            vote_count=movie_data.get('vote_count'),
            poster_path=movie_data.get('poster_path'),
            backdrop_path=movie_data.get('backdrop_path'),
            adult=movie_data.get('adult', False),
            status=movie_data.get('status'),
            original_language=movie_data.get('original_language'),
            production_companies=json.dumps(movie_data.get('production_companies', [])),
            production_countries=json.dumps(movie_data.get('production_countries', [])),
            spoken_languages=json.dumps(movie_data.get('spoken_languages', []))
        )
    
    def _update_movie(self, movie: Movie, movie_data: Dict[str, Any]):
        """Actualiza una película existente con nuevos datos."""
        movie.title = movie_data.get('title', movie.title)
        movie.original_title = movie_data.get('original_title', movie.original_title)
        movie.overview = movie_data.get('overview', movie.overview)
        movie.tagline = movie_data.get('tagline', movie.tagline)
        movie.release_date = self._parse_date(movie_data.get('release_date')) or movie.release_date
        movie.runtime = movie_data.get('runtime', movie.runtime)
        movie.budget = movie_data.get('budget', movie.budget)
        movie.revenue = movie_data.get('revenue', movie.revenue)
        movie.popularity = movie_data.get('popularity', movie.popularity)
        movie.vote_average = movie_data.get('vote_average', movie.vote_average)
        movie.vote_count = movie_data.get('vote_count', movie.vote_count)
        movie.poster_path = movie_data.get('poster_path', movie.poster_path)
        movie.backdrop_path = movie_data.get('backdrop_path', movie.backdrop_path)
        movie.adult = movie_data.get('adult', movie.adult)
        movie.status = movie_data.get('status', movie.status)
        movie.original_language = movie_data.get('original_language', movie.original_language)
        movie.production_companies = json.dumps(movie_data.get('production_companies', []))
        movie.production_countries = json.dumps(movie_data.get('production_countries', []))
        movie.spoken_languages = json.dumps(movie_data.get('spoken_languages', []))
    
    def _save_movie_genres(self, session: Session, movie: Movie, genres_data: List[Dict[str, Any]]):
        """Guarda los géneros de una película."""
        for genre_data in genres_data:
            # Buscar género en la base de datos
            genre = session.query(Genre).filter_by(tmdb_id=genre_data['id']).first()
            if genre and genre not in movie.genres:
                movie.genres.append(genre)
    
    def _save_movie_credits(self, session: Session, movie: Movie, credits_data: Dict[str, Any]):
        """Guarda los créditos de una película."""
        # Limpiar créditos existentes
        session.query(Credit).filter_by(movie_id=movie.id).delete()
        
        # Guardar cast
        if 'cast' in credits_data:
            for i, cast_member in enumerate(credits_data['cast'][:20]):  # Limitar a 20 actores principales
                credit = Credit(
                    movie_id=movie.id,
                    tmdb_person_id=cast_member['id'],
                    name=cast_member['name'],
                    character=cast_member.get('character'),
                    credit_type='cast',
                    order=i,
                    profile_path=cast_member.get('profile_path')
                )
                session.add(credit)
        
        # Guardar crew
        if 'crew' in credits_data:
            for crew_member in credits_data['crew'][:10]:  # Limitar a 10 miembros del crew
                credit = Credit(
                    movie_id=movie.id,
                    tmdb_person_id=crew_member['id'],
                    name=crew_member['name'],
                    job=crew_member.get('job'),
                    department=crew_member.get('department'),
                    credit_type='crew',
                    profile_path=crew_member.get('profile_path')
                )
                session.add(credit)
    
    def _save_movie_keywords(self, session: Session, movie: Movie, keywords_data: Dict[str, Any]):
        """Guarda las palabras clave de una película."""
        # Limpiar palabras clave existentes
        session.query(Keyword).filter_by(movie_id=movie.id).delete()
        
        if 'keywords' in keywords_data:
            for keyword_data in keywords_data['keywords'][:20]:  # Limitar a 20 palabras clave
                keyword = Keyword(
                    movie_id=movie.id,
                    tmdb_keyword_id=keyword_data['id'],
                    name=keyword_data['name']
                )
                session.add(keyword)
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parsea una fecha desde string a datetime."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return None

def main():
    """Función principal para ejecutar el recolector de películas."""
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
    collector = MovieCollector()
    
    # Recolectar películas populares
    success = collector.collect_popular_movies()
    
    if success:
        logger.info("Recolección de películas completada exitosamente")
    else:
        logger.error("Error en la recolección de películas")

if __name__ == "__main__":
    main()
