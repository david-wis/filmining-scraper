import requests
import time
from typing import Dict, Any, Optional, List
from loguru import logger
from src.config import Config

class TMDBAPIClient:
    """Cliente para interactuar con la API de TMDB con reintentos automáticos."""
    
    def __init__(self):
        self.base_url = Config.TMDB_BASE_URL
        self.api_key = Config.TMDB_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.25  # 250ms entre requests
        
        # Configuración de reintentos
        self.max_retries = 3
        self.retry_delay = 1  # segundos
        self.backoff_factor = 2
        
    def _rate_limit(self):
        """Implementa rate limiting para evitar exceder límites de la API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Realiza una petición a la API de TMDB con reintentos automáticos."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params.update({
            'api_key': self.api_key,
            'language': Config.LANGUAGE,
            'region': Config.REGION
        })
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self._rate_limit()
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                logger.debug(f"API Request: {endpoint} - Status: {response.status_code} (attempt {attempt + 1})")
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Timeout en petición API {endpoint} (attempt {attempt + 1}/{self.max_retries + 1})")
                
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Error de conexión en API {endpoint} (attempt {attempt + 1}/{self.max_retries + 1})")
                
            except requests.exceptions.HTTPError as e:
                # Para errores HTTP específicos, no reintentar
                if e.response.status_code in [400, 401, 403, 404]:
                    logger.error(f"Error HTTP no recuperable en API {endpoint}: {e.response.status_code}")
                    return None
                
                last_exception = e
                logger.warning(f"Error HTTP en API {endpoint} (attempt {attempt + 1}/{self.max_retries + 1}): {e.response.status_code}")
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Error en petición API {endpoint} (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}")
            
            # Esperar antes del siguiente intento
            if attempt < self.max_retries:
                delay = self.retry_delay * (self.backoff_factor ** attempt)
                logger.info(f"Esperando {delay} segundos antes del siguiente intento...")
                time.sleep(delay)
        
        # Si llegamos aquí, todos los intentos fallaron
        logger.error(f"Error final en petición API {endpoint} después de {self.max_retries + 1} intentos: {str(last_exception)}")
        return None
    
    def get_popular_movies(self, page: int = 1) -> Optional[Dict[str, Any]]:
        """Obtiene películas populares."""
        return self._make_request("movie/popular", {"page": page})
    
    def get_top_rated_movies(self, page: int = 1) -> Optional[Dict[str, Any]]:
        """Obtiene películas mejor valoradas."""
        return self._make_request("movie/top_rated", {"page": page})
    
    def get_now_playing_movies(self, page: int = 1) -> Optional[Dict[str, Any]]:
        """Obtiene películas en cartelera."""
        return self._make_request("movie/now_playing", {"page": page})
    
    def get_upcoming_movies(self, page: int = 1) -> Optional[Dict[str, Any]]:
        """Obtiene películas próximas a estrenarse."""
        return self._make_request("movie/upcoming", {"page": page})
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene detalles completos de una película."""
        return self._make_request(f"movie/{movie_id}", {
            "append_to_response": "credits,keywords,reviews"
        })
    
    def get_genres(self) -> Optional[Dict[str, Any]]:
        """Obtiene la lista de géneros disponibles."""
        return self._make_request("genre/movie/list")
    
    def get_movie_credits(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene los créditos de una película."""
        return self._make_request(f"movie/{movie_id}/credits")
    
    def get_movie_keywords(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene las palabras clave de una película."""
        return self._make_request(f"movie/{movie_id}/keywords")
    
    def get_movie_reviews(self, movie_id: int, page: int = 1) -> Optional[Dict[str, Any]]:
        """Obtiene las reseñas de una película."""
        return self._make_request(f"movie/{movie_id}/reviews", {"page": page})
    
    def search_movies(self, query: str, page: int = 1) -> Optional[Dict[str, Any]]:
        """Busca películas por título."""
        return self._make_request("search/movie", {
            "query": query,
            "page": page
        })
    
    def get_discover_movies(self, 
                          sort_by: str = "popularity.desc",
                          page: int = 1,
                          year: Optional[int] = None,
                          genre_ids: Optional[List[int]] = None) -> Optional[Dict[str, Any]]:
        """Descubre películas con filtros específicos."""
        params = {
            "sort_by": sort_by,
            "page": page
        }
        
        if year:
            params["primary_release_year"] = year
        
        if genre_ids:
            params["with_genres"] = ",".join(map(str, genre_ids))
        
        return self._make_request("discover/movie", params)
