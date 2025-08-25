import os
from dotenv import load_dotenv
from typing import Optional

# Cargar variables de entorno
load_dotenv()

class Config:
    """Configuración de la aplicación cargada desde variables de entorno."""
    
    # TMDB API Configuration
    TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
    TMDB_BASE_URL: str = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
    
    # Database Configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "movie_database")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    # Application Settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "20"))
    MAX_PAGES: int = int(os.getenv("MAX_PAGES", "500"))
    LANGUAGE: str = os.getenv("LANGUAGE", "en-US")
    REGION: str = os.getenv("REGION", "US")
    
    # Data Collection Settings
    COLLECT_MOVIES: bool = os.getenv("COLLECT_MOVIES", "true").lower() == "true"
    COLLECT_GENRES: bool = os.getenv("COLLECT_GENRES", "true").lower() == "true"
    COLLECT_CREDITS: bool = os.getenv("COLLECT_CREDITS", "true").lower() == "true"
    COLLECT_KEYWORDS: bool = os.getenv("COLLECT_KEYWORDS", "true").lower() == "true"
    COLLECT_REVIEWS: bool = os.getenv("COLLECT_REVIEWS", "false").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/movie_collector.log")
    
    @classmethod
    def validate(cls) -> bool:
        """Valida que las configuraciones críticas estén presentes."""
        if not cls.TMDB_API_KEY:
            print("ERROR: TMDB_API_KEY no está configurada en el archivo .env")
            return False
        
        if not cls.DB_USER or not cls.DB_PASSWORD:
            print("ERROR: DB_USER y DB_PASSWORD deben estar configurados en el archivo .env")
            return False
        
        return True
    
    @classmethod
    def get_database_url(cls) -> str:
        """Retorna la URL de conexión a la base de datos."""
        return f"postgresql://{cls.DB_USER}:{cls.DB_PASSWORD}@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}"
    
    @classmethod
    def print_config(cls):
        """Imprime la configuración actual (sin mostrar contraseñas)."""
        print("=== Configuración de la Aplicación ===")
        print(f"TMDB Base URL: {cls.TMDB_BASE_URL}")
        print(f"Database: {cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Max Pages: {cls.MAX_PAGES}")
        print(f"Language: {cls.LANGUAGE}")
        print(f"Region: {cls.REGION}")
        print(f"Collect Movies: {cls.COLLECT_MOVIES}")
        print(f"Collect Genres: {cls.COLLECT_GENRES}")
        print(f"Collect Credits: {cls.COLLECT_CREDITS}")
        print(f"Collect Keywords: {cls.COLLECT_KEYWORDS}")
        print(f"Collect Reviews: {cls.COLLECT_REVIEWS}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=====================================")
