from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Table, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from datetime import datetime
from typing import List

Base = declarative_base()

# Tabla de relación muchos a muchos entre películas y géneros
movie_genres = Table(
    'movie_genres',
    Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.id'), primary_key=True),
    Column('genre_id', Integer, ForeignKey('genres.id'), primary_key=True)
)

class Movie(Base):
    """Modelo para la tabla de películas."""
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True)
    tmdb_id = Column(Integer, unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    original_title = Column(String(500))
    overview = Column(Text)
    tagline = Column(String(1000))
    release_date = Column(DateTime)
    runtime = Column(Integer)
    budget = Column(BigInteger)
    revenue = Column(BigInteger)
    popularity = Column(Float)
    vote_average = Column(Float)
    vote_count = Column(Integer)
    poster_path = Column(String(500))
    backdrop_path = Column(String(500))
    adult = Column(Boolean, default=False)
    status = Column(String(100))
    original_language = Column(String(10))
    production_companies = Column(Text)  # JSON como string
    production_countries = Column(Text)  # JSON como string
    spoken_languages = Column(Text)      # JSON como string
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relaciones
    genres = relationship("Genre", secondary=movie_genres, back_populates="movies")
    credits = relationship("Credit", back_populates="movie")
    keywords = relationship("Keyword", back_populates="movie")
    
    def __repr__(self):
        return f"<Movie(id={self.id}, title='{self.title}', tmdb_id={self.tmdb_id})>"

class Genre(Base):
    """Modelo para la tabla de géneros."""
    __tablename__ = 'genres'
    
    id = Column(Integer, primary_key=True)
    tmdb_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relaciones
    movies = relationship("Movie", secondary=movie_genres, back_populates="genres")
    
    def __repr__(self):
        return f"<Genre(id={self.id}, name='{self.name}', tmdb_id={self.tmdb_id})>"

class Credit(Base):
    """Modelo para la tabla de créditos (actores, directores, etc.)."""
    __tablename__ = 'credits'
    
    id = Column(Integer, primary_key=True)
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False)
    tmdb_person_id = Column(Integer, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    character = Column(String(500))  # Para actores
    job = Column(String(200))        # Para crew (director, productor, etc.)
    department = Column(String(100)) # Para crew
    credit_type = Column(String(20), nullable=False)  # 'cast' o 'crew'
    order = Column(Integer)          # Orden de aparición para cast
    profile_path = Column(String(500))
    created_at = Column(DateTime, default=func.now())
    
    # Relaciones
    movie = relationship("Movie", back_populates="credits")
    
    def __repr__(self):
        return f"<Credit(id={self.id}, name='{self.name}', type='{self.credit_type}', movie_id={self.movie_id})>"

class Keyword(Base):
    """Modelo para la tabla de palabras clave."""
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False)
    tmdb_keyword_id = Column(Integer, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    # Relaciones
    movie = relationship("Movie", back_populates="keywords")
    
    def __repr__(self):
        return f"<Keyword(id={self.id}, name='{self.name}', movie_id={self.movie_id})>"

class Review(Base):
    """Modelo para la tabla de reseñas."""
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    movie_id = Column(Integer, ForeignKey('movies.id'), nullable=False)
    tmdb_review_id = Column(String(100), unique=True, nullable=False, index=True)
    author = Column(String(200))
    content = Column(Text)
    url = Column(String(500))
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<Review(id={self.id}, author='{self.author}', movie_id={self.movie_id})>"

class ImportProgress(Base):
    """Modelo para rastrear el progreso de importaciones."""
    __tablename__ = 'import_progress'
    
    id = Column(Integer, primary_key=True)
    import_type = Column(String(50), nullable=False)  # 'popular', 'top_rated', etc.
    endpoint = Column(String(100), nullable=False)
    current_page = Column(Integer, nullable=False, default=1)
    total_pages = Column(Integer)
    total_movies = Column(Integer, default=0)
    movies_processed = Column(Integer, default=0)
    movies_new = Column(Integer, default=0)
    movies_updated = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    status = Column(String(20), nullable=False, default='running')  # 'running', 'completed', 'failed', 'paused'
    started_at = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime)
    error_message = Column(Text)
    config_snapshot = Column(Text)  # JSON como string
    estimated_completion = Column(DateTime)
    
    def __repr__(self):
        return f"<ImportProgress(id={self.id}, type='{self.import_type}', page={self.current_page}/{self.total_pages}, status='{self.status}')>"

# Índices adicionales para optimizar consultas
from sqlalchemy import Index

# Índices para búsquedas comunes
Index('idx_movies_title', Movie.title)
Index('idx_movies_release_date', Movie.release_date)
Index('idx_movies_popularity', Movie.popularity)
Index('idx_movies_vote_average', Movie.vote_average)
Index('idx_credits_person_id', Credit.tmdb_person_id)
Index('idx_keywords_name', Keyword.name)

# Índices para import_progress
Index('idx_import_progress_type', ImportProgress.import_type)
Index('idx_import_progress_status', ImportProgress.status)
Index('idx_import_progress_started_at', ImportProgress.started_at)
