#!/usr/bin/env python3
"""
Script para generar embeddings de la columna 'overview' de la tabla movies
y almacenarlos en PostgreSQL con pgvector.

Este script:
1. Conecta a la base de datos PostgreSQL
2. Lee todos los overviews de la tabla movies
3. Genera embeddings usando sentence-transformers
4. Almacena los embeddings en una columna vector en la base de datos
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text, Column, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
import pandas as pd
from tqdm import tqdm
import numpy as np
from loguru import logger

# Configuración
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers no está instalado. Instálalo con: pip install sentence-transformers")

# Configuración de la base de datos
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '25432')
DB_NAME = os.getenv('DB_NAME', 'movie_database')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

# Modelo de embeddings (puedes cambiar esto)
# Opciones populares:
# - 'all-MiniLM-L6-v2': 384 dimensiones, rápido, bueno para la mayoría de casos
# - 'all-mpnet-base-v2': 768 dimensiones, más lento pero mejor calidad
# - 'paraphrase-multilingual-MiniLM-L12-v2': 384 dimensiones, multilingüe
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))  # Tamaño del batch para procesar embeddings


def get_database_engine():
    """Crea y retorna el engine de SQLAlchemy."""
    database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(database_url, pool_pre_ping=True)


def setup_vector_column(engine, dimension=384):
    """
    Crea la columna vector para almacenar embeddings si no existe.
    
    Args:
        engine: SQLAlchemy engine
        dimension: Dimensión de los vectores (384 para all-MiniLM-L6-v2, 768 para all-mpnet-base-v2)
    """
    with engine.connect() as conn:
        # Verificar si la columna ya existe
        check_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'movies' AND column_name = 'overview_embedding'
        """)
        result = conn.execute(check_query).fetchone()
        
        if result:
            logger.info("La columna 'overview_embedding' ya existe")
            # Verificar la dimensión actual
            dim_query = text("""
                SELECT COUNT(*) 
                FROM movies 
                WHERE overview_embedding IS NOT NULL
            """)
            count = conn.execute(dim_query).scalar()
            logger.info(f"Ya hay {count} embeddings almacenados")
        else:
            logger.info(f"Creando columna 'overview_embedding' con dimensión {dimension}")
            # Agregar columna vector
            alter_query = text(f"""
                ALTER TABLE movies 
                ADD COLUMN overview_embedding vector({dimension})
            """)
            conn.execute(alter_query)
            conn.commit()
            logger.info("Columna 'overview_embedding' creada exitosamente")
        
        # Crear índice HNSW para búsqueda rápida por similitud (opcional pero recomendado)
        try:
            index_query = text("""
                CREATE INDEX IF NOT EXISTS movies_overview_embedding_idx 
                ON movies USING hnsw (overview_embedding vector_cosine_ops)
            """)
            conn.execute(index_query)
            conn.commit()
            logger.info("Índice HNSW creado para búsqueda por similitud")
        except Exception as e:
            logger.warning(f"No se pudo crear el índice HNSW (puede que ya exista): {e}")


def get_movies_without_embeddings(engine, batch_size=1000):
    """
    Obtiene películas que no tienen embeddings aún.
    
    Returns:
        DataFrame con id, tmdb_id, title, overview
    """
    query = text("""
        SELECT id, tmdb_id, title, overview
        FROM movies
        WHERE overview IS NOT NULL 
          AND overview != ''
          AND overview_embedding IS NULL
        ORDER BY id
    """)
    
    df = pd.read_sql(query, engine)
    return df


def generate_embeddings(texts, model):
    """
    Genera embeddings para una lista de textos.
    
    Args:
        texts: Lista de strings
        model: Modelo de sentence-transformers
    
    Returns:
        numpy array con los embeddings
    """
    if not texts or len(texts) == 0:
        return np.array([])
    
    # Filtrar textos None o vacíos
    valid_texts = [str(text) if text else "" for text in texts]
    embeddings = model.encode(valid_texts, show_progress_bar=False, batch_size=BATCH_SIZE)
    return embeddings


def save_embeddings_batch(engine, movie_ids, embeddings):
    """
    Guarda un batch de embeddings en la base de datos.
    
    Args:
        engine: SQLAlchemy engine
        movie_ids: Lista de IDs de películas
        embeddings: numpy array de embeddings
    """
    with engine.connect() as conn:
        for movie_id, embedding in zip(movie_ids, embeddings):
            # Convertir numpy array a string de formato pgvector
            embedding_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
            
            # Usar CAST para convertir el string a vector
            update_query = text("""
                UPDATE movies 
                SET overview_embedding = CAST(:embedding AS vector)
                WHERE id = :movie_id
            """)
            conn.execute(update_query, {"embedding": embedding_str, "movie_id": int(movie_id)})
        
        conn.commit()


def main():
    """Función principal."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers no está instalado.")
        logger.info("Instálalo con: pip install sentence-transformers")
        sys.exit(1)
    
    logger.info("Iniciando generación de embeddings para overviews de películas")
    logger.info(f"Modelo: {EMBEDDING_MODEL}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    
    # Conectar a la base de datos
    engine = get_database_engine()
    logger.info("Conectado a la base de datos")
    
    # Cargar modelo
    logger.info(f"Cargando modelo {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dimension = model.get_sentence_embedding_dimension()
    logger.info(f"Modelo cargado. Dimensión de embeddings: {embedding_dimension}")
    
    # Configurar columna vector
    setup_vector_column(engine, dimension=embedding_dimension)
    
    # Obtener películas sin embeddings
    logger.info("Obteniendo películas sin embeddings...")
    df_movies = get_movies_without_embeddings(engine)
    
    if df_movies.empty:
        logger.info("Todas las películas ya tienen embeddings. ¡Listo!")
        return
    
    total_movies = len(df_movies)
    logger.info(f"Procesando {total_movies} películas")
    
    # Procesar en batches
    processed = 0
    batch_size = BATCH_SIZE
    
    with tqdm(total=total_movies, desc="Generando embeddings") as pbar:
        for i in range(0, total_movies, batch_size):
            batch = df_movies.iloc[i:i+batch_size]
            
            # Generar embeddings
            overviews = batch['overview'].tolist()
            embeddings = generate_embeddings(overviews, model)
            
            # Guardar en base de datos
            movie_ids = batch['id'].tolist()
            save_embeddings_batch(engine, movie_ids, embeddings)
            
            processed += len(batch)
            pbar.update(len(batch))
            
            logger.debug(f"Procesadas {processed}/{total_movies} películas")
    
    logger.info(f"✅ Completado! {processed} embeddings generados y guardados")
    
    # Verificar resultados
    with engine.connect() as conn:
        count_query = text("""
            SELECT COUNT(*) 
            FROM movies 
            WHERE overview_embedding IS NOT NULL
        """)
        total_with_embeddings = conn.execute(count_query).scalar()
        logger.info(f"Total de películas con embeddings: {total_with_embeddings}")


if __name__ == "__main__":
    main()

