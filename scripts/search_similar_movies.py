#!/usr/bin/env python3
"""
Script de ejemplo para buscar películas similares usando embeddings.

Este script muestra cómo usar los embeddings generados para encontrar
películas similares basándose en la similitud semántica de sus overviews.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
import pandas as pd
from sentence_transformers import SentenceTransformer

# Configuración de la base de datos
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '25432')
DB_NAME = os.getenv('DB_NAME', 'movie_database')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')

# Modelo de embeddings (debe ser el mismo usado para generar los embeddings)
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')


def get_database_engine():
    """Crea y retorna el engine de SQLAlchemy."""
    database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(database_url, pool_pre_ping=True)


def search_similar_movies(engine, query_text, top_k=10):
    """
    Busca películas similares a un texto de consulta.
    
    Args:
        engine: SQLAlchemy engine
        query_text: Texto de búsqueda (ej: "película de acción con superhéroes")
        top_k: Número de resultados a retornar
    
    Returns:
        DataFrame con las películas más similares
    """
    # Cargar modelo para generar embedding de la consulta
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query_text, show_progress_bar=False)
    embedding_str = '[' + ','.join(map(str, query_embedding.tolist())) + ']'
    
    # Buscar películas similares usando distancia coseno
    # El operador <#> retorna el producto interno negativo
    # Para similitud coseno: 1 - (producto_interno_negativo) cuando los vectores están normalizados
    # Usamos 1 - distance para obtener similitud, pero necesitamos normalizar
    query = text("""
        SELECT 
            id,
            tmdb_id,
            title,
            overview,
            release_date,
            vote_average,
            popularity,
            1 - (overview_embedding <#> CAST(:query_embedding AS vector)) AS similarity
        FROM movies
        WHERE overview_embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT :top_k
    """)
    
    df = pd.read_sql(
        query, 
        engine, 
        params={"query_embedding": embedding_str, "top_k": top_k}
    )
    
    # Asegurar que la similitud esté en el rango [0, 1]
    df['similarity'] = df['similarity'].clip(0, 1)
    
    return df


def find_similar_to_movie(engine, movie_id, top_k=10):
    """
    Encuentra películas similares a una película específica.
    
    Args:
        engine: SQLAlchemy engine
        movie_id: ID de la película de referencia
        top_k: Número de resultados a retornar
    
    Returns:
        DataFrame con las películas más similares (excluyendo la película original)
    """
    query = text("""
        SELECT 
            m1.id,
            m1.tmdb_id,
            m1.title,
            m1.overview,
            m1.release_date,
            m1.vote_average,
            m1.popularity,
            1 - (m1.overview_embedding <#> m2.overview_embedding) AS similarity
        FROM movies m1
        CROSS JOIN movies m2
        WHERE m2.id = :movie_id
          AND m1.id != :movie_id
          AND m1.overview_embedding IS NOT NULL
          AND m2.overview_embedding IS NOT NULL
        ORDER BY similarity DESC
        LIMIT :top_k
    """)
    
    df = pd.read_sql(
        query,
        engine,
        params={"movie_id": movie_id, "top_k": top_k}
    )
    
    # Asegurar que la similitud esté en el rango [0, 1]
    df['similarity'] = df['similarity'].clip(0, 1)
    
    return df


def main():
    """Función principal con ejemplos de uso."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Buscar películas similares usando embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Búsqueda por texto
  python search_similar_movies.py --query "película de acción con superhéroes"
  
  # Búsqueda por ID de película
  python search_similar_movies.py --movie-id 123
  
  # Especificar número de resultados
  python search_similar_movies.py --query "ciencia ficción" --top-k 20
        """
    )
    parser.add_argument('--query', type=str, help='Texto de búsqueda (ej: "película de acción")')
    parser.add_argument('--movie-id', type=int, help='ID de película para buscar similares')
    parser.add_argument('--top-k', type=int, default=10, help='Número de resultados (default: 10)')
    parser.add_argument('--examples', action='store_true', help='Mostrar ejemplos predefinidos')
    
    args = parser.parse_args()
    
    engine = get_database_engine()
    
    print("=" * 70)
    print("Búsqueda de Películas Similares usando Embeddings")
    print("=" * 70)
    print()
    
    if args.examples or (not args.query and not args.movie_id):
        # Modo ejemplos
        print("Ejemplo 1: Buscar películas similares a un texto")
        print("-" * 70)
        query = "película de ciencia ficción con robots y futuro distópico"
        print(f"Consulta: '{query}'")
        print()
        
        results = search_similar_movies(engine, query, top_k=5)
        print(f"Top 5 películas similares:")
        for idx, row in results.iterrows():
            year = row['release_date'].year if pd.notna(row['release_date']) else 'N/A'
            print(f"\n{idx+1}. {row['title']} ({year})")
            print(f"   Similitud: {row['similarity']:.3f} | Rating: {row['vote_average']:.1f}/10")
            print(f"   {row['overview'][:120]}...")
        
        print("\n" + "=" * 70)
        print()
        
        # Ejemplo 2: Encontrar películas similares a una película específica
        print("Ejemplo 2: Encontrar películas similares a una película específica")
        print("-" * 70)
        
        # Buscar una película popular como ejemplo
        sample_query = text("""
            SELECT id, title, overview 
            FROM movies 
            WHERE overview_embedding IS NOT NULL 
              AND vote_average > 7.5
            ORDER BY popularity DESC
            LIMIT 1
        """)
        sample = pd.read_sql(sample_query, engine).iloc[0]
        
        print(f"Película de referencia: '{sample['title']}' (ID: {sample['id']})")
        print(f"Overview: {sample['overview'][:150]}...")
        print()
        
        similar = find_similar_to_movie(engine, int(sample['id']), top_k=5)
        print(f"Top 5 películas similares:")
        for idx, row in similar.iterrows():
            year = row['release_date'].year if pd.notna(row['release_date']) else 'N/A'
            print(f"\n{idx+1}. {row['title']} ({year})")
            print(f"   Similitud: {row['similarity']:.3f} | Rating: {row['vote_average']:.1f}/10")
            print(f"   {row['overview'][:120]}...")
        
        print("\n" + "=" * 70)
        print("¡Búsqueda completada!")
        
    elif args.query:
        # Búsqueda por texto
        print(f"Buscando películas similares a: '{args.query}'")
        print("-" * 70)
        print()
        
        results = search_similar_movies(engine, args.query, top_k=args.top_k)
        print(f"Top {len(results)} películas similares:\n")
        
        for idx, row in results.iterrows():
            year = row['release_date'].year if pd.notna(row['release_date']) else 'N/A'
            print(f"{idx+1}. {row['title']} ({year})")
            print(f"   Similitud: {row['similarity']:.3f} | Rating: {row['vote_average']:.1f}/10 | Popularidad: {row['popularity']:.1f}")
            print(f"   {row['overview']}")
            print()
    
    elif args.movie_id:
        # Búsqueda por ID de película
        # Primero obtener información de la película
        movie_query = text("""
            SELECT id, title, overview 
            FROM movies 
            WHERE id = :movie_id
        """)
        movie = pd.read_sql(movie_query, engine, params={"movie_id": args.movie_id})
        
        if movie.empty:
            print(f"❌ No se encontró la película con ID: {args.movie_id}")
            return
        
        movie = movie.iloc[0]
        print(f"Buscando películas similares a: '{movie['title']}' (ID: {movie['id']})")
        print(f"Overview: {movie['overview']}")
        print("-" * 70)
        print()
        
        similar = find_similar_to_movie(engine, args.movie_id, top_k=args.top_k)
        print(f"Top {len(similar)} películas similares:\n")
        
        for idx, row in similar.iterrows():
            year = row['release_date'].year if pd.notna(row['release_date']) else 'N/A'
            print(f"{idx+1}. {row['title']} ({year})")
            print(f"   Similitud: {row['similarity']:.3f} | Rating: {row['vote_average']:.1f}/10 | Popularidad: {row['popularity']:.1f}")
            print(f"   {row['overview']}")
            print()


if __name__ == "__main__":
    main()

