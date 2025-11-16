"""
Data loading utilities for semantic analysis.
"""
import pandas as pd
from sqlalchemy import create_engine
from typing import Optional, List


class MovieDataLoader:
    """
    Handles loading and preparing movie data for analysis.
    """
    
    def __init__(self, connection_string: str):
        """
        Initialize data loader.
        
        Args:
            connection_string: SQLAlchemy database connection string
        """
        self.engine = create_engine(connection_string)
    
    def load_movies_with_text(
        self,
        min_budget: int = 100000,
        include_genres: bool = True,
        include_keywords: bool = True
    ) -> pd.DataFrame:
        """
        Load movies with text fields and financial data.
        
        Args:
            min_budget: Minimum budget threshold to filter out bad data
            include_genres: Include genres as aggregated string
            include_keywords: Include keywords as aggregated string
            
        Returns:
            DataFrame with movies and text fields
        """
        # Base query for movies
        query = """
        SELECT 
            m.id,
            m.tmdb_id,
            m.title,
            m.original_title,
            m.overview,
            m.tagline,
            m.release_date,
            m.runtime,
            m.budget,
            m.revenue,
            m.popularity,
            m.vote_average,
            m.vote_count,
            m.original_language,
            m.production_countries
        FROM movies m
        WHERE m.budget >= %(min_budget)s
        AND m.revenue > 0
        AND m.budget IS NOT NULL
        AND m.revenue IS NOT NULL
        """
        
        df = pd.read_sql(
            query,
            self.engine,
            params={'min_budget': min_budget}
        )
        
        # Calculate ROI
        df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        df['is_profitable'] = (df['roi'] > 0).astype(int)
        
        # Add release year
        df['release_date'] = pd.to_datetime(df['release_date'])
        df['release_year'] = df['release_date'].dt.year
        
        print(f"Loaded {len(df)} movies with budget >= ${min_budget:,}")
        
        # Add genres if requested
        if include_genres:
            genres_df = self._load_genres_for_movies(df['id'].tolist())
            df = df.merge(genres_df, on='id', how='left')
            df['genres'] = df['genres'].fillna('')
        
        # Add keywords if requested
        if include_keywords:
            keywords_df = self._load_keywords_for_movies(df['id'].tolist())
            df = df.merge(keywords_df, on='id', how='left')
            df['keywords'] = df['keywords'].fillna('')
        
        return df
    
    def _load_genres_for_movies(self, movie_ids: List[int]) -> pd.DataFrame:
        """
        Load genres for specified movies as aggregated strings.
        
        Args:
            movie_ids: List of movie IDs
            
        Returns:
            DataFrame with movie_id and genres columns
        """
        if not movie_ids:
            return pd.DataFrame(columns=['id', 'genres'])
        
        query = """
        SELECT 
            mg.movie_id as id,
            STRING_AGG(g.name, ' ') as genres
        FROM movie_genres mg
        JOIN genres g ON g.id = mg.genre_id
        WHERE mg.movie_id = ANY(%(movie_ids)s)
        GROUP BY mg.movie_id
        """
        
        df = pd.read_sql(
            query,
            self.engine,
            params={'movie_ids': movie_ids}
        )
        
        return df
    
    def _load_keywords_for_movies(self, movie_ids: List[int]) -> pd.DataFrame:
        """
        Load keywords for specified movies as aggregated strings.
        
        Args:
            movie_ids: List of movie IDs
            
        Returns:
            DataFrame with movie_id and keywords columns
        """
        if not movie_ids:
            return pd.DataFrame(columns=['id', 'keywords'])
        
        query = """
        SELECT 
            k.movie_id as id,
            STRING_AGG(k.name, ' ') as keywords
        FROM keywords k
        WHERE k.movie_id = ANY(%(movie_ids)s)
        AND k.name IS NOT NULL
        AND k.name <> ''
        GROUP BY k.movie_id
        """
        
        df = pd.read_sql(
            query,
            self.engine,
            params={'movie_ids': movie_ids}
        )
        
        return df
    
    def load_keywords_detailed(self) -> pd.DataFrame:
        """
        Load keywords in detailed format (one row per movie-keyword pair).
        
        Returns:
            DataFrame with movie_id and keyword columns
        """
        query = """
        SELECT 
            k.movie_id,
            k.name as keyword
        FROM keywords k
        WHERE k.name IS NOT NULL
        AND k.name <> ''
        ORDER BY k.movie_id, k.name
        """
        
        df = pd.read_sql(query, self.engine)
        print(f"Loaded {len(df)} keyword associations ({df['keyword'].nunique()} unique keywords)")
        
        return df
    
    def get_connection(self):
        """Get database engine for custom queries."""
        return self.engine
