"""
Data loading and preparation module for the Streamlit ROI prediction app.
"""
import pandas as pd
import numpy as np
import streamlit as st
from utils.database import get_database_connection


@st.cache_data
def load_movies_data():
    """
    Load movies data from the database with all necessary joins.
    
    Returns:
        pd.DataFrame: Movies data with genres and other features
    """
    engine = get_database_connection()
    if engine is None:
        return None
    
    # Main query to get movies with genres and embeddings
    query = """
    SELECT 
        m.id, m.tmdb_id, m.title, m.original_title, m.overview, m.tagline,
        m.release_date, m.runtime, m.budget, m.revenue, m.popularity,
        m.vote_average, m.vote_count, m.poster_path, m.backdrop_path,
        m.adult, m.status, m.original_language, m.production_companies,
        m.production_countries, m.spoken_languages, m.created_at, m.updated_at,
        m.overview_embedding,
        STRING_AGG(g.name, ', ') as genres
    FROM movies m
    LEFT JOIN movie_genres mg ON m.id = mg.movie_id
    LEFT JOIN genres g ON mg.genre_id = g.id
    GROUP BY m.id, m.tmdb_id, m.title, m.original_title, m.overview, m.tagline,
             m.release_date, m.runtime, m.budget, m.revenue, m.popularity,
             m.vote_average, m.vote_count, m.poster_path, m.backdrop_path,
             m.adult, m.status, m.original_language, m.production_companies,
             m.production_countries, m.spoken_languages, m.created_at, m.updated_at,
             m.overview_embedding
    ORDER BY m.popularity DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading movies data: {str(e)}")
        return None


@st.cache_data
def load_genres_data():
    """
    Load genres data from the database.
    
    Returns:
        pd.DataFrame: Genres data
    """
    engine = get_database_connection()
    if engine is None:
        return None
    
    query = """
    SELECT g.id, g.tmdb_id, g.name, COUNT(mg.movie_id) as movie_count
    FROM genres g
    LEFT JOIN movie_genres mg ON g.id = mg.genre_id
    GROUP BY g.id, g.tmdb_id, g.name
    ORDER BY movie_count DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading genres data: {str(e)}")
        return None


def prepare_movies_for_modeling(df_movies):
    """
    Prepare movies data for machine learning modeling.
    
    Args:
        df_movies (pd.DataFrame): Raw movies data
        
    Returns:
        pd.DataFrame: Prepared data ready for modeling
    """
    if df_movies is None or df_movies.empty:
        return None
    
    df = df_movies.copy()
    
    # Convert release_date to datetime and extract year
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # Calculate ROI
    df['roi'] = np.where(
        (df['budget'] > 0) & (df['revenue'].notna()),
        (df['revenue'] - df['budget']) / df['budget'],
        np.nan
    )
    
    # Create profitability binary variable
    df['is_profitable'] = np.where(df['roi'] > 0, 1, 0)
    
    # Extract main country from production_countries JSON
    def get_main_country_name(row):
        try:
            import json
            countries = json.loads(row) if isinstance(row, str) else []
            if countries and isinstance(countries, list) and len(countries) > 0:
                return countries[0].get("name", "Unknown")
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    df['main_country'] = df['production_countries'].apply(get_main_country_name)
    
    # Combine tagline and overview for text analysis
    df['text_content'] = (
        df['tagline'].fillna('') + ' ' + df['overview'].fillna('')
    ).str.strip()
    
    # Filter data quality issues (as per notebook analysis)
    # Remove movies with budget < $100,000 (likely data errors)
    df_clean = df[df['budget'] >= 100000].copy()
    
    # Remove movies without valid ROI
    df_clean = df_clean[df_clean['roi'].notna()].copy()
    
    # Remove movies without release year
    df_clean = df_clean[df_clean['release_year'].notna()].copy()
    
    return df_clean


def get_data_summary(df):
    """
    Get summary statistics of the prepared dataset.
    
    Args:
        df (pd.DataFrame): Prepared movies data
        
    Returns:
        dict: Summary statistics
    """
    if df is None or df.empty:
        return {}
    
    summary = {
        'total_movies': len(df),
        'profitable_movies': len(df[df['is_profitable'] == 1]),
        'profitability_rate': df['is_profitable'].mean() * 100,
        'avg_roi': df['roi'].mean(),
        'median_roi': df['roi'].median(),
        'year_range': (df['release_year'].min(), df['release_year'].max()),
        'budget_range': (df['budget'].min(), df['budget'].max()),
        'revenue_range': (df['revenue'].min(), df['revenue'].max()),
        'unique_genres': df['genres'].nunique(),
        'unique_countries': df['main_country'].nunique(),
        'unique_languages': df['original_language'].nunique()
    }
    
    return summary


