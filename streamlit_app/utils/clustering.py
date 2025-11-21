"""
Clustering utilities for thematic clustering of movie embeddings.
Uses UMAP for dimensionality reduction and HDBSCAN for clustering.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
from typing import Tuple, Optional
import os

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


def get_database_connection():
    """
    Create and return a database connection using environment variables or defaults.
    
    Returns:
        sqlalchemy.engine.Engine: Database engine
    """
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '25432')
    db_name = os.getenv('DB_NAME', 'movie_database')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    
    connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None


@st.cache_data
def load_embeddings_and_movie_data(_engine, sample_size: Optional[int] = None, random_seed: int = 42):
    """
    Load embeddings and movie data from the database.
    
    Args:
        _engine: SQLAlchemy engine (prefixed with _ to exclude from cache hashing)
        sample_size: Optional number of movies to sample (None for all)
        random_seed: Random seed for sampling
    
    Returns:
        Tuple of (embeddings array, movie dataframe)
    """
    if _engine is None:
        return None, None
    
    try:
        # Build query to get embeddings and movie data
        # pgvector stores vectors, we need to extract them as arrays
        if sample_size:
            query = text("""
                SELECT 
                    m.id,
                    m.title,
                    m.overview,
                    m.release_date,
                    m.budget,
                    m.revenue,
                    m.vote_average,
                    m.popularity,
                    STRING_AGG(g.name, ', ') as genres,
                    CASE 
                        WHEN m.budget > 0 THEN (m.revenue - m.budget) / m.budget::float
                        ELSE NULL
                    END as roi,
                    m.overview_embedding::text as embedding_str
                FROM movies m
                LEFT JOIN movie_genres mg ON m.id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                WHERE m.overview_embedding IS NOT NULL
                  AND m.budget > 0
                  AND m.revenue > 0
                GROUP BY m.id, m.title, m.overview, m.release_date, m.budget, 
                         m.revenue, m.vote_average, m.popularity, m.overview_embedding
                ORDER BY RANDOM()
                LIMIT :sample_size
            """)
            params = {"sample_size": sample_size}
        else:
            query = text("""
                SELECT 
                    m.id,
                    m.title,
                    m.overview,
                    m.release_date,
                    m.budget,
                    m.revenue,
                    m.vote_average,
                    m.popularity,
                    STRING_AGG(g.name, ', ') as genres,
                    CASE 
                        WHEN m.budget > 0 THEN (m.revenue - m.budget) / m.budget::float
                        ELSE NULL
                    END as roi,
                    m.overview_embedding::text as embedding_str
                FROM movies m
                LEFT JOIN movie_genres mg ON m.id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.id
                WHERE m.overview_embedding IS NOT NULL
                  AND m.budget > 0
                  AND m.revenue > 0
                GROUP BY m.id, m.title, m.overview, m.release_date, m.budget, 
                         m.revenue, m.vote_average, m.popularity, m.overview_embedding
            """)
            params = {}
        
        df = pd.read_sql(query, _engine, params=params)
        
        if df.empty:
            return None, None
        
        # Parse embedding strings to numpy arrays
        # The embedding_str is in format: [0.123, 0.456, ...]
        embeddings = []
        valid_indices = []
        
        for idx, row in df.iterrows():
            try:
                # Remove brackets and split by comma
                embedding_str = row['embedding_str'].strip('[]')
                embedding_array = np.array([float(x.strip()) for x in embedding_str.split(',')])
                embeddings.append(embedding_array)
                valid_indices.append(idx)
            except Exception as e:
                st.warning(f"Error parsing embedding for movie {row['id']}: {e}")
                continue
        
        if not embeddings:
            return None, None
        
        # Filter dataframe to only valid rows
        df_valid = df.loc[valid_indices].copy()
        embeddings_array = np.array(embeddings)
        
        return embeddings_array, df_valid.reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None, None


def perform_umap_reduction(embeddings: np.ndarray, n_components: int = 2, 
                          n_neighbors: int = 15, min_dist: float = 0.1,
                          random_state: int = 42) -> np.ndarray:
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        n_components: Number of dimensions for reduction (2 or 3 for visualization)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        random_state: Random seed
    
    Returns:
        Reduced embeddings array
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is not installed. Install it with: pip install umap-learn")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        metric='cosine'
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def perform_hdbscan_clustering(embeddings: np.ndarray, min_cluster_size: int = 10,
                               min_samples: Optional[int] = None,
                               cluster_selection_epsilon: float = 0.0) -> np.ndarray:
    """
    Perform HDBSCAN clustering on embeddings.
    
    Args:
        embeddings: Array of embeddings (can be original or UMAP-reduced)
        min_cluster_size: Minimum size of clusters
        min_samples: Minimum samples in neighborhood (defaults to min_cluster_size)
        cluster_selection_epsilon: Distance threshold for cluster selection
    
    Returns:
        Cluster labels array (-1 for noise points)
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed. Install it with: pip install hdbscan")
    
    if min_samples is None:
        min_samples = min_cluster_size
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean' if embeddings.shape[1] <= 3 else 'cosine',
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    return cluster_labels


def analyze_clusters(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Analyze ROI statistics by cluster.
    
    Args:
        df: DataFrame with movie data including ROI
        cluster_labels: Array of cluster labels
    
    Returns:
        DataFrame with cluster statistics
    """
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # Calculate statistics per cluster
    cluster_stats = []
    
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c >= 0])
    
    for cluster_id in unique_clusters:
        cluster_movies = df_clustered[df_clustered['cluster'] == cluster_id]
        
        stats = {
            'cluster_id': cluster_id,
            'n_movies': len(cluster_movies),
            'roi_mean': cluster_movies['roi'].mean(),
            'roi_median': cluster_movies['roi'].median(),
                'roi_std': cluster_movies['roi'].std(),
            'roi_min': cluster_movies['roi'].min(),
            'roi_max': cluster_movies['roi'].max(),
            'budget_mean': cluster_movies['budget'].mean(),
                'revenue_mean': cluster_movies['revenue'].mean(),
                'revenue_median': cluster_movies['revenue'].median(),
                'revenue_std': cluster_movies['revenue'].std(),
            'vote_average_mean': cluster_movies['vote_average'].mean(),
        }
        
        # Get most common genres in cluster
        if 'genres' in cluster_movies.columns:
            all_genres = cluster_movies['genres'].dropna().str.split(', ').explode()
            top_genres = all_genres.value_counts().head(3)
            stats['top_genres'] = ', '.join(top_genres.index.tolist())
        
        cluster_stats.append(stats)
    
    # Add noise cluster stats if present
    noise_movies = df_clustered[df_clustered['cluster'] == -1]
    if len(noise_movies) > 0:
        stats = {
            'cluster_id': -1,
            'n_movies': len(noise_movies),
            'roi_mean': noise_movies['roi'].mean(),
            'roi_median': noise_movies['roi'].median(),
            'roi_std': noise_movies['roi'].std(),
            'roi_min': noise_movies['roi'].min(),
            'roi_max': noise_movies['roi'].max(),
            'budget_mean': noise_movies['budget'].mean(),
            'revenue_mean': noise_movies['revenue'].mean(),
            'revenue_median': noise_movies['revenue'].median(),
            'revenue_std': noise_movies['revenue'].std(),
            'vote_average_mean': noise_movies['vote_average'].mean(),
            'top_genres': 'Noise'
        }
        cluster_stats.append(stats)
    
    return pd.DataFrame(cluster_stats)


def get_cluster_representative_movies(df: pd.DataFrame, cluster_labels: np.ndarray,
                                      embeddings: np.ndarray, n_per_cluster: int = 3) -> dict:
    """
    Get representative movies for each cluster (closest to cluster centroid).
    
    Args:
        df: DataFrame with movie data
        cluster_labels: Array of cluster labels
        embeddings: Original embeddings array
        n_per_cluster: Number of representative movies per cluster
    
    Returns:
        Dictionary mapping cluster_id to list of representative movies
    """
    representatives = {}
    
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c >= 0])
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_df = df[cluster_mask].reset_index(drop=True)
        
        # Calculate centroid
        centroid = cluster_embeddings.mean(axis=0)
        
        # Calculate distances to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # Get closest movies
        closest_indices = np.argsort(distances)[:n_per_cluster]
        representatives[cluster_id] = cluster_df.iloc[closest_indices].to_dict('records')
    
    return representatives

