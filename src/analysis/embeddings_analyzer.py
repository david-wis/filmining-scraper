"""
Sentence embeddings analysis for movie text features.
Uses transformer-based models (BERT, Sentence-BERT) to capture semantic meaning.
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not installed. Install with: pip install sentence-transformers")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


class EmbeddingsAnalyzer:
    """
    Sentence embeddings based semantic analysis for movies.
    
    Uses pre-trained transformer models to create dense vector representations
    that capture semantic meaning and context.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = None
    ):
        """
        Initialize embeddings analyzer.
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       Popular options:
                       - 'all-MiniLM-L6-v2': Fast, good quality (384 dim)
                       - 'all-mpnet-base-v2': Best quality (768 dim)
                       - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
            device: Device to run model on ('cpu', 'cuda', or None for auto)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embeddings = None
        self.documents = None
        self.pca_model = None
        self.cluster_model = None
        
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            print(f"Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"✓ Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        self.load_model()
        
        # Filter out None/empty texts
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
        
        print(f"Encoding {len(valid_texts)} valid texts (out of {len(texts)} total)...")
        
        # Encode
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Create full array with zeros for invalid texts
        full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
        full_embeddings[valid_indices] = embeddings
        
        self.embeddings = full_embeddings
        return full_embeddings
    
    def compute_similarity(
        self,
        embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between embeddings.
        
        Args:
            embeddings: Embeddings array, or None to use stored embeddings
            
        Returns:
            Similarity matrix, shape (n_texts, n_texts)
        """
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            raise ValueError("No embeddings available. Run encode() first.")
        
        return cosine_similarity(embeddings)
    
    def find_similar_movies(
        self,
        query_idx: int,
        top_k: int = 10,
        embeddings: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find most similar movies to a query movie.
        
        Args:
            query_idx: Index of query movie
            top_k: Number of similar movies to return
            embeddings: Embeddings array, or None to use stored
            
        Returns:
            Tuple of (indices, similarities) for top_k most similar movies
        """
        similarities = self.compute_similarity(embeddings)
        query_sims = similarities[query_idx]
        
        # Sort by similarity (excluding self)
        sorted_indices = np.argsort(query_sims)[::-1]
        sorted_indices = sorted_indices[sorted_indices != query_idx]
        
        top_indices = sorted_indices[:top_k]
        top_sims = query_sims[top_indices]
        
        return top_indices, top_sims
    
    def cluster_movies(
        self,
        n_clusters: int = 10,
        embeddings: Optional[np.ndarray] = None,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Cluster movies based on semantic embeddings.
        
        Args:
            n_clusters: Number of clusters
            embeddings: Embeddings array, or None to use stored
            random_state: Random seed
            
        Returns:
            Cluster labels array
        """
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            raise ValueError("No embeddings available. Run encode() first.")
        
        self.cluster_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
        labels = self.cluster_model.fit_predict(embeddings)
        return labels
    
    def reduce_dimensions(
        self,
        n_components: int = 2,
        embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reduce embedding dimensions using PCA for visualization.
        
        Args:
            n_components: Number of dimensions to reduce to (2 or 3)
            embeddings: Embeddings array, or None to use stored
            
        Returns:
            Reduced embeddings
        """
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            raise ValueError("No embeddings available. Run encode() first.")
        
        self.pca_model = PCA(n_components=n_components, random_state=42)
        reduced = self.pca_model.fit_transform(embeddings)
        
        print(f"✓ Reduced to {n_components}D. Explained variance: {self.pca_model.explained_variance_ratio_.sum():.2%}")
        
        return reduced
    
    def analyze_roi_by_clusters(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        roi_column: str = 'roi'
    ) -> pd.DataFrame:
        """
        Analyze ROI statistics by semantic cluster.
        
        Args:
            df: DataFrame with movie data
            cluster_labels: Cluster assignments
            roi_column: Name of ROI column
            
        Returns:
            DataFrame with cluster statistics
        """
        df_analysis = df.copy()
        df_analysis['cluster'] = cluster_labels
        
        cluster_stats = df_analysis.groupby('cluster').agg({
            roi_column: ['count', 'mean', 'median', 'std', 'min', 'max']
        }).round(3)
        
        cluster_stats.columns = ['n_movies', 'roi_mean', 'roi_median', 'roi_std', 'roi_min', 'roi_max']
        cluster_stats = cluster_stats.sort_values('roi_mean', ascending=False)
        
        return cluster_stats
    
    def correlate_embeddings_with_roi(
        self,
        roi_values: pd.Series,
        embeddings: Optional[np.ndarray] = None,
        method: str = 'spearman',
        top_dims: int = 20
    ) -> pd.DataFrame:
        """
        Compute correlation between embedding dimensions and ROI.
        
        Args:
            roi_values: ROI values
            embeddings: Embeddings array, or None to use stored
            method: Correlation method ('pearson' or 'spearman')
            top_dims: Number of top dimensions to return
            
        Returns:
            DataFrame with dimension correlations
        """
        if embeddings is None:
            embeddings = self.embeddings
        
        if embeddings is None:
            raise ValueError("No embeddings available. Run encode() first.")
        
        correlations = []
        p_values = []
        
        for dim in range(embeddings.shape[1]):
            dim_values = embeddings[:, dim]
            
            if np.std(dim_values) > 0 and np.std(roi_values) > 0:
                if method == 'pearson':
                    corr, p_val = pearsonr(dim_values, roi_values)
                elif method == 'spearman':
                    corr, p_val = spearmanr(dim_values, roi_values)
                else:
                    raise ValueError("Method must be 'pearson' or 'spearman'")
                
                correlations.append(corr)
                p_values.append(p_val)
            else:
                correlations.append(0)
                p_values.append(1)
        
        results = pd.DataFrame({
            'dimension': range(len(correlations)),
            'correlation': correlations,
            'p_value': p_values,
            'abs_correlation': np.abs(correlations)
        })
        
        results = results.sort_values('abs_correlation', ascending=False)
        return results.head(top_dims)
    
    def plot_clusters_2d(
        self,
        reduced_embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        roi_values: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Visualize clusters in 2D space.
        
        Args:
            reduced_embeddings: 2D reduced embeddings
            cluster_labels: Cluster assignments
            roi_values: Optional ROI values for coloring
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Colored by cluster
        scatter1 = axes[0].scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=cluster_labels,
            cmap='tab10',
            alpha=0.6,
            s=30
        )
        axes[0].set_xlabel('PC 1')
        axes[0].set_ylabel('PC 2')
        axes[0].set_title('Movies by Semantic Cluster')
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Plot 2: Colored by ROI
        if roi_values is not None:
            scatter2 = axes[1].scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                c=roi_values,
                cmap='RdYlGn',
                alpha=0.6,
                s=30
            )
            axes[1].set_xlabel('PC 1')
            axes[1].set_ylabel('PC 2')
            axes[1].set_title('Movies by ROI')
            plt.colorbar(scatter2, ax=axes[1], label='ROI')
        
        plt.tight_layout()
        return fig
    
    def get_cluster_representative_movies(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        embeddings: np.ndarray,
        n_per_cluster: int = 3
    ) -> Dict[int, pd.DataFrame]:
        """
        Get most representative movies for each cluster (closest to centroid).
        
        Args:
            df: DataFrame with movie data
            cluster_labels: Cluster assignments
            embeddings: Embeddings array
            n_per_cluster: Number of movies per cluster
            
        Returns:
            Dictionary mapping cluster_id to DataFrame of representative movies
        """
        representatives = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0)
            
            # Find closest movies to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_indices = np.argsort(distances)[:n_per_cluster]
            
            # Get original indices
            original_indices = cluster_indices[closest_indices]
            
            representatives[cluster_id] = df.iloc[original_indices]
        
        return representatives
