"""
TF-IDF analysis for movie text features.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr, spearmanr
from typing import Optional, List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns


class TFIDFAnalyzer:
    """
    TF-IDF based semantic analysis for movies.
    
    This class extracts TF-IDF features from movie text and analyzes
    their relationship with ROI and other target variables.
    """
    
    def __init__(
        self,
        max_features: int = 500,
        min_df: int = 5,
        max_df: float = 0.8,
        ngram_range: Tuple[int, int] = (1, 2),
        use_idf: bool = True,
        sublinear_tf: bool = True
    ):
        """
        Initialize TF-IDF analyzer.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency (absolute count or fraction)
            max_df: Maximum document frequency (fraction)
            ngram_range: Range of n-grams to consider (e.g., (1,2) for unigrams and bigrams)
            use_idf: Enable inverse-document-frequency reweighting
            sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf))
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names = None
        self.svd_model = None
        self.documents_df = None
    
    def fit_transform(self, documents: pd.Series) -> np.ndarray:
        """
        Fit TF-IDF vectorizer and transform documents.
        
        Args:
            documents: Series of text documents
            
        Returns:
            TF-IDF matrix (sparse)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            sublinear_tf=self.sublinear_tf,
            stop_words='english'
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Number of features: {len(self.feature_names)}")
        
        return self.tfidf_matrix
    
    def transform(self, documents: pd.Series) -> np.ndarray:
        """
        Transform new documents using fitted vectorizer.
        
        Args:
            documents: Series of text documents
            
        Returns:
            TF-IDF matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        return self.vectorizer.transform(documents)
    
    def get_top_terms_per_document(
        self,
        doc_index: int,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top TF-IDF terms for a specific document.
        
        Args:
            doc_index: Index of the document
            top_n: Number of top terms to return
            
        Returns:
            DataFrame with terms and their TF-IDF scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not computed. Call fit_transform first.")
        
        # Get TF-IDF scores for this document
        doc_vector = self.tfidf_matrix[doc_index].toarray().flatten()
        
        # Get top indices
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        
        # Create DataFrame
        results = pd.DataFrame({
            'term': self.feature_names[top_indices],
            'tfidf_score': doc_vector[top_indices]
        })
        
        return results
    
    def get_top_terms_global(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get globally most important terms across all documents.
        
        Args:
            top_n: Number of top terms to return
            
        Returns:
            DataFrame with terms and their average TF-IDF scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not computed. Call fit_transform first.")
        
        # Calculate mean TF-IDF score for each term
        mean_tfidf = np.asarray(self.tfidf_matrix.mean(axis=0)).flatten()
        
        # Get top indices
        top_indices = mean_tfidf.argsort()[-top_n:][::-1]
        
        # Create DataFrame
        results = pd.DataFrame({
            'term': self.feature_names[top_indices],
            'mean_tfidf': mean_tfidf[top_indices]
        })
        
        return results
    
    def correlate_with_target(
        self,
        target: pd.Series,
        method: str = 'pearson',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Calculate correlation between TF-IDF features and a target variable.
        
        Args:
            target: Target variable (e.g., ROI)
            method: Correlation method ('pearson' or 'spearman')
            top_n: Number of top correlated terms to return
            
        Returns:
            DataFrame with terms and their correlation coefficients
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not computed. Call fit_transform first.")
        
        # Ensure target aligns with matrix
        if len(target) != self.tfidf_matrix.shape[0]:
            raise ValueError("Target length must match number of documents")
        
        correlations = []
        p_values = []
        
        # Calculate correlation for each feature
        tfidf_dense = self.tfidf_matrix.toarray()
        
        for i, term in enumerate(self.feature_names):
            feature_values = tfidf_dense[:, i]
            
            # Only calculate if there's variance
            if np.std(feature_values) > 0 and np.std(target) > 0:
                if method == 'pearson':
                    corr, p_val = pearsonr(feature_values, target)
                elif method == 'spearman':
                    corr, p_val = spearmanr(feature_values, target)
                else:
                    raise ValueError("Method must be 'pearson' or 'spearman'")
                
                correlations.append(corr)
                p_values.append(p_val)
            else:
                correlations.append(0)
                p_values.append(1)
        
        # Create DataFrame
        results = pd.DataFrame({
            'term': self.feature_names,
            'correlation': correlations,
            'p_value': p_values,
            'abs_correlation': np.abs(correlations)
        })
        
        # Sort by absolute correlation
        results = results.sort_values('abs_correlation', ascending=False)
        
        return results.head(top_n)
    
    def analyze_roi_segments(
        self,
        df: pd.DataFrame,
        roi_column: str = 'roi',
        n_segments: int = 4,
        top_terms_per_segment: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze top terms for different ROI segments (quartiles).
        
        Args:
            df: DataFrame with documents and ROI
            roi_column: Name of ROI column
            n_segments: Number of segments to create
            top_terms_per_segment: Top terms to show per segment
            
        Returns:
            Dictionary mapping segment names to DataFrames with top terms
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not computed. Call fit_transform first.")
        
        # Create ROI segments
        df = df.copy()
        df['roi_segment'] = pd.qcut(
            df[roi_column],
            q=n_segments,
            labels=[f'Q{i+1}' for i in range(n_segments)],
            duplicates='drop'
        )
        
        results = {}
        
        # Analyze each segment
        for segment in df['roi_segment'].unique():
            segment_mask = df['roi_segment'] == segment
            segment_indices = df[segment_mask].index
            
            # Get mean TF-IDF for this segment
            segment_tfidf = self.tfidf_matrix[segment_indices].mean(axis=0)
            segment_tfidf = np.asarray(segment_tfidf).flatten()
            
            # Get top terms
            top_indices = segment_tfidf.argsort()[-top_terms_per_segment:][::-1]
            
            results[str(segment)] = pd.DataFrame({
                'term': self.feature_names[top_indices],
                'mean_tfidf': segment_tfidf[top_indices]
            })
        
        return results
    
    def apply_dimensionality_reduction(
        self,
        n_components: int = 50
    ) -> np.ndarray:
        """
        Apply SVD dimensionality reduction to TF-IDF matrix.
        
        Args:
            n_components: Number of components to keep
            
        Returns:
            Reduced feature matrix
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not computed. Call fit_transform first.")
        
        self.svd_model = TruncatedSVD(
            n_components=min(n_components, self.tfidf_matrix.shape[1] - 1),
            random_state=42
        )
        
        reduced_features = self.svd_model.fit_transform(self.tfidf_matrix)
        
        print(f"Reduced features shape: {reduced_features.shape}")
        print(f"Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.3f}")
        
        return reduced_features
    
    def plot_top_terms(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot top TF-IDF terms.
        
        Args:
            top_n: Number of top terms to plot
            figsize: Figure size
        """
        top_terms = self.get_top_terms_global(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_terms)), top_terms['mean_tfidf'])
        plt.yticks(range(len(top_terms)), top_terms['term'])
        plt.xlabel('Mean TF-IDF Score')
        plt.title(f'Top {top_n} Terms by TF-IDF Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_with_roi(
        self,
        target: pd.Series,
        top_n: int = 20,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Plot terms most correlated with target variable.
        
        Args:
            target: Target variable (e.g., ROI)
            top_n: Number of top terms to plot
            method: Correlation method
            figsize: Figure size
        """
        correlations = self.correlate_with_target(target, method=method, top_n=top_n)
        
        # Separate positive and negative correlations
        pos_corr = correlations[correlations['correlation'] > 0].sort_values('correlation')
        neg_corr = correlations[correlations['correlation'] < 0].sort_values('correlation', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Positive correlations
        if len(pos_corr) > 0:
            ax1.barh(range(len(pos_corr)), pos_corr['correlation'], color='green', alpha=0.7)
            ax1.set_yticks(range(len(pos_corr)))
            ax1.set_yticklabels(pos_corr['term'])
            ax1.set_xlabel('Correlation with ROI')
            ax1.set_title('Terms Positively Correlated with ROI')
            ax1.invert_yaxis()
        
        # Negative correlations
        if len(neg_corr) > 0:
            ax2.barh(range(len(neg_corr)), neg_corr['correlation'], color='red', alpha=0.7)
            ax2.set_yticks(range(len(neg_corr)))
            ax2.set_yticklabels(neg_corr['term'])
            ax2.set_xlabel('Correlation with ROI')
            ax2.set_title('Terms Negatively Correlated with ROI')
            ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return correlations
