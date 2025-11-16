"""
Text preprocessing utilities for semantic analysis.
"""
import re
import pandas as pd
import numpy as np
from typing import List, Optional, Union


class TextPreprocessor:
    """
    Handles text preprocessing for movie text fields.
    
    This class provides methods to clean, normalize, and combine
    text fields from movies for semantic analysis.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        min_word_length: int = 2
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_numbers: Remove numeric characters
            min_word_length: Minimum word length to keep
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Filter by minimum word length
        if self.min_word_length > 0:
            words = text.split()
            words = [w for w in words if len(w) >= self.min_word_length]
            text = ' '.join(words)
        
        return text
    
    def combine_text_fields(
        self,
        df: pd.DataFrame,
        fields: List[str],
        weights: Optional[dict] = None,
        separator: str = ' '
    ) -> pd.Series:
        """
        Combine multiple text fields into a single document per movie.
        
        Args:
            df: DataFrame with movie data
            fields: List of column names to combine
            weights: Optional dictionary mapping field names to repetition weights
                    (e.g., {'title': 3} will repeat title text 3 times)
            separator: String to join fields
            
        Returns:
            Series with combined text documents
        """
        if weights is None:
            weights = {}
        
        combined_texts = []
        
        for idx, row in df.iterrows():
            parts = []
            
            for field in fields:
                if field not in df.columns:
                    continue
                
                text = row[field]
                cleaned = self.clean_text(text)
                
                if cleaned:
                    # Apply weight by repeating text
                    repeat_count = weights.get(field, 1)
                    parts.extend([cleaned] * repeat_count)
            
            combined_texts.append(separator.join(parts))
        
        return pd.Series(combined_texts, index=df.index)
    
    def prepare_movie_documents(
        self,
        df: pd.DataFrame,
        include_title: bool = True,
        include_overview: bool = True,
        include_tagline: bool = True,
        include_genres: bool = True,
        include_keywords: bool = True,
        title_weight: int = 2,
        tagline_weight: int = 1
    ) -> pd.DataFrame:
        """
        Prepare movie text documents for analysis.
        
        Args:
            df: DataFrame with movie data (must have 'id' or 'movie_id' column)
            include_title: Include title field
            include_overview: Include overview field
            include_tagline: Include tagline field
            include_genres: Include genres field
            include_keywords: Include keywords field
            title_weight: Weight for title (repetitions)
            tagline_weight: Weight for tagline (repetitions)
            
        Returns:
            DataFrame with 'document' column containing combined text
        """
        # Determine fields to include
        fields = []
        weights = {}
        
        if include_title and 'title' in df.columns:
            fields.append('title')
            weights['title'] = title_weight
        
        if include_tagline and 'tagline' in df.columns:
            fields.append('tagline')
            weights['tagline'] = tagline_weight
        
        if include_overview and 'overview' in df.columns:
            fields.append('overview')
        
        if include_genres and 'genres' in df.columns:
            fields.append('genres')
        
        if include_keywords and 'keywords' in df.columns:
            fields.append('keywords')
        
        # Create result DataFrame
        result_df = df.copy()
        
        # Combine text fields
        result_df['document'] = self.combine_text_fields(
            df, fields, weights=weights
        )
        
        # Filter out empty documents
        result_df = result_df[result_df['document'].str.len() > 0].copy()
        
        return result_df
    
    def format_genres(self, genres_str: str) -> str:
        """
        Format genres string from database format.
        
        Args:
            genres_str: Genres as comma-separated string or similar
            
        Returns:
            Cleaned genres string
        """
        if pd.isna(genres_str) or genres_str is None:
            return ""
        
        # Handle different formats (comma-separated, pipe-separated, etc.)
        genres = str(genres_str)
        genres = genres.replace('|', ' ').replace(',', ' ')
        return self.clean_text(genres)
    
    def format_keywords(self, keywords_str: str) -> str:
        """
        Format keywords string from database format.
        
        Args:
            keywords_str: Keywords as comma-separated string or similar
            
        Returns:
            Cleaned keywords string
        """
        if pd.isna(keywords_str) or keywords_str is None:
            return ""
        
        # Handle different formats
        keywords = str(keywords_str)
        keywords = keywords.replace('|', ' ').replace(',', ' ')
        return self.clean_text(keywords)
