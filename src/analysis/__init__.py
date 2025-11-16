"""
Semantic analysis module for movie text features.

This module provides various text analysis techniques to extract insights
from movie textual data (title, overview, tagline, genres, keywords).
"""

from .text_preprocessor import TextPreprocessor
from .tfidf_analyzer import TFIDFAnalyzer
from .data_loader import MovieDataLoader
from .embeddings_analyzer import EmbeddingsAnalyzer

__all__ = ['TextPreprocessor', 'TFIDFAnalyzer', 'MovieDataLoader', 'EmbeddingsAnalyzer']
