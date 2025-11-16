# Semantic Analysis Module

This module provides a modular and scalable framework for semantic analysis of movie text fields (title, overview, tagline, genres, keywords) and their relationship to ROI.

## 📁 Project Structure

```
src/analysis/
├── __init__.py              # Module exports
├── text_preprocessor.py     # Text cleaning and preprocessing
├── tfidf_analyzer.py        # TF-IDF analysis
├── data_loader.py           # Database loading utilities
└── (future additions)
    ├── embeddings_analyzer.py    # Sentence embeddings (BERT, etc.)
    ├── topic_analyzer.py         # Topic modeling (LDA)
    └── sentiment_analyzer.py     # Sentiment analysis
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Notebook

Open and run `notebooks/semantic_analysis.ipynb` to see a complete TF-IDF analysis.

### 3. Use in Your Own Code

```python
from src.analysis import TextPreprocessor, TFIDFAnalyzer, MovieDataLoader

# Load data
loader = MovieDataLoader('postgresql://user:pass@localhost:5432/dbname')
df = loader.load_movies_with_text(min_budget=100000)

# Preprocess text
preprocessor = TextPreprocessor()
df_docs = preprocessor.prepare_movie_documents(df)

# Run TF-IDF analysis
tfidf = TFIDFAnalyzer(max_features=500)
tfidf_matrix = tfidf.fit_transform(df_docs['document'])

# Analyze correlation with ROI
correlations = tfidf.correlate_with_target(df_docs['roi'])
```

## 🔧 Module Components

### TextPreprocessor

Handles text cleaning and preprocessing:
- Lowercase conversion
- Punctuation removal
- Stop words filtering
- Combining multiple text fields
- Field weighting (e.g., title appears 2x)

**Key methods:**
- `clean_text(text)`: Clean a single text string
- `combine_text_fields(df, fields, weights)`: Combine multiple fields
- `prepare_movie_documents(df, ...)`: Create documents for analysis

### TFIDFAnalyzer

Performs TF-IDF analysis and correlation with ROI:
- Extract TF-IDF features
- Identify important terms globally
- Correlate terms with ROI
- Segment analysis (high vs low ROI)
- Dimensionality reduction (SVD)

**Key methods:**
- `fit_transform(documents)`: Compute TF-IDF matrix
- `get_top_terms_global(top_n)`: Get globally important terms
- `correlate_with_target(target, method)`: Correlate with ROI
- `analyze_roi_segments(df, n_segments)`: Analyze by ROI quartiles
- `apply_dimensionality_reduction(n_components)`: SVD reduction
- `plot_correlation_with_roi(target)`: Visualize correlations

### MovieDataLoader

Loads movie data from PostgreSQL database:
- Filters by budget threshold
- Includes genres and keywords as aggregated strings
- Calculates ROI automatically
- Provides clean data ready for analysis

**Key methods:**
- `load_movies_with_text(min_budget, include_genres, include_keywords)`: Load movies with text fields
- `load_keywords_detailed()`: Load keywords in detailed format

## 📊 Analysis Pipeline

The recommended analysis pipeline:

1. **Data Loading**: Use `MovieDataLoader` to get clean data
2. **Preprocessing**: Use `TextPreprocessor` to create documents
3. **TF-IDF Analysis**: Use `TFIDFAnalyzer` for feature extraction
4. **Correlation Analysis**: Identify terms correlated with ROI
5. **Segmentation**: Compare high vs low ROI vocabularies
6. **Feature Engineering**: Export TF-IDF features for ML models

## 🎯 Use Cases

### 1. Identify Keywords Associated with High ROI

```python
# Get terms positively correlated with ROI
correlations = tfidf.correlate_with_target(df['roi'], method='spearman', top_n=20)
positive_terms = correlations[correlations['correlation'] > 0]
```

### 2. Compare Vocabulary: Successful vs Unsuccessful Movies

```python
# Analyze by ROI quartiles
segments = tfidf.analyze_roi_segments(df, n_segments=4, top_terms_per_segment=15)
high_roi_terms = segments['Q4']  # Top quartile
low_roi_terms = segments['Q1']   # Bottom quartile
```

### 3. Create Text Features for ML Models

```python
# Reduce to 50 dimensions with SVD
reduced_features = tfidf.apply_dimensionality_reduction(n_components=50)

# Use in ML pipeline
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(reduced_features, df['roi'])
```

## 🔮 Future Extensions

The modular design makes it easy to add new analysis methods:

### Sentence Embeddings (BERT)
```python
# Future implementation
from src.analysis import EmbeddingsAnalyzer

embedder = EmbeddingsAnalyzer(model='sentence-transformers/all-MiniLM-L6-v2')
embeddings = embedder.fit_transform(df_docs['document'])
```

### Topic Modeling (LDA)
```python
# Future implementation
from src.analysis import TopicAnalyzer

topic_model = TopicAnalyzer(n_topics=10)
topics = topic_model.fit_transform(df_docs['document'])
```

### Sentiment Analysis
```python
# Future implementation
from src.analysis import SentimentAnalyzer

sentiment = SentimentAnalyzer()
scores = sentiment.analyze(df['overview'])
```

## 📈 Results and Insights

The TF-IDF analysis reveals:

1. **Distinctive terms**: Words that characterize successful vs unsuccessful movies
2. **Genre patterns**: Certain genres have distinct vocabularies correlated with ROI
3. **Thematic elements**: Narrative themes associated with commercial success
4. **Marketing keywords**: Terms in titles/taglines that correlate with performance

### Key Findings (Example)
- Horror and thriller keywords often correlate negatively with ROI
- Family-friendly and adventure terms correlate positively
- Certain franchise-related terms show strong positive correlation
- Generic action terms show mixed correlation

## 🤝 Contributing

To add a new analysis method:

1. Create a new file in `src/analysis/` (e.g., `embeddings_analyzer.py`)
2. Implement the analyzer class with standard methods
3. Add to `__init__.py` exports
4. Create example usage in a notebook
5. Update this README

## 📚 References

- **TF-IDF**: Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval
- **Scikit-learn**: https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
- **Text Preprocessing**: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

## 📝 License

This module is part of the Filmining project for ITBA Data Science course.
