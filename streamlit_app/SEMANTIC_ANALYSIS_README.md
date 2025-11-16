# Semantic Analysis Page - Streamlit App

## Overview

The Semantic Analysis page provides an interactive visualization of textual patterns in movie data and their relationship to ROI (Return on Investment).

## Features

### 🔤 **Top Terms**
- Displays the most important terms across all movies using TF-IDF scoring
- Interactive slider to adjust the number of terms shown (10-50)
- Visual bar chart and data table
- Identifies distinctive vocabulary in the movie corpus

### 🔗 **ROI Correlation**
- Shows terms most strongly correlated with movie ROI
- Supports both Pearson and Spearman correlation methods
- Split visualization for positive and negative correlations
- Helps identify themes associated with commercial success/failure

### 📊 **ROI Segments**
- Compares vocabulary across different ROI quartiles
- Visual comparison of high-ROI vs low-ROI movie terminology
- Adjustable number of segments (2-5)
- Reveals distinct language patterns in successful vs unsuccessful films

### ☁️ **Word Clouds**
- Visual representation of term frequency
- Separate word clouds for high-ROI and low-ROI movies
- Adjustable ROI threshold
- Color-coded (green for high ROI, red for low ROI)

### 📈 **Statistics**
- TF-IDF matrix statistics (shape, sparsity)
- Document length and word count statistics
- ROI distribution histogram
- Sample document previews (highest and lowest ROI movies)

## How to Use

1. **Navigate** to the "📝 Semantic Analysis" page from the sidebar
2. **Wait** for data loading and preprocessing (cached after first run)
3. **Explore** the different tabs for various analyses
4. **Adjust** sliders and options to customize visualizations
5. **Interpret** results to understand textual patterns in your dataset

## Technical Details

### Data Processing
- Combines title, overview, tagline, genres, and keywords into unified documents
- Applies text preprocessing (lowercase, punctuation removal, etc.)
- Weights title 2x for increased importance
- Filters movies with budget < $100,000

### TF-IDF Configuration
- Max features: 500 terms
- Min document frequency: 5 movies
- Max document frequency: 80% of movies
- N-gram range: unigrams and bigrams (1-2)
- Sublinear TF scaling enabled

### Caching
- Data loading is cached for 1 hour (`@st.cache_data`)
- TF-IDF computation is cached (`@st.cache_resource`)
- Improves performance on subsequent page visits

## Dependencies

Required packages (already in `requirements.txt`):
```
scikit-learn>=1.3.0
wordcloud>=1.9.0
matplotlib>=3.7.0
scipy>=1.11.0
```

## Files Modified/Created

1. **`streamlit_app/utils/semantic_analysis.py`** (new)
   - Data loading and preprocessing functions
   - TF-IDF computation
   - Visualization helper functions

2. **`streamlit_app/app.py`** (modified)
   - Added "📝 Semantic Analysis" to navigation
   - Added `show_semantic_analysis_page()` function
   - Route handling for new page

3. **`streamlit_app/requirements.txt`** (modified)
   - Added `wordcloud>=1.9.0`
   - Added `python-dotenv>=1.0.0`

## Example Insights

The semantic analysis can reveal:

- **Genre-specific vocabulary**: Certain terms appear more in specific genres
- **Success indicators**: Words/phrases associated with high ROI
- **Failure patterns**: Terms that correlate with poor performance
- **Temporal trends**: Different eras may have different characteristic vocabulary
- **Marketing language**: Effective tagline and keyword patterns

## Troubleshooting

### "No data available for semantic analysis"
- Check database connection
- Ensure movies have text fields (overview, tagline, etc.)
- Verify budget threshold ($100,000 minimum)

### Slow performance
- First load takes longer (processing ~2000+ movies)
- Subsequent loads are cached and much faster
- Consider reducing `max_features` if needed

### Empty word clouds
- Adjust ROI threshold to ensure both segments have movies
- Check that documents contain sufficient text

## Future Enhancements

Potential additions:
- Sentiment analysis of overviews and taglines
- Topic modeling (LDA) visualization
- Time-series analysis of vocabulary trends
- Genre-specific semantic analysis
- Export TF-IDF features for ML models
- Custom text field selection
