# Sentence Embeddings Analysis for Movie ROI

This module implements **transformer-based sentence embeddings** (BERT) to analyze movie text and predict ROI. Unlike TF-IDF which focuses on word frequency, sentence embeddings capture semantic meaning and context.

## 🎯 What Are Sentence Embeddings?

Sentence embeddings convert text into dense numerical vectors (e.g., 384 or 768 dimensions) that capture semantic meaning. Movies with similar themes, narratives, or emotional tones will have similar embeddings, even if they use different words.

### TF-IDF vs Sentence Embeddings

| Aspect | TF-IDF | Sentence Embeddings |
|--------|--------|---------------------|
| **Representation** | Sparse vectors (thousands of dimensions) | Dense vectors (384-768 dimensions) |
| **Focus** | Word frequency | Semantic meaning |
| **Context** | No word order | Full context understanding |
| **Similarity** | Shared words | Shared meaning |
| **Example** | "happy ending" ≠ "joyful conclusion" | "happy ending" ≈ "joyful conclusion" |

## 📦 Installation

```bash
pip install sentence-transformers
```

This will install PyTorch and the transformers library.

## 🚀 Quick Start

### 1. Using the Notebook

Open `notebooks/embeddings_analysis.ipynb` for a complete walkthrough:

```python
from src.analysis import EmbeddingsAnalyzer, TextPreprocessor, MovieDataLoader

# Load data
loader = MovieDataLoader(engine)
df = loader.load_movies_with_text(min_budget=100000)

# Prepare documents
preprocessor = TextPreprocessor()
documents = preprocessor.prepare_movie_documents(df)

# Generate embeddings
analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2')
embeddings = analyzer.encode(documents)

# Cluster by semantic similarity
cluster_labels = analyzer.cluster_movies(n_clusters=10)
cluster_stats = analyzer.analyze_roi_by_clusters(df, cluster_labels)

print(cluster_stats)
```

### 2. Using the Streamlit App

```bash
cd streamlit_app
streamlit run app.py
```

Navigate to **📝 Semantic Analysis** → **🧠 Sentence Embeddings** tab.

## 🧠 Available Models

The `EmbeddingsAnalyzer` supports multiple pre-trained models:

### Recommended Models

1. **all-MiniLM-L6-v2** (Default)
   - Dimensions: 384
   - Speed: ⚡⚡⚡ Fast
   - Quality: ⭐⭐⭐ Good
   - Best for: Quick analysis, large datasets
   - Use case: ~10,000 movies in a few minutes

2. **all-mpnet-base-v2**
   - Dimensions: 768
   - Speed: ⚡⚡ Moderate
   - Quality: ⭐⭐⭐⭐ Best
   - Best for: Highest quality embeddings
   - Use case: Production models, when accuracy matters most

3. **paraphrase-multilingual-MiniLM-L12-v2**
   - Dimensions: 384
   - Speed: ⚡⚡ Moderate
   - Quality: ⭐⭐⭐ Good
   - Best for: Multilingual movie datasets
   - Use case: International films

### Changing Models

```python
# Fast and efficient
analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2')

# Best quality
analyzer = EmbeddingsAnalyzer(model_name='all-mpnet-base-v2')

# Multilingual
analyzer = EmbeddingsAnalyzer(model_name='paraphrase-multilingual-MiniLM-L12-v2')
```

## 📊 Analysis Methods

### 1. Semantic Clustering

Group movies by narrative themes and analyze ROI patterns:

```python
# Cluster movies
cluster_labels = analyzer.cluster_movies(n_clusters=15)

# Analyze ROI by cluster
cluster_stats = analyzer.analyze_roi_by_clusters(df, cluster_labels)
print(cluster_stats.head())
```

**Output:**
```
        n_movies  roi_mean  roi_median  roi_std  roi_min  roi_max
cluster                                                           
5             42      3.45        2.80     2.10    -0.50    12.30
2             68      2.90        2.40     1.85    -0.30    10.20
...
```

### 2. Find Similar Movies

Discover movies with similar semantic content:

```python
# Find similar movies
similar_indices, similarities = analyzer.find_similar_movies(
    query_idx=0,  # Index of query movie
    top_k=10
)

for idx, sim in zip(similar_indices, similarities):
    print(f"{df.iloc[idx]['title']}: {sim:.3f}")
```

**Example Output:**
```
The Dark Knight: 0.892
Batman Begins: 0.856
The Dark Knight Rises: 0.824
```

### 3. Dimension Correlation Analysis

Identify which semantic dimensions predict ROI:

```python
dim_corr = analyzer.correlate_embeddings_with_roi(
    df['roi'],
    method='spearman',
    top_dims=20
)

print(dim_corr.head())
```

**Interpretation:**
- **Positive correlation**: Higher values in this dimension → Higher ROI
- **Negative correlation**: Higher values in this dimension → Lower ROI
- **P-value < 0.05**: Statistically significant relationship

### 4. Visualization in 2D

Reduce embeddings to 2D for visualization:

```python
# Reduce to 2D
embeddings_2d = analyzer.reduce_dimensions(n_components=2)

# Plot
fig = analyzer.plot_clusters_2d(
    embeddings_2d,
    cluster_labels,
    roi_values=df['roi'].values
)
```

### 5. Representative Movies

Get movies closest to each cluster's centroid:

```python
representatives = analyzer.get_cluster_representative_movies(
    df, cluster_labels, embeddings, n_per_cluster=3
)

# Display representatives for cluster 0
print(representatives[0][['title', 'roi', 'overview']])
```

## 💡 How Embeddings Relate to ROI

### 1. Semantic Similarity to Successful Movies

Movies semantically similar to high-ROI movies tend to have higher ROI:

```python
# Find high-ROI movies
high_roi_movies = df[df['roi'] > df['roi'].quantile(0.75)]

# Compute average embedding
success_profile = embeddings[high_roi_movies.index].mean(axis=0)

# Score all movies by similarity to success profile
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(embeddings, success_profile.reshape(1, -1))
```

### 2. Narrative Patterns

Certain narrative structures correlate with financial success:

- **"Underdog story"** theme → Positive ROI correlation
- **"Revenge narrative"** → Variable, depends on execution
- **"Family-friendly adventure"** → Consistently high ROI

Embeddings capture these patterns even when different words are used.

### 3. Emotional Tone

The emotional context embedded in text correlates with audience engagement:

```python
# Cluster analysis reveals:
# - Cluster 3: Inspirational/hopeful → High ROI (mean: 3.2)
# - Cluster 7: Dark/gritty → Variable ROI (mean: 1.8)
# - Cluster 12: Lighthearted/comedic → High ROI (mean: 2.9)
```

### 4. Genre-Crossing Themes

Embeddings find successful patterns across genres:

- An action movie and a drama might cluster together if they share an "underdog" theme
- This reveals ROI-predictive patterns that genre labels miss

### 5. Use as ML Features

Embeddings can be used directly in prediction models:

```python
# Export embeddings
embedding_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(384)])
embedding_df['roi'] = df['roi']

# Train model
from sklearn.ensemble import RandomForestRegressor
X = embedding_df[[col for col in embedding_df.columns if col.startswith('emb_')]]
y = embedding_df['roi']

model = RandomForestRegressor()
model.fit(X, y)
```

## 🎬 Real-World Examples

### Example 1: Finding the "Marvel Formula"

```python
# Get Marvel movies
marvel_movies = df[df['overview'].str.contains('superhero|Marvel', na=False, case=False)]
marvel_embeddings = embeddings[marvel_movies.index]

# Find average Marvel embedding
marvel_profile = marvel_embeddings.mean(axis=0)

# Find non-Marvel movies similar to Marvel
similarity = cosine_similarity(embeddings, marvel_profile.reshape(1, -1))
similar_to_marvel = df.iloc[similarity.argsort(axis=0)[-20:][::-1]]

print("Movies semantically similar to Marvel (but not Marvel):")
print(similar_to_marvel[['title', 'roi', 'genres']])
```

### Example 2: "What Makes a High-ROI Drama?"

```python
# Get dramas
dramas = df[df['genres'].str.contains('Drama', na=False)]
drama_embeddings = embeddings[dramas.index]
drama_roi = df.loc[dramas.index, 'roi']

# Cluster dramas
drama_clusters = analyzer.cluster_movies(n_clusters=8, embeddings=drama_embeddings)

# Analyze
cluster_stats = analyzer.analyze_roi_by_clusters(dramas, drama_clusters)
print("Drama clusters by ROI:")
print(cluster_stats)

# Get representatives from highest ROI cluster
best_cluster = cluster_stats['roi_mean'].idxmax()
representatives = analyzer.get_cluster_representative_movies(
    dramas, drama_clusters, drama_embeddings, n_per_cluster=5
)
print(f"\nRepresentative high-ROI dramas (cluster {best_cluster}):")
print(representatives[best_cluster][['title', 'roi', 'overview']])
```

## 📈 Performance Tips

### For Large Datasets (>5,000 movies)

1. **Use batch processing**:
   ```python
   embeddings = analyzer.encode(documents, batch_size=64)
   ```

2. **Enable GPU** (if available):
   ```python
   analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2', device='cuda')
   ```

3. **Use smaller model**:
   ```python
   analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2')  # 384 dim
   # vs
   analyzer = EmbeddingsAnalyzer(model_name='all-mpnet-base-v2')  # 768 dim
   ```

### Memory Optimization

For 10,000 movies with 384 dimensions:
- Memory: ~30 MB for embeddings
- Generation time: ~2-5 minutes (CPU)
- Generation time: ~30 seconds (GPU)

## 🔬 Advanced Usage

### Custom Similarity Search

```python
def find_movies_like(query_text, top_k=10):
    """Find movies similar to a custom query."""
    # Encode query
    query_embedding = analyzer.model.encode([query_text])[0]
    
    # Compute similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings
    )[0]
    
    # Get top matches
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    return results[['title', 'roi', 'genres', 'similarity']]

# Example
results = find_movies_like("A dark hero fighting crime in a gritty city")
print(results)
```

### Semantic ROI Prediction

```python
def predict_roi_by_similarity(movie_idx, n_similar=20):
    """Predict ROI based on similar movies."""
    similar_indices, similarities = analyzer.find_similar_movies(
        movie_idx, top_k=n_similar
    )
    
    # Weighted average of similar movies' ROI
    similar_roi = df.iloc[similar_indices]['roi'].values
    predicted_roi = np.average(similar_roi, weights=similarities)
    
    return predicted_roi

# Test
actual_roi = df.iloc[0]['roi']
predicted_roi = predict_roi_by_similarity(0)
print(f"Actual: {actual_roi:.2f}, Predicted: {predicted_roi:.2f}")
```

## 🆚 Comparison with Other Methods

| Method | Captures Meaning | Interpretability | Speed | Best For |
|--------|------------------|------------------|-------|----------|
| **Keywords** | ❌ No | ✅✅✅ High | ⚡⚡⚡ | Exact matches |
| **TF-IDF** | ⚠️ Partial | ✅✅ Medium | ⚡⚡⚡ | Distinctive words |
| **Topic Modeling** | ⚠️ Partial | ✅✅ Medium | ⚡⚡ | Theme discovery |
| **Sentence Embeddings** | ✅✅✅ Full | ⚠️ Low | ⚡ | Semantic similarity |

**Recommendation**: Use multiple methods together:
1. **Keywords**: Quick genre/theme filters
2. **TF-IDF**: Find distinctive vocabulary
3. **Embeddings**: Capture semantic patterns for prediction

## 📚 Further Reading

- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)
- [sentence-transformers Documentation](https://www.sbert.net/)
- [Pre-trained Models](https://www.sbert.net/docs/pretrained_models.html)

## 🐛 Troubleshooting

### "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### CUDA Compatibility Error
```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**Cause**: PyTorch was installed with CUDA 12 support, but your GPU only supports CUDA 11.

**Solution**: Force CPU usage:
```python
# Option 1: Explicitly use CPU
analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2', device='cpu')

# Option 2: Install PyTorch for your CUDA version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then reinstall sentence-transformers
pip install --upgrade --force-reinstall sentence-transformers
```

**Note**: CPU mode is slower but works on all systems. For 1000 movies, expect ~1-2 minutes on CPU vs ~10 seconds on GPU.

### Out of Memory Error
```python
# Use smaller batch size
embeddings = analyzer.encode(documents, batch_size=16)

# Or use smaller model
analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2')
```

### Slow Performance
```python
# Enable GPU (if available)
analyzer = EmbeddingsAnalyzer(device='cuda')

# Or sample your data
df_sample = df.sample(n=1000, random_state=42)
```

## 🎯 Next Steps

1. ✅ Generate embeddings for your movie dataset
2. ✅ Cluster movies and analyze ROI patterns
3. ✅ Use embeddings as features in ML models
4. ✅ Combine with budget, cast, and temporal features
5. ✅ Build similarity-based recommendation system
6. ✅ Experiment with different embedding models

---

**Questions?** Check the notebook `notebooks/embeddings_analysis.ipynb` for detailed examples!
