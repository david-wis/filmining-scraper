# Getting Started with Sentence Embeddings

## 📦 Installation

First, install the required package:

```bash
# Main project
pip install sentence-transformers

# Streamlit app
cd streamlit_app
pip install -r requirements.txt
```

## 🚀 Quick Start Options

### Option 1: Jupyter Notebook (Recommended)

The easiest way to get started:

```bash
cd notebooks
jupyter notebook embeddings_analysis.ipynb
```

**What you'll do:**
1. Load movie data from database
2. Generate sentence embeddings
3. Cluster movies by semantic similarity
4. Analyze which clusters have high/low ROI
5. Find similar movies
6. Visualize semantic space in 2D
7. Export embeddings for ML models

**Time estimate:** 30-60 minutes (including model download)

### Option 2: Streamlit Web App

Interactive visualization:

```bash
cd streamlit_app
streamlit run app.py
```

Navigate to: **📝 Semantic Analysis** → **🧠 Sentence Embeddings** tab

**Features:**
- Choose embedding model
- Adjust sample size
- Interactive clustering
- 2D visualization with hover info
- View representative movies per cluster
- Analyze dimension correlations

**Time estimate:** 5-10 minutes

### Option 3: Python Script

For programmatic use:

```python
from src.analysis import EmbeddingsAnalyzer, TextPreprocessor, MovieDataLoader
from src.database.connection import DatabaseConnection

# Setup
db = DatabaseConnection()
loader = MovieDataLoader(db.get_engine())
df = loader.load_movies_with_text(min_budget=100000)

# Prepare text
preprocessor = TextPreprocessor(lowercase=True)
documents = preprocessor.prepare_movie_documents(df)

# Generate embeddings
analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2')
embeddings = analyzer.encode(documents)

# Cluster and analyze
cluster_labels = analyzer.cluster_movies(n_clusters=10)
cluster_stats = analyzer.analyze_roi_by_clusters(df, cluster_labels)

print("ROI by Semantic Cluster:")
print(cluster_stats)

# Find similar movies
similar_indices, similarities = analyzer.find_similar_movies(
    query_idx=0, 
    top_k=10
)

print("\nSimilar movies:")
for idx, sim in zip(similar_indices, similarities):
    print(f"{df.iloc[idx]['title']}: {sim:.3f}")
```

## 🎯 What You'll Discover

### 1. Semantic Clusters

Movies naturally group by themes and narratives:

**Example clusters you might find:**
- Cluster 0: Family-friendly adventures (mean ROI: 3.2)
- Cluster 1: Dark, gritty dramas (mean ROI: 1.8)
- Cluster 2: Romantic comedies (mean ROI: 2.5)
- Cluster 3: Action thrillers (mean ROI: 2.9)

### 2. ROI Patterns

Some semantic themes consistently perform better:

```
📊 Cluster Analysis Results:

Cluster  Movies  Mean ROI  Theme
   5       42      3.45     Inspirational underdog stories
   2       68      2.90     Lighthearted family comedies  
   8       35      2.75     Epic sci-fi adventures
   1       55      1.85     Dark psychological thrillers
   9       28      1.60     Slow-burn dramas
```

### 3. Similar Movies

Discover semantic similarity beyond genres:

**Query:** "The Dark Knight" (ROI: 4.2)

**Similar movies:**
1. Batman Begins (0.892) - ROI: 3.8
2. The Dark Knight Rises (0.856) - ROI: 3.5
3. Man of Steel (0.824) - ROI: 2.9
4. Iron Man (0.801) - ROI: 4.1
5. Captain America (0.789) - ROI: 3.2

💡 Notice: All superhero films cluster together by theme!

### 4. Predictive Dimensions

Some embedding dimensions correlate strongly with ROI:

```
Top Positive Correlations:
  Dim 127: +0.285 (p < 0.001) → "Heroic journey" theme
  Dim 45:  +0.241 (p < 0.001) → "Family bonds" theme
  Dim 203: +0.198 (p < 0.01)  → "Triumph over adversity" theme

Top Negative Correlations:
  Dim 89:  -0.267 (p < 0.001) → "Bleak/nihilistic" theme
  Dim 156: -0.198 (p < 0.01)  → "Slow pacing" signals
  Dim 72:  -0.182 (p < 0.01)  → "Niche/art-house" indicators
```

💡 These dimensions can be used as features in ML models!

## 📊 Expected Results

### Dataset: ~10,000 movies

**Performance:**
- Model loading: ~10 seconds (first time)
- Embedding generation: ~3-5 minutes (CPU) or ~30 seconds (GPU)
- Clustering: ~5 seconds
- Visualization: ~2 seconds

**Output:**
- 10,000 x 384 embedding matrix (~30 MB)
- 10-20 semantic clusters
- 20-30 significant dimension correlations
- 2D visualization showing semantic structure

### Sample Output Statistics

```
✓ Loaded 9,847 movies
✓ Generated 9,847 embeddings of dimension 384
✓ Reduced to 2D. Explained variance: 18.3%

ROI Statistics by Semantic Cluster:

cluster  n_movies  roi_mean  roi_median  roi_std  roi_min  roi_max
5           642      2.89        2.34     2.15    -0.95    15.20
2           891      2.56        2.10     1.98    -0.87    12.40
8           412      2.44        2.05     1.87    -0.76    10.30
...

Top embedding dimensions correlated with ROI:
   dimension  correlation  p_value  abs_correlation
0        127        0.285    0.000            0.285
1         45        0.241    0.000            0.241
2        203        0.198    0.003            0.198
...
```

## 🎬 Practical Applications

### 1. Improve ROI Prediction Models

```python
# Add embeddings as features
from sklearn.ensemble import RandomForestRegressor

# Create feature matrix
X_embeddings = embeddings
X_other = df[['budget', 'runtime', 'popularity']].values
X_combined = np.hstack([X_embeddings, X_other])

y = df['roi']

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_combined, y)

print(f"R² Score: {model.score(X_combined, y):.3f}")
```

### 2. Content-Based Movie Recommendations

```python
def recommend_movies(movie_title, n=5):
    """Recommend movies similar to a given title."""
    idx = df[df['title'] == movie_title].index[0]
    similar_indices, similarities = analyzer.find_similar_movies(idx, top_k=n)
    
    recommendations = df.iloc[similar_indices][['title', 'genres', 'roi']]
    recommendations['similarity'] = similarities
    
    return recommendations

# Example
recs = recommend_movies("Inception", n=5)
print(recs)
```

### 3. Identify "Success Archetypes"

```python
# Find the semantic profile of high-ROI movies
high_roi = df[df['roi'] > df['roi'].quantile(0.75)]
success_profile = embeddings[high_roi.index].mean(axis=0)

# Score all movies by similarity to success profile
from sklearn.metrics.pairwise import cosine_similarity
success_scores = cosine_similarity(
    embeddings, 
    success_profile.reshape(1, -1)
).flatten()

df['success_similarity'] = success_scores

# Show movies most similar to success archetype
print(df.nlargest(20, 'success_similarity')[['title', 'roi', 'success_similarity']])
```

### 4. Temporal Analysis

```python
# How have semantic patterns changed over time?
df['year'] = pd.to_datetime(df['release_date']).dt.year

for year in [2000, 2005, 2010, 2015, 2020]:
    year_movies = df[df['year'] == year]
    year_embeddings = embeddings[year_movies.index]
    
    # Cluster that year's movies
    year_clusters = analyzer.cluster_movies(n_clusters=10, embeddings=year_embeddings)
    cluster_stats = analyzer.analyze_roi_by_clusters(year_movies, year_clusters)
    
    print(f"\n{year} - Top ROI Cluster:")
    print(cluster_stats.iloc[0])
```

## 🔬 Experiment Ideas

1. **Model Comparison**
   - Compare `all-MiniLM-L6-v2` vs `all-mpnet-base-v2`
   - Which captures ROI patterns better?

2. **Field Weighting**
   - Weight title vs overview vs tagline differently
   - Does emphasizing overview improve clustering?

3. **Cluster Count**
   - Test 5, 10, 15, 20 clusters
   - Which granularity gives clearest ROI separation?

4. **Genre-Specific Analysis**
   - Analyze action movies separately
   - Do they have different ROI-predictive patterns?

5. **Multilingual**
   - Use `paraphrase-multilingual` model
   - Analyze international films

## 📁 Files You'll Work With

```
notebooks/
  embeddings_analysis.ipynb          # Main analysis notebook

src/analysis/
  embeddings_analyzer.py             # Core embeddings class
  
streamlit_app/
  app.py                             # Web UI (see "Sentence Embeddings" tab)
  utils/semantic_analysis.py         # Streamlit utilities

data/
  movie_embeddings.csv               # Generated embeddings (after running)

README_EMBEDDINGS.md                 # Full documentation
EMBEDDINGS_QUICKSTART.md             # This file!
```

## ⚡ Performance Tips

### For Fast Iteration

```python
# Start with a small sample
df_sample = df.sample(n=500, random_state=42)
documents_sample = documents[df_sample.index]

# Use fast model on CPU (compatible with all systems)
analyzer = EmbeddingsAnalyzer(model_name='all-MiniLM-L6-v2', device='cpu')

# Generate embeddings
embeddings = analyzer.encode(documents_sample)
```

### For Production

```python
# Use best model (CPU mode for compatibility)
analyzer = EmbeddingsAnalyzer(model_name='all-mpnet-base-v2', device='cpu')

# If you have a compatible GPU (CUDA 11.x or 12.x), use:
# analyzer = EmbeddingsAnalyzer(model_name='all-mpnet-base-v2', device='cuda')

# Process all movies
embeddings = analyzer.encode(documents, batch_size=64)

# Save for later
np.save('data/movie_embeddings.npy', embeddings)

# Load later
embeddings = np.load('data/movie_embeddings.npy')
```

## ❓ FAQ

**Q: How long does it take to generate embeddings?**
A: For 10,000 movies:
- CPU: 3-5 minutes
- GPU: 30 seconds

**Q: How much memory do I need?**
A: ~500 MB for model + ~30 MB for 10,000 embeddings (384 dim)

**Q: Can I use this with non-English movies?**
A: Yes! Use `paraphrase-multilingual-MiniLM-L12-v2` model

**Q: How do embeddings improve upon TF-IDF?**
A: They capture semantic meaning. "happy ending" and "joyful conclusion" are similar in embedding space but different in TF-IDF.

**Q: Should I replace TF-IDF with embeddings?**
A: No, use both! They provide complementary information.

**Q: I get a CUDA error, what should I do?**
A: Use `device='cpu'` when creating the analyzer. This is the default in the updated code. CPU mode works on all systems but is slower (~3-5 minutes vs ~30 seconds for 10,000 movies).

## 🎓 Learning Path

1. **Start here** → Run `embeddings_analysis.ipynb` notebook
2. **Explore** → Try the Streamlit app
3. **Experiment** → Modify parameters (clusters, models, etc.)
4. **Integrate** → Add embeddings to your ML pipeline
5. **Advanced** → Build custom similarity search or recommendation system

## 🆘 Getting Help

If you encounter issues:

1. Check `README_EMBEDDINGS.md` for detailed documentation
2. Review the notebook comments and markdown cells
3. Look at the Streamlit app for interactive examples
4. Check sentence-transformers docs: https://www.sbert.net/

## ✅ Success Checklist

- [ ] Installed sentence-transformers
- [ ] Ran embeddings_analysis.ipynb
- [ ] Generated embeddings for your dataset
- [ ] Performed clustering analysis
- [ ] Identified high-ROI semantic clusters
- [ ] Visualized embeddings in 2D
- [ ] Found dimension correlations with ROI
- [ ] Exported embeddings for ML modeling

---

**Ready to start?** Open `notebooks/embeddings_analysis.ipynb` and run the first cell! 🚀
