"""
Semantic analysis utilities for Streamlit app.
Handles text analysis, TF-IDF, embeddings, and visualization.
"""
import sys
import os

# Add parent directory (project root) to path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from src.analysis import TextPreprocessor, TFIDFAnalyzer, MovieDataLoader, EmbeddingsAnalyzer
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt


@st.cache_data(ttl=3600)
def load_semantic_data():
    """Load movie data with text fields for semantic analysis."""
    try:
        # Use database connection from environment or default
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:25432/movie_database')
        loader = MovieDataLoader(db_url)
        
        df = loader.load_movies_with_text(
            min_budget=100000,
            include_genres=True,
            include_keywords=True
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def prepare_documents(_df):
    """Prepare text documents for analysis."""
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,
        min_word_length=2
    )
    
    df_docs = preprocessor.prepare_movie_documents(
        _df,
        include_title=True,
        include_overview=True,
        include_tagline=True,
        include_genres=True,
        include_keywords=True,
        title_weight=2,
        tagline_weight=1
    )
    
    return df_docs


@st.cache_resource
def compute_tfidf(_documents):
    """Compute TF-IDF matrix (cached)."""
    tfidf = TFIDFAnalyzer(
        max_features=500,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        use_idf=True,
        sublinear_tf=True
    )
    
    tfidf_matrix = tfidf.fit_transform(_documents)
    return tfidf, tfidf_matrix


def create_wordcloud(text_data, title="Word Cloud", colormap='viridis'):
    """Create a word cloud visualization."""
    # Combine all text
    text = ' '.join(text_data.dropna().astype(str))
    
    if not text.strip():
        return None
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        relative_scaling=0.5,
        min_font_size=10
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig


def plot_term_correlations(correlations_df, title="Term Correlations with ROI", top_n=15):
    """Plot term correlations with ROI."""
    # Separate positive and negative
    pos_corr = correlations_df[correlations_df['correlation'] > 0].head(top_n)
    neg_corr = correlations_df[correlations_df['correlation'] < 0].head(top_n)
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Positive Correlations', 'Negative Correlations'),
        horizontal_spacing=0.15
    )
    
    # Positive correlations
    if len(pos_corr) > 0:
        fig.add_trace(
            go.Bar(
                y=pos_corr['term'],
                x=pos_corr['correlation'],
                orientation='h',
                marker_color='green',
                name='Positive',
                hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Negative correlations
    if len(neg_corr) > 0:
        fig.add_trace(
            go.Bar(
                y=neg_corr['term'],
                x=neg_corr['correlation'],
                orientation='h',
                marker_color='red',
                name='Negative',
                hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text=title,
        showlegend=False,
        height=500,
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="Correlation", row=1, col=1)
    fig.update_xaxes(title_text="Correlation", row=1, col=2)
    fig.update_yaxes(title_text="Term", row=1, col=1)
    fig.update_yaxes(title_text="Term", row=1, col=2)
    
    return fig


def plot_top_terms(terms_df, title="Top TF-IDF Terms", top_n=20):
    """Plot top TF-IDF terms."""
    top_terms = terms_df.head(top_n).sort_values('mean_tfidf')
    
    fig = go.Figure(go.Bar(
        y=top_terms['term'],
        x=top_terms['mean_tfidf'],
        orientation='h',
        marker_color='steelblue',
        hovertemplate='<b>%{y}</b><br>TF-IDF Score: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Mean TF-IDF Score",
        yaxis_title="Term",
        height=600,
        hovermode='closest'
    )
    
    return fig


def plot_segment_comparison(segments_dict, segment_names=['Q1', 'Q4'], top_n=15):
    """Compare terms between two ROI segments."""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Low ROI ({segment_names[0]})', f'High ROI ({segment_names[1]})'),
        horizontal_spacing=0.15
    )
    
    # Low ROI segment
    if segment_names[0] in segments_dict:
        low_roi = segments_dict[segment_names[0]].head(top_n).sort_values('mean_tfidf')
        fig.add_trace(
            go.Bar(
                y=low_roi['term'],
                x=low_roi['mean_tfidf'],
                orientation='h',
                marker_color='red',
                name='Low ROI',
                hovertemplate='<b>%{y}</b><br>TF-IDF: %{x:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # High ROI segment
    if segment_names[1] in segments_dict:
        high_roi = segments_dict[segment_names[1]].head(top_n).sort_values('mean_tfidf')
        fig.add_trace(
            go.Bar(
                y=high_roi['term'],
                x=high_roi['mean_tfidf'],
                orientation='h',
                marker_color='green',
                name='High ROI',
                hovertemplate='<b>%{y}</b><br>TF-IDF: %{x:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text="Vocabulary Comparison: Low ROI vs High ROI",
        showlegend=False,
        height=600,
        hovermode='closest'
    )
    
    fig.update_xaxes(title_text="TF-IDF Score", row=1, col=1)
    fig.update_xaxes(title_text="TF-IDF Score", row=1, col=2)
    
    return fig


# ============================================================================
# EMBEDDINGS ANALYSIS FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embeddings_model(model_name='all-MiniLM-L6-v2'):
    """Load and cache the sentence embeddings model."""
    try:
        # Force CPU to avoid CUDA compatibility issues
        analyzer = EmbeddingsAnalyzer(model_name=model_name, device='cpu')
        analyzer.load_model()
        return analyzer
    except Exception as e:
        st.error(f"Error loading embeddings model: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def compute_embeddings(_analyzer, documents, batch_size=32):
    """Compute sentence embeddings for documents."""
    embeddings = _analyzer.encode(
        documents.tolist(),
        batch_size=batch_size,
        show_progress=False
    )
    return embeddings


def plot_embeddings_2d(embeddings, cluster_labels, roi_values, df_subset=None):
    """Plot 2D embeddings with clusters and ROI."""
    from sklearn.decomposition import PCA
    
    # Reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': embeddings_2d[:, 0],
        'PC2': embeddings_2d[:, 1],
        'Cluster': cluster_labels,
        'ROI': roi_values
    })
    
    if df_subset is not None:
        plot_df['Title'] = df_subset['title'].values
        plot_df['Budget'] = df_subset['budget'].values
        plot_df['Revenue'] = df_subset['revenue'].values
    
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Colored by Cluster', 'Colored by ROI'),
        horizontal_spacing=0.12
    )
    
    # Plot 1: Clusters
    hover_data = ['ROI']
    has_movie_data = df_subset is not None and not df_subset.empty
    if has_movie_data:
        hover_data.extend(['Title', 'Budget', 'Revenue'])
    
    for cluster_id in np.unique(cluster_labels):
        mask = plot_df['Cluster'] == cluster_id
        cluster_data = plot_df[mask]
        
        fig.add_trace(
            go.Scatter(
                x=cluster_data['PC1'],
                y=cluster_data['PC2'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=6, opacity=0.6),
                customdata=cluster_data[hover_data].values if has_movie_data else None,
                hovertemplate='<b>Cluster %{fullData.name}</b><br>' +
                             'PC1: %{x:.2f}<br>' +
                             'PC2: %{y:.2f}<br>' +
                             'ROI: %{customdata[0]:.2f}<br>' +
                             ('<b>%{customdata[1]}</b><br>Budget: $%{customdata[2]:,.0f}<br>Revenue: $%{customdata[3]:,.0f}' if has_movie_data else '') +
                             '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Plot 2: ROI
    fig.add_trace(
        go.Scatter(
            x=plot_df['PC1'],
            y=plot_df['PC2'],
            mode='markers',
            marker=dict(
                size=6,
                color=plot_df['ROI'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='ROI', x=1.15),
                opacity=0.7
            ),
            customdata=plot_df[hover_data].values if has_movie_data else None,
            hovertemplate='<b>Movie</b><br>' +
                         'PC1: %{x:.2f}<br>' +
                         'PC2: %{y:.2f}<br>' +
                         'ROI: %{marker.color:.2f}<br>' +
                         ('<b>%{customdata[1]}</b><br>Budget: $%{customdata[2]:,.0f}<br>Revenue: $%{customdata[3]:,.0f}' if has_movie_data else '') +
                         '<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text='PC 1', row=1, col=1)
    fig.update_yaxes(title_text='PC 2', row=1, col=1)
    fig.update_xaxes(title_text='PC 1', row=1, col=2)
    fig.update_yaxes(title_text='PC 2', row=1, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(x=0.45, y=1.0, orientation='h'),
        title_text=f'Semantic Embeddings Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})'
    )
    
    return fig


def plot_cluster_roi_stats(cluster_stats):
    """Plot ROI statistics by cluster."""
    fig = go.Figure()
    
    # Add bars for mean ROI
    fig.add_trace(go.Bar(
        x=[f"C{i}" for i in cluster_stats.index],
        y=cluster_stats['roi_mean'],
        error_y=dict(type='data', array=cluster_stats['roi_std']),
        marker_color='steelblue',
        hovertemplate='<b>Cluster %{x}</b><br>' +
                     'Mean ROI: %{y:.2f}<br>' +
                     'Median: %{customdata[0]:.2f}<br>' +
                     'N Movies: %{customdata[1]}<br>' +
                     '<extra></extra>',
        customdata=cluster_stats[['roi_median', 'n_movies']].values
    ))
    
    # Add overall mean line
    overall_mean = cluster_stats['roi_mean'].mean()
    fig.add_hline(
        y=overall_mean,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Overall Mean: {overall_mean:.2f}',
        annotation_position='top right'
    )
    
    fig.update_layout(
        title='Average ROI by Semantic Cluster',
        xaxis_title='Cluster',
        yaxis_title='Mean ROI',
        height=500,
        hovermode='closest'
    )
    
    return fig


def display_cluster_representatives(representatives, cluster_stats, top_n=5):
    """Display representative movies for top clusters."""
    st.markdown("### 🎯 Representative Movies by Cluster")
    st.markdown("Showing movies closest to each cluster's semantic centroid.")
    
    top_clusters = cluster_stats.head(top_n).index
    
    for cluster_id in top_clusters:
        cluster_roi = cluster_stats.loc[cluster_id, 'roi_mean']
        n_movies = cluster_stats.loc[cluster_id, 'n_movies']
        
        with st.expander(f"**Cluster {cluster_id}** - Mean ROI: {cluster_roi:.2f} ({n_movies} movies)"):
            rep_movies = representatives[cluster_id]
            
            for idx, row in rep_movies.iterrows():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**🎬 {row['title']}**")
                    if pd.notna(row.get('genres')):
                        st.markdown(f"*{row['genres']}*")
                    if pd.notna(row.get('overview')):
                        overview = row['overview'][:200] + '...' if len(row['overview']) > 200 else row['overview']
                        st.markdown(f"> {overview}")
                
                with col2:
                    st.metric("ROI", f"{row['roi']:.2f}")
                    if pd.notna(row.get('budget')) and row['budget'] > 0:
                        st.caption(f"Budget: ${row['budget']:,.0f}")
                
                st.divider()
