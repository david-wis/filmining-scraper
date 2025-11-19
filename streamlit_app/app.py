"""
Streamlit ROI Prediction App for Movies
Main application file for predicting movie ROI using Random Forest.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import os

# Import custom modules
from utils.database import test_database_connection
from models.data_loader import load_movies_data, load_genres_data, prepare_movies_for_modeling, get_data_summary
from models.feature_engineering import FeatureEngineer
from models.model_trainer import ROIModelTrainer


# Page configuration
st.set_page_config(
    page_title="Movie ROI Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie ROI Predictor</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Predict the Return on Investment (ROI) of movies using machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Predict ROI", "📊 Data Analysis", "📝 Semantic Analysis", "🎯 Thematic Clustering", "🤖 Model Training", "📈 Model Performance", "🔬 Sensitivity Analysis"]
    )
    
    # Database connection check
    if not test_database_connection():
        st.error("❌ Cannot connect to database. Please check your database configuration.")
        return
    
    # Load data
    with st.spinner("Loading data..."):
        df_movies = load_movies_data()
        df_genres = load_genres_data()
    
    if df_movies is None or df_movies.empty:
        st.error("❌ No data available. Please check your database.")
        return
    
    # Prepare data for modeling
    df_clean = prepare_movies_for_modeling(df_movies)
    data_summary = get_data_summary(df_clean)
    
    # Initialize session state
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ROIModelTrainer()
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = FeatureEngineer()
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page(data_summary, df_clean)
    elif page == "🔮 Predict ROI":
        show_prediction_page(df_clean, df_genres)
    elif page == "📊 Data Analysis":
        show_data_analysis_page(df_clean, df_genres)
    elif page == "📝 Semantic Analysis":
        show_semantic_analysis_page()
    elif page == "🎯 Thematic Clustering":
        show_clustering_page()
    elif page == "🤖 Model Training":
        show_model_training_page(df_clean)
    elif page == "📈 Model Performance":
        show_model_performance_page()
    elif page == "🔬 Sensitivity Analysis":
        show_sensitivity_analysis_page(df_clean, df_genres)


def show_home_page(data_summary, df_clean):
    """Display the home page with dataset overview."""
    
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Movies",
            value=f"{data_summary.get('total_movies', 0):,}",
            help="Movies with valid financial data"
        )
    
    with col2:
        st.metric(
            label="Profitable Movies",
            value=f"{data_summary.get('profitable_movies', 0):,}",
            delta=f"{data_summary.get('profitability_rate', 0):.1f}%",
            help="Percentage of profitable movies"
        )
    
    with col3:
        st.metric(
            label="Average ROI",
            value=f"{data_summary.get('avg_roi', 0):.2f}",
            help="Mean Return on Investment"
        )
    
    with col4:
        st.metric(
            label="Year Range",
            value=f"{data_summary.get('year_range', (0, 0))[0]}-{data_summary.get('year_range', (0, 0))[1]}",
            help="Release years covered"
        )
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Unique Genres", f"{data_summary.get('unique_genres', 0)}")
    with col2:
        st.metric("Unique Countries", f"{data_summary.get('unique_countries', 0)}")
    with col3:
        st.metric("Unique Languages", f"{data_summary.get('unique_languages', 0)}")
    
    # Quick insights
    st.header("🔍 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top genres
        if not df_clean.empty:
            genre_counts = df_clean['genres'].str.split(', ').explode().value_counts().head(10)
            fig_genres = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation='h',
                title="Top 10 Genres",
                labels={'x': 'Number of Movies', 'y': 'Genre'}
            )
            fig_genres.update_layout(height=400)
            st.plotly_chart(fig_genres, width="stretch")
    
    with col2:
        # ROI distribution
        if not df_clean.empty:
            fig_roi = px.histogram(
                df_clean,
                x='roi',
                nbins=50,
                title="ROI Distribution",
                labels={'roi': 'ROI', 'count': 'Number of Movies'}
            )
            fig_roi.update_layout(height=400)
            st.plotly_chart(fig_roi, width="stretch")
    
    # Instructions
    st.header("🚀 How to Use This App")
    
    st.markdown("""
    This application helps you predict the Return on Investment (ROI) of movies using machine learning. Here's how to get started:
    
    1. **🔮 Predict ROI**: Use the prediction page to input movie characteristics and get ROI predictions
    2. **📊 Data Analysis**: Explore the dataset with interactive visualizations
    3. **🤖 Model Training**: Train or retrain the Random Forest model with different parameters
    4. **📈 Model Performance**: View model metrics and feature importance
    
    The model uses features like budget, runtime, genres, country, and language to predict ROI.
    """)


def show_prediction_page(df_clean, df_genres):
    """Display the ROI prediction page."""
    
    st.header("🔮 Predict Movie ROI")
    
    if not st.session_state.model_trainer.is_trained:
        st.warning("⚠️ Model not trained yet. Please train the model first in the 'Model Training' page.")
        return
    
    st.markdown("Enter the characteristics of a movie to predict its ROI:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic movie information
        st.subheader("📝 Basic Information")
        
        title = st.text_input("Movie Title", value="My Movie")
        budget = st.number_input("Budget (USD)", min_value=100000, max_value=500000000, value=10000000, step=1000000)
        runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=300, value=120, step=5)
        
        # Language and country
        languages = df_clean['original_language'].value_counts().head(10).index.tolist()
        original_language = st.selectbox("Original Language", languages)
        
        countries = df_clean['main_country'].value_counts().head(10).index.tolist()
        main_country = st.selectbox("Main Country", countries)
        
    with col2:
        # Content and genres
        st.subheader("🎭 Content")
        
        adult = st.checkbox("Adult Content", value=False)
        
        # Genres
        st.subheader("� Genres")
        all_genres = df_genres['name'].tolist() if df_genres is not None else []
        selected_genres = st.multiselect("Select Genres", all_genres, default=["Drama"])
    
    # Create prediction
    if st.button("🔮 Predict ROI", type="primary"):
        
        # Prepare movie data
        movie_data = {
            'title': title,
            'budget': budget,
            'runtime': runtime,
            'release_date': '2024-01-01',  # Placeholder - not used in features
            'original_language': original_language,
            'main_country': main_country,
            'vote_average': 0,  # Not used in prediction
            'vote_count': 0,  # Not used in prediction
            'adult': adult,
            'status': 'Released',  # Not used in prediction
            'genres': ', '.join(selected_genres),
            'production_countries': f'[{{"name": "{main_country}"}}]'
        }
        
        try:
            # Create features for prediction
            X_pred = st.session_state.feature_engineer.create_prediction_features(movie_data)
            
            # Make prediction
            predicted_roi = st.session_state.model_trainer.predict_roi(X_pred)[0]
            
            # Calculate predicted revenue
            predicted_revenue = budget * (1 + predicted_roi)
            predicted_profit = predicted_revenue - budget
            
            # Display results
            st.success("✅ Prediction completed!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted ROI</h3>
                    <h2 style="color: {'green' if predicted_roi > 0 else 'red'};">
                        {predicted_roi:.2f} ({predicted_roi*100:.1f}%)
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Revenue</h3>
                    <h2 style="color: #1f77b4;">
                        ${predicted_revenue:,.0f}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Predicted Profit</h3>
                    <h2 style="color: {'green' if predicted_profit > 0 else 'red'};">
                        ${predicted_profit:,.0f}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretation
            st.subheader("📊 Prediction Interpretation")
            
            if predicted_roi > 0:
                st.success(f"🎉 This movie is predicted to be **profitable** with a {predicted_roi*100:.1f}% return on investment!")
            else:
                st.error(f"📉 This movie is predicted to be **unprofitable** with a {predicted_roi*100:.1f}% return on investment.")
            
            # Confidence interval (simplified)
            confidence_interval = 0.2  # 20% margin of error
            lower_bound = predicted_roi * (1 - confidence_interval)
            upper_bound = predicted_roi * (1 + confidence_interval)
            
            st.info(f"📈 **Confidence Interval**: {lower_bound:.2f} to {upper_bound:.2f} ROI (±{confidence_interval*100:.0f}%)")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")


def show_data_analysis_page(df_clean, df_genres):
    """Display the data analysis page."""
    
    st.header("📊 Data Analysis")
    
    if df_clean.empty:
        st.error("No data available for analysis.")
        return
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Overview", "Financial Analysis", "Genre Analysis", "Temporal Analysis", "Country Analysis"]
    )
    
    if analysis_type == "Overview":
        show_overview_analysis(df_clean)
    elif analysis_type == "Financial Analysis":
        show_financial_analysis(df_clean)
    elif analysis_type == "Genre Analysis":
        show_genre_analysis(df_clean)
    elif analysis_type == "Temporal Analysis":
        show_temporal_analysis(df_clean)
    elif analysis_type == "Country Analysis":
        show_country_analysis(df_clean)


def show_overview_analysis(df_clean):
    """Show overview analysis."""
    
    st.subheader("📈 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI distribution
        fig_roi = px.histogram(
            df_clean, x='roi', nbins=50,
            title="ROI Distribution",
            labels={'roi': 'ROI', 'count': 'Number of Movies'}
        )
        st.plotly_chart(fig_roi, width="stretch")
    
    with col2:
        # Budget vs Revenue
        fig_budget = px.scatter(
            df_clean, x='budget', y='revenue',
            title="Budget vs Revenue",
            labels={'budget': 'Budget (USD)', 'revenue': 'Revenue (USD)'},
            opacity=0.6
        )
        fig_budget.update_layout(
            xaxis_type="log",
            yaxis_type="log"
        )
        st.plotly_chart(fig_budget, width="stretch")
    
    # Top movies by ROI
    st.subheader("🏆 Top Movies by ROI")
    top_movies = df_clean.nlargest(10, 'roi')[['title', 'release_year', 'budget', 'revenue', 'roi']]
    st.dataframe(top_movies, width="stretch")


def show_financial_analysis(df_clean):
    """Show financial analysis."""
    
    st.subheader("💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Budget distribution
        fig_budget = px.box(
            df_clean, y='budget',
            title="Budget Distribution",
            labels={'budget': 'Budget (USD)'}
        )
        fig_budget.update_layout(yaxis_type="log")
        st.plotly_chart(fig_budget, width="stretch")
    
    with col2:
        # Revenue distribution
        fig_revenue = px.box(
            df_clean, y='revenue',
            title="Revenue Distribution",
            labels={'revenue': 'Revenue (USD)'}
        )
        fig_revenue.update_layout(yaxis_type="log")
        st.plotly_chart(fig_revenue, width="stretch")
    
    # Financial metrics
    st.subheader("📊 Financial Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Budget", f"${df_clean['budget'].mean():,.0f}")
    with col2:
        st.metric("Avg Revenue", f"${df_clean['revenue'].mean():,.0f}")
    with col3:
        st.metric("Avg ROI", f"{df_clean['roi'].mean():.2f}")
    with col4:
        st.metric("Profitability Rate", f"{df_clean['is_profitable'].mean()*100:.1f}%")


def show_genre_analysis(df_clean):
    """Show genre analysis."""
    
    st.subheader("🎭 Genre Analysis")
    
    # Genre frequency
    genre_counts = df_clean['genres'].str.split(', ').explode().value_counts().head(15)
    
    fig_genres = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        title="Genre Frequency",
        labels={'x': 'Number of Movies', 'y': 'Genre'}
    )
    st.plotly_chart(fig_genres, width="stretch")
    
    # Genre vs ROI
    genre_roi = []
    for genre in genre_counts.index:
        genre_movies = df_clean[df_clean['genres'].str.contains(genre, na=False)]
        avg_roi = genre_movies['roi'].mean()
        genre_roi.append({'genre': genre, 'avg_roi': avg_roi, 'count': len(genre_movies)})
    
    genre_roi_df = pd.DataFrame(genre_roi)
    genre_roi_df = genre_roi_df[genre_roi_df['count'] >= 10]  # Filter genres with at least 10 movies
    
    fig_genre_roi = px.bar(
        genre_roi_df, x='avg_roi', y='genre',
        orientation='h',
        title="Average ROI by Genre (min 10 movies)",
        labels={'avg_roi': 'Average ROI', 'genre': 'Genre'}
    )
    st.plotly_chart(fig_genre_roi, width="stretch")


def show_temporal_analysis(df_clean):
    """Show temporal analysis."""
    
    st.subheader("📅 Temporal Analysis")
    
    # Movies by year
    yearly_counts = df_clean.groupby('release_year').size().reset_index(name='count')
    
    fig_yearly = px.line(
        yearly_counts, x='release_year', y='count',
        title="Movies Released by Year",
        labels={'release_year': 'Year', 'count': 'Number of Movies'}
    )
    st.plotly_chart(fig_yearly, width="stretch")
    
    # ROI by decade
    df_clean['decade'] = (df_clean['release_year'] // 10) * 10
    decade_roi = df_clean.groupby('decade')['roi'].mean().reset_index()
    
    fig_decade = px.bar(
        decade_roi, x='decade', y='roi',
        title="Average ROI by Decade",
        labels={'decade': 'Decade', 'roi': 'Average ROI'}
    )
    st.plotly_chart(fig_decade, width="stretch")


def show_country_analysis(df_clean):
    """Show country analysis."""
    
    st.subheader("🌍 Country Analysis")
    
    # Top countries by movie count
    country_counts = df_clean['main_country'].value_counts().head(15)
    
    fig_countries = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        title="Movies by Country",
        labels={'x': 'Number of Movies', 'y': 'Country'}
    )
    st.plotly_chart(fig_countries, width="stretch")
    
    # ROI by country
    country_roi = df_clean.groupby('main_country')['roi'].agg(['mean', 'count']).reset_index()
    country_roi = country_roi[country_roi['count'] >= 20]  # Filter countries with at least 20 movies
    country_roi = country_roi.sort_values('mean', ascending=False).head(15)
    
    fig_country_roi = px.bar(
        country_roi, x='mean', y='main_country',
        orientation='h',
        title="Average ROI by Country (min 20 movies)",
        labels={'mean': 'Average ROI', 'main_country': 'Country'}
    )
    st.plotly_chart(fig_country_roi, width="stretch")


def show_model_training_page(df_clean):
    """Display the model training page."""
    
    st.header("🤖 Model Training")
    
    if df_clean.empty:
        st.error("No data available for training.")
        return
    
    st.markdown("Train a Random Forest model to predict movie ROI.")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        optimize_hyperparams = st.checkbox("Optimize Hyperparameters", value=True)
    
    with col2:
        random_state = st.number_input("Random State", min_value=0, max_value=1000, value=42)
        # ROI truncation percentiles
        roi_lower_pct = st.number_input(
            "ROI lower percentile (0-100)", min_value=0, max_value=100, value=1, step=1, key="roi_lower_pct"
        )
        roi_upper_pct = st.number_input(
            "ROI upper percentile (0-100)", min_value=0, max_value=100, value=99, step=1, key="roi_upper_pct"
        )
        st.caption("Values outside the selected percentile range will be truncated to the percentile value before training.")
    
    # Debug features section
    st.divider()
    st.subheader("🔍 Debug Features")
    
    show_features = st.checkbox("Show all features before training", value=False)
    enable_feature_tinkering = st.checkbox("Enable feature tinkering (manually select features)", value=False)
    
    # Initialize feature selection in session state
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    
    if show_features:
        with st.spinner("Preparing features preview..."):
            try:
                # Prepare features to show what will be used
                df_features_preview = st.session_state.feature_engineer.create_features(df_clean)
                X_preview, _, _, _, feature_names = st.session_state.feature_engineer.prepare_modeling_data(
                    df_features_preview, test_size=test_size, random_state=random_state
                )
                
                st.success(f"✅ Total features available: **{len(feature_names)}**")
                
                # Group features by category
                # Numeric features we want to show individually (keep names unchanged)
                numeric_names = ['budget_per_minute', 'budget_log', 'runtime']
                numeric_features = [f for f in feature_names if f in numeric_names]

                # Runtime binary partitions (classification features)
                runtime_blocks = [f for f in feature_names if f.startswith('runtime_') and f != 'runtime_category']

                # Classification groups
                genre_features = [f for f in feature_names if f.startswith('genre_')]
                country_features = [f for f in feature_names if f.startswith('country_')]
                language_features = [f for f in feature_names if f.startswith('language_')]

                # Other features (not numeric and not in classification groups)
                classification_prefixes = ['genre_', 'country_', 'language_', 'runtime_']
                other_features = [f for f in feature_names if f not in numeric_features and not any(f.startswith(p) for p in classification_prefixes)]

                feature_categories = {
                    'Numeric Features': numeric_features,
                    'Runtime Blocks': runtime_blocks,
                    'Genres': genre_features,
                    'Countries': country_features,
                    'Languages': language_features,
                    'Other': other_features
                }
                
                # Initialize selected features with all features if not set
                if st.session_state.selected_features is None:
                    st.session_state.selected_features = set(feature_names)
                
                # Initialize enabled_feature_categories based on available display items
                enabled = set()
                display_order = [
                    'budget_per_minute', 'budget_log', 'runtime',
                    'runtime_blocks', 'genres', 'countries', 'languages', 'other'
                ]
                for itm in display_order:
                    if itm == 'budget_per_minute' and any(f == 'budget_per_minute' for f in feature_names):
                        enabled.add(itm)
                    if itm == 'budget_log' and any(f == 'budget_log' for f in feature_names):
                        enabled.add(itm)
                    if itm == 'runtime' and any(f == 'runtime' for f in feature_names):
                        enabled.add(itm)
                    if itm == 'runtime_blocks' and feature_categories.get('Runtime Blocks'):
                        enabled.add(itm)
                    if itm == 'genres' and feature_categories.get('Genres'):
                        enabled.add(itm)
                    if itm == 'countries' and feature_categories.get('Countries'):
                        enabled.add(itm)
                    if itm == 'languages' and feature_categories.get('Languages'):
                        enabled.add(itm)
                    if itm == 'other' and feature_categories.get('Other'):
                        enabled.add(itm)
                if 'enabled_feature_categories' not in st.session_state:
                    st.session_state.enabled_feature_categories = enabled

                # Render display items as checkboxes (disabled when not tinkering)
                for item in display_order:
                    # Map item to underlying features
                    if item == 'budget_per_minute':
                        underlying = [f for f in feature_names if f == 'budget_per_minute']
                    elif item == 'budget_log':
                        underlying = [f for f in feature_names if f == 'budget_log']
                    elif item == 'runtime':
                        underlying = [f for f in feature_names if f == 'runtime']
                    elif item == 'runtime_blocks':
                        underlying = feature_categories.get('Runtime Blocks', [])
                    elif item == 'genres':
                        underlying = feature_categories.get('Genres', [])
                    elif item == 'countries':
                        underlying = feature_categories.get('Countries', [])
                    elif item == 'languages':
                        underlying = feature_categories.get('Languages', [])
                    else:
                        underlying = feature_categories.get('Other', [])

                    # Build label: for classification groups include inline class list
                    if item in ('genres', 'countries', 'languages', 'runtime_blocks'):
                        short_names = []
                        if item == 'runtime_blocks':
                            for feat in underlying:
                                s = feat.replace('runtime_', '')
                                short_names.append(s.replace('_', '-'))
                        elif item == 'genres':
                            short_names = [f.replace('genre_', '') for f in underlying]
                        elif item == 'countries':
                            short_names = [f.replace('country_', '') for f in underlying]
                        elif item == 'languages':
                            short_names = [f.replace('language_', '') for f in underlying]

                        label = f"{item}: {', '.join(short_names)}"
                    else:
                        label = item

                    safe_key = f"feat_{item}"
                    is_enabled = item in st.session_state.enabled_feature_categories
                    checked = st.checkbox(label, value=is_enabled, key=safe_key, disabled=not enable_feature_tinkering)
                    # Only update state when tinkering is enabled
                    if enable_feature_tinkering:
                        if checked:
                            st.session_state.enabled_feature_categories.add(item)
                        else:
                            st.session_state.enabled_feature_categories.discard(item)

                # Rebuild selected_features from enabled items
                selected = set()

                # numeric features (include only if enabled)
                if 'budget_per_minute' in st.session_state.enabled_feature_categories:
                    selected.update([f for f in numeric_features if f == 'budget_per_minute'])
                if 'budget_log' in st.session_state.enabled_feature_categories:
                    selected.update([f for f in numeric_features if f == 'budget_log'])
                if 'runtime' in st.session_state.enabled_feature_categories:
                    selected.update([f for f in numeric_features if f == 'runtime'])

                # include 'Other' if enabled
                if 'other' in st.session_state.enabled_feature_categories:
                    selected.update(feature_categories.get('Other', []))

                # include classification groups if enabled
                if 'runtime_blocks' in st.session_state.enabled_feature_categories:
                    selected.update(feature_categories.get('Runtime Blocks', []))
                if 'genres' in st.session_state.enabled_feature_categories:
                    selected.update(feature_categories.get('Genres', []))
                if 'countries' in st.session_state.enabled_feature_categories:
                    selected.update(feature_categories.get('Countries', []))
                if 'languages' in st.session_state.enabled_feature_categories:
                    selected.update(feature_categories.get('Languages', []))

                st.session_state.selected_features = selected

                # Show summary of selected features
                selected_count = len(st.session_state.selected_features)
                st.metric("Selected Features", f"{selected_count} / {len(feature_names)}")

                if selected_count == 0:
                    st.error("❌ Please select at least one feature to train the model.")
                    return
                
                
            except Exception as e:
                st.error(f"❌ Error preparing features preview: {str(e)}")
                return
    
    st.divider()
    
    if st.button("🚀 Train Model", type="primary"):
        
        with st.spinner("Training model..."):
            try:
                # Prepare features
                df_features = st.session_state.feature_engineer.create_features(df_clean)
                X_train, X_test, y_train, y_test, feature_names = st.session_state.feature_engineer.prepare_modeling_data(
                    df_features, test_size=test_size, random_state=random_state
                )
                # Apply ROI truncation based on selected percentiles
                try:
                    # Validate percentiles
                    if roi_lower_pct >= roi_upper_pct:
                        st.error("❌ ROI lower percentile must be smaller than ROI upper percentile.")
                        return

                    lower_val = y_train.quantile(roi_lower_pct / 100.0)
                    upper_val = y_train.quantile(roi_upper_pct / 100.0)

                    train_below = int((y_train < lower_val).sum())
                    train_above = int((y_train > upper_val).sum())
                    test_below = int((y_test < lower_val).sum())
                    test_above = int((y_test > upper_val).sum())

                    # Clip target values to percentile bounds
                    y_train = y_train.clip(lower=lower_val, upper=upper_val)
                    y_test = y_test.clip(lower=lower_val, upper=upper_val)

                    st.info(
                        f"ROI truncation applied: {roi_lower_pct}th -> {lower_val:.3f}, {roi_upper_pct}th -> {upper_val:.3f}. "
                        f"Train clipped: below={train_below}, above={train_above}; Test clipped: below={test_below}, above={test_above}."
                    )
                except Exception as e:
                    st.error(f"❌ Error applying ROI truncation: {e}")
                    return
                
                # Apply feature selection if enabled
                if enable_feature_tinkering and st.session_state.selected_features:
                    selected_features_list = list(st.session_state.selected_features)
                    
                    # Filter to only include features that exist in the dataset
                    valid_selected_features = [f for f in selected_features_list if f in X_train.columns]
                    
                    if len(valid_selected_features) == 0:
                        st.error("❌ None of the selected features are available in the dataset.")
                        return
                    
                    st.info(f"🔍 Training with {len(valid_selected_features)} selected features (out of {len(feature_names)} total)")
                    
                    # Subset the data to selected features
                    X_train = X_train[valid_selected_features]
                    X_test = X_test[valid_selected_features]
                    feature_names = valid_selected_features
                else:
                    st.info(f"🔍 Training with all {len(feature_names)} features")
                
                # Train model
                metrics = st.session_state.model_trainer.train_model(
                    X_train, y_train, X_test, y_test, optimize_hyperparams=optimize_hyperparams
                )
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                st.subheader("📊 Model Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("R² Score (Test)", f"{metrics['test_r2']:.3f}")
                with col2:
                    st.metric("RMSE (Test)", f"{metrics['test_rmse']:.3f}")
                with col3:
                    st.metric("MAE (Test)", f"{metrics['test_mae']:.3f}")
                
                # Cross-validation scores
                cv_mean = metrics['cv_scores'].mean()
                cv_std = metrics['cv_scores'].std()
                
                st.metric("CV R² Score", f"{cv_mean:.3f} ± {cv_std:.3f}")
                
                # Feature importance
                st.subheader("🔍 Feature Importance")
                importance_plot = st.session_state.model_trainer.get_feature_importance_plot()
                if importance_plot:
                    st.plotly_chart(importance_plot, width="stretch")
                
                # Save model option
                if st.button("💾 Save Model"):
                    model_path = "models/roi_model.pkl"
                    os.makedirs("models", exist_ok=True)
                    st.session_state.model_trainer.save_model(model_path)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")


def show_model_performance_page():
    """Display the model performance page."""
    
    st.header("📈 Model Performance")
    
    if not st.session_state.model_trainer.is_trained:
        st.warning("⚠️ No model trained yet. Please train a model first.")
        return
    
    # Model summary
    summary = st.session_state.model_trainer.get_model_summary()
    
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Type", summary['model_type'])
        st.metric("N Estimators", summary['n_estimators'])
        st.metric("Max Depth", summary['max_depth'])
    
    with col2:
        st.metric("Min Samples Split", summary['min_samples_split'])
        st.metric("Min Samples Leaf", summary['min_samples_leaf'])
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    
    metrics = summary['training_metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Train R²", f"{metrics['train_r2']:.3f}")
        st.metric("Test R²", f"{metrics['test_r2']:.3f}")
    
    with col2:
        st.metric("Train RMSE", f"{metrics['train_rmse']:.3f}")
        st.metric("Test RMSE", f"{metrics['test_rmse']:.3f}")
    
    with col3:
        st.metric("Train MAE", f"{metrics['train_mae']:.3f}")
        st.metric("Test MAE", f"{metrics['test_mae']:.3f}")
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    importance_plot = st.session_state.model_trainer.get_feature_importance_plot()
    if importance_plot:
        st.plotly_chart(importance_plot, width="stretch")
    
    # Load model option
    st.subheader("💾 Model Management")
    
    uploaded_file = st.file_uploader("Load Model", type=['pkl'])
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with open("temp_model.pkl", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load model
            st.session_state.model_trainer.load_model("temp_model.pkl")
            
            # Clean up
            os.remove("temp_model.pkl")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")


def show_sensitivity_analysis_page(df_clean, df_genres):
    """Display sensitivity analysis page showing how ROI changes with individual features."""
    
    st.header("🔬 Sensitivity Analysis")
    
    if not st.session_state.model_trainer.is_trained:
        st.warning("⚠️ Model not trained yet. Please train the model first in the 'Model Training' page.")
        return
    
    st.markdown("""
    This page shows how predicted ROI changes when varying individual features while keeping others constant.
    This helps understand which factors have the most impact on ROI predictions.
    """)
    
    # Get baseline values from user
    st.subheader("📝 Baseline Movie Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        baseline_budget = st.number_input("Baseline Budget (USD)", min_value=100000, max_value=500000000, value=10000000, step=1000000)
        baseline_runtime = st.number_input("Baseline Runtime (minutes)", min_value=60, max_value=300, value=120, step=5)
        
        languages = df_clean['original_language'].value_counts().head(10).index.tolist()
        baseline_language = st.selectbox("Baseline Language", languages, index=0 if 'en' not in languages else languages.index('en'))
        
    with col2:
        countries = df_clean['main_country'].value_counts().head(10).index.tolist()
        baseline_country = st.selectbox("Baseline Country", countries)
        
        baseline_adult = st.checkbox("Baseline Adult Content", value=False)
        
        all_genres = df_genres['name'].tolist() if df_genres is not None else []
        baseline_genres = st.multiselect("Baseline Genres", all_genres, default=["Drama"])
    
    # Create baseline movie data
    baseline_movie = {
        'title': 'Baseline Movie',
        'budget': baseline_budget,
        'runtime': baseline_runtime,
        'release_date': '2024-01-01',
        'original_language': baseline_language,
        'main_country': baseline_country,
        'vote_average': 0,
        'vote_count': 0,
        'adult': baseline_adult,
        'status': 'Released',
        'genres': ', '.join(baseline_genres),
        'production_countries': f'[{{"name": "{baseline_country}"}}]'
    }
    
    # Analysis type selection
    st.subheader("📊 Select Analysis Type")
    
    analysis_tabs = st.tabs(["💰 Budget Impact", "⏱️ Runtime Impact", "🌍 Country Impact", "🗣️ Language Impact", "🎭 Genre Impact"])
    
    # Budget Analysis
    with analysis_tabs[0]:
        st.markdown("### How ROI changes with different budgets")
        
        # Show fixed values
        with st.expander("📌 Fixed Baseline Values", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Runtime:** {baseline_runtime} minutes")
                st.write(f"**Language:** {baseline_language}")
                st.write(f"**Country:** {baseline_country}")
            with col2:
                st.write(f"**Adult Content:** {baseline_adult}")
                st.write(f"**Genres:** {', '.join(baseline_genres)}")
        
        # Generate budget range
        budget_min = st.number_input("Min Budget", min_value=100000, value=100000, step=100000, key="budget_min")
        budget_max = st.number_input("Max Budget", min_value=budget_min, value=100000000, step=1000000, key="budget_max")
        budget_steps = st.slider("Number of points", min_value=10, max_value=100, value=100, key="budget_steps")
        
        budgets = np.linspace(budget_min, budget_max, budget_steps)
        roi_predictions = []
        
        with st.spinner("Calculating predictions..."):
            for budget in budgets:
                movie = baseline_movie.copy()
                movie['budget'] = budget
                
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    predicted_roi = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    roi_predictions.append(predicted_roi)
                except Exception as e:
                    st.error(f"Error predicting for budget {budget}: {str(e)}")
                    roi_predictions.append(None)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=budgets, y=roi_predictions, mode='lines+markers', name='Predicted ROI'))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title="ROI vs Budget",
            xaxis_title="Budget (USD)",
            yaxis_title="Predicted ROI",
            height=500
        )
        st.plotly_chart(fig, width="stretch")
        
        # Summary statistics (ignore failed predictions)
        valid_preds = [p for p in roi_predictions if p is not None and not (isinstance(p, float) and np.isnan(p))]
        col1, col2, col3 = st.columns(3)
        if valid_preds:
            with col1:
                st.metric("Min ROI", f"{min(valid_preds):.2f}")
            with col2:
                st.metric("Max ROI", f"{max(valid_preds):.2f}")
            with col3:
                st.metric("ROI Range", f"{max(valid_preds) - min(valid_preds):.2f}")
        else:
            with col1:
                st.metric("Min ROI", "N/A")
            with col2:
                st.metric("Max ROI", "N/A")
            with col3:
                st.metric("ROI Range", "N/A")
    
    # Runtime Analysis
    with analysis_tabs[1]:
        st.markdown("### How ROI changes with different runtimes")
        
        # Show fixed values
        with st.expander("📌 Fixed Baseline Values", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Budget:** ${baseline_budget:,.0f}")
                st.write(f"**Language:** {baseline_language}")
                st.write(f"**Country:** {baseline_country}")
            with col2:
                st.write(f"**Adult Content:** {baseline_adult}")
                st.write(f"**Genres:** {', '.join(baseline_genres)}")
        
        runtime_min = st.number_input("Min Runtime (minutes)", min_value=45, value=60, step=5, key="runtime_min")
        runtime_max = st.number_input("Max Runtime (minutes)", min_value=runtime_min, value=270, step=5, key="runtime_max")
        runtime_steps = st.slider("Number of points", min_value=10, max_value=50, value=50, key="runtime_steps")
        
        runtimes = np.linspace(runtime_min, runtime_max, runtime_steps)
        roi_predictions = []
        
        with st.spinner("Calculating predictions..."):
            for runtime in runtimes:
                movie = baseline_movie.copy()
                movie['runtime'] = int(runtime)
                
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    predicted_roi = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    roi_predictions.append(predicted_roi)
                except Exception as e:
                    st.error(f"Error predicting for runtime {runtime}: {str(e)}")
                    roi_predictions.append(None)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=runtimes, y=roi_predictions, mode='lines+markers', name='Predicted ROI'))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title="ROI vs Runtime",
            xaxis_title="Runtime (minutes)",
            yaxis_title="Predicted ROI",
            height=500
        )
        st.plotly_chart(fig, width="stretch")
        
        # Summary statistics (ignore failed predictions)
        valid_preds = [p for p in roi_predictions if p is not None and not (isinstance(p, float) and np.isnan(p))]
        col1, col2, col3 = st.columns(3)
        if valid_preds:
            with col1:
                st.metric("Min ROI", f"{min(valid_preds):.2f}")
            with col2:
                st.metric("Max ROI", f"{max(valid_preds):.2f}")
            with col3:
                st.metric("ROI Range", f"{max(valid_preds) - min(valid_preds):.2f}")
        else:
            with col1:
                st.metric("Min ROI", "N/A")
            with col2:
                st.metric("Max ROI", "N/A")
            with col3:
                st.metric("ROI Range", "N/A")
    
    # Country Analysis
    with analysis_tabs[2]:
        st.markdown("### How ROI changes across different countries")
        
        # Show fixed values
        with st.expander("📌 Fixed Baseline Values", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Budget:** ${baseline_budget:,.0f}")
                st.write(f"**Runtime:** {baseline_runtime} minutes")
                st.write(f"**Language:** {baseline_language}")
            with col2:
                st.write(f"**Adult Content:** {baseline_adult}")
                st.write(f"**Genres:** {', '.join(baseline_genres)}")
        
        top_n_countries = st.slider("Number of top countries to analyze", min_value=5, max_value=20, value=10, key="country_n")
        countries = df_clean['main_country'].value_counts().head(top_n_countries).index.tolist()
        
        roi_predictions = []
        
        with st.spinner("Calculating predictions..."):
            for country in countries:
                movie = baseline_movie.copy()
                movie['main_country'] = country
                movie['production_countries'] = f'[{{"name": "{country}"}}]'
                
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    predicted_roi = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    roi_predictions.append(predicted_roi)
                except Exception as e:
                    st.error(f"Error predicting for country {country}: {str(e)}")
                    roi_predictions.append(None)
        
    # Plot (replace None with NaN so plot handles missing values)
    y_vals = [np.nan if p is None else p for p in roi_predictions]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=countries, y=y_vals, name='Predicted ROI'))
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig.update_layout(
        title="ROI by Production Country",
        xaxis_title="Country",
        yaxis_title="Predicted ROI",
        height=500
    )
    st.plotly_chart(fig, width="stretch")

    # Best and worst (handle missing values)
    country_roi_df = pd.DataFrame({'Country': countries, 'ROI': [np.nan if p is None else p for p in roi_predictions]})
    country_roi_df = country_roi_df.sort_values('ROI', ascending=False, na_position='last')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 Countries by ROI**")
        st.dataframe(country_roi_df.head(5), width="stretch")
    with col2:
        st.markdown("**Bottom 5 Countries by ROI**")
        st.dataframe(country_roi_df.tail(5), width="stretch")
    
    # Language Analysis
    with analysis_tabs[3]:
        st.markdown("### How ROI changes across different languages")
        
        # Show fixed values
        with st.expander("📌 Fixed Baseline Values", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Budget:** ${baseline_budget:,.0f}")
                st.write(f"**Runtime:** {baseline_runtime} minutes")
                st.write(f"**Country:** {baseline_country}")
            with col2:
                st.write(f"**Adult Content:** {baseline_adult}")
                st.write(f"**Genres:** {', '.join(baseline_genres)}")
        
        top_n_languages = st.slider("Number of top languages to analyze", min_value=5, max_value=15, value=10, key="language_n")
        languages_list = df_clean['original_language'].value_counts().head(top_n_languages).index.tolist()
        
        roi_predictions = []
        
        with st.spinner("Calculating predictions..."):
            for lang in languages_list:
                movie = baseline_movie.copy()
                movie['original_language'] = lang
                
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    predicted_roi = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    roi_predictions.append(predicted_roi)
                except Exception as e:
                    st.error(f"Error predicting for language {lang}: {str(e)}")
                    roi_predictions.append(None)
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=languages_list, y=roi_predictions, name='Predicted ROI'))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title="ROI by Original Language",
            xaxis_title="Language",
            yaxis_title="Predicted ROI",
            height=500
        )
        st.plotly_chart(fig, width="stretch")
        
        # Best and worst
        language_roi_df = pd.DataFrame({'Language': languages_list, 'ROI': roi_predictions}).sort_values('ROI', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 Languages by ROI**")
            st.dataframe(language_roi_df.head(5), width="stretch")
        with col2:
            st.markdown("**Bottom 5 Languages by ROI**")
            st.dataframe(language_roi_df.tail(5), width="stretch")
    
    # Genre Analysis
    with analysis_tabs[4]:
        st.markdown("### How ROI changes with individual genres")
        
        st.info("This analysis shows ROI when adding each genre individually to the baseline configuration.")
        
        # Show fixed values
        with st.expander("📌 Fixed Baseline Values", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Budget:** ${baseline_budget:,.0f}")
                st.write(f"**Runtime:** {baseline_runtime} minutes")
                st.write(f"**Language:** {baseline_language}")
            with col2:
                st.write(f"**Country:** {baseline_country}")
                st.write(f"**Adult Content:** {baseline_adult}")
        
        genres_to_test = all_genres if all_genres else []
        roi_predictions = []
        
        with st.spinner("Calculating predictions..."):
            for genre in genres_to_test:
                movie = baseline_movie.copy()
                movie['genres'] = genre  # Single genre
                
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    predicted_roi = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    roi_predictions.append(predicted_roi)
                except Exception as e:
                    roi_predictions.append(None)
        
        # Plot
        genre_roi_df = pd.DataFrame({'Genre': genres_to_test, 'ROI': roi_predictions}).sort_values('ROI', ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=genre_roi_df['Genre'], y=genre_roi_df['ROI'], name='Predicted ROI'))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title="ROI by Genre (Individual)",
            xaxis_title="Genre",
            yaxis_title="Predicted ROI",
            height=500
        )
        st.plotly_chart(fig, width="stretch")
        
        # Best and worst
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 Genres by ROI**")
            st.dataframe(genre_roi_df.head(5), width="stretch")
        with col2:
            st.markdown("**Bottom 5 Genres by ROI**")
            st.dataframe(genre_roi_df.tail(5), width="stretch")


def show_semantic_analysis_page():
    """Display semantic analysis of movie text fields."""
    from utils.semantic_analysis import (
        load_semantic_data, prepare_documents, compute_tfidf,
        create_wordcloud, plot_term_correlations, plot_top_terms,
        plot_segment_comparison, load_embeddings_model, compute_embeddings,
        plot_embeddings_2d, plot_cluster_roi_stats, display_cluster_representatives
    )
    
    st.header("📝 Semantic Analysis of Movie Text")
    
    st.markdown("""
    This page analyzes the textual content of movies (title, overview, tagline, genres, keywords) 
    to discover patterns and their relationship with ROI using both **TF-IDF** and **Sentence Embeddings**.
    """)
    
    # Load data
    with st.spinner("Loading movie text data..."):
        df_semantic = load_semantic_data()
    
    if df_semantic is None or df_semantic.empty:
        st.error("No data available for semantic analysis.")
        return
    
    # Show data summary
    st.subheader("📊 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Movies Analyzed", f"{len(df_semantic):,}")
    with col2:
        st.metric("Avg ROI", f"{df_semantic['roi'].mean():.2f}")
    with col3:
        st.metric("With Keywords", f"{df_semantic['keywords'].notna().sum():,}")
    with col4:
        st.metric("With Overview", f"{df_semantic['overview'].notna().sum():,}")
    
    # Prepare documents
    with st.spinner("Preprocessing text..."):
        df_docs = prepare_documents(df_semantic)
    
    if df_docs.empty:
        st.error("Could not prepare documents for analysis.")
        return
    
    st.success(f"✓ Prepared {len(df_docs):,} documents for analysis")
    
    # Compute TF-IDF
    with st.spinner("Computing TF-IDF features..."):
        tfidf, tfidf_matrix = compute_tfidf(df_docs['document'])
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔤 Top Terms (TF-IDF)",
        "🔗 ROI Correlation",
        "📊 ROI Segments",
        "☁️ Word Clouds",
        "🧠 Sentence Embeddings",
        "📈 Statistics"
    ])
    
    # Tab 1: Top Terms
    with tab1:
        st.subheader("Most Important Terms (TF-IDF)")
        st.markdown("""
        These are the most distinctive terms across all movies, 
        based on TF-IDF (Term Frequency-Inverse Document Frequency) scoring.
        """)
        
        top_n = st.slider("Number of terms to show", 10, 50, 25, key="top_terms_slider")
        
        top_terms = tfidf.get_top_terms_global(top_n=top_n)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_terms = plot_top_terms(top_terms, top_n=top_n)
            st.plotly_chart(fig_terms, width="stretch")
        
        with col2:
            st.dataframe(
                top_terms.head(20),
                width="stretch",
                height=600
            )
    
    # Tab 2: ROI Correlation
    with tab2:
        st.subheader("Terms Correlated with ROI")
        st.markdown("""
        This shows which terms are most strongly correlated (positively or negatively) with movie ROI.
        """)
        
        correlation_method = st.selectbox(
            "Correlation method",
            ["spearman", "pearson"],
            help="Spearman is more robust to outliers"
        )
        
        top_corr_n = st.slider("Terms to show", 10, 30, 15, key="corr_slider")
        
        with st.spinner("Calculating correlations..."):
            roi_values = df_docs.loc[df_docs.index, 'roi']
            correlations = tfidf.correlate_with_target(
                roi_values,
                method=correlation_method,
                top_n=top_corr_n * 2  # Get more to split pos/neg
            )
        
        fig_corr = plot_term_correlations(correlations, top_n=top_corr_n)
        st.plotly_chart(fig_corr, width="stretch")
        
        # Show data table
        with st.expander("📋 View Correlation Data"):
            st.dataframe(
                correlations[['term', 'correlation', 'p_value']].head(30),
                width="stretch"
            )
    
    # Tab 3: ROI Segments
    with tab3:
        st.subheader("Vocabulary by ROI Segment")
        st.markdown("""
        Compare the most characteristic terms used in high-ROI vs low-ROI movies.
        """)
        
        n_segments = st.slider("Number of segments", 2, 5, 4, key="segments_slider")
        top_seg_n = st.slider("Terms per segment", 10, 25, 15, key="seg_terms_slider")
        
        with st.spinner("Analyzing segments..."):
            segments = tfidf.analyze_roi_segments(
                df_docs,
                roi_column='roi',
                n_segments=n_segments,
                top_terms_per_segment=top_seg_n
            )
        
        # Show comparison of extreme segments
        if n_segments >= 4:
            fig_seg = plot_segment_comparison(
                segments,
                segment_names=['Q1', 'Q4'],
                top_n=top_seg_n
            )
            st.plotly_chart(fig_seg, width="stretch")
        
        # Show all segments
        with st.expander("📊 View All Segments"):
            for seg_name in sorted(segments.keys()):
                st.markdown(f"**{seg_name}**")
                st.dataframe(
                    segments[seg_name].head(10),
                    width="stretch"
                )
                st.markdown("---")
    
    # Tab 4: Word Clouds
    with tab4:
        st.subheader("Word Cloud Visualizations")
        st.markdown("""
        Visual representation of the most common terms in different ROI segments.
        """)
        
        # ROI threshold selection
        roi_threshold = st.slider(
            "ROI threshold for high/low split",
            float(df_docs['roi'].min()),
            float(df_docs['roi'].max()),
            float(df_docs['roi'].median()),
            0.1
        )
        
        high_roi_docs = df_docs[df_docs['roi'] >= roi_threshold]['document']
        low_roi_docs = df_docs[df_docs['roi'] < roi_threshold]['document']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**High ROI (>= {roi_threshold:.2f})** - {len(high_roi_docs)} movies")
            with st.spinner("Generating word cloud..."):
                fig_wc_high = create_wordcloud(
                    high_roi_docs,
                    title="High ROI Movies",
                    colormap='Greens'
                )
                if fig_wc_high:
                    st.pyplot(fig_wc_high)
                else:
                    st.info("Not enough data for word cloud")
        
        with col2:
            st.markdown(f"**Low ROI (< {roi_threshold:.2f})** - {len(low_roi_docs)} movies")
            with st.spinner("Generating word cloud..."):
                fig_wc_low = create_wordcloud(
                    low_roi_docs,
                    title="Low ROI Movies",
                    colormap='Reds'
                )
                if fig_wc_low:
                    st.pyplot(fig_wc_low)
                else:
                    st.info("Not enough data for word cloud")
    
    # Tab 5: Sentence Embeddings
    with tab5:
        st.subheader("🧠 Sentence Embeddings Analysis")
        
        st.markdown("""
        **Sentence Embeddings** use transformer models (BERT) to capture semantic meaning.
        Unlike TF-IDF which counts words, embeddings understand context and themes.
        """)
        
        # Model selection
        model_choice = st.selectbox(
            "Select Embedding Model",
            options=[
                'all-MiniLM-L6-v2 (Fast, 384 dim)',
                'all-mpnet-base-v2 (Best quality, 768 dim)'
            ],
            help="Smaller models are faster but larger models capture more nuance"
        )
        
        model_name = model_choice.split(' ')[0]
        
        # Sample size for performance
        sample_size = st.slider(
            "Number of movies to analyze",
            min_value=100,
            max_value=min(2000, len(df_docs)),
            value=min(500, len(df_docs)),
            step=100,
            help="Analyzing fewer movies is faster. Start small!"
        )
        
        if st.button("🚀 Generate Embeddings", type="primary"):
            # Sample data
            df_sample = df_docs.sample(n=sample_size, random_state=42)
            
            # Load model
            with st.spinner(f"Loading {model_name} model..."):
                embeddings_analyzer = load_embeddings_model(model_name)
            
            if embeddings_analyzer is None:
                st.error("Failed to load embeddings model. Make sure sentence-transformers is installed.")
                st.code("pip install sentence-transformers")
                return
            
            # Compute embeddings
            with st.spinner(f"Computing embeddings for {sample_size} movies..."):
                embeddings = compute_embeddings(
                    embeddings_analyzer,
                    df_sample['document'],
                    batch_size=32
                )
            
            st.success(f"✓ Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
            
            # Cluster movies
            n_clusters = st.slider("Number of clusters", 5, 20, 10)
            
            with st.spinner("Clustering movies by semantic similarity..."):
                cluster_labels = embeddings_analyzer.cluster_movies(
                    n_clusters=n_clusters,
                    embeddings=embeddings
                )
            
            # Analyze ROI by cluster
            cluster_stats = embeddings_analyzer.analyze_roi_by_clusters(
                df_sample.reset_index(drop=True),
                cluster_labels,
                roi_column='roi'
            )
            
            # Display results
            st.markdown("### 📊 Cluster Analysis Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top 5 Clusters by ROI**")
                st.dataframe(
                    cluster_stats.head(5).style.format({
                        'roi_mean': '{:.2f}',
                        'roi_median': '{:.2f}',
                        'roi_std': '{:.2f}',
                        'roi_min': '{:.2f}',
                        'roi_max': '{:.2f}'
                    }),
                    width="stretch"
                )
            
            with col2:
                st.markdown("**Cluster Size Distribution**")
                fig_sizes = px.bar(
                    x=cluster_stats.index.astype(str),
                    y=cluster_stats['n_movies'],
                    labels={'x': 'Cluster', 'y': 'Number of Movies'},
                    title='Movies per Cluster'
                )
                st.plotly_chart(fig_sizes, width="stretch")
            
            # ROI by cluster chart
            st.markdown("### 📈 ROI by Semantic Cluster")
            fig_cluster_roi = plot_cluster_roi_stats(cluster_stats)
            st.plotly_chart(fig_cluster_roi, width="stretch")
            
            # 2D visualization
            st.markdown("### 🗺️ Semantic Space Visualization")
            fig_2d = plot_embeddings_2d(
                embeddings,
                cluster_labels,
                df_sample.reset_index(drop=True)['roi'].values,
                df_sample.reset_index(drop=True)
            )
            st.plotly_chart(fig_2d, width="stretch")
            
            # Representative movies
            representatives = embeddings_analyzer.get_cluster_representative_movies(
                df_sample.reset_index(drop=True),
                cluster_labels,
                embeddings,
                n_per_cluster=3
            )
            
            display_cluster_representatives(representatives, cluster_stats, top_n=5)
            
            # Correlation analysis
            st.markdown("### 🔬 Embedding Dimensions vs ROI")
            
            with st.spinner("Computing dimension correlations..."):
                dim_corr = embeddings_analyzer.correlate_embeddings_with_roi(
                    df_sample.reset_index(drop=True)['roi'],
                    embeddings=embeddings,
                    method='spearman',
                    top_dims=20
                )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_dims = go.Figure()
                colors = ['green' if x > 0 else 'red' for x in dim_corr['correlation']]
                fig_dims.add_trace(go.Bar(
                    x=dim_corr['correlation'],
                    y=[f"Dim {d}" for d in dim_corr['dimension']],
                    orientation='h',
                    marker_color=colors,
                    hovertemplate='<b>Dimension %{y}</b><br>' +
                                 'Correlation: %{x:.3f}<br>' +
                                 '<extra></extra>'
                ))
                fig_dims.update_layout(
                    title='Top 20 Embedding Dimensions by ROI Correlation',
                    xaxis_title='Correlation with ROI',
                    yaxis_title='',
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_dims, width="stretch")
            
            with col2:
                st.markdown("**Key Insights**")
                st.metric("Top Positive Corr.", f"{dim_corr['correlation'].max():.3f}")
                st.metric("Top Negative Corr.", f"{dim_corr['correlation'].min():.3f}")
                st.metric("Significant Dims (p<0.05)", f"{(dim_corr['p_value'] < 0.05).sum()}")
                
                st.markdown("""
                **What this means:**
                - Certain semantic dimensions predict ROI
                - Green bars: Higher values → Higher ROI
                - Red bars: Higher values → Lower ROI
                - Use these dims as ML features!
                """)
        
        else:
            st.info("👆 Click the button above to start the embeddings analysis!")
    
    # Tab 6: Statistics
    with tab6:
        st.subheader("Analysis Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**TF-IDF Matrix**")
            st.write(f"- Shape: {tfidf_matrix.shape}")
            st.write(f"- Features: {tfidf_matrix.shape[1]}")
            st.write(f"- Documents: {tfidf_matrix.shape[0]}")
            sparsity = (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
            st.write(f"- Sparsity: {sparsity:.2f}%")
            
            st.markdown("**Document Statistics**")
            doc_lengths = df_docs['document'].str.len()
            st.write(f"- Avg length: {doc_lengths.mean():.0f} chars")
            st.write(f"- Median length: {doc_lengths.median():.0f} chars")
            
            word_counts = df_docs['document'].str.split().str.len()
            st.write(f"- Avg words: {word_counts.mean():.0f}")
            st.write(f"- Median words: {word_counts.median():.0f}")
        
        with col2:
            st.markdown("**ROI Distribution**")
            fig_roi_dist = px.histogram(
                df_docs,
                x='roi',
                nbins=50,
                title="ROI Distribution in Analyzed Movies"
            )
            st.plotly_chart(fig_roi_dist, width="stretch")
        
        # Sample documents
        with st.expander("📄 View Sample Documents"):
            st.markdown("**Highest ROI Movie**")
            idx_max = df_docs['roi'].idxmax()
            st.write(f"Title: {df_docs.loc[idx_max, 'title']}")
            st.write(f"ROI: {df_docs.loc[idx_max, 'roi']:.2f}")
            st.write(f"Document preview: {df_docs.loc[idx_max, 'document'][:300]}...")
            
            st.markdown("---")
            st.markdown("**Lowest ROI Movie**")
            idx_min = df_docs['roi'].idxmin()
            st.write(f"Title: {df_docs.loc[idx_min, 'title']}")
            st.write(f"ROI: {df_docs.loc[idx_min, 'roi']:.2f}")
            st.write(f"Document preview: {df_docs.loc[idx_min, 'document'][:300]}...")


def show_clustering_page():
    """Display thematic clustering page using UMAP and HDBSCAN."""
    from utils.clustering import (
        get_database_connection, load_embeddings_and_movie_data,
        perform_umap_reduction, perform_hdbscan_clustering,
        analyze_clusters, get_cluster_representative_movies
    )
    
    st.header("🎯 Thematic Clustering Analysis")
    
    st.markdown("""
    This page performs **thematic clustering** of movies based on their overview embeddings.
    We use **UMAP** for dimensionality reduction and **HDBSCAN** for clustering to discover
    thematic groups of movies and analyze their ROI patterns.
    """)
    
    # Check dependencies
    try:
        import umap
        import hdbscan
    except ImportError as e:
        st.error(f"❌ Missing dependency: {str(e)}")
        st.info("Please install required packages: `pip install umap-learn hdbscan`")
        return
    
    # Database connection
    engine = get_database_connection()
    if engine is None:
        st.error("❌ Cannot connect to database.")
        return
    
    # Load data section
    st.subheader("📥 Load Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.number_input(
            "Number of movies to analyze",
            min_value=100,
            max_value=10000,
            value=10000,
            step=100,
            help="Analyzing fewer movies is faster. Start with 1000 for good results."
        )
    
    with col2:
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=1000,
            value=42,
            help="For reproducible results"
        )
    
    if st.button("🔄 Load Embeddings", type="primary"):
        with st.spinner("Loading embeddings from database..."):
            embeddings, df_movies = load_embeddings_and_movie_data(
                engine, sample_size=sample_size, random_seed=random_seed
            )
        
        if embeddings is None or df_movies is None:
            st.error("❌ Failed to load embeddings. Make sure embeddings are generated in the database.")
            return
        
        st.session_state['clustering_embeddings'] = embeddings
        st.session_state['clustering_movies'] = df_movies
        st.success(f"✅ Loaded {len(df_movies):,} movies with embeddings of dimension {embeddings.shape[1]}")
    
    # Check if data is loaded
    if 'clustering_embeddings' not in st.session_state:
        st.info("👆 Click the button above to load embeddings from the database.")
        return
    
    embeddings = st.session_state['clustering_embeddings']
    df_movies = st.session_state['clustering_movies']
    
    # Clustering configuration
    st.subheader("⚙️ Clustering Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**UMAP Parameters**")
        umap_n_neighbors = st.slider(
            "n_neighbors",
            min_value=5,
            max_value=50,
            value=50,
            help="Number of neighbors for UMAP"
        )
        umap_min_dist = st.slider(
            "min_dist",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Minimum distance in UMAP"
        )
        umap_n_components = st.selectbox(
            "Reduction dimensions",
            [2, 3],
            index=0,
            help="2D for visualization, 3D for more detail"
        )
    
    with col2:
        st.markdown("**HDBSCAN Parameters**")
        min_cluster_size = st.slider(
            "min_cluster_size",
            min_value=5,
            max_value=50,
            value=20,
            help="Minimum size of clusters"
        )
        min_samples = st.slider(
            "min_samples",
            min_value=1,
            max_value=20,
            value=1,
            help="Minimum samples in neighborhood"
        )
        cluster_epsilon = st.slider(
            "cluster_selection_epsilon",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05,
            help="Distance threshold for cluster selection"
        )
    
    with col3:
        st.markdown("**Processing Options**")
        use_umap_reduction = st.checkbox(
            "Use UMAP reduction before clustering",
            value=True,
            help="Reduce dimensions with UMAP before clustering (recommended)"
        )
        show_noise = st.checkbox(
            "Show noise points",
            value=True,
            help="Show movies that don't belong to any cluster"
        )
    
    # Perform clustering
    if st.button("🚀 Perform Clustering", type="primary"):
        with st.spinner("Performing UMAP reduction and HDBSCAN clustering..."):
            try:
                # UMAP reduction
                if use_umap_reduction:
                    reduced_embeddings = perform_umap_reduction(
                        embeddings,
                        n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors,
                        min_dist=umap_min_dist,
                        random_state=random_seed
                    )
                    clustering_input = reduced_embeddings
                else:
                    clustering_input = embeddings
                    reduced_embeddings = None
                
                # HDBSCAN clustering
                cluster_labels = perform_hdbscan_clustering(
                    clustering_input,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_epsilon
                )
                
                # Store results
                st.session_state['cluster_labels'] = cluster_labels
                st.session_state['reduced_embeddings'] = reduced_embeddings
                st.session_state['clustering_input'] = clustering_input
                st.session_state['umap_n_components'] = umap_n_components
                st.session_state['use_umap_reduction'] = use_umap_reduction
                
                # Analyze clusters
                cluster_stats = analyze_clusters(df_movies, cluster_labels)
                st.session_state['cluster_stats'] = cluster_stats
                
                n_clusters = len([c for c in np.unique(cluster_labels) if c >= 0])
                n_noise = np.sum(cluster_labels == -1)
                
                st.success(f"✅ Clustering completed! Found {n_clusters} clusters and {n_noise} noise points.")
                
            except Exception as e:
                st.error(f"❌ Error during clustering: {str(e)}")
                return
    
    # Check if clustering is done
    if 'cluster_labels' not in st.session_state:
        st.info("👆 Configure parameters and click 'Perform Clustering' to start.")
        return
    
    cluster_labels = st.session_state['cluster_labels']
    cluster_stats = st.session_state['cluster_stats']
    reduced_embeddings = st.session_state.get('reduced_embeddings')
    clustering_input = st.session_state.get('clustering_input', embeddings)

    # Symbol palette for cluster markers (many distinct shapes)
    # Allowed marker symbols for Plotly 3D scatter are limited; use the intersection
    # of supported symbols to avoid runtime ValueErrors. If you need more shapes,
    # consider using 2D plots or mapping additional visual encodings.
    SYMBOL_SEQUENCE = [
        'circle', 'circle-open', 'cross', 'diamond', 'diamond-open',
        'square', 'square-open', 'x'
    ]

    import colorsys

    def make_hsv_palette(n, s=0.7, v=0.9):
        """Return n hex colors evenly spaced in HSV space."""
        if n <= 0:
            return []
        colors = []
        for i in range(n):
            h = float(i) / float(n)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def _make_color_legend_fig(cluster_ids, color_map):
        import plotly.graph_objects as go
        fig = go.Figure()
        # create one small marker per cluster for legend
        for cid in cluster_ids:
            # color_map keys are strings (cluster labels), so lookup by str(cid)
            fig.add_trace(go.Scatter(
                x=[cid], y=[0], mode='markers',
                marker=dict(color=color_map.get(str(cid), '#888'), size=12),
                name=f"Cluster {cid}",
                showlegend=True
            ))
        fig.update_layout(
            height=60,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            legend=dict(orientation='h')
        )
        return fig

    def _make_shape_legend_fig(cluster_ids, symbol_seq):
        import plotly.graph_objects as go
        fig = go.Figure()
        for i, cid in enumerate(cluster_ids):
            sym = symbol_seq[i % len(symbol_seq)]
            fig.add_trace(go.Scatter(
                x=[i], y=[0], mode='markers',
                marker=dict(color='#444', size=12, symbol=sym),
                name=f"Cluster {cid}",
                showlegend=True
            ))
        fig.update_layout(
            height=60,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            legend=dict(orientation='h')
        )
        return fig
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cluster Statistics",
        "📈 ROI Analysis by Cluster",
        "🗺️ Visualization",
        "🎬 Cluster Representatives"
    ])
    
    # Tab 1: Cluster Statistics
    with tab1:
        st.subheader("📊 Cluster Statistics")
        
        # Summary metrics
        n_clusters = len([c for c in np.unique(cluster_labels) if c >= 0])
        n_noise = np.sum(cluster_labels == -1)
        total_movies = len(cluster_labels)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Clusters", n_clusters)
        with col2:
            st.metric("Noise Points", n_noise)
        with col3:
            st.metric("Clustered Movies", total_movies - n_noise)
        with col4:
            st.metric("Clustering Rate", f"{(total_movies - n_noise) / total_movies * 100:.1f}%")
        
        # Cluster statistics table
        st.markdown("### Cluster Details")
        
        # Format cluster stats for display
        display_stats = cluster_stats.copy()
        if 'top_genres' in display_stats.columns:
            display_stats = display_stats[['cluster_id', 'n_movies', 'roi_mean', 'roi_median', 
                                          'roi_std', 'budget_mean', 'revenue_mean', 
                                          'vote_average_mean', 'top_genres']]
        else:
            display_stats = display_stats[['cluster_id', 'n_movies', 'roi_mean', 'roi_median', 
                                          'roi_std', 'budget_mean', 'revenue_mean', 
                                          'vote_average_mean']]
        
        display_stats = display_stats.sort_values('roi_mean', ascending=False)
        display_stats.columns = ['Cluster ID', 'Movies', 'ROI Mean', 'ROI Median', 
                                'ROI Std', 'Budget Mean', 'Revenue Mean', 
                                'Vote Avg', 'Top Genres'] if 'top_genres' in display_stats.columns else \
                              ['Cluster ID', 'Movies', 'ROI Mean', 'ROI Median', 
                               'ROI Std', 'Budget Mean', 'Revenue Mean', 'Vote Avg']
        
        st.dataframe(
            display_stats.style.format({
                'ROI Mean': '{:.2f}',
                'ROI Median': '{:.2f}',
                'ROI Std': '{:.2f}',
                'Budget Mean': '{:,.0f}',
                'Revenue Mean': '{:,.0f}',
                'Vote Avg': '{:.2f}'
            }),
            width="stretch",
            height=400
        )
    
    # Tab 2: ROI Analysis by Cluster
    with tab2:
        st.subheader("📈 ROI Analysis by Cluster")
        
        # Filter out noise if requested
        display_stats = cluster_stats[cluster_stats['cluster_id'] >= 0] if not show_noise else cluster_stats
        
        # ROI Mean by Cluster
        st.markdown("### Average ROI by Cluster")
        
        fig_roi_mean = px.bar(
            display_stats.sort_values('roi_mean', ascending=True),
            x='roi_mean',
            y='cluster_id',
            orientation='h',
            color='roi_mean',
            color_continuous_scale='RdYlGn',
            labels={'roi_mean': 'Average ROI', 'cluster_id': 'Cluster ID'},
            title='Average ROI by Cluster'
        )
        fig_roi_mean.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_roi_mean.update_layout(height=600)
        st.plotly_chart(fig_roi_mean, width="stretch")
        
        # ROI Distribution by Cluster
        st.markdown("### ROI Distribution by Cluster")
        
        df_with_clusters = df_movies.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        if not show_noise:
            df_with_clusters = df_with_clusters[df_with_clusters['cluster'] >= 0]
        
        fig_roi_dist = px.box(
            df_with_clusters,
            x='cluster',
            y='roi',
            color='cluster',
            color_discrete_sequence=px.colors.qualitative.Dark24,
            labels={'cluster': 'Cluster ID', 'roi': 'ROI'},
            title='ROI Distribution by Cluster'
        )
        fig_roi_dist.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_roi_dist.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_roi_dist, width="stretch")
        
        # Cluster size vs ROI
        st.markdown("### Cluster Size vs Average ROI")

        fig_size_roi = px.scatter(
            display_stats,
            x='n_movies',
            y='roi_mean',
            size='n_movies',
            color='roi_mean',
            color_continuous_scale='RdYlGn',
            # Use different marker shapes per cluster to avoid relying solely on color gradients
            symbol='cluster_id',
            symbol_sequence=SYMBOL_SEQUENCE,
            hover_data=['cluster_id', 'roi_median', 'vote_average_mean'],
            labels={'n_movies': 'Number of Movies', 'roi_mean': 'Average ROI', 'cluster_id': 'Cluster ID'},
            title='Cluster Size vs Average ROI'
        )
        fig_size_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_size_roi.update_layout(height=500)
        st.plotly_chart(fig_size_roi, width="stretch")
        
        # Top and Bottom Clusters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Top 5 Clusters by ROI**")
            top_clusters = display_stats.nlargest(5, 'roi_mean')
            st.dataframe(
                top_clusters[['cluster_id', 'n_movies', 'roi_mean', 'roi_median']].style.format({
                    'roi_mean': '{:.2f}',
                    'roi_median': '{:.2f}'
                }),
                width="stretch"
            )
        
        with col2:
            st.markdown("**📉 Bottom 5 Clusters by ROI**")
            bottom_clusters = display_stats.nsmallest(5, 'roi_mean')
            st.dataframe(
                bottom_clusters[['cluster_id', 'n_movies', 'roi_mean', 'roi_median']].style.format({
                    'roi_mean': '{:.2f}',
                    'roi_median': '{:.2f}'
                }),
                width="stretch"
            )
    
    # Tab 3: Visualization
    with tab3:
        st.subheader("🗺️ Cluster Visualization")
        
        # Get the actual number of components used from session state
        umap_n_components = st.session_state.get('umap_n_components', 2)
        use_umap_reduction = st.session_state.get('use_umap_reduction', True)
        actual_n_components = umap_n_components if use_umap_reduction else 2
        
        if reduced_embeddings is None:
            st.warning("⚠️ UMAP reduction was not used. Using original embeddings for visualization.")
            # Use PCA for visualization if UMAP wasn't used
            from sklearn.decomposition import PCA
            pca = PCA(n_components=actual_n_components, random_state=random_seed)
            vis_embeddings = pca.fit_transform(embeddings)
        else:
            vis_embeddings = reduced_embeddings
        
        # Color by cluster or ROI
        color_by = st.radio(
            "Color by:",
            ["Cluster", "ROI Individual", "ROI Cluster Average"],
            horizontal=True
        )
        
        # Always calculate truncation bounds based on individual ROI values
        roi_individual = df_movies['roi'].values
        
        # Calculate cluster average ROI if needed (but truncation uses individual values)
        if color_by == "ROI Cluster Average":
            # Create a mapping from cluster_id to average ROI
            df_with_clusters = pd.DataFrame({
                'cluster': cluster_labels,
                'roi': roi_individual
            })
            cluster_avg_roi = df_with_clusters.groupby('cluster')['roi'].mean().to_dict()
            # Map each point to its cluster's average ROI
            roi_values_for_color = np.array([cluster_avg_roi.get(c, 0) for c in cluster_labels])
        else:
            roi_values_for_color = roi_individual
        
        # Truncate ROI values for color mapping when coloring by ROI
        # IMPORTANT: Truncation bounds are always calculated from individual ROI values
        if color_by in ["ROI Individual", "ROI Cluster Average"]:
            # Show statistics and allow manual adjustment
            with st.expander("⚙️ Color Scale Settings", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Automatic (Percentile-based)**")
                    st.caption("Bounds calculated from individual ROI values")
                    percentile_method = st.selectbox(
                        "Percentile range",
                        ["1st-99th (most inclusive)", "5th-95th (recommended)", "10th-90th (more restrictive)"],
                        index=1
                    )
                    
                    if percentile_method == "1st-99th (most inclusive)":
                        lower_percentile, upper_percentile = 1, 99
                    elif percentile_method == "5th-95th (recommended)":
                        lower_percentile, upper_percentile = 5, 95
                    else:
                        lower_percentile, upper_percentile = 10, 90
                    
                    # Calculate bounds from INDIVIDUAL ROI values
                    lower_bound = np.percentile(roi_individual, lower_percentile)
                    upper_bound = np.percentile(roi_individual, upper_percentile)
                
                with col2:
                    st.markdown("**Manual Override**")
                    use_manual = st.checkbox("Use manual limits", value=False)
                    if use_manual:
                        min_roi = float(roi_individual.min())
                        max_roi = float(roi_individual.max())
                        
                        lower_bound = st.number_input(
                            "Lower bound",
                            min_value=min_roi,
                            max_value=max_roi,
                            value=float(lower_bound),
                            step=0.1
                        )
                        upper_bound = st.number_input(
                            "Upper bound",
                            min_value=min_roi,
                            max_value=max_roi,
                            value=float(upper_bound),
                            step=0.1
                        )
                
                # Show statistics for both individual and cluster average (if applicable)
                st.markdown("**ROI Statistics (Individual Values):**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min", f"{roi_individual.min():.2f}")
                with col2:
                    st.metric("Max", f"{roi_individual.max():.2f}")
                with col3:
                    st.metric("Median", f"{np.median(roi_individual):.2f}")
                with col4:
                    st.metric("Mean", f"{roi_individual.mean():.2f}")
                
                if color_by == "ROI Cluster Average":
                    st.markdown("**ROI Statistics (Cluster Averages):**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min", f"{roi_values_for_color.min():.2f}")
                    with col2:
                        st.metric("Max", f"{roi_values_for_color.max():.2f}")
                    with col3:
                        st.metric("Median", f"{np.median(roi_values_for_color):.2f}")
                    with col4:
                        st.metric("Mean", f"{roi_values_for_color.mean():.2f}")
                
                st.write(f"**Color scale range:** [{lower_bound:.2f}, {upper_bound:.2f}] (based on individual ROI)")
            
            # Truncate values for color mapping using bounds calculated from individual ROI
            roi_for_color = np.clip(roi_values_for_color, lower_bound, upper_bound)
            n_outliers = np.sum((roi_values_for_color < lower_bound) | (roi_values_for_color > upper_bound))
            
            if n_outliers > 0:
                color_type = "cluster average ROI" if color_by == "ROI Cluster Average" else "ROI"
                st.info(f"📊 Truncated {n_outliers} {color_type} values for color scale (bounds: [{lower_bound:.2f}, {upper_bound:.2f}], calculated from individual ROI)")
        else:
            roi_for_color = roi_values_for_color
        
        # Create visualization dataframe
        if actual_n_components == 3:
            # 3D visualization
            df_viz = pd.DataFrame({
                'x': vis_embeddings[:, 0],
                'y': vis_embeddings[:, 1],
                'z': vis_embeddings[:, 2],
                'cluster': cluster_labels,
                'roi': roi_for_color if color_by in ["ROI Individual", "ROI Cluster Average"] else df_movies['roi'].values,
                'roi_original': df_movies['roi'].values,  # Keep original for hover
                'roi_cluster_avg': roi_for_color if color_by == "ROI Cluster Average" else None,
                'title': df_movies['title'].values,
                'vote_average': df_movies['vote_average'].values
            })
            
            # Filter noise if requested
            if not show_noise:
                df_viz = df_viz[df_viz['cluster'] >= 0]
            
            if color_by == "Cluster":
                # Build HSV palette for cluster colors and a mapping
                cluster_ids = sorted(df_viz['cluster'].unique())
                # Create a categorical label column so Plotly treats clusters as discrete
                df_viz['cluster_label'] = df_viz['cluster'].astype(str)
                # Define explicit category order (string keys) to guarantee deterministic mapping
                cluster_label_order = [str(cid) for cid in cluster_ids]
                df_viz['cluster_label'] = pd.Categorical(df_viz['cluster_label'], categories=cluster_label_order, ordered=True)
                n_cls = len(cluster_ids)
                hsv_colors = make_hsv_palette(n_cls)
                # Color map must have string keys matching the cluster_label values
                color_map = {str(cid): hsv_colors[i] for i, cid in enumerate(cluster_ids)}

                fig = px.scatter_3d(
                    df_viz,
                    x='x',
                    y='y',
                    z='z',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    symbol='cluster_label',
                    symbol_sequence=SYMBOL_SEQUENCE,
                    size='vote_average',
                    hover_data=['title', 'roi_original', 'vote_average'],
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'z': 'UMAP Dimension 3', 'cluster_label': 'Cluster'},
                    title='Movie Clusters in 3D Space (colored by cluster)',
                    category_orders={'cluster_label': cluster_label_order}
                )
            else:
                title_suffix = "by cluster average ROI" if color_by == "ROI Cluster Average" else "by ROI"
                hover_data = ['title', 'cluster', 'vote_average', 'roi_original']
                if color_by == "ROI Cluster Average":
                    hover_data.append('roi_cluster_avg')
                
                fig = px.scatter_3d(
                    df_viz,
                    x='x',
                    y='y',
                    z='z',
                    color='roi',
                    symbol='cluster',
                    symbol_sequence=SYMBOL_SEQUENCE,
                    size='vote_average',
                    hover_data=hover_data,
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'z': 'UMAP Dimension 3', 
                           'roi': 'ROI (truncated)', 'roi_original': 'ROI Individual', 
                           'roi_cluster_avg': 'ROI Cluster Avg'},
                    title=f'Movie Clusters in 3D Space (colored {title_suffix})',
                    color_continuous_scale='RdYlGn'
                )
        else:
            # 2D visualization
            df_viz = pd.DataFrame({
                'x': vis_embeddings[:, 0],
                'y': vis_embeddings[:, 1],
                'cluster': cluster_labels,
                'roi': roi_for_color if color_by in ["ROI Individual", "ROI Cluster Average"] else df_movies['roi'].values,
                'roi_original': df_movies['roi'].values,  # Keep original for hover
                'roi_cluster_avg': roi_for_color if color_by == "ROI Cluster Average" else None,
                'title': df_movies['title'].values,
                'vote_average': df_movies['vote_average'].values
            })
            
            # Filter noise if requested
            if not show_noise:
                df_viz = df_viz[df_viz['cluster'] >= 0]
            
            if color_by == "Cluster":
                # Build HSV palette for cluster colors and a mapping
                cluster_ids = sorted(df_viz['cluster'].unique())
                # Create a categorical label column so Plotly treats clusters as discrete
                df_viz['cluster_label'] = df_viz['cluster'].astype(str)
                # Define explicit category order (string keys) to guarantee deterministic mapping
                cluster_label_order = [str(cid) for cid in cluster_ids]
                df_viz['cluster_label'] = pd.Categorical(df_viz['cluster_label'], categories=cluster_label_order, ordered=True)
                n_cls = len(cluster_ids)
                hsv_colors = make_hsv_palette(n_cls)
                # Color map must have string keys matching the cluster_label values
                color_map = {str(cid): hsv_colors[i] for i, cid in enumerate(cluster_ids)}

                fig = px.scatter(
                    df_viz,
                    x='x',
                    y='y',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    symbol='cluster_label',
                    symbol_sequence=SYMBOL_SEQUENCE,
                    size='vote_average',
                    hover_data=['title', 'roi_original', 'vote_average'],
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'cluster_label': 'Cluster'},
                    title='Movie Clusters in 2D Space (colored by cluster)',
                    category_orders={'cluster_label': cluster_label_order}
                )
            else:
                title_suffix = "by cluster average ROI" if color_by == "ROI Cluster Average" else "by ROI"
                hover_data = ['title', 'cluster', 'vote_average', 'roi_original']
                if color_by == "ROI Cluster Average":
                    hover_data.append('roi_cluster_avg')
                
                fig = px.scatter(
                    df_viz,
                    x='x',
                    y='y',
                    color='roi',
                    symbol='cluster',
                    symbol_sequence=SYMBOL_SEQUENCE,
                    size='vote_average',
                    hover_data=hover_data,
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 
                           'roi': 'ROI (truncated)', 'roi_original': 'ROI Individual',
                           'roi_cluster_avg': 'ROI Cluster Avg'},
                    title=f'Movie Clusters in 2D Space (colored {title_suffix})',
                    color_continuous_scale='RdYlGn'
                )
        
        # Hide the built-in legend on the main chart and show two separate legend panels
        fig.update_layout(height=700, showlegend=False)

        # If we colored by cluster we build legend panels (colors and shapes)
        if color_by == "Cluster":
            try:
                # cluster_ids and color_map were defined above in the Cluster branches
                # If they aren't in scope, fall back to unique values from df_viz
                cluster_ids = cluster_ids if 'cluster_ids' in locals() else sorted(df_viz['cluster'].unique())
                color_map = color_map if 'color_map' in locals() else {cid: c for cid, c in zip(cluster_ids, make_hsv_palette(len(cluster_ids)))}

                color_legend_fig = _make_color_legend_fig(cluster_ids, color_map)
                shape_legend_fig = _make_shape_legend_fig(cluster_ids, SYMBOL_SEQUENCE)

                # Display legends horizontally above the main chart
                lcol, rcol = st.columns(2)
                with lcol:
                    st.markdown("**Color legend**")
                    st.plotly_chart(color_legend_fig, use_container_width=True)
                with rcol:
                    st.markdown("**Shape legend**")
                    st.plotly_chart(shape_legend_fig, use_container_width=True)
            except Exception:
                # If anything goes wrong building the separate legends, fall back to built-in legend
                fig.update_layout(showlegend=True)

        st.plotly_chart(fig, width="stretch")
    
    # Tab 4: Cluster Representatives
    with tab4:
        st.subheader("🎬 Cluster Representatives")
        
        st.markdown("""
        Representative movies for each cluster (closest to cluster centroid).
        These movies best represent the thematic content of their cluster.
        """)
        
        n_representatives = st.slider(
            "Number of representative movies per cluster",
            min_value=1,
            max_value=10,
            value=3
        )
        
        if st.button("🔄 Update Representatives"):
            with st.spinner("Calculating cluster representatives..."):
                representatives = get_cluster_representative_movies(
                    df_movies,
                    cluster_labels,
                    embeddings,
                    n_per_cluster=n_representatives
                )
                st.session_state['cluster_representatives'] = representatives
        
        if 'cluster_representatives' in st.session_state:
            representatives = st.session_state['cluster_representatives']
            
            # Sort clusters by ROI
            sorted_clusters = sorted(
                representatives.keys(),
                key=lambda c: cluster_stats[cluster_stats['cluster_id'] == c]['roi_mean'].values[0] if len(cluster_stats[cluster_stats['cluster_id'] == c]) > 0 else -999,
                reverse=True
            )
            
            for cluster_id in sorted_clusters:
                cluster_info = cluster_stats[cluster_stats['cluster_id'] == cluster_id]
                if len(cluster_info) > 0:
                    roi_mean = cluster_info['roi_mean'].values[0]
                    n_movies = cluster_info['n_movies'].values[0]
                else:
                    roi_mean = 0
                    n_movies = 0
                
                with st.expander(f"Cluster {cluster_id} - ROI: {roi_mean:.2f} ({n_movies} movies)"):
                    for movie in representatives[cluster_id]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{movie['title']}**")
                            if pd.notna(movie.get('overview')):
                                st.write(movie['overview'][:200] + "..." if len(str(movie.get('overview', ''))) > 200 else movie.get('overview', ''))
                        with col2:
                            if pd.notna(movie.get('roi')):
                                st.metric("ROI", f"{movie['roi']:.2f}")
                            if pd.notna(movie.get('vote_average')):
                                st.metric("Rating", f"{movie['vote_average']:.1f}")
                        st.markdown("---")
        else:
            st.info("👆 Click 'Update Representatives' to see representative movies for each cluster.")


if __name__ == "__main__":
    main()
