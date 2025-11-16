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
        ["🏠 Home", "🔮 Predict ROI", "📊 Data Analysis", "📝 Semantic Analysis", "🤖 Model Training", "📈 Model Performance", "🔬 Sensitivity Analysis"]
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
    
    if st.button("🚀 Train Model", type="primary"):
        
        with st.spinner("Training model..."):
            try:
                # Prepare features
                df_features = st.session_state.feature_engineer.create_features(df_clean)
                X_train, X_test, y_train, y_test, feature_names = st.session_state.feature_engineer.prepare_modeling_data(
                    df_features, test_size=test_size, random_state=random_state
                )
                
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
        budget_min = st.number_input("Min Budget", min_value=100000, value=1000000, step=100000, key="budget_min")
        budget_max = st.number_input("Max Budget", min_value=budget_min, value=100000000, step=1000000, key="budget_max")
        budget_steps = st.slider("Number of points", min_value=10, max_value=100, value=50, key="budget_steps")
        
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
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min ROI", f"{min(roi_predictions):.2f}")
        with col2:
            st.metric("Max ROI", f"{max(roi_predictions):.2f}")
        with col3:
            st.metric("ROI Range", f"{max(roi_predictions) - min(roi_predictions):.2f}")
    
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
        
        runtime_min = st.number_input("Min Runtime (minutes)", min_value=60, value=80, step=5, key="runtime_min")
        runtime_max = st.number_input("Max Runtime (minutes)", min_value=runtime_min, value=180, step=5, key="runtime_max")
        runtime_steps = st.slider("Number of points", min_value=10, max_value=50, value=25, key="runtime_steps")
        
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
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min ROI", f"{min(roi_predictions):.2f}")
        with col2:
            st.metric("Max ROI", f"{max(roi_predictions):.2f}")
        with col3:
            st.metric("ROI Range", f"{max(roi_predictions) - min(roi_predictions):.2f}")
    
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
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=countries, y=roi_predictions, name='Predicted ROI'))
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title="ROI by Production Country",
            xaxis_title="Country",
            yaxis_title="Predicted ROI",
            height=500
        )
        st.plotly_chart(fig, width="stretch")
        
        # Best and worst
        country_roi_df = pd.DataFrame({'Country': countries, 'ROI': roi_predictions}).sort_values('ROI', ascending=False)
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


if __name__ == "__main__":
    main()
