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
        ["🏠 Home", "🔮 Predict ROI", "📊 Data Analysis", "🤖 Model Training", "📈 Model Performance"]
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
    elif page == "🤖 Model Training":
        show_model_training_page(df_clean)
    elif page == "📈 Model Performance":
        show_model_performance_page()


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
            st.plotly_chart(fig_genres, use_container_width=True)
    
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
            st.plotly_chart(fig_roi, use_container_width=True)
    
    # Instructions
    st.header("🚀 How to Use This App")
    
    st.markdown("""
    This application helps you predict the Return on Investment (ROI) of movies using machine learning. Here's how to get started:
    
    1. **🔮 Predict ROI**: Use the prediction page to input movie characteristics and get ROI predictions
    2. **📊 Data Analysis**: Explore the dataset with interactive visualizations
    3. **🤖 Model Training**: Train or retrain the Random Forest model with different parameters
    4. **📈 Model Performance**: View model metrics and feature importance
    
    The model uses features like budget, runtime, genres, country, language, and ratings to predict ROI.
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
        
        # Release information
        release_year = st.number_input("Release Year", min_value=1900, max_value=2030, value=2024)
        release_month = st.selectbox("Release Month", range(1, 13), index=0)
        
        # Language and country
        languages = df_clean['original_language'].value_counts().head(10).index.tolist()
        original_language = st.selectbox("Original Language", languages)
        
        countries = df_clean['main_country'].value_counts().head(10).index.tolist()
        main_country = st.selectbox("Main Country", countries)
        
    with col2:
        # Ratings and content
        st.subheader("⭐ Ratings & Content")
        
        vote_average = st.slider("Vote Average (0-10)", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
        vote_count = st.number_input("Vote Count", min_value=0, max_value=50000, value=1000, step=100)
        
        adult = st.checkbox("Adult Content", value=False)
        status = st.selectbox("Status", ["Released", "Post Production", "In Production", "Planned"])
        
        # Genres
        st.subheader("🎭 Genres")
        all_genres = df_genres['name'].tolist() if df_genres is not None else []
        selected_genres = st.multiselect("Select Genres", all_genres, default=["Drama"])
    
    # Create prediction
    if st.button("🔮 Predict ROI", type="primary"):
        
        # Prepare movie data
        movie_data = {
            'title': title,
            'budget': budget,
            'runtime': runtime,
            'release_date': f"{release_year}-{release_month:02d}-01",
            'original_language': original_language,
            'main_country': main_country,
            'vote_average': vote_average,
            'vote_count': vote_count,
            'adult': adult,
            'status': status,
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
        st.plotly_chart(fig_roi, use_container_width=True)
    
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
        st.plotly_chart(fig_budget, use_container_width=True)
    
    # Top movies by ROI
    st.subheader("🏆 Top Movies by ROI")
    top_movies = df_clean.nlargest(10, 'roi')[['title', 'release_year', 'budget', 'revenue', 'roi']]
    st.dataframe(top_movies, use_container_width=True)


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
        st.plotly_chart(fig_budget, use_container_width=True)
    
    with col2:
        # Revenue distribution
        fig_revenue = px.box(
            df_clean, y='revenue',
            title="Revenue Distribution",
            labels={'revenue': 'Revenue (USD)'}
        )
        fig_revenue.update_layout(yaxis_type="log")
        st.plotly_chart(fig_revenue, use_container_width=True)
    
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
    st.plotly_chart(fig_genres, use_container_width=True)
    
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
    st.plotly_chart(fig_genre_roi, use_container_width=True)


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
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    # ROI by decade
    df_clean['decade'] = (df_clean['release_year'] // 10) * 10
    decade_roi = df_clean.groupby('decade')['roi'].mean().reset_index()
    
    fig_decade = px.bar(
        decade_roi, x='decade', y='roi',
        title="Average ROI by Decade",
        labels={'decade': 'Decade', 'roi': 'Average ROI'}
    )
    st.plotly_chart(fig_decade, use_container_width=True)


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
    st.plotly_chart(fig_countries, use_container_width=True)
    
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
    st.plotly_chart(fig_country_roi, use_container_width=True)


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
                    st.plotly_chart(importance_plot, use_container_width=True)
                
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
        st.plotly_chart(importance_plot, use_container_width=True)
    
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


if __name__ == "__main__":
    main()
