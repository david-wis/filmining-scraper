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
from models.profitability_trainer import ProfitabilityModelTrainer


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
    
    # Sidebar navigation: top-level pages and per-page sections
    st.sidebar.title("Navigation")
    top_section = st.sidebar.selectbox("Page:", ["Revenue", "Profitability"]) 

    # When in Revenue page show its sections; when in Profitability show profitability sections
    # Revenue sections (legacy)
    revenue_sections = [
        "🏠 Home", "🔮 Predict ROI", "📊 Data Analysis",
        "🎯 Thematic Clustering", "🤖 Model Training", "📈 Model Performance", "🔬 Sensitivity Analysis"
    ]
    # Profitability sections
    profitability_sections = [
        "🏠 Profitability Overview", "🔮 Predict Profitability", "🤖 Train Classifier", "📈 Model Performance", "🔬 Sensitivity Analysis", "🎯 Thematic Clustering"
    ]

    if top_section == "Revenue":
        page = st.sidebar.selectbox("Section:", revenue_sections)
    else:
        page = st.sidebar.selectbox("Section:", profitability_sections)
    # Note: per-plot selectors are used instead of a global sidebar plot selector.
    
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
    if 'profitability_trainer' not in st.session_state:
        st.session_state.profitability_trainer = ProfitabilityModelTrainer()
    
    # Route based on the selected top page and section
    if top_section == "Profitability":
        # profitability pages
        if page == "🏠 Profitability Overview":
            show_profitability_home(data_summary, df_clean)
        elif page == "🔮 Predict Profitability":
            show_profitability_prediction(df_clean, df_genres)
        elif page == "🤖 Train Classifier":
            show_profitability_training(df_clean)
        elif page == "📈 Model Performance":
            show_profitability_performance()
        elif page == "🔬 Sensitivity Analysis":
            show_profitability_sensitivity_page(df_clean, df_genres)
        elif page == "🎯 Thematic Clustering":
            show_profitability_clustering_page()
    else:
        # revenue pages (legacy)
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


def _get_target_column():
    """Return the currently selected target column used by the feature engineer (defaults to 'roi')."""
    try:
        return getattr(st.session_state.feature_engineer, 'target_column', 'roi')
    except Exception:
        return 'roi'


def _get_target_label():
    """Return a human-friendly label for the selected target ('ROI' or 'Revenue')."""
    col = _get_target_column()
    return 'Revenue' if col == 'revenue' else 'ROI'


def _plot_target_selector(scope_name: str = ""):
    """Render a small per-plot selector that overrides the global sidebar plot target.

    Returns (target_column, target_label) where target_column is 'roi' or 'revenue'.
    The selector default is 'Global (use sidebar)'.
    """
    key = f"plot_target_override_{scope_name}"
    choice = st.selectbox(
        "Plot variable (for this chart)",
        ["ROI", "Revenue"],
        index=0,
        key=key,
        help="Choose whether this chart shows ROI or Revenue"
    )
    return ('revenue', 'Revenue') if choice == 'Revenue' else ('roi', 'ROI')
    


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
        target_col = _get_target_column()
        target_label = _get_target_label()
        avg_val = df_clean[target_col].mean() if (df_clean is not None and target_col in df_clean.columns) else data_summary.get('avg_roi', 0)
        st.metric(
            label=f"Average {target_label}",
            value=f"{avg_val:.2f}",
            help=f"Mean {target_label}"
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
        # Target distribution
        if not df_clean.empty:
            tgt = _get_target_column()
            label = _get_target_label()
            fig_target = px.histogram(
                df_clean,
                x=tgt,
                nbins=50,
                title=f"{label} Distribution",
                labels={tgt: label, 'count': 'Number of Movies'}
            )
            fig_target.update_layout(height=400)
            st.plotly_chart(fig_target, width="stretch")
    
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


def show_profitability_home(data_summary, df_clean):
    """Overview page for profitability section."""
    st.header("📊 Profitability Overview")

    # Ensure threshold exists in session
    if 'profitability_threshold' not in st.session_state:
        st.session_state.profitability_threshold = 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", f"{data_summary.get('total_movies', 0):,}")
    with col2:
        st.metric("Profitable Movies (default)", f"{data_summary.get('profitable_movies', 0):,}", delta=f"{data_summary.get('profitability_rate', 0):.1f}%")
    with col3:
        st.metric("Avg ROI", f"{data_summary.get('avg_roi', 0):.2f}")
    with col4:
        thr = st.session_state.profitability_threshold
        st.metric("Current Threshold", f"{thr:.2f}", help="Threshold (ROI) used to label movies as profitable")

    st.markdown(
        "Use the **Train Classifier** page to set a custom profitability threshold and train a classification model to predict whether a movie is profitable."
    )


def show_profitability_prediction(df_clean, df_genres):
    """Prediction page for profitability classification."""
    st.header("🔮 Predict Profitability")

    if not st.session_state.profitability_trainer.is_trained:
        st.warning("⚠️ Classifier not trained yet. Please train the classifier first in the 'Train Classifier' page.")
        return

    st.markdown("Enter movie features to predict probability of being profitable:")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Movie Title", value="My Movie")
        budget = st.number_input("Budget (USD)", min_value=100000, max_value=500000000, value=10000000, step=1000000)
        runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=300, value=120, step=5)
        languages = df_clean['original_language'].value_counts().head(10).index.tolist()
        original_language = st.selectbox("Original Language", languages)
    with col2:
        countries = df_clean['main_country'].value_counts().head(10).index.tolist()
        main_country = st.selectbox("Main Country", countries)
        adult = st.checkbox("Adult Content", value=False)
        all_genres = df_genres['name'].tolist() if df_genres is not None else []
        selected_genres = st.multiselect("Select Genres", all_genres, default=["Drama"])

    if st.button("Predict Profitability", type="primary"):
        movie_data = {
            'title': title,
            'budget': budget,
            'runtime': runtime,
            'release_date': '2024-01-01',
            'original_language': original_language,
            'main_country': main_country,
            'vote_average': 0,
            'vote_count': 0,
            'adult': adult,
            'status': 'Released',
            'genres': ', '.join(selected_genres),
            'production_countries': f'[{{"name": "{main_country}"}}]'
        }
        try:
            X_pred = st.session_state.feature_engineer.create_prediction_features(movie_data)
            prob = st.session_state.profitability_trainer.predict_proba(X_pred)
            label = st.session_state.profitability_trainer.predict_label(X_pred)
            prob_val = float(prob[0]) if prob is not None else None

            if prob_val is not None:
                st.success(f"Probability of being profitable: {prob_val*100:.1f}%")
            st.info(f"Predicted label: {'Profitable' if int(label[0]) == 1 else 'Not Profitable'}")
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")


def show_profitability_training(df_clean):
    """Training UI for profitability classifier."""
    st.header("🤖 Train Profitability Classifier")

    if df_clean is None or df_clean.empty:
        st.error("No data available for training.")
        return

    st.markdown("Set the profitability threshold (ROI) to create binary labels, then train a classifier.")

    col1, col2 = st.columns(2)
    with col1:
        threshold = st.number_input("Profitability threshold (ROI)", value=0.0, step=0.01, format="%.2f")
        optimize_hyperparams = st.checkbox("Optimize Hyperparameters", value=True)
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)

    # store selected threshold in session for display
    st.session_state.profitability_threshold = float(threshold)
    # Feature selection UI (mirror revenue training flow)
    show_features = st.checkbox("Show all features before training", value=False, key="prof_show_features")
    enable_feature_tinkering = st.checkbox("Enable feature tinkering (manually select features)", value=False, key="prof_enable_feature_tinkering")

    # initialize selected features container for profitability if not present
    if 'prof_selected_features' not in st.session_state:
        st.session_state.prof_selected_features = None

    # Prepare df_local (labels) so we can preview features even before training
    df_local = df_clean.copy()
    if 'roi' not in df_local.columns:
        if 'revenue' in df_local.columns and 'budget' in df_local.columns:
            df_local['roi'] = (df_local['revenue'] - df_local['budget']) / df_local['budget']
    df_local['is_profitable'] = (df_local['roi'] > threshold).astype(int)

    if show_features:
        with st.spinner("Preparing features preview..."):
            try:
                df_features = st.session_state.feature_engineer.create_features(df_local)
                X_preview, _, _, _, feature_names = st.session_state.feature_engineer.prepare_modeling_data(
                    df_features, test_size=test_size, random_state=42
                )

                st.success(f"✅ Total features available: **{len(feature_names)}**")

                # Group features by category (same grouping used in revenue training)
                numeric_names = ['budget_per_minute', 'budget_log', 'runtime']
                numeric_features = [f for f in feature_names if f in numeric_names]

                runtime_blocks = [f for f in feature_names if f.startswith('runtime_') and f != 'runtime_category']
                genre_features = [f for f in feature_names if f.startswith('genre_')]
                country_features = [f for f in feature_names if f.startswith('country_')]
                language_features = [f for f in feature_names if f.startswith('language_')]

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

                # Initialize prof_enabled_feature_categories if missing
                if 'prof_enabled_feature_categories' not in st.session_state:
                    enabled = set()
                    # enable defaults if available
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
                    st.session_state.prof_enabled_feature_categories = enabled

                # Render feature group checkboxes
                display_order = [
                    'budget_per_minute', 'budget_log', 'runtime',
                    'runtime_blocks', 'genres', 'countries', 'languages', 'other'
                ]

                for item in display_order:
                    if item == 'budget_per_minute':
                        underlying = [f for f in feature_names if f == 'budget_per_minute']
                        label = 'budget_per_minute'
                    elif item == 'budget_log':
                        underlying = [f for f in feature_names if f == 'budget_log']
                        label = 'budget_log'
                    elif item == 'runtime':
                        underlying = [f for f in feature_names if f == 'runtime']
                        label = 'runtime'
                    elif item == 'runtime_blocks':
                        underlying = feature_categories.get('Runtime Blocks', [])
                        short_names = [f.replace('runtime_', '') for f in underlying]
                        label = f"Runtime Blocks: {', '.join(short_names)}"
                    elif item == 'genres':
                        underlying = feature_categories.get('Genres', [])
                        short_names = [f.replace('genre_', '') for f in underlying]
                        label = f"Genres: {', '.join(short_names)}"
                    elif item == 'countries':
                        underlying = feature_categories.get('Countries', [])
                        short_names = [f.replace('country_', '') for f in underlying]
                        label = f"Countries: {', '.join(short_names)}"
                    elif item == 'languages':
                        underlying = feature_categories.get('Languages', [])
                        short_names = [f.replace('language_', '') for f in underlying]
                        label = f"Languages: {', '.join(short_names)}"
                    else:
                        underlying = feature_categories.get('Other', [])
                        label = 'Other'

                    safe_key = f"prof_feat_{item}"
                    is_enabled = item in st.session_state.prof_enabled_feature_categories
                    checked = st.checkbox(label, value=is_enabled, key=safe_key, disabled=not enable_feature_tinkering)
                    if enable_feature_tinkering:
                        if checked:
                            st.session_state.prof_enabled_feature_categories.add(item)
                        else:
                            st.session_state.prof_enabled_feature_categories.discard(item)

                # Rebuild selected features from enabled items
                selected = set()
                if 'budget_per_minute' in st.session_state.prof_enabled_feature_categories:
                    selected.update([f for f in numeric_features if f == 'budget_per_minute'])
                if 'budget_log' in st.session_state.prof_enabled_feature_categories:
                    selected.update([f for f in numeric_features if f == 'budget_log'])
                if 'runtime' in st.session_state.prof_enabled_feature_categories:
                    selected.update([f for f in numeric_features if f == 'runtime'])
                if 'other' in st.session_state.prof_enabled_feature_categories:
                    selected.update(feature_categories.get('Other', []))
                if 'runtime_blocks' in st.session_state.prof_enabled_feature_categories:
                    selected.update(feature_categories.get('Runtime Blocks', []))
                if 'genres' in st.session_state.prof_enabled_feature_categories:
                    selected.update(feature_categories.get('Genres', []))
                if 'countries' in st.session_state.prof_enabled_feature_categories:
                    selected.update(feature_categories.get('Countries', []))
                if 'languages' in st.session_state.prof_enabled_feature_categories:
                    selected.update(feature_categories.get('Languages', []))

                st.session_state.prof_selected_features = selected

                selected_count = len(st.session_state.prof_selected_features)
                st.metric("Selected Features", f"{selected_count} / {len(feature_names)}")

                if selected_count == 0:
                    st.error("❌ Please select at least one feature to train the classifier.")
                    # don't return; allow user to change selection

            except Exception as e:
                st.error(f"❌ Error preparing features preview: {str(e)}")

    # Training button
    if st.button("🚀 Train Classifier", type="primary"):
        with st.spinner("Preparing data and training classifier..."):
            try:
                # create a copy and set binary target based on threshold
                # (recompute to ensure latest threshold)
                df_local = df_clean.copy()
                if 'roi' not in df_local.columns:
                    if 'revenue' in df_local.columns and 'budget' in df_local.columns:
                        df_local['roi'] = (df_local['revenue'] - df_local['budget']) / df_local['budget']
                df_local['is_profitable'] = (df_local['roi'] > threshold).astype(int)

                # instruct feature engineer to use the new target
                st.session_state.feature_engineer.target_column = 'is_profitable'

                # create features and prepare modeling data
                df_features = st.session_state.feature_engineer.create_features(df_local)
                X_train, X_test, y_train, y_test, feature_names = st.session_state.feature_engineer.prepare_modeling_data(
                    df_features, test_size=test_size, random_state=42
                )

                # Apply feature selection if tinkering enabled and selection provided
                if enable_feature_tinkering and st.session_state.prof_selected_features:
                    selected_features_list = list(st.session_state.prof_selected_features)
                    # filter to only features present
                    valid_selected_features = [f for f in selected_features_list if f in df_features.columns]
                    if len(valid_selected_features) == 0:
                        st.error("❌ None of the selected features are available in the dataset.")
                        return

                    st.info(f"🔍 Training with {len(valid_selected_features)} selected features (out of {len(feature_names)} total)")

                    # prepare unscaled data for selected features and refit scaler
                    X_unscaled = df_features[valid_selected_features].fillna(0).replace([np.inf, -np.inf], 0)
                    y_full = df_features[st.session_state.feature_engineer.target_column]
                    valid_indices = ~y_full.isna()
                    X_unscaled = X_unscaled[valid_indices]
                    y_full = y_full[valid_indices]

                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import StandardScaler

                    X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(
                        X_unscaled, y_full, test_size=test_size, random_state=42
                    )

                    st.session_state.feature_engineer.scaler = StandardScaler()
                    X_train = pd.DataFrame(
                        st.session_state.feature_engineer.scaler.fit_transform(X_train_unscaled),
                        columns=valid_selected_features,
                        index=X_train_unscaled.index
                    )
                    X_test = pd.DataFrame(
                        st.session_state.feature_engineer.scaler.transform(X_test_unscaled),
                        columns=valid_selected_features,
                        index=X_test_unscaled.index
                    )

                    feature_names = valid_selected_features
                    st.session_state.feature_engineer.feature_columns = valid_selected_features

                # train classifier
                metrics = st.session_state.profitability_trainer.train_model(
                    X_train, y_train, X_test, y_test, optimize_hyperparams=optimize_hyperparams
                )

                st.success("✅ Classifier trained successfully!")

                # show key metrics
                st.subheader("📊 Training Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
                with col2:
                    st.metric("Test ROC AUC", f"{metrics.get('test_roc_auc', 0):.3f}" if metrics.get('test_roc_auc') is not None else "N/A")
                with col3:
                    st.metric("Test F1", f"{metrics.get('test_f1', 0):.3f}")

            except Exception as e:
                st.error(f"❌ Error training classifier: {e}")


def show_profitability_performance():
    """Show classifier performance and feature importance."""
    st.header("📈 Profitability Model Performance")

    trainer = st.session_state.profitability_trainer
    if not trainer.is_trained:
        st.warning("⚠️ No classifier trained yet. Train a classifier first.")
        return

    metrics = trainer.training_metrics

    st.subheader("📊 Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.3f}")
    with col2:
        st.metric("Test Precision", f"{metrics.get('test_precision', 0):.3f}")
    with col3:
        st.metric("Test Recall", f"{metrics.get('test_recall', 0):.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test F1", f"{metrics.get('test_f1', 0):.3f}")
    with col2:
        roc = metrics.get('test_roc_auc', None)
        st.metric("ROC AUC", f"{roc:.3f}" if roc is not None else "N/A")

    # Confusion matrix
    cm = metrics.get('confusion_matrix', None)
    if cm is not None:
        st.subheader("Confusion Matrix (Test)")
        cm_df = pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"])
        st.dataframe(cm_df)

    # Feature importance
    st.subheader("🔍 Feature Importance")
    fig = trainer.get_feature_importance_plot()
    if fig:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Feature importance not available.")


def show_profitability_sensitivity_page(df_clean, df_genres):
    """Sensitivity analysis for profitability classifier.

    Shows how the probability of being profitable changes when varying single features
    (budget, runtime, country, language, genre) while keeping others at baseline.
    """
    st.header("🔬 Profitability Sensitivity Analysis")

    trainer = st.session_state.profitability_trainer
    if not trainer.is_trained:
        st.warning("⚠️ Classifier not trained yet. Please train the classifier first in the 'Train Classifier' page.")
        return

    # Baseline movie configuration
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
        baseline_genres = st.multiselect("Baseline Genres", all_genres, default=[all_genres[0]] if all_genres else [])

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
        'genres': ', '.join(baseline_genres) if baseline_genres else '',
        'production_countries': f'[{{"name": "{baseline_country}"}}]'
    }

    st.markdown("This page shows how the *probability* of being profitable changes when varying a single feature.")

    analysis_tabs = st.tabs(["💰 Budget Impact", "⏱️ Runtime Impact", "🌍 Country Impact", "🗣️ Language Impact", "🎭 Genre Impact"])

    # Budget Impact
    with analysis_tabs[0]:
        st.markdown("### How probability of profitability changes with budget")
        budget_min = st.number_input("Min Budget", min_value=100000, value=100000, step=100000, key="prof_budget_min")
        budget_max = st.number_input("Max Budget", min_value=budget_min, value=100000000, step=1000000, key="prof_budget_max")
        budget_steps = st.slider("Number of points", min_value=10, max_value=200, value=100, key="prof_budget_steps")

        budgets = np.linspace(budget_min, budget_max, budget_steps)
        probs = []
        with st.spinner("Calculating probabilities..."):
            for b in budgets:
                movie = baseline_movie.copy()
                movie['budget'] = float(b)
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    p = trainer.predict_proba(X_pred)
                    probs.append(float(p[0]) if p is not None else np.nan)
                except Exception:
                    probs.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=budgets, y=probs, mode='lines+markers', name='Prob(Profitable)'))
        fig.update_layout(title='Probability of Profitability vs Budget', xaxis_title='Budget (USD)', yaxis_title='Probability')
        st.plotly_chart(fig, width='stretch')

        # Summary
        valid = [p for p in probs if p is not None and not (isinstance(p, float) and np.isnan(p))]
        if valid:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Min Prob', f"{min(valid):.3f}")
            with col2:
                st.metric('Max Prob', f"{max(valid):.3f}")
            with col3:
                st.metric('Range', f"{max(valid)-min(valid):.3f}")
        else:
            st.info('No valid predictions produced.')

    # Runtime Impact
    with analysis_tabs[1]:
        st.markdown("### How probability of profitability changes with runtime")
        runtime_min = st.number_input("Min Runtime (minutes)", min_value=45, value=60, step=5, key='prof_runtime_min')
        runtime_max = st.number_input("Max Runtime (minutes)", min_value=runtime_min, value=240, step=5, key='prof_runtime_max')
        runtime_steps = st.slider("Number of points", min_value=10, max_value=100, value=50, key='prof_runtime_steps')

        runtimes = np.linspace(runtime_min, runtime_max, runtime_steps)
        probs = []
        with st.spinner("Calculating probabilities..."):
            for rt in runtimes:
                movie = baseline_movie.copy()
                movie['runtime'] = int(rt)
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    p = trainer.predict_proba(X_pred)
                    probs.append(float(p[0]) if p is not None else np.nan)
                except Exception:
                    probs.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=runtimes, y=probs, mode='lines+markers', name='Prob(Profitable)'))
        fig.update_layout(title='Probability of Profitability vs Runtime', xaxis_title='Runtime (minutes)', yaxis_title='Probability')
        st.plotly_chart(fig, width='stretch')

    # Country Impact
    with analysis_tabs[2]:
        st.markdown("### Probability by Production Country")
        top_n_countries = st.slider("Number of top countries to analyze", min_value=5, max_value=20, value=10, key='prof_country_n')
        countries = df_clean['main_country'].value_counts().head(top_n_countries).index.tolist()
        probs = []
        with st.spinner("Calculating probabilities..."):
            for country in countries:
                movie = baseline_movie.copy()
                movie['main_country'] = country
                movie['production_countries'] = f'[{{"name": "{country}"}}]'
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    p = trainer.predict_proba(X_pred)
                    probs.append(float(p[0]) if p is not None else np.nan)
                except Exception:
                    probs.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=countries, y=probs, name='Prob(Profitable)'))
        fig.update_layout(title='Probability of Profitability by Country', xaxis_title='Country', yaxis_title='Probability', height=500)
        st.plotly_chart(fig, width='stretch')

    # Language Impact
    with analysis_tabs[3]:
        st.markdown("### Probability by Original Language")
        top_n_lang = st.slider("Number of top languages to analyze", min_value=5, max_value=20, value=10, key='prof_lang_n')
        languages = df_clean['original_language'].value_counts().head(top_n_lang).index.tolist()
        probs = []
        with st.spinner("Calculating probabilities..."):
            for lang in languages:
                movie = baseline_movie.copy()
                movie['original_language'] = lang
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    p = trainer.predict_proba(X_pred)
                    probs.append(float(p[0]) if p is not None else np.nan)
                except Exception:
                    probs.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=languages, y=probs, name='Prob(Profitable)'))
        fig.update_layout(title='Probability of Profitability by Language', xaxis_title='Language', yaxis_title='Probability', height=500)
        st.plotly_chart(fig, width='stretch')

    # Genre Impact
    with analysis_tabs[4]:
        st.markdown("### Probability when adding individual genres")
        genres_to_test = df_genres['name'].tolist() if df_genres is not None else []
        probs = []
        labels = []
        with st.spinner("Calculating probabilities..."):
            for genre in genres_to_test:
                movie = baseline_movie.copy()
                movie['genres'] = genre
                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    p = trainer.predict_proba(X_pred)
                    probs.append(float(p[0]) if p is not None else np.nan)
                    labels.append(genre)
                except Exception:
                    probs.append(np.nan)
                    labels.append(genre)

        df_plot = pd.DataFrame({'Genre': labels, 'Prob': probs}).sort_values('Prob', ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_plot['Genre'], y=df_plot['Prob'], name='Prob(Profitable)'))
        fig.update_layout(title='Probability of Profitability by Genre (single genre)', xaxis_title='Genre', yaxis_title='Probability', height=600)
        st.plotly_chart(fig, width='stretch')






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
            
            # Make prediction (the trainer returns predictions in the original units of the trained target)
            pred_val = st.session_state.model_trainer.predict_roi(X_pred)[0]

            target_col = st.session_state.feature_engineer.target_column if hasattr(st.session_state, 'feature_engineer') else 'roi'
            if target_col == 'roi':
                predicted_roi = pred_val
                # Calculate predicted revenue
                predicted_revenue = budget * (1 + predicted_roi)
            else:
                # target is revenue
                predicted_revenue = pred_val
                # compute ROI from predicted revenue if budget available
                predicted_roi = (predicted_revenue - budget) / budget if budget and budget > 0 else np.nan

            predicted_profit = predicted_revenue - budget
            
            # Display results
            st.success("✅ Prediction completed!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Show primary predicted value depending on trained target
                if target_col == 'roi':
                    title = 'Predicted ROI'
                    value_html = f"{predicted_roi:.2f} ({predicted_roi*100:.1f}%)"
                    color = 'green' if predicted_roi > 0 else 'red'
                else:
                    title = 'Predicted Revenue'
                    value_html = f"${predicted_revenue:,.0f}"
                    color = '#1f77b4'

                st.markdown(f"""
                <div class="prediction-card">
                    <h3>{title}</h3>
                    <h2 style="color: {color};">
                        {value_html}
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
    
    # Per-plot target override
    tgt_col, tgt_label = _plot_target_selector("overview")

    # Ensure the dataframe has the requested target column (derive if possible)
    def _ensure_target(df, tgt):
        df = df.copy()
        if tgt == 'revenue' and 'revenue' not in df.columns:
            if 'roi' in df.columns and 'budget' in df.columns:
                df['revenue'] = df['budget'] * (1 + df['roi'])
        if tgt == 'roi' and 'roi' not in df.columns:
            if 'revenue' in df.columns and 'budget' in df.columns:
                # avoid divide by zero
                df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        return df

    df_plot = _ensure_target(df_clean, tgt_col)

    col1, col2 = st.columns(2)
    
    with col1:
        # Target distribution
        tgt = tgt_col
        label = tgt_label
        fig_target = px.histogram(
            df_clean, x=tgt, nbins=50,
            title=f"{label} Distribution",
            labels={tgt: label, 'count': 'Number of Movies'}
        )
        st.plotly_chart(fig_target, width="stretch")
    
    with col2:
        # Budget vs selected target
        fig_budget = px.scatter(
            df_plot, x='budget', y=tgt_col,
            title=f"Budget vs {tgt_label}",
            labels={'budget': 'Budget (USD)', tgt_col: f'{tgt_label} ({"USD" if tgt_col=="revenue" else "ratio"})'},
            opacity=0.6
        )
        fig_budget.update_layout(xaxis_type="log")
        if tgt_col == 'revenue':
            fig_budget.update_layout(yaxis_type="log")
        st.plotly_chart(fig_budget, width="stretch")
    
    # Top movies by selected target
    st.subheader(f"🏆 Top Movies by {tgt_label}")
    tgt = tgt_col
    cols = ['title', 'release_year', 'budget']
    if tgt not in cols:
        cols.append(tgt)
    top_movies = df_plot.nlargest(10, tgt)[cols]
    st.dataframe(top_movies, width="stretch")


def show_financial_analysis(df_clean):
    """Show financial analysis."""
    
    st.subheader("💰 Financial Analysis")
    
    # Per-plot target override
    tgt_col, tgt_label = _plot_target_selector("financial")

    # Ensure target column exists for plotting
    def _ensure_target(df, tgt):
        df = df.copy()
        if tgt == 'revenue' and 'revenue' not in df.columns:
            if 'roi' in df.columns and 'budget' in df.columns:
                df['revenue'] = df['budget'] * (1 + df['roi'])
        if tgt == 'roi' and 'roi' not in df.columns:
            if 'revenue' in df.columns and 'budget' in df.columns:
                df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        return df

    df_plot = _ensure_target(df_clean, tgt_col)

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
        # Selected target distribution
        fig_target = px.box(
            df_plot, y=tgt_col,
            title=f"{tgt_label} Distribution",
            labels={tgt_col: f"{tgt_label} ({'USD' if tgt_col=='revenue' else 'ratio'})"}
        )
        if tgt_col == 'revenue':
            fig_target.update_layout(yaxis_type="log")
        st.plotly_chart(fig_target, width="stretch")
    
    # Financial metrics
    st.subheader("📊 Financial Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Budget", f"${df_clean['budget'].mean():,.0f}")
    with col2:
        # Keep Avg Revenue metric for quick reference if revenue exists
        if 'revenue' in df_clean.columns:
            st.metric("Avg Revenue", f"${df_clean['revenue'].mean():,.0f}")
        else:
            st.metric("Avg Revenue", "N/A")
    with col3:
        tgt = tgt_col
        lbl = tgt_label
        avg_val = df_plot[tgt].mean() if tgt in df_plot.columns else np.nan
        # show appropriately formatted value for revenue vs roi
        if tgt == 'revenue':
            st.metric(f"Avg {lbl}", f"${avg_val:,.0f}")
        else:
            st.metric(f"Avg {lbl}", f"{avg_val:.2f}")
    with col4:
        st.metric("Profitability Rate", f"{df_clean['is_profitable'].mean()*100:.1f}%")


def show_genre_analysis(df_clean):
    """Show genre analysis."""
    
    st.subheader("🎭 Genre Analysis")

    # Per-plot target override
    tgt_col, tgt_label = _plot_target_selector("genre")
    
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
    
    # Genre vs selected target
    # Ensure target available
    def _ensure_target(df, tgt):
        df = df.copy()
        if tgt == 'revenue' and 'revenue' not in df.columns:
            if 'roi' in df.columns and 'budget' in df.columns:
                df['revenue'] = df['budget'] * (1 + df['roi'])
        if tgt == 'roi' and 'roi' not in df.columns:
            if 'revenue' in df.columns and 'budget' in df.columns:
                df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        return df

    df_plot = _ensure_target(df_clean, tgt_col)
    genre_roi = []
    for genre in genre_counts.index:
        genre_movies = df_plot[df_plot['genres'].str.contains(genre, na=False)]
        tgt = tgt_col
        avg_val = genre_movies[tgt].mean() if tgt in genre_movies.columns else np.nan
        genre_roi.append({'genre': genre, 'avg_val': avg_val, 'count': len(genre_movies)})
    
    genre_roi_df = pd.DataFrame(genre_roi)
    genre_roi_df = genre_roi_df[genre_roi_df['count'] >= 10]  # Filter genres with at least 10 movies
    
    # use per-plot label
    tgt_label = tgt_label
    fig_genre_roi = px.bar(
        genre_roi_df, x='avg_val', y='genre',
        orientation='h',
        title=f"Average {tgt_label} by Genre (min 10 movies)",
        labels={'avg_val': f'Average {tgt_label}', 'genre': 'Genre'}
    )
    st.plotly_chart(fig_genre_roi, width="stretch")


def show_temporal_analysis(df_clean):
    """Show temporal analysis."""
    
    st.subheader("📅 Temporal Analysis")

    # Per-plot target override
    tgt_col, tgt_label = _plot_target_selector("temporal")

    def _ensure_target(df, tgt):
        df = df.copy()
        if tgt == 'revenue' and 'revenue' not in df.columns:
            if 'roi' in df.columns and 'budget' in df.columns:
                df['revenue'] = df['budget'] * (1 + df['roi'])
        if tgt == 'roi' and 'roi' not in df.columns:
            if 'revenue' in df.columns and 'budget' in df.columns:
                df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        return df

    df_plot = _ensure_target(df_clean, tgt_col)
    
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
    tgt = tgt_col
    decade_roi = df_plot.groupby('decade')[tgt].mean().reset_index()

    fig_decade = px.bar(
        decade_roi, x='decade', y=tgt,
        title=f"Average {tgt_label} by Decade",
        labels={'decade': 'Decade', tgt: f'Average {tgt_label}'}
    )
    st.plotly_chart(fig_decade, width="stretch")


def show_country_analysis(df_clean):
    """Show country analysis."""
    
    st.subheader("🌍 Country Analysis")

    # Per-plot target override
    tgt_col, tgt_label = _plot_target_selector("country")

    def _ensure_target(df, tgt):
        df = df.copy()
        if tgt == 'revenue' and 'revenue' not in df.columns:
            if 'roi' in df.columns and 'budget' in df.columns:
                df['revenue'] = df['budget'] * (1 + df['roi'])
        if tgt == 'roi' and 'roi' not in df.columns:
            if 'revenue' in df.columns and 'budget' in df.columns:
                df['roi'] = (df['revenue'] - df['budget']) / df['budget']
        return df

    df_plot = _ensure_target(df_clean, tgt_col)
    
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
    
    # Target by country
    tgt = tgt_col
    country_roi = df_plot.groupby('main_country')[tgt].agg(['mean', 'count']).reset_index()
    country_roi = country_roi[country_roi['count'] >= 20]  # Filter countries with at least 20 movies
    country_roi = country_roi.sort_values('mean', ascending=False).head(15)
    tgt_label = _get_target_label()

    fig_country_roi = px.bar(
        country_roi, x='mean', y='main_country',
        orientation='h',
        title=f"Average {tgt_label} by Country (min 20 movies)",
        labels={'mean': f'Average {tgt_label}', 'main_country': 'Country'}
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
        # Target variable selection (ROI or Revenue)
        target_variable = st.selectbox("Target variable", ["ROI", "Revenue"], index=0, help="Choose whether to predict ROI or revenue directly")
        # Truncation percentiles (applies to the selected target)
        roi_lower_pct = st.number_input(
            f"{target_variable} lower percentile (0-100)", min_value=0, max_value=100, value=1, step=1, key="roi_lower_pct"
        )
        roi_upper_pct = st.number_input(
            f"{target_variable} upper percentile (0-100)", min_value=0, max_value=100, value=99, step=1, key="roi_upper_pct"
        )
        # Ensure FeatureEngineer uses the selected target column
        selected_col = 'roi' if target_variable == 'ROI' else 'revenue'
        st.session_state.feature_engineer.target_column = selected_col
        # Target transform selection (applies to the selected target variable)
        target_transform_label = st.selectbox(
            "Target transform",
            [
                'Raw',
                'Signed log (sign * log1p(abs(x)))',
                'Log + shift (log1p(x + shift))',
                'asinh (arcsinh)'
            ],
            index=1,
            help="Choose how to transform the target before training"
        )
        transform_params = {}
        if target_transform_label.startswith('Log + shift'):
            # default shift: ensure positive inside log for min value in selected target
            selected_col = 'roi' if target_variable == 'ROI' else 'revenue'
            min_target = float(df_clean[selected_col].min()) if selected_col in df_clean.columns else 0.0
            default_shift = max(1e-6, -min_target + 1e-6) if min_target <= 0 else 0.0
            shift_val = st.number_input("Shift (added before log1p)", value=float(default_shift), step=0.1)
            transform_params['shift'] = float(shift_val)
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
                        f"{target_variable} truncation applied: {roi_lower_pct}th -> {lower_val:.3f}, {roi_upper_pct}th -> {upper_val:.3f}. "
                        f"Train clipped: below={train_below}, above={train_above}; Test clipped: below={test_below}, above={test_above}."
                    )
                except Exception as e:
                    st.error(f"❌ Error applying {target_variable} truncation: {e}")
                    return
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
                    
                    # Get the unscaled data before subsetting
                    # We need to refit the scaler on only the selected features
                    df_features = st.session_state.feature_engineer.create_features(df_clean)
                    
                    # Get unscaled data for the selected features
                    X_unscaled = df_features[valid_selected_features].fillna(0).replace([np.inf, -np.inf], 0)
                    y_full = df_features[st.session_state.feature_engineer.target_column]
                    
                    # Remove rows with missing target
                    valid_indices = ~y_full.isna()
                    X_unscaled = X_unscaled[valid_indices]
                    y_full = y_full[valid_indices]
                    
                    # Split data with selected features only
                    from sklearn.model_selection import train_test_split
                    X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(
                        X_unscaled, y_full, test_size=test_size, random_state=random_state
                    )
                    
                    # Refit the scaler on selected features only
                    from sklearn.preprocessing import StandardScaler
                    st.session_state.feature_engineer.scaler = StandardScaler()
                    X_train = pd.DataFrame(
                        st.session_state.feature_engineer.scaler.fit_transform(X_train_unscaled),
                        columns=valid_selected_features,
                        index=X_train_unscaled.index
                    )
                    X_test = pd.DataFrame(
                        st.session_state.feature_engineer.scaler.transform(X_test_unscaled),
                        columns=valid_selected_features,
                        index=X_test_unscaled.index
                    )
                    
                    feature_names = valid_selected_features
                    
                    # Update feature_columns to reflect the selected features
                    st.session_state.feature_engineer.feature_columns = valid_selected_features
                else:
                    st.info(f"🔍 Training with all {len(feature_names)} features")
                
                # Map UI label to internal transform key
                label_to_key = {
                    'Raw': 'raw',
                    'Signed log (sign * log1p(abs(x)))': 'signed_log1p',
                    'Log + shift (log1p(x + shift))': 'log_plus_shift',
                    'asinh (arcsinh)': 'asinh'
                }
                selected_transform = label_to_key.get(target_transform_label, 'signed_log1p')

                # record which target name the trainer is using (for UI labeling)
                st.session_state.model_trainer.target_name = selected_col

                # Train model
                metrics = st.session_state.model_trainer.train_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    optimize_hyperparams=optimize_hyperparams,
                    target_transform=selected_transform,
                    transform_params=transform_params or None,
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
    
    tgt_col = _get_target_column()
    tgt_label = _get_target_label()

    st.markdown(f"""
    This page shows how predicted {tgt_label} changes when varying individual features while keeping others constant.
    This helps understand which factors have the most impact on {tgt_label} predictions.
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
        
    # Per-plot target selection (ROI or Revenue)
    plot_choice = st.selectbox("Show plot as", ["ROI", "Revenue"], index=0, key="budget_plot_target")
    plot_col = 'revenue' if plot_choice == 'Revenue' else 'roi'
    plot_label = 'Revenue' if plot_choice == 'Revenue' else 'ROI'

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
                pred_raw = st.session_state.model_trainer.predict_roi(X_pred)[0]
                roi_predictions.append(pred_raw)
            except Exception as e:
                st.error(f"Error predicting for budget {budget}: {str(e)}")
                roi_predictions.append(None)

    # Convert predictions to chosen plot target taking into account how the model was trained
    trained_target = tgt_col  # 'roi' or 'revenue' as used when training
    
    # Debug: show what's happening
    st.write(f"**Debug:** Model trained on: `{trained_target}`, Plotting: `{plot_col}`, Sample raw prediction: `{roi_predictions[0] if roi_predictions else 'N/A'}`")
    
    # raw predictions are in trained_target units
    if trained_target == plot_col:
        # no conversion needed
        pred_vals = [np.nan if p is None else p for p in roi_predictions]
    elif trained_target == 'roi' and plot_col == 'revenue':
        # convert ROI -> revenue per budget point: revenue = budget * (ROI + 1)
        pred_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else (p + 1) * b for p, b in zip(roi_predictions, budgets)]
    elif trained_target == 'revenue' and plot_col == 'roi':
        # convert revenue -> ROI per budget point: ROI = (revenue - budget) / budget
        pred_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else (p - b) / b if b != 0 else np.nan for p, b in zip(roi_predictions, budgets)]
    else:
        pred_vals = [np.nan if p is None else p for p in roi_predictions]
    
    st.write(f"**Debug:** Sample converted value: `{pred_vals[0] if pred_vals else 'N/A'}`")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=budgets, y=pred_vals, mode='lines+markers', name=f'Predicted {plot_label}'))
    if plot_col == 'roi':
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    fig.update_layout(
        title=f"{plot_label} vs Budget",
        xaxis_title="Budget (USD)",
        yaxis_title=f"Predicted {plot_label}",
        height=500
    )
    st.plotly_chart(fig, width="stretch")

    # Summary statistics (ignore failed predictions)
    valid_preds = [p for p in pred_vals if p is not None and not (isinstance(p, float) and np.isnan(p))]
    col1, col2, col3 = st.columns(3)
    if valid_preds:
        with col1:
            if plot_col == 'revenue':
                st.metric(f"Min {plot_label}", f"${min(valid_preds):,.0f}")
            else:
                st.metric(f"Min {plot_label}", f"{min(valid_preds):.2f}")
        with col2:
            if plot_col == 'revenue':
                st.metric(f"Max {plot_label}", f"${max(valid_preds):,.0f}")
            else:
                st.metric(f"Max {plot_label}", f"{max(valid_preds):.2f}")
        with col3:
            if plot_col == 'revenue':
                st.metric(f"{plot_label} Range", f"${max(valid_preds) - min(valid_preds):,.0f}")
            else:
                st.metric(f"{plot_label} Range", f"{max(valid_preds) - min(valid_preds):.2f}")
    else:
        with col1:
            st.metric(f"Min {tgt_label}", "N/A")
        with col2:
            st.metric(f"Max {tgt_label}", "N/A")
            with col3:
                st.metric(f"{tgt_label} Range", "N/A")
    
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
        
        # Per-plot target selection (ROI or Revenue)
        plot_choice = st.selectbox("Show plot as", ["ROI", "Revenue"], index=0, key="runtime_plot_target")
        plot_col = 'revenue' if plot_choice == 'Revenue' else 'roi'
        plot_label = 'Revenue' if plot_choice == 'Revenue' else 'ROI'

        runtimes = np.linspace(runtime_min, runtime_max, runtime_steps)
        roi_preds = []

        with st.spinner("Calculating predictions..."):
            for runtime in runtimes:
                movie = baseline_movie.copy()
                movie['runtime'] = int(runtime)

                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    pred_val = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    roi_preds.append(pred_val)
                except Exception as e:
                    st.error(f"Error predicting for runtime {runtime}: {str(e)}")
                    roi_preds.append(None)

        # Convert to chosen plot target (use baseline budget when converting between ROI and Revenue)
        trained_target = tgt_col
        if trained_target == plot_col:
            plot_vals = [np.nan if p is None else p for p in roi_preds]
        elif trained_target == 'roi' and plot_col == 'revenue':
            b = baseline_budget
            plot_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else p * (b + 1) for p in roi_preds]
        elif trained_target == 'revenue' and plot_col == 'roi':
            b = baseline_budget
            plot_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else (p - b) / b if b != 0 else np.nan for p in roi_preds]
        else:
            plot_vals = [np.nan if p is None else p for p in roi_preds]

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=runtimes, y=plot_vals, mode='lines+markers', name=f'Predicted {plot_label}'))
        if plot_col == 'roi':
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title=f"{plot_label} vs Runtime",
            xaxis_title="Runtime (minutes)",
            yaxis_title=f"Predicted {plot_label}",
            height=500
        )
        st.plotly_chart(fig, width="stretch")

        # Summary statistics (ignore failed predictions)
        valid_preds = [p for p in plot_vals if p is not None and not (isinstance(p, float) and np.isnan(p))]
        col1, col2, col3 = st.columns(3)
        if valid_preds:
            with col1:
                if plot_col == 'revenue':
                    st.metric(f"Min {plot_label}", f"${min(valid_preds):,.0f}")
                else:
                    st.metric(f"Min {plot_label}", f"{min(valid_preds):.2f}")
            with col2:
                if plot_col == 'revenue':
                    st.metric(f"Max {plot_label}", f"${max(valid_preds):,.0f}")
                else:
                    st.metric(f"Max {plot_label}", f"{max(valid_preds):.2f}")
            with col3:
                if plot_col == 'revenue':
                    st.metric(f"{plot_label} Range", f"${max(valid_preds) - min(valid_preds):,.0f}")
                else:
                    st.metric(f"{plot_label} Range", f"{max(valid_preds) - min(valid_preds):.2f}")
        else:
            with col1:
                st.metric(f"Min {plot_label}", "N/A")
            with col2:
                st.metric(f"Max {plot_label}", "N/A")
            with col3:
                st.metric(f"{plot_label} Range", "N/A")
    
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
        
        pred_vals = []

        with st.spinner("Calculating predictions..."):
            for country in countries:
                movie = baseline_movie.copy()
                movie['main_country'] = country
                movie['production_countries'] = f'[{{"name": "{country}"}}]'

                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    pred = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    pred_vals.append(pred)
                except Exception as e:
                    st.error(f"Error predicting for country {country}: {str(e)}")
                    pred_vals.append(None)

        # Per-plot target selection (ROI or Revenue)
        plot_choice = st.selectbox("Show plot as", ["ROI", "Revenue"], index=0, key="country_plot_target")
        plot_col = 'revenue' if plot_choice == 'Revenue' else 'roi'
        plot_label = 'Revenue' if plot_choice == 'Revenue' else 'ROI'

        # Plot (replace None with NaN so plot handles missing values)
        trained_target = tgt_col
        if trained_target == plot_col:
            y_vals = [np.nan if p is None else p for p in pred_vals]
        elif trained_target == 'roi' and plot_col == 'revenue':
            b = baseline_budget
            y_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else p * (b + 1) for p in pred_vals]
        elif trained_target == 'revenue' and plot_col == 'roi':
            b = baseline_budget
            y_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else (p - b) / b if b != 0 else np.nan for p in pred_vals]
        else:
            y_vals = [np.nan if p is None else p for p in pred_vals]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=countries, y=y_vals, name=f'Predicted {plot_label}'))
        if plot_col == 'roi':
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title=f"{plot_label} by Production Country",
            xaxis_title="Country",
            yaxis_title=f"Predicted {plot_label}",
            height=500
        )
        st.plotly_chart(fig, width="stretch")

        # Best and worst (handle missing values)
        country_df = pd.DataFrame({'Country': countries, plot_label: y_vals})
        # Sort by the predicted value (NaNs last)
        country_df = country_df.sort_values(plot_label, ascending=False, na_position='last')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Top 5 Countries by {plot_label}**")
            st.dataframe(country_df.head(5), width="stretch")
        with col2:
            st.markdown(f"**Bottom 5 Countries by {plot_label}**")
            st.dataframe(country_df.tail(5), width="stretch")
    
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

        pred_vals = []

        with st.spinner("Calculating predictions..."):
            for lang in languages_list:
                movie = baseline_movie.copy()
                movie['original_language'] = lang

                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    pred = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    pred_vals.append(pred)
                except Exception as e:
                    st.error(f"Error predicting for language {lang}: {str(e)}")
                    pred_vals.append(None)

        # Per-plot target selection (ROI or Revenue)
        plot_choice = st.selectbox("Show plot as", ["ROI", "Revenue"], index=0, key="language_plot_target")
        plot_col = 'revenue' if plot_choice == 'Revenue' else 'roi'
        plot_label = 'Revenue' if plot_choice == 'Revenue' else 'ROI'

        # Plot
        trained_target = tgt_col
        if trained_target == plot_col:
            y_vals = [np.nan if p is None else p for p in pred_vals]
        elif trained_target == 'roi' and plot_col == 'revenue':
            b = baseline_budget
            y_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else (p * (b + 1)) for p in pred_vals]
        elif trained_target == 'revenue' and plot_col == 'roi':
            b = baseline_budget
            y_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else ((p - b) / b if (b is not None and b != 0) else np.nan) for p in pred_vals]
        else:
            y_vals = [np.nan if p is None else p for p in pred_vals]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=languages_list, y=y_vals, name=f'Predicted {plot_label}'))
        if plot_col == 'roi':
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title=f"{plot_label} by Original Language",
            xaxis_title="Language",
            yaxis_title=f"Predicted {plot_label}",
            height=500
        )
        st.plotly_chart(fig, width="stretch")

        # Best and worst
        language_df = pd.DataFrame({'Language': languages_list, plot_label: y_vals}).sort_values(plot_label, ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Top 5 Languages by {plot_label}**")
            st.dataframe(language_df.head(5), width="stretch")
        with col2:
            st.markdown(f"**Bottom 5 Languages by {plot_label}**")
            st.dataframe(language_df.tail(5), width="stretch")
    
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
        pred_vals = []

        with st.spinner("Calculating predictions..."):
            for genre in genres_to_test:
                movie = baseline_movie.copy()
                movie['genres'] = genre  # Single genre

                try:
                    X_pred = st.session_state.feature_engineer.create_prediction_features(movie)
                    pred = st.session_state.model_trainer.predict_roi(X_pred)[0]
                    pred_vals.append(pred)
                except Exception as e:
                    pred_vals.append(None)

        # Per-plot target selection (ROI or Revenue)
        plot_choice = st.selectbox("Show plot as", ["ROI", "Revenue"], index=0, key="genre_plot_target")
        plot_col = 'revenue' if plot_choice == 'Revenue' else 'roi'
        plot_label = 'Revenue' if plot_choice == 'Revenue' else 'ROI'

        # Plot
        trained_target = tgt_col
        if trained_target == plot_col:
            genre_vals = [np.nan if p is None else p for p in pred_vals]
        elif trained_target == 'roi' and plot_col == 'revenue':
            b = baseline_budget
            genre_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else (p * (b + 1)) for p in pred_vals]
        elif trained_target == 'revenue' and plot_col == 'roi':
            b = baseline_budget
            genre_vals = [np.nan if p is None or (isinstance(p, float) and np.isnan(p)) else ((p - b) / b if (b is not None and b != 0) else np.nan) for p in pred_vals]
        else:
            genre_vals = [np.nan if p is None else p for p in pred_vals]

        genre_df = pd.DataFrame({'Genre': genres_to_test, plot_label: genre_vals}).sort_values(plot_label, ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=genre_df['Genre'], y=genre_df[plot_label], name=f'Predicted {plot_label}'))
        if plot_col == 'roi':
            fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig.update_layout(
            title=f"{plot_label} by Genre (Individual)",
            xaxis_title="Genre",
            yaxis_title=f"Predicted {plot_label}",
            height=500
        )
        st.plotly_chart(fig, width="stretch")

        # Best and worst
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Top 5 Genres by {plot_label}**")
            st.dataframe(genre_df.head(5), width="stretch")
        with col2:
            st.markdown(f"**Bottom 5 Genres by {plot_label}**")
            st.dataframe(genre_df.tail(5), width="stretch")


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
        tgt = _get_target_column()
        lbl = _get_target_label()
        avg_val = df_semantic[tgt].mean() if tgt in df_semantic.columns else df_semantic['roi'].mean()
        if tgt == 'revenue':
            st.metric(f"Avg {lbl}", f"${avg_val:,.0f}")
        else:
            st.metric(f"Avg {lbl}", f"{avg_val:.2f}")
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
            tgt = _get_target_column()
            roi_values = df_docs.loc[df_docs.index, tgt]
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
            tgt = _get_target_column()
            segments = tfidf.analyze_roi_segments(
                df_docs,
                roi_column=tgt,
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
        tgt = _get_target_column()
        tgt_label = _get_target_label()
        roi_threshold = st.slider(
            f"{tgt_label} threshold for high/low split",
            float(df_docs[tgt].min()),
            float(df_docs[tgt].max()),
            float(df_docs[tgt].median()),
            0.1
        )

        high_roi_docs = df_docs[df_docs[tgt] >= roi_threshold]['document']
        low_roi_docs = df_docs[df_docs[tgt] < roi_threshold]['document']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**High {tgt_label} (>= {roi_threshold:.2f})** - {len(high_roi_docs)} movies")
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
            st.markdown(f"**Low {tgt_label} (< {roi_threshold:.2f})** - {len(low_roi_docs)} movies")
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
            
            # Analyze target by cluster
            tgt = _get_target_column()
            cluster_stats = embeddings_analyzer.analyze_roi_by_clusters(
                df_sample.reset_index(drop=True),
                cluster_labels,
                roi_column=tgt
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
                df_sample.reset_index(drop=True)[tgt].values,
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
                    df_sample.reset_index(drop=True)[tgt],
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
            st.markdown(f"**{_get_target_label()} Distribution**")
            tgt = _get_target_column()
            fig_roi_dist = px.histogram(
                df_docs,
                x=tgt,
                nbins=50,
                title=f"{_get_target_label()} Distribution in Analyzed Movies"
            )
            st.plotly_chart(fig_roi_dist, width="stretch")
        
        # Sample documents
        with st.expander("📄 View Sample Documents"):
            st.markdown("**Highest ROI Movie**")
            tgt = _get_target_column()
            idx_max = df_docs[tgt].idxmax()
            st.write(f"Title: {df_docs.loc[idx_max, 'title']}")
            val_max = df_docs.loc[idx_max, tgt]
            if tgt == 'revenue':
                st.write(f"{_get_target_label()}: ${val_max:,.0f}")
            else:
                st.write(f"{_get_target_label()}: {val_max:.2f}")
            st.write(f"Document preview: {df_docs.loc[idx_max, 'document'][:300]}...")
            
            st.markdown("---")
            st.markdown("**Lowest ROI Movie**")
            idx_min = df_docs[tgt].idxmin()
            st.write(f"Title: {df_docs.loc[idx_min, 'title']}")
            val_min = df_docs.loc[idx_min, tgt]
            if tgt == 'revenue':
                st.write(f"{_get_target_label()}: ${val_min:,.0f}")
            else:
                st.write(f"{_get_target_label()}: {val_min:.2f}")
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

    def _make_continuous_colorbar_fig(min_val, max_val, colorscale='Viridis'):
        """Return a small figure that displays a continuous colorbar for a numeric range.

        Used to show cluster-mean profitability color scale next to the shape legend.
        """
        import plotly.graph_objects as go
        # simple 1x2 heatmap to produce a colorbar
        z = [[min_val, max_val]]
        fig = go.Figure(data=go.Heatmap(
            z=z,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(len=0.9, thickness=14, outlinewidth=0)
        ))
        fig.update_layout(
            height=80,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    # Allow user to disable shape encoding across clustering visualizations
    shape_mode = st.selectbox(
        "Shape encoding:",
        ["Enabled (use shapes)", "Disabled (single symbol)"],
        index=0,
        key="clust_shapes_select",
        help="Turn off shapes to use the same marker for all points (useful to avoid busy legends)."
    )
    shapes_enabled = (shape_mode == "Enabled (use shapes)")

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
        # Use dynamic target label (ROI or Revenue) for headings
        tgt_col = _get_target_column()
        tgt_label = _get_target_label()

        has_genres = 'top_genres' in display_stats.columns

        # Always include both ROI and Revenue summary stats so users can compare
        # Build a canonical set of source columns (keep only those present in dataframe)
        canonical_src = [
            'cluster_id', 'n_movies',
            'roi_mean', 'roi_median', 'roi_std',
            'revenue_mean', 'revenue_median', 'revenue_std',
            'budget_mean', 'vote_average_mean'
        ]

        src_cols = [c for c in canonical_src if c in display_stats.columns]

        # Define display names in a sensible order depending on selected target
        if tgt_col == 'revenue':
            # Prioritize revenue columns first, then ROI
            display_names_order = [
                'Cluster ID', 'Movies',
                'Revenue Mean', 'Revenue Median', 'Revenue Std',
                'ROI Mean', 'ROI Median', 'ROI Std',
                'Budget Mean', 'Vote Avg'
            ]
        else:
            # Prioritize ROI columns first, then revenue
            display_names_order = [
                'Cluster ID', 'Movies',
                'ROI Mean', 'ROI Median', 'ROI Std',
                'Revenue Mean', 'Revenue Median', 'Revenue Std',
                'Budget Mean', 'Vote Avg'
            ]

        # Filter display_names_order to only include names that map to available src_cols
        # Build mapping from src_col -> display name
        src_to_name = {
            'cluster_id': 'Cluster ID',
            'n_movies': 'Movies',
            'roi_mean': 'ROI Mean',
            'roi_median': 'ROI Median',
            'roi_std': 'ROI Std',
            'revenue_mean': 'Revenue Mean',
            'revenue_median': 'Revenue Median',
            'revenue_std': 'Revenue Std',
            'budget_mean': 'Budget Mean',
            'vote_average_mean': 'Vote Avg'
        }

        # Build the final ordered list of source columns that exist
        final_src_cols = []
        for disp in display_names_order:
            # find the source column that maps to this display name
            for k, v in src_to_name.items():
                if v == disp and k in src_cols:
                    final_src_cols.append(k)
                    break

        # Append genres column if present
        if has_genres:
            final_src_cols.append('top_genres')
            display_names = [src_to_name.get(c, c) for c in final_src_cols[:-1]] + ['Top Genres']
        else:
            display_names = [src_to_name.get(c, c) for c in final_src_cols]

        # Subset and sort using the primary numeric column (first metric after Cluster ID/Movies)
        sort_key = final_src_cols[2] if len(final_src_cols) > 2 else final_src_cols[0]
        display_stats = display_stats.loc[:, [c for c in final_src_cols if c in display_stats.columns]]
        display_stats = display_stats.sort_values(sort_key, ascending=False)

        display_stats_display = display_stats.copy()
        # Rename columns to human-friendly labels when lengths match
        if len(display_names) == display_stats_display.shape[1]:
            display_stats_display.columns = display_names

        # Build a formatting map for whichever display columns are present
        format_map = {}
        col_names = list(display_stats_display.columns)
        if 'ROI Mean' in col_names:
            format_map['ROI Mean'] = '{:.2f}'
        if 'ROI Median' in col_names:
            format_map['ROI Median'] = '{:.2f}'
        if 'ROI Std' in col_names:
            format_map['ROI Std'] = '{:.2f}'
        if 'Revenue Mean' in col_names:
            format_map['Revenue Mean'] = '{:,.0f}'
        if 'Revenue Median' in col_names:
            format_map['Revenue Median'] = '{:,.0f}'
        if 'Revenue Std' in col_names:
            format_map['Revenue Std'] = '{:,.0f}'
        if 'Budget Mean' in col_names:
            format_map['Budget Mean'] = '{:,.0f}'
        if 'Vote Avg' in col_names:
            format_map['Vote Avg'] = '{:.2f}'

        st.dataframe(
            display_stats_display.style.format(format_map),
            width="stretch",
            height=400
        )
    
    # Tab 2: ROI Analysis by Cluster
    with tab2:
        st.subheader("📈 ROI Analysis by Cluster")
        
        # Filter out noise if requested
        display_stats = cluster_stats[cluster_stats['cluster_id'] >= 0] if not show_noise else cluster_stats
        
        # Target Mean by Cluster
        st.markdown(f"### Average {tgt_label} by Cluster")

        fig_roi_mean = px.bar(
            display_stats.sort_values('roi_mean', ascending=True),
            x='roi_mean',
            y='cluster_id',
            orientation='h',
            color='roi_mean',
            color_continuous_scale='RdYlGn',
            labels={'roi_mean': f'Average {tgt_label}', 'cluster_id': 'Cluster ID'},
            title=f'Average {tgt_label} by Cluster'
        )
        fig_roi_mean.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_roi_mean.update_layout(height=600)
        st.plotly_chart(fig_roi_mean, width="stretch")

        # Target Distribution by Cluster
        st.markdown(f"### {tgt_label} Distribution by Cluster")

        df_with_clusters = df_movies.copy()
        df_with_clusters['cluster'] = cluster_labels

        if not show_noise:
            df_with_clusters = df_with_clusters[df_with_clusters['cluster'] >= 0]

        fig_roi_dist = px.box(
            df_with_clusters,
            x='cluster',
            y=tgt_col,
            color='cluster',
            color_discrete_sequence=px.colors.qualitative.Dark24,
            labels={'cluster': 'Cluster ID', tgt_col: tgt_label},
            title=f'{tgt_label} Distribution by Cluster'
        )
        if tgt_col == 'roi':
            fig_roi_dist.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_roi_dist.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_roi_dist, width="stretch")

        # Cluster size vs Target mean
        st.markdown(f"### Cluster Size vs Average {tgt_label}")

        fig_size_roi = px.scatter(
            display_stats,
            x='n_movies',
            y='roi_mean',
            size='n_movies',
            color='roi_mean',
            color_continuous_scale='RdYlGn',
            # Use different marker shapes per cluster to avoid relying solely on color gradients
            **({'symbol': 'cluster_id', 'symbol_sequence': SYMBOL_SEQUENCE} if shapes_enabled else {}),
            hover_data=['cluster_id', 'roi_median', 'vote_average_mean'],
            labels={'n_movies': 'Number of Movies', 'roi_mean': f'Average {tgt_label}', 'cluster_id': 'Cluster ID'},
            title=f'Cluster Size vs Average {tgt_label}'
        )
        if tgt_col == 'roi':
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
        # prefer persisted values from clustering step; fall back to current UI selections
        persisted_n_comp = st.session_state.get('prof_umap_n_components', None)
        persisted_use_umap = st.session_state.get('prof_use_umap_reduction', None)
        umap_n_components = persisted_n_comp if persisted_n_comp is not None else st.session_state.get('umap_n_components', 2)
        use_umap_reduction = persisted_use_umap if persisted_use_umap is not None else st.session_state.get('use_umap_reduction', True)
        # If reduced embeddings exist, derive actual components from their shape to avoid mismatch
        if reduced_embeddings is not None:
            actual_n_components = int(reduced_embeddings.shape[1])
        else:
            actual_n_components = umap_n_components if use_umap_reduction else 2
        
        if reduced_embeddings is None:
            st.warning("⚠️ UMAP reduction was not used. Using original embeddings for visualization.")
            # Use PCA for visualization if UMAP wasn't used
            from sklearn.decomposition import PCA
            pca = PCA(n_components=actual_n_components, random_state=random_seed)
            vis_embeddings = pca.fit_transform(embeddings)
        else:
            vis_embeddings = reduced_embeddings
        
        # Per-plot target selector for clustering visualization
        tgt_col, tgt_label = _plot_target_selector("clustering")

        # Color by cluster or selected target (Individual / Cluster Average)
        value_individual = f"{tgt_label} Individual"
        value_clusteravg = f"{tgt_label} Cluster Average"
        color_by = st.radio(
            "Color by:",
            ["Cluster", value_individual, value_clusteravg],
            horizontal=True
        )

        # Individual target values from movies
        target_individual = df_movies[tgt_col].values

        # Calculate cluster-average target if needed (but truncation uses individual values)
        if color_by == value_clusteravg:
            df_with_clusters = pd.DataFrame({'cluster': cluster_labels, 'target': target_individual})
            cluster_avg_target = df_with_clusters.groupby('cluster')['target'].mean().to_dict()
            target_values_for_color = np.array([cluster_avg_target.get(c, 0) for c in cluster_labels])
        else:
            target_values_for_color = target_individual

        # Truncate values for color mapping when coloring by the selected target
        if color_by in [value_individual, value_clusteravg]:
            with st.expander("⚙️ Color Scale Settings", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Automatic (Percentile-based)**")
                    st.caption("Bounds calculated from individual values")
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
                    lower_bound = np.percentile(target_individual, lower_percentile)
                    upper_bound = np.percentile(target_individual, upper_percentile)
                with col2:
                    st.markdown("**Manual Override**")
                    use_manual = st.checkbox("Use manual limits", value=False)
                    if use_manual:
                        min_val = float(target_individual.min())
                        max_val = float(target_individual.max())
                        lower_bound = st.number_input("Lower bound", min_value=min_val, max_value=max_val, value=float(lower_bound), step=0.1)
                        upper_bound = st.number_input("Upper bound", min_value=min_val, max_value=max_val, value=float(upper_bound), step=0.1)

                st.markdown(f"**{tgt_label} Statistics (Individual Values):**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min", f"{target_individual.min():.2f}")
                with col2:
                    st.metric("Max", f"{target_individual.max():.2f}")
                with col3:
                    st.metric("Median", f"{np.median(target_individual):.2f}")
                with col4:
                    st.metric("Mean", f"{target_individual.mean():.2f}")

                if color_by == value_clusteravg:
                    st.markdown(f"**{tgt_label} Statistics (Cluster Averages):**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Min", f"{target_values_for_color.min():.2f}")
                    with col2:
                        st.metric("Max", f"{target_values_for_color.max():.2f}")
                    with col3:
                        st.metric("Median", f"{np.median(target_values_for_color):.2f}")
                    with col4:
                        st.metric("Mean", f"{target_values_for_color.mean():.2f}")

                st.write(f"**Color scale range:** [{lower_bound:.2f}, {upper_bound:.2f}] (based on individual values)")

            target_for_color = np.clip(target_values_for_color, lower_bound, upper_bound)
            n_outliers = np.sum((target_values_for_color < lower_bound) | (target_values_for_color > upper_bound))
            if n_outliers > 0:
                color_type = f"cluster average {tgt_label}" if color_by == value_clusteravg else tgt_label
                st.info(f"📊 Truncated {n_outliers} {color_type} values for color scale (bounds: [{lower_bound:.2f}, {upper_bound:.2f}], calculated from individual values)")
        else:
            target_for_color = target_values_for_color
        
        # Create visualization dataframe
        if actual_n_components == 3:
            # 3D visualization
            df_viz = pd.DataFrame({
                'x': vis_embeddings[:, 0],
                'y': vis_embeddings[:, 1],
                'z': vis_embeddings[:, 2],
                'cluster': cluster_labels,
                'target': target_for_color if color_by in [value_individual, value_clusteravg] else df_movies[tgt_col].values,
                'target_original': df_movies[tgt_col].values,  # Keep original for hover
                'target_cluster_avg': target_for_color if color_by == value_clusteravg else None,
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

                # decide whether to encode cluster as shapes or not
                symbol_args = {'symbol': 'cluster_label', 'symbol_sequence': SYMBOL_SEQUENCE} if shapes_enabled else {}

                fig = px.scatter_3d(
                    df_viz,
                    x='x',
                    y='y',
                    z='z',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    size='vote_average',
                    hover_data=['title', 'target_original', 'vote_average'],
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'z': 'UMAP Dimension 3', 'cluster_label': 'Cluster'},
                    title='Movie Clusters in 3D Space (colored by cluster)',
                    category_orders={'cluster_label': cluster_label_order},
                    **symbol_args
                )

                # if shapes disabled, force a single marker symbol
                if not shapes_enabled:
                    try:
                        fig.update_traces(marker_symbol='circle')
                    except Exception:
                        pass
            else:
                title_suffix = f"by cluster average {tgt_label}" if color_by == value_clusteravg else f"by {tgt_label}"
                hover_data = ['title', 'cluster', 'vote_average', 'target_original']
                if color_by == value_clusteravg:
                    hover_data.append('target_cluster_avg')

                # decide whether to encode cluster as shapes or not for target-colored plots
                symbol_args = {'symbol': 'cluster', 'symbol_sequence': SYMBOL_SEQUENCE} if shapes_enabled else {}

                fig = px.scatter_3d(
                    df_viz,
                    x='x',
                    y='y',
                    z='z',
                    color='target',
                    size='vote_average',
                    hover_data=hover_data,
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'z': 'UMAP Dimension 3', 
                           'target': f'{tgt_label} (truncated)', 'target_original': f'{tgt_label} Individual', 
                           'target_cluster_avg': f'{tgt_label} Cluster Avg'},
                    title=f'Movie Clusters in 3D Space (colored {title_suffix})',
                    color_continuous_scale='RdYlGn',
                    **symbol_args
                )

                if not shapes_enabled:
                    try:
                        fig.update_traces(marker_symbol='circle')
                    except Exception:
                        pass
        else:
            # 2D visualization
            df_viz = pd.DataFrame({
                'x': vis_embeddings[:, 0],
                'y': vis_embeddings[:, 1],
                'cluster': cluster_labels,
                'target': target_for_color if color_by in [value_individual, value_clusteravg] else df_movies[tgt_col].values,
                'target_original': df_movies[tgt_col].values,  # Keep original for hover
                'target_cluster_avg': target_for_color if color_by == value_clusteravg else None,
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

                # decide whether to encode cluster as shapes or not
                symbol_args = {'symbol': 'cluster_label', 'symbol_sequence': SYMBOL_SEQUENCE} if shapes_enabled else {}

                fig = px.scatter(
                    df_viz,
                    x='x',
                    y='y',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    size='vote_average',
                    hover_data=['title', 'target_original', 'vote_average'],
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'cluster_label': 'Cluster'},
                    title='Movie Clusters in 2D Space (colored by cluster)',
                    category_orders={'cluster_label': cluster_label_order},
                    **symbol_args
                )

                if not shapes_enabled:
                    try:
                        fig.update_traces(marker_symbol='circle')
                    except Exception:
                        pass
            else:
                title_suffix = f"by cluster average {tgt_label}" if color_by == value_clusteravg else f"by {tgt_label}"
                hover_data = ['title', 'cluster', 'vote_average', 'target_original']
                if color_by == value_clusteravg:
                    hover_data.append('target_cluster_avg')

                symbol_args = {'symbol': 'cluster', 'symbol_sequence': SYMBOL_SEQUENCE} if shapes_enabled else {}

                fig = px.scatter(
                    df_viz,
                    x='x',
                    y='y',
                    color='target',
                    size='vote_average',
                    hover_data=hover_data,
                    labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 
                           'target': f'{tgt_label} (truncated)', 'target_original': f'{tgt_label} Individual',
                           'target_cluster_avg': f'{tgt_label} Cluster Avg'},
                    title=f'Movie Clusters in 2D Space (colored {title_suffix})',
                    color_continuous_scale='RdYlGn',
                    **symbol_args
                )

                if not shapes_enabled:
                    try:
                        fig.update_traces(marker_symbol='circle')
                    except Exception:
                        pass
        
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
                # only prepare shape legend if shapes are enabled
                if shapes_enabled:
                    shape_legend_fig = _make_shape_legend_fig(cluster_ids, SYMBOL_SEQUENCE)

                # Display legends horizontally above the main chart
                if shapes_enabled:
                    lcol, rcol = st.columns(2)
                    with lcol:
                        st.markdown("**Color legend**")
                        st.plotly_chart(color_legend_fig, use_container_width=True)
                    with rcol:
                        st.markdown("**Shape legend**")
                        st.plotly_chart(shape_legend_fig, use_container_width=True)
                else:
                    # show only color legend full-width
                    st.markdown("**Color legend**")
                    st.plotly_chart(color_legend_fig, use_container_width=True)
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


def show_profitability_clustering_page():
    """Thematic clustering focused on profitability.

    This mirrors the main thematic clustering page but computes and displays
    profitability rates (based on ROI threshold) per cluster instead of focusing
    on raw ROI statistics.
    """
    from utils.clustering import (
        get_database_connection, load_embeddings_and_movie_data,
        perform_umap_reduction, perform_hdbscan_clustering,
        analyze_clusters, get_cluster_representative_movies
    )

    st.header("🎯 Thematic Clustering (Profitability)")

    st.markdown("""
    This page clusters movies by their overview embeddings (UMAP + HDBSCAN) and
    reports cluster-level profitability rates (fraction of movies exceeding the
    configured profitability threshold).
    """)

    # Dependency check
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

    # Load data options
    st.subheader("📥 Load Data")
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.number_input("Number of movies to analyze", min_value=100, max_value=10000, value=3000, step=100)
    with col2:
        random_seed = st.number_input("Random seed", min_value=0, max_value=1000, value=42)

    if st.button("🔄 Load Embeddings (Profitability)"):
        with st.spinner("Loading embeddings from database..."):
            embeddings, df_movies = load_embeddings_and_movie_data(engine, sample_size=sample_size, random_seed=random_seed)
        if embeddings is None or df_movies is None:
            st.error("❌ Failed to load embeddings. Make sure embeddings are generated in the database.")
            return
        st.session_state['prof_clustering_embeddings'] = embeddings
        st.session_state['prof_clustering_movies'] = df_movies
        st.success(f"✅ Loaded {len(df_movies):,} movies with embeddings of dimension {embeddings.shape[1]}")

    if 'prof_clustering_embeddings' not in st.session_state:
        st.info("👆 Click the button above to load embeddings from the database.")
        return

    embeddings = st.session_state['prof_clustering_embeddings']
    df_movies = st.session_state['prof_clustering_movies']

    # Ensure ROI & profitability label exist
    if 'roi' not in df_movies.columns:
        if 'revenue' in df_movies.columns and 'budget' in df_movies.columns:
            df_movies['roi'] = (df_movies['revenue'] - df_movies['budget']) / df_movies['budget']

    # Use threshold from session (fallback to 0.0)
    threshold = float(st.session_state.get('profitability_threshold', 0.0))
    df_movies['is_profitable'] = (df_movies['roi'] > threshold).astype(int)

    # Clustering configuration (minimal subset)
    st.subheader("⚙️ Clustering Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        umap_n_neighbors = st.slider("n_neighbors", 5, 50, 30)
        umap_min_dist = st.slider("min_dist", 0.0, 1.0, 0.1, step=0.05)
        umap_n_components = st.selectbox("Reduction dimensions", [2, 3], index=0)
    with col2:
        min_cluster_size = st.slider("min_cluster_size", 5, 100, 20)
        min_samples = st.slider("min_samples", 1, 20, 1)
        cluster_epsilon = st.slider("cluster_selection_epsilon", 0.0, 0.5, 0.0, step=0.05)
    with col3:
        use_umap_reduction = st.checkbox("Use UMAP reduction before clustering", value=True)
        show_noise = st.checkbox("Show noise points", value=True)

    if st.button("🚀 Perform Clustering (Profitability)"):
        with st.spinner("Performing UMAP reduction and HDBSCAN clustering..."):
            try:
                if use_umap_reduction:
                    reduced_embeddings = perform_umap_reduction(
                        embeddings, n_components=umap_n_components,
                        n_neighbors=umap_n_neighbors, min_dist=umap_min_dist,
                        random_state=random_seed
                    )
                    clustering_input = reduced_embeddings
                else:
                    clustering_input = embeddings
                    reduced_embeddings = None

                cluster_labels = perform_hdbscan_clustering(
                    clustering_input,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_epsilon=cluster_epsilon
                )

                # store
                st.session_state['prof_cluster_labels'] = cluster_labels
                st.session_state['prof_reduced_embeddings'] = reduced_embeddings
                st.session_state['prof_clustering_input'] = clustering_input
                # persist UMAP config so visualization knows how many components were produced
                st.session_state['prof_umap_n_components'] = umap_n_components
                st.session_state['prof_use_umap_reduction'] = use_umap_reduction

                # Analyze clusters (base ROI stats)
                cluster_stats = analyze_clusters(df_movies, cluster_labels)

                # Compute profitability rate per cluster and merge
                df_with_clusters = df_movies.copy()
                df_with_clusters['cluster'] = cluster_labels
                profit_rates = df_with_clusters.groupby('cluster')['is_profitable'].mean().reset_index()
                profit_rates.columns = ['cluster_id', 'profitability_rate']

                # Merge (left join) so we keep cluster_stats order and add profitability_rate
                cluster_stats = cluster_stats.merge(profit_rates, on='cluster_id', how='left')
                st.session_state['prof_cluster_stats'] = cluster_stats

                n_clusters = len([c for c in np.unique(cluster_labels) if c >= 0])
                n_noise = np.sum(cluster_labels == -1)
                st.success(f"✅ Clustering completed! Found {n_clusters} clusters and {n_noise} noise points.")
            except Exception as e:
                st.error(f"❌ Error during clustering: {str(e)}")
                return

    if 'prof_cluster_labels' not in st.session_state:
        st.info("👆 Configure parameters and click 'Perform Clustering (Profitability)' to start.")
        return

    cluster_labels = st.session_state['prof_cluster_labels']
    cluster_stats = st.session_state.get('prof_cluster_stats')
    reduced_embeddings = st.session_state.get('prof_reduced_embeddings')
    clustering_input = st.session_state.get('prof_clustering_input', embeddings)

    # Tabs: stats, profitability, viz, representatives
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cluster Statistics",
        "📈 Profitability by Cluster",
        "🗺️ Visualization",
        "🎬 Cluster Representatives"
    ])

    # Tab 1: Cluster Statistics (include profitability_rate)
    with tab1:
        st.subheader("📊 Cluster Statistics (Profitability)")
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

        if cluster_stats is not None:
            display_stats = cluster_stats.copy()
            # Prefer showing profitability_rate and later ROI columns
            cols_priority = [c for c in ['cluster_id', 'n_movies', 'profitability_rate', 'roi_mean', 'roi_median', 'budget_mean', 'vote_average_mean'] if c in display_stats.columns]
            display_stats = display_stats.loc[:, cols_priority]
            # format
            fmt = {}
            if 'profitability_rate' in display_stats.columns:
                fmt['profitability_rate'] = '{:.3f}'
            if 'roi_mean' in display_stats.columns:
                fmt['roi_mean'] = '{:.2f}'
            if 'budget_mean' in display_stats.columns:
                fmt['budget_mean'] = '{:,.0f}'

            st.dataframe(display_stats.style.format(fmt), width='stretch', height=400)

    # Tab 2: Profitability by cluster
    with tab2:
        st.subheader("📈 Profitability by Cluster")
        if cluster_stats is None or 'profitability_rate' not in cluster_stats.columns:
            st.info("Run clustering to compute profitability rates.")
        else:
            display = cluster_stats[cluster_stats['cluster_id'] >= 0].sort_values('profitability_rate', ascending=True)
            fig = px.bar(
                display,
                x='profitability_rate',
                y='cluster_id',
                orientation='h',
                color='profitability_rate',
                color_continuous_scale='Viridis',
                labels={'profitability_rate': 'Profitability Rate', 'cluster_id': 'Cluster ID'},
                title='Profitability Rate by Cluster'
            )
            st.plotly_chart(fig, width='stretch')

            st.markdown("### Cluster Size vs Profitability Rate")
            fig2 = px.scatter(
                display,
                x='n_movies',
                y='profitability_rate',
                size='n_movies',
                color='profitability_rate',
                color_continuous_scale='Viridis',
                hover_data=['cluster_id'],
                labels={'n_movies': 'Number of Movies', 'profitability_rate': 'Profitability Rate'},
                title='Cluster Size vs Profitability Rate'
            )
            st.plotly_chart(fig2, width='stretch')

    # Tab 3: Visualization (select coloring)
    with tab3:
        st.subheader("🗺️ Cluster Visualization (Profitability)")
        umap_n_components = st.session_state.get('umap_n_components', 2)
        use_umap_reduction = st.session_state.get('use_umap_reduction', True)
        # Prefer the actual dimensionality of reduced embeddings (if available).
        # Some flows persist a 3D UMAP result; use its real shape to decide 2D vs 3D plotting.
        if reduced_embeddings is not None and hasattr(reduced_embeddings, 'shape'):
            try:
                actual_n_components = int(reduced_embeddings.shape[1])
            except Exception:
                actual_n_components = umap_n_components if use_umap_reduction else 2
        else:
            actual_n_components = umap_n_components if use_umap_reduction else 2

        if reduced_embeddings is None:
            st.warning("⚠️ UMAP reduction was not used. Using original embeddings for visualization.")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=actual_n_components, random_state=42)
            vis_embeddings = pca.fit_transform(embeddings)
        else:
            vis_embeddings = reduced_embeddings

        # Color options: Cluster, Individual Profitability, Cluster Mean Profitability
        color_by = st.radio(
            "Color by:",
            ["Cluster", "Profitability (Individual)", "Profitability (Cluster Mean)"],
            horizontal=True
        )

        # Shape encoding: allow user to disable shape encoding and use a single symbol
        shape_mode = st.selectbox(
            "Shape encoding:",
            ["Enabled (use shapes)", "Disabled (single symbol)"],
            index=0,
            key="prof_shapes_select",
            help="Turn off shapes to use the same marker for all points (useful to avoid busy legends)."
        )
        shapes_enabled = (shape_mode == "Enabled (use shapes)")

        # Build viz dataframe
        if actual_n_components == 3:
            df_viz = pd.DataFrame({
                'x': vis_embeddings[:, 0], 'y': vis_embeddings[:, 1], 'z': vis_embeddings[:, 2],
                'cluster': cluster_labels,
                'is_profitable': df_movies['is_profitable'].values if 'is_profitable' in df_movies.columns else 0,
                'title': df_movies['title'].values,
                'vote_average': df_movies['vote_average'].values
            })
        else:
            df_viz = pd.DataFrame({
                'x': vis_embeddings[:, 0], 'y': vis_embeddings[:, 1],
                'cluster': cluster_labels,
                'is_profitable': df_movies['is_profitable'].values if 'is_profitable' in df_movies.columns else 0,
                'title': df_movies['title'].values,
                'vote_average': df_movies['vote_average'].values
            })

        if not show_noise:
            df_viz = df_viz[df_viz['cluster'] >= 0]

        # Helper for colors and shapes
        SYMBOL_SEQUENCE = ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']

        def make_hsv_palette(n, s=0.7, v=0.9):
            import colorsys
            if n <= 0:
                return []
            colors = []
            for i in range(n):
                h = float(i) / float(n)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
            return colors

        # Prepare color values depending on selection
        if color_by == "Cluster":
            # Map cluster ids to categorical string labels for Plotly
            df_viz['cluster_label'] = df_viz['cluster'].astype(str)
            cluster_ids = sorted(df_viz['cluster'].unique())
            cluster_label_order = [str(cid) for cid in cluster_ids]
            df_viz['cluster_label'] = pd.Categorical(df_viz['cluster_label'], categories=cluster_label_order, ordered=True)
            n_cls = len(cluster_ids)
            hsv_colors = make_hsv_palette(n_cls)
            color_map = {str(cid): hsv_colors[i] for i, cid in enumerate(cluster_ids)}

            # decide whether to encode cluster as shapes or not
            if shapes_enabled:
                symbol_args = {'symbol': 'cluster_label', 'symbol_sequence': SYMBOL_SEQUENCE}
            else:
                symbol_args = {}

            if actual_n_components == 3:
                fig = px.scatter_3d(
                    df_viz,
                    x='x', y='y', z='z',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    size='vote_average',
                    hover_data=['title'],
                    title='Movie Clusters in 3D Space (colored by cluster)',
                    category_orders={'cluster_label': cluster_label_order},
                    **symbol_args
                )
            else:
                fig = px.scatter(
                    df_viz,
                    x='x', y='y',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    size='vote_average',
                    hover_data=['title'],
                    title='Movie Clusters in 2D Space (colored by cluster)',
                    category_orders={'cluster_label': cluster_label_order},
                    **symbol_args
                )

            # if shapes disabled, force a single marker symbol for all traces
            if not shapes_enabled:
                try:
                    fig.update_traces(marker_symbol='circle')
                except Exception:
                    pass

            # helper legend builders (local copies)
            def _make_color_legend_fig(cluster_ids, color_map):
                import plotly.graph_objects as go
                fig = go.Figure()
                for cid in cluster_ids:
                    fig.add_trace(go.Scatter(
                        x=[cid], y=[0], mode='markers',
                        marker=dict(color=color_map.get(str(cid), '#888'), size=12),
                        name=f"Cluster {cid}", showlegend=True
                    ))
                fig.update_layout(
                    height=60, margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
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
                        name=f"Cluster {cid}", showlegend=True
                    ))
                fig.update_layout(
                    height=60, margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    legend=dict(orientation='h')
                )
                return fig

            def _make_continuous_colorbar_fig(min_val, max_val, colorscale='Viridis'):
                import plotly.graph_objects as go
                z = [[min_val, max_val]]
                fig = go.Figure(data=go.Heatmap(
                    z=z,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(len=0.9, thickness=14, outlinewidth=0)
                ))
                fig.update_layout(
                    height=80, margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False), yaxis=dict(visible=False)
                )
                return fig

            # show separate legends for colors/shapes if possible
            try:
                color_legend_fig = _make_color_legend_fig(cluster_ids, {str(k): v for k, v in color_map.items()})
                shape_legend_fig = _make_shape_legend_fig(cluster_ids, SYMBOL_SEQUENCE)
                lcol, rcol = st.columns(2)
                with lcol:
                    st.markdown("**Color legend**")
                    st.plotly_chart(color_legend_fig, use_container_width=True)
                with rcol:
                    st.markdown("**Shape legend**")
                    st.plotly_chart(shape_legend_fig, use_container_width=True)
            except Exception:
                fig.update_layout(showlegend=True)

        elif color_by == "Profitability (Cluster Mean)":
            # compute mean profitability per cluster and map to each movie
            cluster_map = df_viz.groupby('cluster')['is_profitable'].mean().to_dict()
            df_viz['cluster_mean_profit'] = df_viz['cluster'].map(cluster_map)
            color_col = 'cluster_mean_profit'
            title_suffix = 'cluster mean profitability'

            # prepare categorical cluster labels for shape encoding
            cluster_ids = sorted(df_viz['cluster'].unique())
            cluster_label_order = [str(cid) for cid in cluster_ids]
            df_viz['cluster_label'] = df_viz['cluster'].astype(str)
            df_viz['cluster_label'] = pd.Categorical(df_viz['cluster_label'], categories=cluster_label_order, ordered=True)

            # decide whether to encode cluster as shapes or not
            if shapes_enabled:
                symbol_args = {'symbol': 'cluster_label', 'symbol_sequence': SYMBOL_SEQUENCE}
            else:
                symbol_args = {}

            if actual_n_components == 3:
                fig = px.scatter_3d(
                    df_viz, x='x', y='y', z='z', color=color_col,
                    size='vote_average', hover_data=['title', 'cluster'], color_continuous_scale='Viridis',
                    title=f'Clusters colored by {title_suffix}', category_orders={'cluster_label': cluster_label_order},
                    **symbol_args
                )
            else:
                fig = px.scatter(
                    df_viz, x='x', y='y', color=color_col, size='vote_average',
                    hover_data=['title', 'cluster'], color_continuous_scale='Viridis',
                    title=f'Clusters colored by {title_suffix}', category_orders={'cluster_label': cluster_label_order},
                    **symbol_args
                )

            # Hide in-plot legends for symbol traces so the continuous colorbar remains
            try:
                # if shapes disabled we don't need symbol legends
                if shapes_enabled:
                    fig.update_traces(showlegend=False)
                else:
                    # force a single marker symbol for all traces
                    fig.update_traces(marker_symbol='circle')
            except Exception:
                pass

        else:
            # Profitability (Individual): color by cluster, shape by profitable/not profitable
            title_suffix = 'profitability (individual 0/1)'

            # prepare categorical cluster labels for color encoding
            cluster_ids = sorted(df_viz['cluster'].unique())
            cluster_label_order = [str(cid) for cid in cluster_ids]
            df_viz['cluster_label'] = df_viz['cluster'].astype(str)
            df_viz['cluster_label'] = pd.Categorical(df_viz['cluster_label'], categories=cluster_label_order, ordered=True)

            # build discrete color map per cluster (reuse HSV palette)
            n_cls = len(cluster_ids)
            hsv_colors = make_hsv_palette(n_cls)
            color_map = {str(cid): hsv_colors[i] for i, cid in enumerate(cluster_ids)}

            # prepare profitable/not profitable shape labels
            df_viz['is_profitable_label'] = df_viz['is_profitable'].apply(lambda v: 'Profitable' if int(v) == 1 else 'Not Profitable')
            profit_order = ['Not Profitable', 'Profitable']
            df_viz['is_profitable_label'] = pd.Categorical(df_viz['is_profitable_label'], categories=profit_order, ordered=True)

            # symbol sequence for profit labels (keep two distinct symbols)
            PROF_SYMBOLS = ['x', 'circle']

            # decide whether to encode profitability as shapes or not
            if shapes_enabled:
                profit_symbol_args = {'symbol': 'is_profitable_label', 'symbol_sequence': PROF_SYMBOLS}
            else:
                profit_symbol_args = {}

            if actual_n_components == 3:
                fig = px.scatter_3d(
                    df_viz,
                    x='x', y='y', z='z',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    size='vote_average',
                    hover_data=['title', 'cluster', 'is_profitable_label'],
                    title=f'Clusters colored by cluster (shapes = {title_suffix})',
                    category_orders={'cluster_label': cluster_label_order, 'is_profitable_label': profit_order},
                    **profit_symbol_args
                )
            else:
                fig = px.scatter(
                    df_viz,
                    x='x', y='y',
                    color='cluster_label',
                    color_discrete_map=color_map,
                    size='vote_average',
                    hover_data=['title', 'cluster', 'is_profitable_label'],
                    title=f'Clusters colored by cluster (shapes = {title_suffix})',
                    category_orders={'cluster_label': cluster_label_order, 'is_profitable_label': profit_order},
                    **profit_symbol_args
                )

            # if shapes disabled force a single marker for all traces
            if not shapes_enabled:
                try:
                    fig.update_traces(marker_symbol='circle')
                except Exception:
                    pass

        # Display small legends side-by-side to avoid overlap between color and shape legends
        try:
            if color_by == "Profitability (Cluster Mean)":
                # show shape legend for clusters and a compact colorbar info panel
                try:
                    # build only the external shape legend (clusters -> symbol). The
                    # continuous colorbar for cluster mean profitability is left inside
                    # the main plot itself to avoid duplicating/overlapping legends.
                    shape_legend_fig = _make_shape_legend_fig(cluster_ids, SYMBOL_SEQUENCE)

                    # Render the external shape legend on the page (outside the plot)
                    # — show it full-width above the plot so it is always visible.
                    st.markdown("**Shape legend (clusters)**")
                    st.plotly_chart(shape_legend_fig, use_container_width=True)
                except Exception:
                    pass

            elif color_by == "Profitability (Individual)":
                # show discrete color legend for clusters and shape legend for profit labels side-by-side
                try:
                    color_legend_fig = _make_color_legend_fig(cluster_ids, {str(k): v for k, v in color_map.items()})
                    # build profit-shape legend only if shapes are enabled
                    def _make_profit_shape_legend(profit_order, prof_symbols):
                        import plotly.graph_objects as go
                        figp = go.Figure()
                        for i, label in enumerate(profit_order):
                            sym = prof_symbols[i % len(prof_symbols)]
                            figp.add_trace(go.Scatter(
                                x=[i], y=[0], mode='markers',
                                marker=dict(color='#444', size=12, symbol=sym),
                                name=label, showlegend=True
                            ))
                        figp.update_layout(
                            height=60, margin=dict(l=0, r=0, t=0, b=0),
                            xaxis=dict(visible=False), yaxis=dict(visible=False),
                            legend=dict(orientation='h')
                        )
                        return figp

                    if shapes_enabled:
                        shape_legend_fig = _make_profit_shape_legend(profit_order, PROF_SYMBOLS)
                        lcol, rcol = st.columns(2)
                        with lcol:
                            st.markdown("**Color legend (clusters)**")
                            st.plotly_chart(color_legend_fig, use_container_width=True)
                        with rcol:
                            st.markdown("**Shape legend (profitability)**")
                            st.plotly_chart(shape_legend_fig, use_container_width=True)
                    else:
                        # only show color legend when shapes disabled
                        st.markdown("**Color legend (clusters)**")
                        st.plotly_chart(color_legend_fig, use_container_width=True)
                except Exception:
                    pass
        except Exception:
            # fail silently if legends cannot be built
            pass

        # Render the main figure in the previously reserved left column if present,
        # otherwise render normally. This ensures the external shape legend (right
        # column) appears beside the plot rather than overlapping it.
        try:
            # if we created columns above, render into the left column
            if 'lcol' in locals():
                with lcol:
                    fig.update_layout(height=700, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig.update_layout(height=700, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, width='stretch')
        except Exception:
            # fallback
            fig.update_layout(height=700, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width='stretch')

    # Tab 4: Representatives
    with tab4:
        st.subheader("🎬 Cluster Representatives (Profitability)")
        n_representatives = st.slider("Number of representative movies per cluster", 1, 10, 3)
        if st.button("🔄 Update Representatives (Profitability)"):
            with st.spinner("Calculating cluster representatives..."):
                representatives = get_cluster_representative_movies(df_movies, cluster_labels, embeddings, n_per_cluster=n_representatives)
                st.session_state['prof_cluster_representatives'] = representatives

        if 'prof_cluster_representatives' in st.session_state:
            representatives = st.session_state['prof_cluster_representatives']
            sorted_clusters = sorted(representatives.keys())
            for cluster_id in sorted_clusters:
                cluster_info = cluster_stats[cluster_stats['cluster_id'] == cluster_id]
                roi_mean = float(cluster_info['roi_mean'].values[0]) if len(cluster_info) > 0 and 'roi_mean' in cluster_info.columns else 0.0
                n_movies = int(cluster_info['n_movies'].values[0]) if len(cluster_info) > 0 else 0
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
            st.info("👆 Click 'Update Representatives (Profitability)' to see representative movies for each cluster.")


if __name__ == "__main__":
    main()
