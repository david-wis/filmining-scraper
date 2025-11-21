"""
Feature engineering module for the Streamlit ROI prediction app.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class FeatureEngineer:
    """
    Feature engineering class for movie ROI prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'roi'
        
    def create_features(self, df):
        """
        Create engineered features from raw movie data.
        
        Args:
            df (pd.DataFrame): Raw movie data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        df_features = df.copy()
        
        # 1. Temporal features
        df_features['release_year'] = pd.to_datetime(df_features['release_date']).dt.year
        df_features['release_month'] = pd.to_datetime(df_features['release_date']).dt.month
        df_features['release_quarter'] = pd.to_datetime(df_features['release_date']).dt.quarter
        df_features['release_decade'] = (df_features['release_year'] // 10) * 10
        
        # 2. Budget and revenue features
        df_features['budget_log'] = np.log1p(df_features['budget'])
        # Only create revenue_log if revenue column exists (for training data)
        if 'revenue' in df_features.columns:
            df_features['revenue_log'] = np.log1p(df_features['revenue'])
        else:
            # For prediction data, set revenue_log to 0 (will be ignored in prediction)
            df_features['revenue_log'] = 0
        # Avoid division by zero and handle infinite values
        df_features['budget_per_minute'] = np.where(
            df_features['runtime'] > 0,
            df_features['budget'] / df_features['runtime'],
            0
        )
        # Replace infinite values with 0
        df_features['budget_per_minute'] = df_features['budget_per_minute'].replace([np.inf, -np.inf], 0)
        
        # 3. Rating and popularity features
        df_features['vote_confidence'] = np.log1p(df_features['vote_count'])
        df_features['rating_popularity_score'] = df_features['vote_average'] * df_features['vote_confidence']
        
        # 4. Genre features (one-hot encoding for top genres)
        top_genres = self._get_top_genres(df_features)
        for genre in top_genres:
            df_features[f'genre_{genre.lower().replace(" ", "_")}'] = df_features['genres'].str.contains(genre, na=False).astype(int)
        
        # 5. Country features
        top_countries = self._get_top_countries(df_features)
        for country in top_countries:
            df_features[f'country_{country.lower().replace(" ", "_")}'] = (df_features['main_country'] == country).astype(int)
        
        # 6. Language features
        top_languages = self._get_top_languages(df_features)
        for lang in top_languages:
            df_features[f'language_{lang}'] = (df_features['original_language'] == lang).astype(int)
        
        # 7. Status features
        df_features['status_released'] = (df_features['status'] == 'Released').astype(int)
        df_features['status_post_production'] = (df_features['status'] == 'Post Production').astype(int)
        
        # 8. Adult content feature
        df_features['is_adult'] = df_features['adult'].astype(int)
        
        # 9. Runtime categories
        df_features['runtime_category'] = pd.cut(
            df_features['runtime'], 
            bins=[0, 90, 120, 150, float('inf')], 
            labels=['short', 'medium', 'long', 'very_long']
        )

        # 9b. Runtime binary windows centered every 30 minutes
        # Create binary features covering the runtime distribution with windows
        # centered on 30, 60, 90, ... minutes. Each window spans [center-15, center+15)
        # The last window uses <= upper bound so the max runtime is included.
        try:
            runtimes = df_features['runtime'].dropna().astype(float)
            if not runtimes.empty:
                max_rt = float(runtimes.max())
                # create centers at 30, 60, ... up to ceil(max_rt/30)*30
                last_center = int(np.ceil(max_rt / 30.0) * 30)
                centers = list(range(30, last_center + 1, 30))

                for i, center in enumerate(centers):
                    lower = center - 15
                    upper = center + 15
                    col_name = f"runtime_{int(lower)}_{int(upper)}"
                    if i == len(centers) - 1:
                        # include the upper bound in the last window
                        df_features[col_name] = ((df_features['runtime'] >= lower) & (df_features['runtime'] <= upper)).astype(int)
                    else:
                        df_features[col_name] = ((df_features['runtime'] >= lower) & (df_features['runtime'] < upper)).astype(int)
            else:
                # No runtime info available; create a default window to avoid downstream failures
                df_features['runtime_15_45'] = 0
        except Exception:
            # On any unexpected error, add a safe default column
            df_features['runtime_15_45'] = 0
        
        # 10. Budget categories
        df_features['budget_category'] = pd.cut(
            df_features['budget'], 
            bins=[0, 1000000, 10000000, 50000000, float('inf')], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return df_features
    
    def _get_top_genres(self, df, top_n=10):
        """Get top N genres by frequency."""
        all_genres = []
        for genres_str in df['genres'].dropna():
            if isinstance(genres_str, str):
                all_genres.extend([g.strip() for g in genres_str.split(',')])
        
        genre_counts = pd.Series(all_genres).value_counts()
        return genre_counts.head(top_n).index.tolist()
    
    def _get_top_countries(self, df, top_n=10):
        """Get top N countries by frequency."""
        return df['main_country'].value_counts().head(top_n).index.tolist()
    
    def _get_top_languages(self, df, top_n=10):
        """Get top N languages by frequency."""
        return df['original_language'].value_counts().head(top_n).index.tolist()
    
    def prepare_modeling_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data for machine learning modeling.
        
        Args:
            df (pd.DataFrame): Data with engineered features
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        # Select numerical features for modeling
        # Note: Excluded post-release features (release_year, vote_average, vote_count, status)
        # as they are not known at production time
        # Also excluded revenue_log to avoid data leakage (revenue is what we're predicting)
        numerical_features = [
            'budget_log', 'budget_per_minute', 'runtime',
            'is_adult'
        ]
        
        # Add genre features
        genre_features = [col for col in df.columns if col.startswith('genre_')]
        country_features = [col for col in df.columns if col.startswith('country_')]
        language_features = [col for col in df.columns if col.startswith('language_')]
        
        # Combine all features
        # Pick up runtime binary windows but exclude the categorical 'runtime_category'
        runtime_binary_features = [col for col in df.columns if col.startswith('runtime_') and col != 'runtime_category']
        # Combine all features (include runtime binary windows)
        feature_columns = numerical_features + genre_features + country_features + language_features + runtime_binary_features
        
        # Filter features that exist in the dataframe
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Prepare feature matrix
        X = df[feature_columns].fillna(0)
        
        # Replace infinite values with 0
        X = X.replace([np.inf, -np.inf], 0)
        
        # Ensure all values are finite
        X = X.fillna(0)
        
        y = df[self.target_column]
        
        # Remove rows with missing target
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)
        
        self.feature_columns = feature_columns
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def create_prediction_features(self, movie_data):
        """
        Create features for a single movie prediction.
        
        Args:
            movie_data (dict): Dictionary with movie features
            
        Returns:
            pd.DataFrame: Single row with features for prediction
        """
        # Create a DataFrame with the movie data
        df_pred = pd.DataFrame([movie_data])
        
        # Apply the same feature engineering
        df_pred = self.create_features(df_pred)
        
        # Select the same features used in training
        if self.feature_columns:
            # Ensure all expected features exist, create missing ones with 0
            for feature in self.feature_columns:
                if feature not in df_pred.columns:
                    df_pred[feature] = 0
            
            # IMPORTANT: Only select the features that were used during training
            X_pred = df_pred[self.feature_columns].fillna(0)
        else:
            # Fallback to default features if not trained yet
            default_features = [
                'release_year', 'budget_log', 'vote_confidence', 'rating_popularity_score',
                'status_released', 'is_adult'
            ]
            # Only use features that exist
            default_features = [f for f in default_features if f in df_pred.columns]
            X_pred = df_pred[default_features].fillna(0)
        
        # Replace infinite values with 0
        X_pred = X_pred.replace([np.inf, -np.inf], 0)
        X_pred = X_pred.fillna(0)
        
        # Scale features using the same scaler from training
        X_pred_scaled = self.scaler.transform(X_pred)
        X_pred_scaled = pd.DataFrame(X_pred_scaled, columns=X_pred.columns)
        
        return X_pred_scaled
