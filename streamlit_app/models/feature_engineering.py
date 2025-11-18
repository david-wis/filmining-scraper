"""
Feature engineering module for the Streamlit ROI prediction app.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


class FeatureEngineer:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'roi'
        # Sentence transformer is only used for single predictions, not for training
        self.sentence_transformer = None
        self.embedding_dim = 384  # MiniLM-L6 produces 384-dimensional embeddings
        self.text_feature_names = []
        
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
        
        # 10. Budget categories
        df_features['budget_category'] = pd.cut(
            df_features['budget'], 
            bins=[0, 1000000, 10000000, 50000000, float('inf')], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # 11. Text features from overview
        if 'text_content' in df_features.columns:
            # Fill missing text with empty string
            df_features['text_content'] = df_features['text_content'].fillna('')
        elif 'overview' in df_features.columns:
            # Use overview for text content
            df_features['text_content'] = df_features['overview'].fillna('').astype(str)
        else:
            # No text content available
            df_features['text_content'] = ''
        
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
        
        # Process text features if available - load from database instead of generating
        text_features_array = None
        if 'overview_embedding' in df.columns:
            # Use pre-computed embeddings from database
            # Filter out rows with null embeddings
            valid_embeddings = df['overview_embedding'].notna()
            if valid_embeddings.any():
                # Convert string representation to numpy array if needed
                embeddings_list = []
                for emb in df['overview_embedding']:
                    if emb is not None:
                        if isinstance(emb, str):
                            # Parse string representation like '[0.1, 0.2, ...]'
                            emb = np.array([float(x) for x in emb.strip('[]').split(',')])
                        elif isinstance(emb, list):
                            emb = np.array(emb)
                        embeddings_list.append(emb)
                    else:
                        # For null embeddings, use zero vector
                        embeddings_list.append(np.zeros(self.embedding_dim))
                
                text_features_array = np.array(embeddings_list)
                self.text_feature_names = [f'text_emb_{i}' for i in range(self.embedding_dim)]
        
        # Combine all features
        feature_columns = numerical_features + genre_features + country_features + language_features
        
        # Filter features that exist in the dataframe
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Prepare feature matrix
        X = df[feature_columns].fillna(0)
        
        # Replace infinite values with 0
        X = X.replace([np.inf, -np.inf], 0)
        
        # Ensure all values are finite
        X = X.fillna(0)
        
        # Combine with text features if available
        if text_features_array is not None:
            text_df = pd.DataFrame(
                text_features_array,
                columns=self.text_feature_names,
                index=X.index
            )
            X = pd.concat([X, text_df], axis=1)
            feature_columns = feature_columns + self.text_feature_names
        
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
            # Get non-text features
            non_text_features = [f for f in self.feature_columns if not f.startswith('text_')]
            
            # Ensure all expected non-text features exist, create missing ones with 0
            for feature in non_text_features:
                if feature not in df_pred.columns:
                    df_pred[feature] = 0
            
            X_pred = df_pred[non_text_features].fillna(0)
            
            # Process text features if they were used in training
            if self.text_feature_names:
                # Initialize sentence transformer only when needed for single predictions
                if self.sentence_transformer is None:
                    self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                
                text_content = df_pred['text_content'].fillna('').iloc[0]
                # Generate semantic embedding for the text
                embedding = self.sentence_transformer.encode([text_content])
                text_df = pd.DataFrame(
                    embedding,
                    columns=self.text_feature_names,
                    index=X_pred.index
                )
                X_pred = pd.concat([X_pred, text_df], axis=1)
        else:
            # Fallback to default features if not trained yet
            default_features = [
                'release_year', 'budget_log', 'vote_confidence', 'rating_popularity_score',
                'status_released', 'is_adult'
            ]
            X_pred = df_pred[default_features].fillna(0)
        
        # Replace infinite values with 0
        X_pred = X_pred.replace([np.inf, -np.inf], 0)
        X_pred = X_pred.fillna(0)
        
        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)
        X_pred_scaled = pd.DataFrame(X_pred_scaled, columns=X_pred.columns)
        
        return X_pred_scaled
    
    def save_feature_engineer(self, filepath):
        """
        Save feature engineer configuration to file.
        
        Args:
            filepath (str): Path to save the feature engineer
        """
        engineer_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'embedding_dim': self.embedding_dim,
            'text_feature_names': self.text_feature_names
        }
        # Note: sentence_transformer model is not saved, will be reloaded on load
        
        joblib.dump(engineer_data, filepath)
    
    def load_feature_engineer(self, filepath):
        """
        Load feature engineer configuration from file.
        
        Args:
            filepath (str): Path to load the feature engineer from
        """
        engineer_data = joblib.load(filepath)
        self.scaler = engineer_data['scaler']
        self.label_encoders = engineer_data['label_encoders']
        self.feature_columns = engineer_data['feature_columns']
        self.target_column = engineer_data['target_column']
        # Handle both old (TF-IDF) and new (embeddings) formats
        self.embedding_dim = engineer_data.get('embedding_dim', 384)
        self.text_feature_names = engineer_data['text_feature_names']
        # Don't initialize sentence_transformer until needed (for single predictions only)
        self.sentence_transformer = None
