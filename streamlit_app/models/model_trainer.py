"""
Model training module for the Streamlit ROI prediction app.
"""
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ROIModelTrainer:
    """
    Random Forest model trainer for movie ROI prediction.
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        self.is_trained = False
        
    def train_model(self, X_train, y_train, X_test, y_test, optimize_hyperparams=True):
        """
        Train Random Forest model for ROI prediction.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            optimize_hyperparams (bool): Whether to optimize hyperparameters
            
        Returns:
            dict: Training metrics
        """
        if optimize_hyperparams:
            st.info("🔍 Optimizing hyperparameters...")
            self.model = self._optimize_hyperparameters(X_train, y_train)
        else:
            # Use default parameters
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.training_metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'cv_scores': cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        
        return self.training_metrics
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize Random Forest hyperparameters using GridSearchCV.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            RandomForestRegressor: Optimized model
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        st.success(f"✅ Best parameters: {grid_search.best_params_}")
        st.success(f"✅ Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict_roi(self, X):
        """
        Predict ROI for given features.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.array: Predicted ROI values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance_plot(self, top_n=20):
        """
        Create feature importance plot.
        
        Args:
            top_n (int): Number of top features to show
            
        Returns:
            plotly.graph_objects.Figure: Feature importance plot
        """
        if self.feature_importance is None:
            return None
        
        top_features = self.feature_importance.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def get_prediction_analysis_plot(self, y_true, y_pred, title="Predictions vs Actual"):
        """
        Create prediction analysis plot.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.array): Predicted values
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Prediction analysis plot
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predictions vs Actual', 'Residuals', 'Distribution of Residuals', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scatter plot: Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred, mode='markers',
                name='Predictions', opacity=0.6
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=1
        )
        
        # Residuals plot
        residuals = y_true - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals, mode='markers',
                name='Residuals', opacity=0.6
            ),
            row=1, col=2
        )
        
        # Zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # Distribution of residuals
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals Distribution', nbinsx=30),
            row=2, col=1
        )
        
        # Q-Q plot (simplified)
        from scipy import stats
        qq_data = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq_data[0][0], y=qq_data[0][1],
                mode='markers', name='Q-Q Plot'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=False
        )
        
        return fig
    
    def save_model(self, filepath):
        """
        Save trained model to file.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'feature_columns': self.feature_importance['feature'].tolist()
        }
        
        joblib.dump(model_data, filepath)
        st.success(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model from file.
        
        Args:
            filepath (str): Path to load the model from
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_importance = model_data['feature_importance']
            self.training_metrics = model_data['training_metrics']
            self.is_trained = True
            st.success(f"✅ Model loaded from {filepath}")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    def get_model_summary(self):
        """
        Get model performance summary.
        
        Returns:
            dict: Model summary
        """
        if not self.is_trained:
            return {}
        
        return {
            'model_type': 'Random Forest Regressor',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'training_metrics': self.training_metrics
        }


