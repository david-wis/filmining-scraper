"""
Profitability model trainer (classification) for the Streamlit app.
"""
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import plotly.express as px


class ProfitabilityModelTrainer:
    """Random Forest classifier trainer for movie profitability."""

    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        self.is_trained = False

    def train_model(self, X_train, y_train, X_test, y_test, optimize_hyperparams=True):
        """
        Train a RandomForestClassifier on binary profitability labels.
        Returns a dict with classification metrics.
        """
        if optimize_hyperparams:
            st.info("🔍 Optimizing hyperparameters for classifier...")
            self.model = self._optimize_hyperparameters(X_train, y_train)
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_proba = None
        try:
            y_test_proba = self.model.predict_proba(X_test)[:, 1]
        except Exception:
            y_test_proba = None

        # Metrics
        metrics = {
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'train_precision': float(precision_score(y_train, y_train_pred, zero_division=0)),
            'train_recall': float(recall_score(y_train, y_train_pred, zero_division=0)),
            'train_f1': float(f1_score(y_train, y_train_pred, zero_division=0)),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
            'test_precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_test_pred, zero_division=0)),
            'cv_scores': cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        }

        if y_test_proba is not None:
            try:
                metrics['test_roc_auc'] = float(roc_auc_score(y_test, y_test_proba))
            except Exception:
                metrics['test_roc_auc'] = None
        else:
            metrics['test_roc_auc'] = None

        # Confusion matrix (test)
        try:
            cm = confusion_matrix(y_test, y_test_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except Exception:
            metrics['confusion_matrix'] = None

        self.training_metrics = metrics

        # Feature importance
        try:
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        except Exception:
            self.feature_importance = None

        self.is_trained = True
        return metrics

    def _optimize_hyperparameters(self, X_train, y_train):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        st.success(f"✅ Best classifier params: {grid_search.best_params_}")
        st.success(f"✅ Best CV ROC AUC: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting")
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception:
            return None

    def predict_label(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before predicting")
        return self.model.predict(X)

    def get_feature_importance_plot(self, top_n=20):
        if self.feature_importance is None:
            return None
        top_features = self.feature_importance.head(top_n)
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importance (Classification)',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        return fig

    def save_model(self, filepath):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'feature_engineer_state': st.session_state.feature_engineer.__dict__ if 'feature_engineer' in st.session_state else {}
        }
        joblib.dump(model_data, filepath)
        st.success(f"✅ Classifier saved to {filepath}")

    def load_model(self, filepath):
        try:
            model_data = joblib.load(filepath)
            self.model = model_data.get('model')
            self.feature_importance = model_data.get('feature_importance')
            self.training_metrics = model_data.get('training_metrics', {})
            self.is_trained = True
            st.success(f"✅ Classifier loaded from {filepath}")
            if 'feature_engineer_state' in model_data and 'feature_engineer' in st.session_state:
                st.session_state.feature_engineer.__dict__.update(model_data['feature_engineer_state'])
        except Exception as e:
            st.error(f"❌ Error loading classifier: {e}")

    def get_model_summary(self):
        if not self.is_trained:
            return {}
        return {
            'model_type': 'Random Forest Classifier',
            'n_estimators': getattr(self.model, 'n_estimators', None),
            'max_depth': getattr(self.model, 'max_depth', None),
            'min_samples_split': getattr(self.model, 'min_samples_split', None),
            'min_samples_leaf': getattr(self.model, 'min_samples_leaf', None),
            'training_metrics': self.training_metrics
        }
