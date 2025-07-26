import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import base64


class ModelTrainer:
    def __init__(self, task_type):
        self.task_type = task_type
        self.model = None
        self.model_name = None
        self.is_trained = False
        self.predictions = None
        self.probabilities = None
        self.metrics = {}
        
    def get_available_models(self):
        """Get list of available models based on task type"""
        if self.task_type == "Classification":
            return {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Support Vector Machine": SVC(random_state=42, probability=True)
            }
        else:  # Regression
            return {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Support Vector Machine": SVR()
            }
    
    def select_model(self):
        """Interface for model selection"""
        st.subheader("🤖 Choose ML Model")
        
        available_models = self.get_available_models()
        
        # Model descriptions
        model_descriptions = {
            "Random Forest": "Ensemble of decision trees. Good for both tasks, handles missing values well.",
            "Logistic Regression": "Linear model for classification. Fast and interpretable.",
            "Linear Regression": "Linear model for regression. Simple and interpretable.",
            "Support Vector Machine": "Finds optimal decision boundary. Good for complex patterns."
        }
        
        self.model_name = st.selectbox(
            "Select a machine learning model:",
            options=list(available_models.keys()),
            help="Choose the algorithm to train on your data"
        )
        
        if self.model_name:
            st.info(f"**{self.model_name}:** {model_descriptions.get(self.model_name, '')}")
            self.model = available_models[self.model_name]
            return True
        
        return False
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train the selected model"""
        if self.model is None:
            st.error("Please select a model first")
            return False
        
        try:
            with st.spinner("Training model..."):
                # Train the model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                self.predictions = self.model.predict(X_test)
                
                # Get probabilities for classification
                if self.task_type == "Classification" and hasattr(self.model, 'predict_proba'):
                    self.probabilities = self.model.predict_proba(X_test)
                
                # Calculate metrics
                self._calculate_metrics(y_test)
                
                self.is_trained = True
                return True
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def _calculate_metrics(self, y_test):
        """Calculate performance metrics"""
        if self.task_type == "Classification":
            self.metrics = {
                'accuracy': accuracy_score(y_test, self.predictions),
                'precision': precision_score(y_test, self.predictions, average='weighted'),
                'recall': recall_score(y_test, self.predictions, average='weighted'),
                'f1_score': f1_score(y_test, self.predictions, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, self.predictions)
            }
        else:  # Regression
            self.metrics = {
                'r2_score': r2_score(y_test, self.predictions),
                'mean_squared_error': mean_squared_error(y_test, self.predictions),
                'root_mean_squared_error': np.sqrt(mean_squared_error(y_test, self.predictions)),
                'mean_absolute_error': mean_absolute_error(y_test, self.predictions)
            }
    
    def display_metrics(self):
        """Display model performance metrics"""
        if not self.is_trained:
            return
        
        st.subheader("📈 Model Performance")
        
        if self.task_type == "Classification":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{self.metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{self.metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{self.metrics['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{self.metrics['f1_score']:.3f}")
                
        else:  # Regression
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score", f"{self.metrics['r2_score']:.3f}")
            with col2:
                st.metric("RMSE", f"{self.metrics['root_mean_squared_error']:.3f}")
            with col3:
                st.metric("MAE", f"{self.metrics['mean_absolute_error']:.3f}")
    
    def plot_confusion_matrix(self, class_names=None):
        """Plot confusion matrix for classification"""
        if self.task_type != "Classification" or not self.is_trained:
            return
        
        cm = self.metrics['confusion_matrix']
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=class_names if class_names else [f"Class {i}" for i in range(cm.shape[1])],
            y=class_names if class_names else [f"Class {i}" for i in range(cm.shape[0])]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_prediction_vs_actual(self, y_test):
        """Plot predictions vs actual values for regression"""
        if self.task_type != "Regression" or not self.is_trained:
            return
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_test,
            y=self.predictions,
            mode='markers',
            name='Predictions',
            opacity=0.6
        ))
        
        # Perfect prediction line
        min_val = min(min(y_test), min(self.predictions))
        max_val = max(max(y_test), max(self.predictions))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Predictions vs Actual Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_residuals(self, y_test):
        """Plot residuals for regression"""
        if self.task_type != "Regression" or not self.is_trained:
            return
        
        residuals = y_test - self.predictions
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Residual Plot", "Residual Distribution"]
        )
        
        # Residual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=self.predictions,
                y=residuals,
                mode='markers',
                name='Residuals',
                opacity=0.6
            ),
            row=1, col=1
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Histogram of residuals
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='Distribution',
                nbinsx=20
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Residual Analysis",
            showlegend=False,
            height=400
        )
        
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance if available"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, feature_names, top_n=15):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(feature_names)
        
        if importance_df is None:
            st.info("Feature importance not available for this model")
            return
        
        # Take top N features
        top_features = importance_df.head(top_n)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title=f"Top {min(top_n, len(top_features))} Feature Importance",
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        
        fig.update_layout(height=max(400, len(top_features) * 25))
        st.plotly_chart(fig, use_container_width=True)
        
        return importance_df
    
    def cross_validate(self, X_train, y_train, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            return None
        
        if self.task_type == "Classification":
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
        
        cv_results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores,
            'scoring_metric': scoring
        }
        
        return cv_results
    
    def display_cross_validation(self, cv_results):
        """Display cross-validation results"""
        if cv_results is None:
            return
        
        st.subheader("🔄 Cross-Validation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                f"Mean {cv_results['scoring_metric'].title()}",
                f"{cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}"
            )
        
        with col2:
            fig = go.Figure(data=go.Box(
                y=cv_results['scores'],
                name="CV Scores",
                boxpoints='all'
            ))
            
            fig.update_layout(
                title="Cross-Validation Score Distribution",
                yaxis_title=cv_results['scoring_metric'].title(),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def save_model(self, filename):
        """Save trained model"""
        if not self.is_trained:
            return None
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'task_type': self.task_type,
            'metrics': self.metrics
        }
        
        # Save to bytes
        buffer = io.BytesIO()
        joblib.dump(model_data, buffer)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def load_model(self, model_data):
        """Load saved model"""
        try:
            loaded_data = joblib.load(io.BytesIO(model_data))
            self.model = loaded_data['model']
            self.model_name = loaded_data['model_name']
            self.task_type = loaded_data['task_type']
            self.metrics = loaded_data['metrics']
            self.is_trained = True
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_single(self, X_single):
        """Make prediction for a single sample"""
        if not self.is_trained:
            return None
        
        prediction = self.model.predict(X_single.reshape(1, -1))[0]
        
        result = {'prediction': prediction}
        
        if self.task_type == "Classification" and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_single.reshape(1, -1))[0]
            result['probabilities'] = probabilities
        
        return result 