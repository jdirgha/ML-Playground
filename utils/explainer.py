import pandas as pd
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import io
import base64
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# Set global Matplotlib style for dark theme
plt.style.use('dark_background')
plt.rcParams.update({
    'text.color': '#ECECF1',
    'axes.labelcolor': '#ECECF1',
    'xtick.color': '#ECECF1',
    'ytick.color': '#ECECF1',
    'axes.titlecolor': '#FFFFFF'
})


class ModelExplainer:
    def __init__(self, model, X_train, X_test, task_type, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.task_type = task_type
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.base_value = None
        
    def initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            with st.spinner("Initializing explainer..."):
                # Ensure data types are compatible with newer numpy versions
                X_train_clean = self.X_train.copy()
                X_test_clean = self.X_test.copy()
                
                # Convert any boolean columns to int to avoid np.bool deprecation
                for col in X_train_clean.columns:
                    if X_train_clean[col].dtype == bool:
                        X_train_clean[col] = X_train_clean[col].astype(int)
                    if X_test_clean[col].dtype == bool:
                        X_test_clean[col] = X_test_clean[col].astype(int)
                
                # Choose appropriate explainer based on model type
                if isinstance(self.model, (RandomForestClassifier, RandomForestRegressor)):
                    self.explainer = shap.TreeExplainer(self.model)
                elif isinstance(self.model, (LogisticRegression, LinearRegression)):
                    self.explainer = shap.LinearExplainer(self.model, X_train_clean)
                else:
                    # Use KernelExplainer as fallback (slower but works with any model)
                    background = shap.sample(X_train_clean, min(100, len(X_train_clean)))
                    self.explainer = shap.KernelExplainer(self.model.predict, background)
                
                # Store clean data for later use
                self.X_train_clean = X_train_clean
                self.X_test_clean = X_test_clean
                
                return True
                
        except Exception as e:
            st.error(f"Error initializing explainer: {str(e)}")
            return False
    
    def calculate_shap_values(self, n_samples=None):
        """Calculate SHAP values for test set"""
        if self.explainer is None:
            return False
        
        try:
            with st.spinner("Calculating SHAP values..."):
                # Limit samples for performance
                if n_samples is None:
                    n_samples = min(100, len(self.X_test_clean))
                
                sample_data = self.X_test_clean.iloc[:n_samples] if hasattr(self.X_test_clean, 'iloc') else self.X_test_clean[:n_samples]
                
                # Calculate SHAP values
                self.shap_values = self.explainer.shap_values(sample_data)
                
                # Handle different formats of SHAP values
                if isinstance(self.shap_values, list):
                    # Multi-class classification
                    if self.task_type == "Classification":
                        # Use the first class for visualization
                        self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
                
                # Get base value
                if hasattr(self.explainer, 'expected_value'):
                    self.base_value = self.explainer.expected_value
                    if isinstance(self.base_value, np.ndarray):
                        self.base_value = self.base_value[0] if len(self.base_value) > 0 else 0
                else:
                    self.base_value = 0
                
                return True
                
        except Exception as e:
            st.error(f"Error calculating SHAP values: {str(e)}")
            return False
    
    def plot_summary_plot(self):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            return
        
        st.subheader("🎯 Global Feature Importance")
        
        try:
            # Use limited sample for performance
            sample_data = self.X_test_clean.iloc[:len(self.shap_values)] if hasattr(self.X_test_clean, 'iloc') else self.X_test_clean[:len(self.shap_values)]
            
            # Create the bar plot (no colorbar issues)
            fig, ax = plt.subplots(figsize=(10, 8))
            
            shap.summary_plot(
                self.shap_values, 
                sample_data,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Also create a beeswarm plot with explicit colorbar handling
            st.subheader("📊 Feature Impact Distribution")
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create the beeswarm plot
            shap.summary_plot(
                self.shap_values, 
                sample_data,
                feature_names=self.feature_names,
                show=False
            )
            
            # Ensure proper layout
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error creating summary plot: {str(e)}")
            st.info("This might be due to the dataset size or feature types. Try with a smaller dataset or different features.")
    
    def plot_feature_importance_interactive(self):
        """Create interactive feature importance plot"""
        if self.shap_values is None:
            return
        
        try:
            # Calculate mean absolute SHAP values
            feature_importance = np.abs(self.shap_values).mean(0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            # Create interactive bar plot
            fig = px.bar(
                importance_df.tail(15),  # Top 15 features
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance (Mean |SHAP Value|)",
                labels={'importance': 'Mean |SHAP Value|', 'feature': 'Features'},
                color='importance',
                template="plotly_dark",
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=600,
                xaxis=dict(tickfont=dict(color='#ECECF1')),
                yaxis=dict(tickfont=dict(color='#ECECF1'))
            )
            st.plotly_chart(fig, use_container_width=True)
            
            return importance_df
            
        except Exception as e:
            st.error(f"Error creating interactive importance plot: {str(e)}")
            return None
    
    def plot_dependence_plot(self, feature_idx=None, interaction_idx=None):
        """Create SHAP dependence plot"""
        if self.shap_values is None:
            return
        
        st.subheader("🔍 Feature Dependence Analysis")
        
        try:
            # Let user select feature
            if feature_idx is None:
                feature_names = list(self.feature_names)
                selected_feature = st.selectbox(
                    "Select feature for dependence analysis:",
                    options=feature_names,
                    help="Shows how this feature's values affect predictions"
                )
                feature_idx = feature_names.index(selected_feature)
            
            # Create dependence plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sample_data = self.X_test_clean.iloc[:len(self.shap_values)] if hasattr(self.X_test_clean, 'iloc') else self.X_test_clean[:len(self.shap_values)]
            
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                sample_data,
                feature_names=self.feature_names,
                interaction_index=interaction_idx,
                show=False,
                ax=ax
            )
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        except Exception as e:
            st.error(f"Error creating dependence plot: {str(e)}")
    
    def explain_single_prediction(self, sample_idx=None):
        """Explain a single prediction"""
        if self.shap_values is None:
            return
        
        st.subheader("🔎 Individual Prediction Explanation")
        
        try:
            # Let user select sample
            if sample_idx is None:
                sample_idx = st.selectbox(
                    "Select sample to explain:",
                    options=range(min(20, len(self.shap_values))),
                    format_func=lambda x: f"Sample {x + 1}",
                    help="Choose a specific prediction to understand"
                )
            
            if sample_idx >= len(self.shap_values):
                st.error("Sample index out of range")
                return
            
            # Get sample data
            sample_data = self.X_test_clean.iloc[sample_idx] if hasattr(self.X_test_clean, 'iloc') else self.X_test_clean[sample_idx]
            sample_shap = self.shap_values[sample_idx]
            
            # Show sample information
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sample Features:**")
                sample_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Value': sample_data.values if hasattr(sample_data, 'values') else sample_data
                })
                st.dataframe(sample_df.astype(str))
            
            with col2:
                # Make prediction
                prediction = self.model.predict(sample_data.values.reshape(1, -1))[0]
                
                st.write("**Prediction Information:**")
                
                # Special handling for survival prediction
                if hasattr(self.model, 'predict_proba') and self.task_type == "Classification":
                    proba = self.model.predict_proba(sample_data.values.reshape(1, -1))[0]
                    if len(proba) == 2:  # Binary classification (like survival)
                        survival_prob = proba[1] * 100
                        death_prob = proba[0] * 100
                        
                        st.write(f"**🛳️ Survival Prediction:**")
                        if prediction == 1:
                            st.success(f"✅ **SURVIVED** (Probability: {survival_prob:.1f}%)")
                        else:
                            st.error(f"❌ **DID NOT SURVIVE** (Probability: {death_prob:.1f}%)")
                        
                        st.write(f"• Survival chance: {survival_prob:.1f}%")
                        st.write(f"• Death chance: {death_prob:.1f}%")
                    else:
                        st.write(f"Predicted Class: {prediction}")
                else:
                    st.write(f"Predicted Value: {prediction:.3f}")
                
                st.write(f"Base Value: {self.base_value:.3f}")
                st.write(f"Prediction - Base: {prediction - self.base_value:.3f}")
            
            # Create waterfall plot using plotly
            self._create_waterfall_plot(sample_shap, sample_data, prediction)
            
            # Create force plot alternative
            self._create_force_plot_alternative(sample_shap, sample_data, prediction)
            
        except Exception as e:
            st.error(f"Error explaining prediction: {str(e)}")
    
    def _create_waterfall_plot(self, shap_values, sample_data, prediction):
        """Create waterfall plot for individual prediction"""
        try:
            # Prepare data for waterfall plot
            feature_contributions = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': shap_values,
                'feature_value': sample_data.values if hasattr(sample_data, 'values') else sample_data
            })
            
            # Sort by absolute SHAP value
            feature_contributions = feature_contributions.reindex(
                feature_contributions['shap_value'].abs().sort_values(ascending=False).index
            )
            
            # Take top features
            top_features = feature_contributions.head(10)
            
            # Create waterfall chart
            fig = go.Figure()
            
            # Start with base value
            cumulative = [self.base_value]
            x_labels = ['Base Value']
            colors = ['blue']
            
            # Add each feature contribution
            for _, row in top_features.iterrows():
                cumulative.append(cumulative[-1] + row['shap_value'])
                x_labels.append(f"{row['feature']}<br>({row['feature_value']:.2f})")
                colors.append('green' if row['shap_value'] > 0 else 'red')
            
            # Add final prediction
            x_labels.append('Prediction')
            colors.append('blue')
            
            # Create bars
            for i in range(len(cumulative)):
                if i == 0:
                    # Base value
                    fig.add_trace(go.Bar(
                        x=[x_labels[i]],
                        y=[cumulative[i]],
                        name='Base',
                        marker_color=colors[i],
                        showlegend=False
                    ))
                elif i == len(cumulative) - 1:
                    # Final prediction line
                    continue
                else:
                    # Feature contributions
                    height = cumulative[i] - cumulative[i-1]
                    base = cumulative[i-1] if height > 0 else cumulative[i]
                    
                    fig.add_trace(go.Bar(
                        x=[x_labels[i]],
                        y=[abs(height)],
                        base=base,
                        name=f"Feature {i}",
                        marker_color=colors[i],
                        showlegend=False
                    ))
            
            # Add prediction line
            fig.add_hline(
                y=prediction,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Final Prediction: {prediction:.3f}"
            )
            
            fig.update_layout(
                title="Feature Contribution to Prediction (Waterfall)",
                xaxis_title="Features",
                yaxis_title="Prediction Value",
                template="plotly_dark",
                height=500,
                xaxis=dict(tickfont=dict(color='#ECECF1')),
                yaxis=dict(tickfont=dict(color='#ECECF1'))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating waterfall plot: {str(e)}")
    
    def _create_force_plot_alternative(self, shap_values, sample_data, prediction):
        """Create alternative to SHAP force plot"""
        try:
            # Prepare data
            feature_contributions = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': shap_values,
                'feature_value': sample_data.values if hasattr(sample_data, 'values') else sample_data
            })
            
            # Sort by SHAP value
            feature_contributions = feature_contributions.sort_values('shap_value')
            
            # Take top positive and negative contributors
            top_features = pd.concat([
                feature_contributions.head(5),  # Most negative
                feature_contributions.tail(5)   # Most positive
            ])
            
            # Create horizontal bar chart
            fig = px.bar(
                top_features,
                x='shap_value',
                y='feature',
                orientation='h',
                color='shap_value',
                color_continuous_scale='RdBu',
                title="Feature Contributions (SHAP Values)",
                labels={'shap_value': 'SHAP Value', 'feature': 'Features'},
                template="plotly_dark",
                hover_data=['feature_value']
            )
            
            # Add base value line
            fig.add_vline(
                x=0,
                line_dash="dash",
                line_color="black",
                annotation_text="Base Value"
            )
            
            fig.update_layout(
                height=400,
                xaxis=dict(tickfont=dict(color='#ECECF1')),
                yaxis=dict(tickfont=dict(color='#ECECF1'))
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating force plot alternative: {str(e)}")
    
    def create_explanation_report(self, sample_idx):
        """Create a detailed explanation report"""
        if self.shap_values is None:
            return None
        
        try:
            # Get sample data
            sample_data = self.X_test.iloc[sample_idx] if hasattr(self.X_test, 'iloc') else self.X_test[sample_idx]
            sample_shap = self.shap_values[sample_idx]
            
            # Make prediction
            prediction = self.model.predict(sample_data.values.reshape(1, -1))[0]
            
            # Create report
            report = {
                'sample_index': sample_idx,
                'prediction': prediction,
                'base_value': self.base_value,
                'feature_values': dict(zip(self.feature_names, sample_data.values if hasattr(sample_data, 'values') else sample_data)),
                'shap_values': dict(zip(self.feature_names, sample_shap)),
                'top_positive_features': [],
                'top_negative_features': []
            }
            
            # Get top contributing features
            feature_contributions = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': sample_shap
            }).sort_values('shap_value', ascending=False)
            
            # Top positive contributors
            positive_features = feature_contributions[feature_contributions['shap_value'] > 0].head(5)
            report['top_positive_features'] = positive_features.to_dict('records')
            
            # Top negative contributors
            negative_features = feature_contributions[feature_contributions['shap_value'] < 0].tail(5)
            report['top_negative_features'] = negative_features.to_dict('records')
            
            return report
            
        except Exception as e:
            st.error(f"Error creating explanation report: {str(e)}")
            return None
    
    def display_global_insights(self):
        """Display global model insights"""
        if self.shap_values is None:
            return
        
        st.subheader("🌍 Global Model Insights")
        
        try:
            # Calculate global feature importance
            global_importance = np.abs(self.shap_values).mean(0)
            
            # Create insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Most Important Features:**")
                top_features = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': global_importance
                }).sort_values('Importance', ascending=False).head(5)
                
                for _, row in top_features.iterrows():
                    st.write(f"• {row['Feature']}: {row['Importance']:.4f}")
            
            with col2:
                st.write("**Model Behavior:**")
                st.write(f"• Average |SHAP|: {np.abs(self.shap_values).mean():.4f}")
                st.write(f"• Max |SHAP|: {np.abs(self.shap_values).max():.4f}")
                st.write(f"• Features analyzed: {len(self.feature_names)}")
                st.write(f"• Samples explained: {len(self.shap_values)}")
            
            # Feature correlation with predictions
            predictions = self.model.predict(self.X_test.iloc[:len(self.shap_values)] if hasattr(self.X_test, 'iloc') else self.X_test[:len(self.shap_values)])
            
            st.write("**Feature-Prediction Relationships:**")
            
            # Calculate correlations
            sample_data = self.X_test.iloc[:len(self.shap_values)] if hasattr(self.X_test, 'iloc') else self.X_test[:len(self.shap_values)]
            correlations = []
            
            for i, feature in enumerate(self.feature_names):
                if hasattr(sample_data, 'iloc'):
                    feature_values = sample_data.iloc[:, i]
                else:
                    feature_values = sample_data[:, i]
                
                corr = np.corrcoef(feature_values, predictions)[0, 1]
                if not np.isnan(corr):
                    correlations.append({'Feature': feature, 'Correlation': corr})
            
            corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False).head(10)
            
            if not corr_df.empty:
                fig = px.bar(
                    corr_df,
                    x='Correlation',
                    y='Feature',
                    orientation='h',
                    title="Feature-Prediction Correlations",
                    color='Correlation',
                    template="plotly_dark",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                fig.update_layout(
                    xaxis=dict(tickfont=dict(color='#ECECF1')),
                    yaxis=dict(tickfont=dict(color='#ECECF1'))
                )
            
        except Exception as e:
            st.error(f"Error displaying global insights: {str(e)}")
    
    def export_explanations(self):
        """Export explanations as downloadable content"""
        if self.shap_values is None:
            return None
        
        try:
            # Create summary DataFrame
            explanations_df = pd.DataFrame(self.shap_values, columns=self.feature_names)
            explanations_df['base_value'] = self.base_value
            
            # Add predictions
            sample_data = self.X_test.iloc[:len(self.shap_values)] if hasattr(self.X_test, 'iloc') else self.X_test[:len(self.shap_values)]
            predictions = self.model.predict(sample_data)
            explanations_df['prediction'] = predictions
            
            return explanations_df
            
        except Exception as e:
            st.error(f"Error exporting explanations: {str(e)}")
            return None 