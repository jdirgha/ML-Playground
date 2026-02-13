import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px

# Import custom modules
from utils.data_handler import DataHandler, load_sample_dataset
from utils.model_trainer import ModelTrainer
from utils.explainer import ModelExplainer
from utils.deployment import generate_fastapi_app, generate_requirements
from utils.ai_advisor import AIAdvisor
import utils.ui as ui

# Page configuration
st.set_page_config(
    page_title="Explainable ML Playground",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom UI styles
ui.set_custom_style()

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = None
    
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    
    if 'step' not in st.session_state:
        st.session_state.step = 1
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    if 'explanations_ready' not in st.session_state:
        st.session_state.explanations_ready = False
    
    if 'preprocessing_complete' not in st.session_state:
        st.session_state.preprocessing_complete = False
    
    if 'model_training_complete' not in st.session_state:
        st.session_state.model_training_complete = False


def create_bottom_navigation(current_step, show_next=True, next_step=None, next_label=None, next_disabled=False, show_back=True):
    """Create bottom navigation buttons for each step"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if show_back and current_step > 1:
            if st.button("← Back", key="back_bottom_" + str(current_step)):
                st.session_state.step = current_step - 1
                st.rerun()
    
    with col2:
        st.write(f"Step {current_step} of 5")
    
    with col3:
        if show_next and not next_disabled:
            next_step_num = next_step if next_step else current_step + 1
            label = next_label if next_label else "Next Step →"
            if st.button(label, key="next_bottom_" + str(current_step)):
                st.session_state.step = next_step_num
                st.rerun()
        elif next_disabled and show_next:
            disabled_key = "disabled_bottom_" + str(current_step)
            st.button(next_label or "Complete current step first", disabled=True, key=disabled_key)


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    ui.create_header()
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        
        # Step indicators with clickable navigation
        steps = [
            ("1. Data Upload", 1),
            ("2. Target Selection", 2), 
            ("3. Model Training", 3),
            ("4. Explanations", 4),
            ("5. Analysis & Deployment", 5)
        ]
        
        st.write("*Click on completed steps to navigate*")
        
        for step_name, step_num in steps:
            # Determine if step is accessible
            step_accessible = step_num <= st.session_state.step
            
            # Different styling based on current step and accessibility
            if step_num == st.session_state.step:
                # Current step - highlighted
                st.write(f"**→ {step_name}**")
            elif step_accessible:
                # Completed step - clickable
                nav_key = "nav_step_" + str(step_num)
                button_text = f"✓ {step_name}"
                if st.button(button_text, key=nav_key):
                    st.session_state.step = step_num
                    st.rerun()
            else:
                # Future step - disabled
                st.write(f"   {step_name}")
        
        st.markdown("---")
        
        # Progress bar
        progress = (st.session_state.step - 1) / 4  # 5 steps total, so max progress is 4/4
        st.progress(progress)
        st.caption(f"Step {st.session_state.step} of 5")
        
        st.markdown("---")
        
        # Actions
        if st.button("Reset Application", help="Start over with a fresh session"):
            # Reset only necessary keys, keep system keys
            keys_to_reset = [
                'data_handler', 'model_trainer', 'explainer', 'step', 'data_loaded',
                'model_trained', 'explanations_ready', 'preprocessing_complete', 
                'model_training_complete'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Show status
        st.markdown("---")
        st.markdown("**Status:**")
        if st.session_state.data_loaded:
            st.success(f"✅ Data: {st.session_state.data_handler.filename}")
        if st.session_state.model_trained:
            st.success(f"✅ Model: {st.session_state.model_trainer.model_name}")

    # Render Content
    if st.session_state.step == 1:
        step_1_upload()
    elif st.session_state.step == 2:
        step_2_target()
    elif st.session_state.step == 3:
        step_3_training()
    elif st.session_state.step == 4:
        step_4_explanations()
    elif st.session_state.step == 5:
        step_5_analysis()


def step_1_upload():
    """Step 1: Data Upload"""
    ui.create_step_header(1, "Data Upload", "Upload your dataset")
    
    # Option to use sample data or upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a tabular dataset with features and a target variable"
        )
        
        if uploaded_file is not None:
            if st.session_state.data_handler.load_data(uploaded_file):
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
    
    with col2:
        st.subheader("Or Try Sample Datasets")
        sample_datasets = [
            "None",
            "Titanic Survival",
            "Iris Classification",
            "California Housing Regression", 
            "Wine Classification"
        ]
        
        selected_sample = st.selectbox(
            "Choose a sample dataset:",
            options=sample_datasets,
            help="Pre-loaded datasets to explore the app features"
        )
        
        if selected_sample != "None":
            sample_data = load_sample_dataset(selected_sample)
            if sample_data is not None:
                st.session_state.data_handler.data = sample_data
                st.session_state.data_handler.filename = f"Sample: {selected_sample}"
                st.session_state.data_loaded = True
                st.success("Sample data loaded successfully!")
    
    # Display data preview if loaded
    if st.session_state.data_loaded:
        st.session_state.data_handler.display_data_preview()
    
    # Bottom navigation
    create_bottom_navigation(
        current_step=1, 
        show_back=False,  # First step, no back button
        next_label="Proceed to Target Selection",
        next_disabled=not st.session_state.data_loaded
    )


def step_2_target():
    """Step 2: Target and Feature Selection"""
    ui.create_step_header(2, "Target Selection", "Select the target column you want to predict.")
    
    if not st.session_state.data_loaded:
        st.error("Please upload data first!")
        create_bottom_navigation(current_step=2, show_next=False)
        return
    
    # Check if preprocessing is already complete
    if st.session_state.get('preprocessing_complete', False):
        st.success("Data preprocessing completed!")
        
        # Show summary
        summary = st.session_state.data_handler.get_preprocessing_summary()
        if summary:
            st.subheader("Preprocessing Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", summary['train_size'])
            with col2:
                st.metric("Test Samples", summary['test_size'])
            with col3:
                st.metric("Features", summary['n_features'])
            
            if summary['task_type'] == "Classification":
                st.write(f"**Number of Classes:** {summary['n_classes']}")
                st.write("**Class Distribution:**", summary['class_distribution'])
        
        # Bottom navigation
        create_bottom_navigation(
            current_step=2,
            next_label="Proceed to Model Training"
        )
        return
    
    # Target selection interface
    target_selected = st.session_state.data_handler.select_target_and_features()
    
    if target_selected:
        # Data preprocessing
        st.subheader("Data Preprocessing")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        with col2:
            random_state = st.number_input("Random seed", 1, 100, 42)
        
        if st.button("Preprocess Data", key="preprocess_data", type="primary"):
            with st.spinner("Preprocessing data..."):
                if st.session_state.data_handler.preprocess_data(test_size, random_state):
                    # Mark preprocessing as complete
                    st.session_state.preprocessing_complete = True
                    st.success("Data preprocessing completed!")
                    st.rerun()
    
    
            # AI Advisor Recommendations
            st.markdown("---")
            with st.expander("🤖 AI Data Analyst Insights", expanded=True):
                advisor = AIAdvisor(st.session_state.data_handler)
                
                # 1. Data Quality Checks
                st.subheader("1. Data Quality Analysis")
                alerts = advisor.analyze_data_quality()
                for alert in alerts:
                    st.write(alert)
                    
                # 2. Model Recommendation
                st.subheader("2. Model Recommendation")
                model_name, reason = advisor.suggest_model()
                st.info(f"💡 **Recommended Model:** {model_name}\n\n**Reason:** {reason}")
                
                # 3. Feature Engineering Tips
                st.subheader("3. Feature Engineering Tips")
                tips = advisor.get_feature_advice()
                for tip in tips:
                    st.write(tip)

    # Bottom navigation
    create_bottom_navigation(
        current_step=2,
        next_label="Proceed to Model Training" if st.session_state.get('preprocessing_complete', False) else "Preprocess data first",
        next_disabled=not st.session_state.get('preprocessing_complete', False)
    )


def step_3_training():
    """Step 3: Model Training"""
    ui.create_step_header(3, "Model Training", "Train a Machine Learning model on your data.")
    
    # Check if data is preprocessed
    if not st.session_state.get('preprocessing_complete', False) or st.session_state.data_handler.X_train is None:
        st.error("Please preprocess data first!")
        create_bottom_navigation(current_step=3, show_next=False)
        return
    
    # Check if model training is already complete
    if st.session_state.get('model_training_complete', False) and st.session_state.model_trained:
        st.success("Model training completed!")
        
        # Display metrics
        st.session_state.model_trainer.display_metrics()
        
        # Display visualizations
        if st.session_state.data_handler.task_type == "Classification":
            st.session_state.model_trainer.plot_confusion_matrix()
        else:
            st.session_state.model_trainer.plot_prediction_vs_actual(st.session_state.data_handler.y_test)
            st.session_state.model_trainer.plot_residuals(st.session_state.data_handler.y_test)
        
        # Feature importance
        feature_names = st.session_state.data_handler.get_preprocessing_summary()['feature_names']
        st.session_state.model_trainer.plot_feature_importance(feature_names)
        
        # Bottom navigation
        create_bottom_navigation(
            current_step=3,
            next_label="Explore Explanations"
        )
        return
    
    # Initialize model trainer
    if st.session_state.model_trainer is None:
        st.session_state.model_trainer = ModelTrainer(st.session_state.data_handler.task_type)
    
    # Model selection
    if st.session_state.model_trainer.select_model():
        
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            perform_cv = st.checkbox("Perform Cross-Validation", value=True)
        with col2:
            cv_folds = st.number_input("CV Folds", 3, 10, 5) if perform_cv else 5
        
        # Train model button
        if st.button("Train Model", type="primary"):
            
            # Cross-validation first
            if perform_cv:
                with st.spinner("Performing cross-validation..."):
                    cv_results = st.session_state.model_trainer.cross_validate(
                        st.session_state.data_handler.X_train,
                        st.session_state.data_handler.y_train,
                        cv=cv_folds
                    )
                    
                    if cv_results:
                        st.session_state.model_trainer.display_cross_validation(cv_results)
            
            # Train final model
            success = st.session_state.model_trainer.train_model(
                st.session_state.data_handler.X_train,
                st.session_state.data_handler.y_train,
                st.session_state.data_handler.X_test,
                st.session_state.data_handler.y_test
            )
            
            if success:
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
                
                # Display metrics
                st.session_state.model_trainer.display_metrics()
                
                # Display visualizations
                if st.session_state.data_handler.task_type == "Classification":
                    st.session_state.model_trainer.plot_confusion_matrix()
                else:
                    st.session_state.model_trainer.plot_prediction_vs_actual(st.session_state.data_handler.y_test)
                    st.session_state.model_trainer.plot_residuals(st.session_state.data_handler.y_test)
                
                # Feature importance
                feature_names = st.session_state.data_handler.get_preprocessing_summary()['feature_names']
                st.session_state.model_trainer.plot_feature_importance(feature_names)
                
                # Mark training as complete for navigation
                st.session_state.model_training_complete = True
                
    # Bottom navigation
    create_bottom_navigation(
        current_step=3,
        next_label="Explore Explanations" if st.session_state.get('model_training_complete', False) else "Train model first",
        next_disabled=not st.session_state.get('model_training_complete', False)
    )


def step_4_explanations():
    """Step 4: Model Explanations"""
    ui.create_step_header(4, "Explanations", "Understand model predictions using SHAP (Explainable AI).")
    
    if not st.session_state.model_trained or not st.session_state.get('model_training_complete', False):
        st.error("Please train a model first!")
        create_bottom_navigation(current_step=4, show_next=False)
        return
    
    # Initialize explainer
    if st.session_state.explainer is None:
        feature_names = st.session_state.data_handler.get_preprocessing_summary()['feature_names']
        st.session_state.explainer = ModelExplainer(
            st.session_state.model_trainer.model,
            st.session_state.data_handler.X_train,
            st.session_state.data_handler.X_test,
            st.session_state.data_handler.task_type,
            feature_names
        )
    
    # Initialize and calculate SHAP values
    if not st.session_state.explanations_ready:
        if st.button("Generate Explanations", type="primary"):
            with st.spinner("Generating SHAP explanations..."):
                if st.session_state.explainer.initialize_explainer():
                    if st.session_state.explainer.calculate_shap_values():
                        st.session_state.explanations_ready = True
                        st.success("Explanations generated successfully!")
                        st.rerun()
    
    if st.session_state.explanations_ready:
        
        # Explanation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Global Explanations", 
            "Individual Predictions", 
            "Feature Dependencies",
            "Model Insights"
        ])
        
        with tab1:
            st.session_state.explainer.plot_summary_plot()
            st.session_state.explainer.plot_feature_importance_interactive()
        
        with tab2:
            st.session_state.explainer.explain_single_prediction()
        
        with tab3:
            st.session_state.explainer.plot_dependence_plot()
        
        with tab4:
            st.session_state.explainer.display_global_insights()
    
    # Bottom navigation
    create_bottom_navigation(
        current_step=4,
        next_label="Advanced Analysis" if st.session_state.explanations_ready else "Generate explanations first",
        next_disabled=not st.session_state.explanations_ready
    )


def step_5_analysis():
    """Step 5: Advanced Analysis and Export"""
    ui.create_step_header(5, "Deployment & Analysis", "Analyze performance, compare models, and deploy as API.")
    
    if not st.session_state.explanations_ready:
        st.error("Please generate explanations first!")
        create_bottom_navigation(current_step=5, show_next=False)
        return
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Model Analysis", "Export Results", "🚀 One-Click Deployment", "Model Comparison"])
    
    with tab1:
        st.subheader("Comprehensive Model Analysis")
        
        # Performance summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Performance Summary:**")
            metrics = st.session_state.model_trainer.metrics
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    st.write(f"• {metric.replace('_', ' ').title()}: {value:.4f}")
        
        with col2:
            st.write("**Dataset Information:**")
            summary = st.session_state.data_handler.get_preprocessing_summary()
            st.write(f"• Task Type: {summary['task_type']}")
            st.write(f"• Total Features: {summary['n_features']}")
            st.write(f"• Training Samples: {summary['train_size']}")
            st.write(f"• Test Samples: {summary['test_size']}")
        
        # Additional visualizations
        if st.checkbox("Show Feature Correlation Analysis"):
            # Simple correlation analysis
            feature_names = st.session_state.data_handler.get_preprocessing_summary()['feature_names']
            if len(feature_names) <= 20:  # Only for manageable number of features
                try:
                    correlation_matrix = st.session_state.data_handler.X_train.corr()
                    fig = px.imshow(correlation_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate correlation matrix: {str(e)}")
                    st.info("This may happen if the features are not numeric or if there are too few samples.")
    
    with tab2:
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export explanations
            if st.button("Export SHAP Explanations"):
                explanations_df = st.session_state.explainer.export_explanations()
                if explanations_df is not None:
                    csv = explanations_df.to_csv(index=False)
                    st.download_button(
                        label="Download Explanations CSV",
                        data=csv,
                        file_name="shap_explanations.csv",
                        mime="text/csv"
                    )
        
        with col2:
            # Export predictions
            if st.button("Export Predictions"):
                predictions_df = pd.DataFrame({
                    'prediction': st.session_state.model_trainer.predictions,
                    'actual': st.session_state.data_handler.y_test
                })
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="model_predictions.csv",
                    mime="text/csv"
                )
        
        # Generate report
        if st.button("Generate Analysis Report"):
            report = generate_analysis_report()
            st.download_button(
                label="Download Analysis Report",
                data=report,
                file_name="ml_analysis_report.txt",
                mime="text/plain"
            )
    

    
    with tab3:
        st.subheader("🚀 Deploy Your Model")
        st.markdown("Generate production-ready code to serve your model as an API.")
        
        if st.session_state.model_trained:
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("👇 **Step 1: Download Model**")
                # Reuse the save logic
                model_data = st.session_state.model_trainer.save_model("model.pkl")
                st.download_button(
                    label="Download Model File (model.pkl)",
                    data=model_data,
                    file_name="model.pkl",
                    mime="application/octet-stream",
                    key="deploy_model_dl"
                )
                
            with col2:
                st.info("👇 **Step 2: Download API Code**")
                
                # Generate API code
                feature_names = st.session_state.data_handler.get_preprocessing_summary()['feature_names']
                api_code = generate_fastapi_app(
                    st.session_state.model_trainer.model_name,
                    feature_names,
                    st.session_state.data_handler.task_type
                )
                
                st.download_button(
                    label="Download API Code (fastapi_app.py)",
                    data=api_code,
                    file_name="fastapi_app.py",
                    mime="text/x-python",
                    key="deploy_api_dl"
                )
                
                # Requirements
                reqs = generate_requirements()
                st.download_button(
                    label="Download requirements.txt",
                    data=reqs,
                    file_name="requirements.txt",
                    mime="text/plain",
                    key="deploy_reqs_dl"
                )
                
            st.markdown("---")
            st.subheader("How to run your API locally:")
            st.code("""
# 1. Install requirements
pip install -r requirements.txt

# 2. Run the server
python fastapi_app.py

# 3. Test the API
# Open http://localhost:8000/docs in your browser
            """, language="bash")
            
        else:
            st.warning("Please train a model first to generate deployment code.")

    with tab4:
        st.subheader("⚔️ Model Comparison & Ensemble")
        
        if st.session_state.model_trained:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("Compare different algorithms to find the best performer for your data.")
                
                if st.button("🚀 Train & Compare All Models"):
                    with st.spinner("Training all models... This might take a moment."):
                        results = st.session_state.model_trainer.train_all_models(
                            st.session_state.data_handler.X_train,
                            st.session_state.data_handler.y_train,
                            st.session_state.data_handler.X_test,
                            st.session_state.data_handler.y_test
                        )
                        st.success("Comparison complete!")
                
                # Display results if available
                if hasattr(st.session_state.model_trainer, 'comparison_results') and \
                   isinstance(st.session_state.model_trainer.comparison_results, pd.DataFrame):
                    
                    results_df = st.session_state.model_trainer.comparison_results
                    
                    # Formatting for display (hide model object)
                    display_df = results_df.drop(columns=['model_obj'])
                    
                    st.write("### 🏆 Leaderboard")
                    try:
                        st.dataframe(display_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                    except Exception:
                        st.dataframe(display_df, use_container_width=True)
                    
                    # Bar chart of performance
                    metric_col = display_df.columns[1] # Accuracy or R2
                    fig = px.bar(
                        display_df, 
                        x='Model', 
                        y=metric_col, 
                        color=metric_col,
                        title=f"Model Performance ({metric_col})",
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.info("ℹ️ **Ensemble Learning** combine multiple models to improve accuracy and reduce overfitting.")
                
                if hasattr(st.session_state.model_trainer, 'comparison_results') and \
                   isinstance(st.session_state.model_trainer.comparison_results, pd.DataFrame):
                    
                    st.subheader("🤝 Create Ensemble")
                    top_n = st.slider("Combine Top N Models", 2, len(st.session_state.model_trainer.comparison_results), 3)
                    
                    if st.button("Build Voting Ensemble"):
                        with st.spinner("Building ensemble model..."):
                            ensemble_res = st.session_state.model_trainer.create_ensemble(
                                st.session_state.data_handler.X_train,
                                st.session_state.data_handler.y_train,
                                st.session_state.data_handler.X_test,
                                st.session_state.data_handler.y_test,
                                top_n=top_n
                            )
                            
                            if ensemble_res:
                                st.balloons()
                                st.success(f"Ensemble Built Successfully!")
                                st.metric(f"Ensemble {ensemble_res['metric']}", f"{ensemble_res['score']:.4f}")
                                st.write("**Models Combined:**")
                                for m in ensemble_res['models_used']:
                                    st.write(f"- {m}")
        else:
            st.warning("Please train a base model first in Step 3 to initialize the process.")
    
    # Bottom navigation (final step)
    create_bottom_navigation(
        current_step=5,
        show_next=False  # Last step, no next button
    )


def generate_analysis_report():
    """Generate a comprehensive analysis report"""
    report_content = []
    report_content.append("=== EXPLAINABLE ML PLAYGROUND - ANALYSIS REPORT ===\n")
    report_content.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Dataset information
    if st.session_state.data_loaded:
        summary = st.session_state.data_handler.get_preprocessing_summary()
        report_content.append("DATASET INFORMATION:\n")
        report_content.append(f"- Task Type: {summary['task_type']}\n")
        report_content.append(f"- Target Column: {summary['target_column']}\n")
        report_content.append(f"- Total Features: {summary['n_features']}\n")
        report_content.append(f"- Training Samples: {summary['train_size']}\n")
        report_content.append(f"- Test Samples: {summary['test_size']}\n\n")
    
    # Model information
    if st.session_state.model_trained:
        report_content.append("MODEL INFORMATION:\n")
        report_content.append(f"- Model Type: {st.session_state.model_trainer.model_name}\n")
        report_content.append(f"- Task: {st.session_state.model_trainer.task_type}\n\n")
        
        # Performance metrics
        report_content.append("PERFORMANCE METRICS:\n")
        for metric, value in st.session_state.model_trainer.metrics.items():
            if metric != 'confusion_matrix':
                report_content.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}\n")
        report_content.append("\n")
    
    # Feature importance (if available)
    if st.session_state.model_trained and hasattr(st.session_state.model_trainer.model, 'feature_importances_'):
        feature_names = st.session_state.data_handler.get_preprocessing_summary()['feature_names']
        importance_df = st.session_state.model_trainer.get_feature_importance(feature_names)
        if importance_df is not None:
            report_content.append("TOP 10 FEATURE IMPORTANCE:\n")
            for _, row in importance_df.head(10).iterrows():
                report_content.append(f"- {row['feature']}: {row['importance']:.4f}\n")
            report_content.append("\n")
    
    report_content.append("=== END OF REPORT ===")
    
    return "\n".join(report_content)


if __name__ == "__main__":
    main() 