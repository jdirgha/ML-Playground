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

# Page configuration
st.set_page_config(
    page_title="Explainable ML Playground",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple font styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    .stMarkdown, .stText {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

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
            if st.button("Back", key="back_bottom_" + str(current_step)):
                st.session_state.step = current_step - 1
                st.experimental_rerun()
    
    with col2:
        st.write(f"Step {current_step} of 5")
    
    with col3:
        if show_next and not next_disabled:
            next_step_num = next_step if next_step else current_step + 1
            label = next_label if next_label else "Next Step"
            if st.button(label, key="next_bottom_" + str(current_step)):
                st.session_state.step = next_step_num
                st.experimental_rerun()
        elif next_disabled and show_next:
            disabled_key = "disabled_bottom_" + str(current_step)
            st.button(next_label or "Complete current step first", disabled=True, key=disabled_key)


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.title("🔬 Explainable ML Playground")
    st.markdown("Build, train and explain machine learning models with interactive insights")
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        
        # Step indicators with clickable navigation
        steps = [
            ("📁 Data Upload", 1),
            ("🎯 Target Selection", 2), 
            ("🤖 Model Training", 3),
            ("🔍 Explanations", 4),
            ("📊 Analysis", 5)
        ]
        
        st.write("*Click on completed steps to navigate*")
        
        for step_name, step_num in steps:
            # Determine if step is accessible
            step_accessible = step_num <= st.session_state.step
            
            # Different styling based on current step and accessibility
            if step_num == st.session_state.step:
                # Current step - highlighted
                st.write("**🔄 " + step_name + "**")
            elif step_accessible:
                # Completed step - clickable
                nav_key = "nav_step_" + str(step_num)
                button_text = "✅ " + step_name
                if st.button(button_text, key=nav_key):
                    st.session_state.step = step_num
                    st.experimental_rerun()
            else:
                # Future step - disabled
                st.write("⏳ " + step_name)
        
        st.markdown("---")
        
        # Progress bar
        progress = (st.session_state.step - 1) / 4  # 5 steps total, so max progress is 4/4
        st.progress(progress)
        st.caption(f"Step {st.session_state.step} of 5")
        
        st.markdown("---")
        
        # Actions
        if st.button("🔄 Reset Application", help="Start over with a fresh session"):
            # Reset only necessary keys, keep system keys
            keys_to_reset = [
                'data_handler', 'model_trainer', 'explainer', 'step', 'data_loaded',
                'model_trained', 'explanations_ready', 'preprocessing_complete', 
                'model_training_complete'
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_rerun()
        
        if st.session_state.model_trained:
            if st.button("📥 Download Model"):
                model_data = st.session_state.model_trainer.save_model("model.pkl")
                if model_data:
                    st.download_button(
                        label="💾 Download Model File",
                        data=model_data,
                        file_name="trained_model.pkl",
                        mime="application/octet-stream"
                    )
    
    # Main content area
    if st.session_state.step == 1:
        step_1_data_upload()
    elif st.session_state.step == 2:
        step_2_target_selection()
    elif st.session_state.step == 3:
        step_3_model_training()
    elif st.session_state.step == 4:
        step_4_explanations()
    elif st.session_state.step == 5:
        step_5_analysis()


def step_1_data_upload():
    """Step 1: Data Upload"""
    st.header("📁 Step 1: Data Upload")
    
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
                st.success("✅ Data loaded successfully!")
    
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
                st.session_state.data_loaded = True
                st.success("✅ Sample data loaded successfully!")
    
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


def step_2_target_selection():
    """Step 2: Target and Feature Selection"""
    st.header("🎯 Step 2: Target Variable Selection")
    
    if not st.session_state.data_loaded:
        st.error("Please upload data first!")
        create_bottom_navigation(current_step=2, show_next=False)
        return
    
    # Check if preprocessing is already complete
    if st.session_state.get('preprocessing_complete', False):
        st.success("✅ Data preprocessing completed!")
        
        # Show summary
        summary = st.session_state.data_handler.get_preprocessing_summary()
        if summary:
            st.subheader("📋 Preprocessing Summary")
            
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
    if st.session_state.data_handler.select_target_and_features():
        
        # Data preprocessing
        st.subheader("🔧 Data Preprocessing")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        with col2:
            random_state = st.number_input("Random seed", 1, 100, 42)
        
        if st.button("🔄 Preprocess Data", key="preprocess_data"):
            if st.session_state.data_handler.preprocess_data(test_size, random_state):
                
                # Mark preprocessing as complete
                st.session_state.preprocessing_complete = True
                
                st.success("✅ Data preprocessing completed!")
                st.experimental_rerun()
    
    # Bottom navigation
    create_bottom_navigation(
        current_step=2,
        next_label="Proceed to Model Training" if st.session_state.get('preprocessing_complete', False) else "Preprocess data first",
        next_disabled=not st.session_state.get('preprocessing_complete', False)
    )


def step_3_model_training():
    """Step 3: Model Training"""
    st.header("🤖 Step 3: Model Training")
    
    # Check if data is preprocessed
    if not st.session_state.get('preprocessing_complete', False) or st.session_state.data_handler.X_train is None:
        st.error("Please preprocess data first!")
        create_bottom_navigation(current_step=3, show_next=False)
        return
    
    # Check if model training is already complete
    if st.session_state.get('model_training_complete', False) and st.session_state.model_trained:
        st.success("✅ Model training completed!")
        
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
        
        st.subheader("⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            perform_cv = st.checkbox("Perform Cross-Validation", value=True)
        with col2:
            cv_folds = st.number_input("CV Folds", 3, 10, 5) if perform_cv else 5
        
        # Train model button
        if st.button("🚀 Train Model"):
            
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
                st.success("✅ Model trained successfully!")
                
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
    st.header("🔍 Step 4: Model Explanations")
    
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
        if st.button("🧠 Generate Explanations"):
            if st.session_state.explainer.initialize_explainer():
                if st.session_state.explainer.calculate_shap_values():
                    st.session_state.explanations_ready = True
                    st.success("✅ Explanations generated successfully!")
                    st.experimental_rerun()
    
    if st.session_state.explanations_ready:
        
        # Explanation tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌍 Global Explanations", 
            "🔎 Individual Predictions", 
            "📊 Feature Dependencies",
            "🎯 Model Insights"
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
    st.header("📊 Step 5: Advanced Analysis & Export")
    
    if not st.session_state.explanations_ready:
        st.error("Please generate explanations first!")
        create_bottom_navigation(current_step=5, show_next=False)
        return
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📈 Model Analysis", "📥 Export Results", "🔧 Model Comparison"])
    
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
            if st.button("📊 Export SHAP Explanations"):
                explanations_df = st.session_state.explainer.export_explanations()
                if explanations_df is not None:
                    csv = explanations_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Explanations CSV",
                        data=csv,
                        file_name="shap_explanations.csv",
                        mime="text/csv"
                    )
        
        with col2:
            # Export predictions
            if st.button("🎯 Export Predictions"):
                predictions_df = pd.DataFrame({
                    'prediction': st.session_state.model_trainer.predictions,
                    'actual': st.session_state.data_handler.y_test
                })
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="model_predictions.csv",
                    mime="text/csv"
                )
        
        # Generate report
        if st.button("📋 Generate Analysis Report"):
            report = generate_analysis_report()
            st.download_button(
                label="📥 Download Analysis Report",
                data=report,
                file_name="ml_analysis_report.txt",
                mime="text/plain"
            )
    
    with tab3:
        st.subheader("Model Comparison")
        st.info("💡 Feature coming soon: Compare multiple models side by side!")
        
        # Placeholder for future model comparison features
        st.write("**Planned Features:**")
        st.write("• Train multiple models simultaneously")
        st.write("• Compare performance metrics")
        st.write("• Compare explanation differences")
        st.write("• Model ensemble capabilities")
    
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
        report_content.append(f"• Task Type: {summary['task_type']}\n")
        report_content.append(f"• Target Column: {summary['target_column']}\n")
        report_content.append(f"• Total Features: {summary['n_features']}\n")
        report_content.append(f"• Training Samples: {summary['train_size']}\n")
        report_content.append(f"• Test Samples: {summary['test_size']}\n\n")
    
    # Model information
    if st.session_state.model_trained:
        report_content.append("MODEL INFORMATION:\n")
        report_content.append(f"• Model Type: {st.session_state.model_trainer.model_name}\n")
        report_content.append(f"• Task: {st.session_state.model_trainer.task_type}\n\n")
        
        # Performance metrics
        report_content.append("PERFORMANCE METRICS:\n")
        for metric, value in st.session_state.model_trainer.metrics.items():
            if metric != 'confusion_matrix':
                report_content.append(f"• {metric.replace('_', ' ').title()}: {value:.4f}\n")
        report_content.append("\n")
    
    # Feature importance (if available)
    if st.session_state.model_trained and hasattr(st.session_state.model_trainer.model, 'feature_importances_'):
        feature_names = st.session_state.data_handler.get_preprocessing_summary()['feature_names']
        importance_df = st.session_state.model_trainer.get_feature_importance(feature_names)
        if importance_df is not None:
            report_content.append("TOP 10 FEATURE IMPORTANCE:\n")
            for _, row in importance_df.head(10).iterrows():
                report_content.append(f"• {row['feature']}: {row['importance']:.4f}\n")
            report_content.append("\n")
    
    report_content.append("=== END OF REPORT ===")
    
    return "\n".join(report_content)


if __name__ == "__main__":
    main() 