
import pandas as pd
import numpy as np

class AIAdvisor:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.data = data_handler.data
        self.target = data_handler.target_column
        self.task_type = data_handler.task_type
        
    def analyze_data_quality(self):
        """Analyze data quality and return alerts"""
        alerts = []
        
        if self.data is None:
            return ["No data loaded"]
            
        # Check for missing values
        missing_percent = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
        if missing_percent > 5:
            alerts.append(f"⚠️ High missing values detected ({missing_percent:.1f}% overall). Preprocessing will impute these.")
            
        # Check for class imbalance (Classification only)
        if self.task_type == "Classification" and self.target:
            target_counts = self.data[self.target].value_counts(normalize=True)
            min_class_ratio = target_counts.min()
            if min_class_ratio < 0.1:
                alerts.append(f"⚠️ **Severe class imbalance** detected (smallest class is only {min_class_ratio*100:.1f}%). Consider using SMOTE or class weights.")
            elif min_class_ratio < 0.2:
                alerts.append(f"ℹ️ **Moderate class imbalance** detected. Stratified split recommended (already enabled).")
        
        # Check target cardinality
        if self.target:
            unique_targets = self.data[self.target].nunique()
            if self.data[self.target].dtype == 'object' and unique_targets > 100:
                alerts.append(f"❌ **Unsuitable Target**: The target column '{self.target}' has {unique_targets} unique text values. This is usually a 'Name' or 'ID' column and cannot be predicted reliably.")
            elif self.task_type == "Regression" and self.data[self.target].dtype == 'object':
                alerts.append(f"ℹ️ **Text Target in Regression**: Predicting text labels in a regression task. We've automatically encoded them to numbers for you.")
                
        # Check for high dimensionality
        if self.data.shape[1] > 20:
            alerts.append(f"ℹ️ High dimensionality ({self.data.shape[1]} features). Feature selection might improve performance.")
            
        # Check for small dataset
        if self.data.shape[0] < 100:
            alerts.append("⚠️ Small dataset (< 100 samples). Model might overfit. Cross-validation is crucial.")
            
        if not alerts:
            alerts.append("✅ Data quality looks good!")
            
        return alerts

    def suggest_model(self):
        """Suggest the best model based on data characteristics"""
        if self.data is None or self.target is None:
            return None, None
            
        n_samples = self.data.shape[0]
        n_features = self.data.shape[1]
        
        # Simple rule-based logic
        if self.task_type == "Classification":
            if n_samples < 1000:
                if n_features > 10:
                    return "Random Forest", "Handling high dimensionality and interactions well on smaller data."
                else:
                    return "Logistic Regression", "Simple and interpretable for small, low-dimensional data."
            else:
                return "Random Forest", "Robust performance on larger datasets."
        else:
            # Regression
            if n_samples < 50:
                return "Linear Regression", "Avoid overfitting on very small data."
            else:
                return "Random Forest", "Capturing non-linear relationships effectively."
                
    def get_feature_advice(self):
        """Give advice on feature engineering"""
        advice = []
        
        # Check for skewness in numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if self.target in numeric_cols:
            numeric_cols = numeric_cols.drop(self.target)
            
        for col in numeric_cols:
            # Check skewness (simple check: mean vs median)
            mean_val = self.data[col].mean()
            median_val = self.data[col].median()
            std_val = self.data[col].std()
            
            if abs(mean_val - median_val) > std_val:
                advice.append(f"• Feature '{col}' appears skewed. Log transformation might help.")
                
        # Check for high cardinality categorical
        cat_cols = self.data.select_dtypes(include=['object']).columns
        if self.target in cat_cols:
            cat_cols = cat_cols.drop(self.target)
            
        for col in cat_cols:
            if self.data[col].nunique() > 10:
                 advice.append(f"• Feature '{col}' has many unique values ({self.data[col].nunique()}). Target encoding might be better than Label Encoding.")
                 
        if not advice:
            advice.append("• Features look standard. Standard Scaling (enabled) is recommended.")
            
        return advice
