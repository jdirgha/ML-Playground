import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import io


class DataHandler:
    def __init__(self):
        self.data = None
        self.filename = None
        self.target_column = None
        self.feature_columns = None
        self.task_type = None
        self.label_encoders = {}
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self, uploaded_file):
        """Load CSV or Excel data with robust encoding handling"""
        try:
            if uploaded_file.name.endswith('.csv'):
                # Robust CSV reading with multiple encoding fallbacks
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
                success = False
                
                for enc in encodings:
                    try:
                        # Reset file pointer to beginning for each attempt
                        uploaded_file.seek(0)
                        self.data = pd.read_csv(uploaded_file, encoding=enc)
                        self.filename = uploaded_file.name
                        success = True
                        break
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
                
                if not success:
                    st.error("Could not decode CSV file. Please ensure it's a valid CSV with standard encoding.")
                    return False
                    
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
                self.filename = uploaded_file.name
            else:
                st.error("Please upload a CSV or Excel file")
                return False
            
            # Basic validation
            if self.data is None or self.data.empty:
                st.error("The uploaded file is empty or could not be read.")
                return False
            
            if len(self.data.columns) < 2:
                st.error("Dataset must have at least 2 columns (features and target)")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def get_data_info(self):
        """Get basic information about the dataset"""
        if self.data is None:
            return None
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns)
        }
        return info
    
    def display_data_preview(self):
        """Display data preview and statistics"""
        if self.data is None:
            return
        
        st.subheader("📊 Data Preview")
        
        # Check if this is Titanic dataset and show survival stats prominently
        if 'Survived' in self.data.columns:
            st.info("🛳️ **Titanic Survival Dataset** - Predict passenger survival")
            
            # Show survival statistics prominently
            col1, col2, col3 = st.columns(3)
            with col1:
                total_passengers = len(self.data)
                st.metric("Total Passengers", total_passengers)
            
            with col2:
                survivors = self.data['Survived'].sum()
                st.metric("Survivors", survivors, f"{(survivors/total_passengers*100):.1f}%")
            
            with col3:
                deaths = total_passengers - survivors
                st.metric("Deaths", deaths, f"{(deaths/total_passengers*100):.1f}%")
            
            st.write("**🎯 Goal**: Predict if a passenger survived (1) or died (0) based on their characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", self.data.shape)
            st.write("**First 5 rows:**")
            st.dataframe(self.data.head())
        
        with col2:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': self.data.dtypes.index,
                'Type': [str(dtype) for dtype in self.data.dtypes.values]
            })
            st.dataframe(dtype_df)
            
            # Show survival breakdown if Titanic dataset
            if 'Survived' in self.data.columns:
                st.write("**Survival by Features:**")
                if 'Sex' in self.data.columns:
                    survival_by_sex = self.data.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
                    st.write("By Gender:")
                    st.dataframe(survival_by_sex.astype(str))
                
                if 'Pclass' in self.data.columns:
                    survival_by_class = self.data.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
                    st.write("By Class:")
                    st.dataframe(survival_by_class.astype(str))
            else:
                pass  # Placeholder for non-Titanic datasets
            
            # Data Overview Metrics (for all datasets)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", self.data.shape[0])
            with col2:
                st.metric("Columns", self.data.shape[1])
            with col3:
                missing_total = self.data.isnull().sum().sum()
                st.metric("Missing Values", missing_total, delta_color="inverse")
            
            # Missing Values Analysis
            if missing_total > 0:
                with st.expander("⚠️ Missing Value Details"):
                    missing_df = pd.DataFrame({
                        'Column': self.data.columns,
                        'Missing Count': self.data.isnull().sum().values,
                        'Percentage': (self.data.isnull().sum() / len(self.data) * 100)
                    })
                    missing_filtered = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                    
                    try:
                        st.dataframe(missing_filtered.style.format({'Percentage': "{:.1f}%"}), use_container_width=True)
                    except Exception:
                        # Fallback if jinja2 is missing/outdated
                        st.dataframe(missing_filtered, use_container_width=True)
            
            # Data Types
            with st.expander("📋 Column Types"):
                 dtype_counts = self.data.dtypes.astype(str).value_counts()
                 st.bar_chart(dtype_counts)
    
    def select_target_and_features(self):
        """Interface for selecting target column and task type"""
        if self.data is None:
            return False
        
        st.subheader("🎯 Select Target Variable")
        
        # Target column selection
        self.target_column = st.selectbox(
            "Choose the column to predict:",
            options=self.data.columns.tolist(),
            help="This is the variable your model will learn to predict"
        )
        
        if self.target_column:
            # Show target variable statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Target Variable Preview:**")
                st.write(f"Column: {self.target_column}")
                st.write(f"Data Type: {self.data[self.target_column].dtype}")
                st.write(f"Unique Values: {self.data[self.target_column].nunique()}")
                
            with col2:
                st.write("**Value Distribution:**")
                if self.data[self.target_column].nunique() <= 20:
                    value_counts = self.data[self.target_column].value_counts()
                    st.write(value_counts)
                    
                    # Special handling for Titanic survival
                    if self.target_column == 'Survived':
                        st.write("**Meaning:**")
                        st.write("• **0** = Did not survive (died)")
                        st.write("• **1** = Survived")
                        st.write(f"• **Survival rate**: {(value_counts.get(1, 0)/len(self.data)*100):.1f}%")
                else:
                    st.write(self.data[self.target_column].describe())
                    
                # High cardinality warning for categorical targets
                if self.data[self.target_column].dtype == 'object' and self.data[self.target_column].nunique() > 100:
                    st.warning("⚠️ **High Cardinality Target**: You selected a column with many unique text values (like names or IDs). This is usually not suitable for prediction. Consider selecting a different target column.")
            
            # Task type selection
            st.subheader("📋 Select Prediction Type")
            
            # Auto-suggest task type
            unique_values = self.data[self.target_column].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(self.data[self.target_column])
            
            if unique_values <= 10 and not is_numeric:
                suggested_task = "Classification"
            elif unique_values <= 10 and is_numeric:
                suggested_task = "Classification" if unique_values <= 5 else "Regression"
            else:
                suggested_task = "Regression"
            
            self.task_type = st.radio(
                f"What type of prediction? (Suggested: {suggested_task})",
                options=["Classification", "Regression"],
                help="Classification: Predict categories (yes/no, fraud/not fraud)\nRegression: Predict numbers (price, temperature)"
            )
            
            # Feature columns (all except target)
            self.feature_columns = [col for col in self.data.columns if col != self.target_column]
            
            st.write(f"**Features to use:** {len(self.feature_columns)} columns")
            with st.expander("View feature columns"):
                st.write(self.feature_columns)
            
            return True
        
        return False
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """Preprocess data for machine learning"""
        if self.data is None or self.target_column is None:
            return False
        
        try:
            # Separate features and target
            X = self.data[self.feature_columns].copy()
            y = self.data[self.target_column].copy()
            
            # --- ROBUSTNESS IMPROVEMENTS ---
            
            # 1. Drop features with single value (zero variance)
            for col in X.columns:
                if X[col].nunique() <= 1:
                    X.drop(columns=[col], inplace=True)
                    if col in self.feature_columns:
                        self.feature_columns.remove(col)
                        
            # 2. Identify and drop high-cardinality non-numeric columns (likely IDs)
            # Threshold: if unique values > 90% of rows and type is object
            threshold_ratio = 0.9
            for col in X.select_dtypes(include=['object']).columns:
                if X[col].nunique() / len(X) > threshold_ratio:
                    X.drop(columns=[col], inplace=True)
                    if col in self.feature_columns:
                        self.feature_columns.remove(col)
            
            # 3. Handle Date/Time columns (simple approach: drop them for now)
            # In a real app, we would extract year/month/day
            for col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    X.drop(columns=[col], inplace=True)
                    if col in self.feature_columns:
                        self.feature_columns.remove(col)

            # Update feature columns list after dropping
            self.feature_columns = list(X.columns)
            
            # -------------------------------
            
            # Handle missing values
            # For numeric columns, fill with median
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                X[col].fillna(X[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
            
            # Encode categorical variables
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            
            # Handle target variable
            # --- FORCED ENCODING FIX ---
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
                self.label_encoders['target'] = le_target
                st.info(f"ℹ️ Encoded target column '{self.target_column}' into numeric values.")
            elif self.task_type == "Classification" and y.nunique() <= 10:
                # Still encode if it's classification with few numbers to ensure 0, 1, 2... format
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                self.label_encoders['target'] = le_target
            # ---------------------------
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y if self.task_type == "Classification" else None
            )
            
            # Scale features for certain algorithms
            self.scaler = StandardScaler()
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            return True
            
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return False
    
    def get_preprocessing_summary(self):
        """Get summary of preprocessing steps"""
        if self.X_train is None:
            return None
        
        summary = {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'n_features': self.X_train.shape[1],
            'task_type': self.task_type,
            'target_column': self.target_column,
            'encoded_columns': list(self.label_encoders.keys()),
            'feature_names': list(self.X_train.columns)
        }
        
        if self.task_type == "Classification":
            summary['n_classes'] = len(np.unique(self.y_train))
            summary['class_distribution'] = pd.Series(self.y_train).value_counts().to_dict()
        
        return summary
    
    def get_sample_prediction_data(self, n_samples=5):
        """Get sample data for prediction explanations"""
        if self.X_test is None:
            return None
        
        # Get random samples from test set
        sample_indices = np.random.choice(len(self.X_test), min(n_samples, len(self.X_test)), replace=False)
        samples = self.X_test.iloc[sample_indices]
        actual_values = self.y_test.iloc[sample_indices] if hasattr(self.y_test, 'iloc') else self.y_test[sample_indices]
        
        return samples, actual_values, sample_indices


def load_sample_dataset(dataset_name):
    """Load predefined sample datasets"""
    if dataset_name == "Iris Classification":
        from sklearn.datasets import load_iris
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['species'] = iris.target
        data['species'] = data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return data
    
    elif dataset_name == "California Housing Regression":
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        data = pd.DataFrame(california.data, columns=california.feature_names)
        data['price'] = california.target
        return data
    
    elif dataset_name == "Wine Classification":
        from sklearn.datasets import load_wine
        wine = load_wine()
        data = pd.DataFrame(wine.data, columns=wine.feature_names)
        data['wine_class'] = wine.target
        return data
    
    elif dataset_name == "Titanic Survival":
        # Load the simple Titanic dataset
        import os
        file_path = os.path.join("sample_data", "titanic.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            return data
        else:
            return None
    
    return None 