# 🔬 Explainable ML Playground

A comprehensive web application for machine learning experimentation with built-in explainability features. Upload your data, train models, and understand predictions with AI-powered explanations.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

<img width="1440" height="816" alt="Playground" src="https://github.com/user-attachments/assets/d44e76d6-fca3-4c93-9ad4-77d4a9021e5e" />


## 🚀 Features

### 📊 **Data Management**
- **File Upload**: Support for CSV and Excel files
- **Sample Datasets**: Pre-loaded datasets for immediate experimentation
- **Data Validation**: Automatic data quality checks and preprocessing
- **Missing Value Handling**: Smart imputation strategies

### 🤖 **Machine Learning**
- **Multiple Algorithms**: Random Forest, Logistic Regression, Linear Regression, SVM
- **Task Types**: Both Classification and Regression supported
- **Automated Preprocessing**: Feature encoding, scaling, and train/test splits
- **Cross-Validation**: Built-in model validation with performance metrics

### 🔍 **Explainability (SHAP Integration)**
- **Global Explanations**: Understand overall model behavior
- **Local Explanations**: Explain individual predictions
- **Feature Importance**: Interactive visualizations
- **Dependence Plots**: Explore feature relationships
- **Waterfall Charts**: Step-by-step prediction breakdown

### 📈 **Visualizations**
- **Performance Metrics**: Accuracy, precision, recall, F1-score, R², RMSE
- **Confusion Matrices**: Classification performance analysis
- **Residual Plots**: Regression model diagnostics
- **Interactive Charts**: Plotly-powered visualizations

### 📥 **Export & Reporting**
- **Model Export**: Save trained models for future use
- **Explanation Export**: Download SHAP values and explanations
- **Analysis Reports**: Comprehensive model performance reports
- **Prediction Export**: Download predictions and actual values

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone or Download** the project files to your local machine

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Application**
```bash
streamlit run app.py
```

4. **Open in Browser**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## 📋 Quick Start Guide

### Step 1: Upload Data
- **Option A**: Upload your own CSV/Excel file
- **Option B**: Use sample datasets (Titanic Survival, Iris, California Housing, Wine)

### Step 2: Configure Target
- Select the column you want to predict
- Choose task type (Classification or Regression)
- Configure preprocessing options

### Step 3: Train Model
- Select ML algorithm
- Configure cross-validation
- Train and evaluate the model

### Step 4: Explore Explanations
- Generate SHAP explanations
- Explore global feature importance
- Analyze individual predictions
- Understand feature dependencies

### Step 5: Export Results
- Download trained models
- Export explanations and predictions
- Generate analysis reports

## 📊 Supported Data Formats

### Input Requirements
- **File Types**: CSV, Excel (.xlsx, .xls)
- **Structure**: Tabular data with headers
- **Minimum**: 2 columns (features + target)
- **Size**: Recommended < 10MB for optimal performance

### Data Types
- **Numeric**: Integers, floats, continuous values
- **Categorical**: Text, categories, binary values
- **Mixed**: Combination of numeric and categorical

### Preprocessing
- **Missing Values**: Automatic imputation (median for numeric, mode for categorical)
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler for algorithms that benefit from scaling

## 🤖 Machine Learning Algorithms

### Classification
- **Random Forest Classifier**: Ensemble method, handles mixed data types well
- **Logistic Regression**: Linear classifier, fast and interpretable
- **Support Vector Machine**: Non-linear classification with RBF kernel

### Regression
- **Random Forest Regressor**: Ensemble method for numerical prediction
- **Linear Regression**: Simple linear relationships
- **Support Vector Regression**: Non-linear regression capabilities

### Performance Metrics

#### Classification
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

#### Regression
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

## 🔍 Explainability Features

### SHAP (SHapley Additive exPlanations)
Our application uses SHAP to provide model-agnostic explanations:

#### Global Explanations
- **Feature Importance**: Which features matter most overall
- **Summary Plots**: Distribution of feature impacts
- **Interaction Effects**: How features work together

#### Local Explanations
- **Individual Predictions**: Why the model made a specific prediction
- **Force Plots**: Push/pull factors for each prediction
- **Waterfall Charts**: Step-by-step contribution breakdown

#### Advanced Analysis
- **Dependence Plots**: Relationship between feature values and predictions
- **Partial Dependence**: Average effect of features
- **Feature Interactions**: How features influence each other

## 📁 Project Structure

```
explainable-ml-playground/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── utils/                 # Utility modules
│   ├── data_handler.py   # Data processing and validation
│   ├── model_trainer.py  # ML model training and evaluation
│   └── explainer.py      # SHAP explanations and visualizations
├── sample_data/          # Sample datasets
│   ├── customer_churn.csv
│   └── house_prices.csv
└── README.md            # This file
```

## 🎯 Use Cases

### Business Analytics
- **Customer Churn**: Predict and understand why customers leave
- **Sales Forecasting**: Predict revenue and understand driving factors
- **Risk Assessment**: Identify high-risk scenarios with explanations

### Research & Education
- **Model Comparison**: Compare different algorithms systematically
- **Feature Analysis**: Understand which variables matter most
- **Teaching Tool**: Demonstrate ML concepts with visual explanations

### Data Science Projects
- **Rapid Prototyping**: Quickly test ideas and hypotheses
- **Model Validation**: Ensure models are learning meaningful patterns
- **Stakeholder Communication**: Explain models to non-technical audiences

## 🔧 Customization

### Adding New Algorithms
1. Modify `utils/model_trainer.py`
2. Add the algorithm to `get_available_models()`
3. Ensure it follows scikit-learn interface

### Custom Datasets
1. Place CSV files in `sample_data/`
2. Add loading logic to `load_sample_dataset()` in `utils/data_handler.py`

### Styling
- Modify CSS in `app.py` for custom appearance
- Adjust color schemes in visualization functions

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**2. Memory Issues with Large Datasets**
- Use smaller datasets (< 10MB recommended)
- Reduce number of SHAP samples in explainer

**3. SHAP Calculation Takes Too Long**
- Reduce sample size in `calculate_shap_values()`
- Use TreeExplainer for tree-based models (faster)

**4. Model Training Fails**
- Check for data quality issues
- Ensure target variable is properly formatted
- Verify sufficient data for train/test split

### Performance Tips
- **Dataset Size**: Keep under 10MB for optimal performance
- **Feature Count**: Too many features (>100) may slow explanations
- **Sample Size**: Large datasets may require sampling for SHAP

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Bug Reports**: Open issues with detailed descriptions
2. **Feature Requests**: Suggest new algorithms or visualizations
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Help improve guides and examples

### Development Setup
```bash
# Clone repository
git clone <repository-url>

# Install development dependencies
pip install -r requirements.txt

# Run with development settings
streamlit run app.py --server.runOnSave true
```

## 📚 Learn More

### Recommended Reading
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Related Projects
- [LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations
- [ELI5](https://github.com/TeamHG-Memex/eli5) - Machine Learning Model Inspection
- [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visual ML Diagnostics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

Need help? Here are your options:

1. **Documentation**: Check this README first
2. **Issues**: Open a GitHub issue for bugs
3. **Discussions**: Use GitHub Discussions for questions
4. **Email**: Contact the maintainers directly

---

**Built with ❤️ using Streamlit, SHAP, and scikit-learn**

*Happy experimenting! 🚀* 
