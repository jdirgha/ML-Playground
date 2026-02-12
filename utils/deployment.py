import json
import pandas as pd

def generate_fastapi_app(model_name, feature_names, task_type):
    """
    Generate the content for a FastAPI application file.
    
    Args:
        model_name (str): Name of the model (e.g., 'Random Forest')
        feature_names (list): List of feature names the model expects
        task_type (str): 'Classification' or 'Regression'
        
    Returns:
        str: The Python code for the FastAPI app
    """
    
    code = f'''
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

# Initialize FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for {model_name} ({task_type})",
    version="1.0.0"
)

# Load the model
try:
    model_data = joblib.load("model.pkl")
    model = model_data['model']
    print(f"Model loaded successfully: {{model_data['model_name']}}")
    
    # Verify expected features
    EXPECTED_FEATURES = {feature_names}
except Exception as e:
    print(f"Error loading model: {{e}}")
    model = None

# Define input data schema
class InputData(BaseModel):
    {_generate_pydantic_fields(feature_names)}

@app.get("/")
def home():
    return {{"message": "ML Model API is running. Use /predict to get predictions."}}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {{"status": "healthy", "model": "{model_name}"}}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input data to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Ensure correct column order
        df = df[EXPECTED_FEATURES]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        result = {{
            "prediction": float(prediction)
        }}
        
        # Add probabilities for classification if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[0]
            result["probabilities"] = probs.tolist()
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    return code

def _generate_pydantic_fields(feature_names):
    """Generate Pydantic fields for the input schema"""
    fields = []
    for feature in feature_names:
        # We assume float for simplicity, but could be more specific if we knew types
        fields.append(f"{feature}: float")
    return "\n    ".join(fields)

def generate_requirements():
    """Generate requirements.txt content for the API"""
    return """
fastapi>=0.68.0
uvicorn>=0.15.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
pydantic>=1.8.0
"""
