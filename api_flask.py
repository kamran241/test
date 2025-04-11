from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io
import logging
import os
from sklearn.base import BaseEstimator

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
BASE_DIR = r"C:\Users\kamra\OneDrive\Desktop\grok model"
MODEL_PATH = os.path.join(BASE_DIR, 'churn_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'feature_columns.pkl')

# Load model, scaler, and feature columns
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature columns file not found at {FEATURES_PATH}")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    
    if not isinstance(model, BaseEstimator):
        raise TypeError(f"Loaded model is not a scikit-learn estimator, got type: {type(model)}")
    
    logger.info("Model, scaler, and feature columns loaded successfully")
    logger.info(f"Model type: {type(model).__name__}")
    logger.info(f"Feature columns: {feature_columns}")
except Exception as e:
    logger.error(f"Failed to load files: {str(e)}")
    raise

def preprocess_data(df, feature_columns, scaler):
    """
    Preprocess test data to match training pipeline.
    
    Args:
        df (pd.DataFrame): Raw test data
        feature_columns (list): Expected feature columns from training
        scaler (StandardScaler): Fitted scaler from training
    
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    # Drop customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        logger.info("Dropped 'customerID' column")
    
    # Define numeric and categorical columns explicitly as per training
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    original_cols = df.columns.tolist()
    categorical_cols = [col for col in original_cols if col not in numeric_cols]
    
    # Convert numeric columns to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
        logger.info(f"{col} values: {df[col].tolist()}")
    
    # Encode binary categorical columns
    binary_cols = [col for col in categorical_cols if df[col].nunique() == 2]
    label_encoder = LabelEncoder()
    for col in binary_cols:
        df[col] = label_encoder.fit_transform(df[col])
        logger.info(f"Encoded {col}: {df[col].tolist()}")
    
    # One-hot encode multi-class categorical columns
    multi_class_cols = [col for col in categorical_cols if df[col].nunique() > 2]
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)
    logger.info(f"One-hot encoded columns: {multi_class_cols}")
    logger.info(f"Columns after one-hot encoding: {df.columns.tolist()}")
    
    # Align with training feature columns
    missing_cols = [col for col in feature_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    extra_cols = [col for col in df.columns if col not in feature_columns]
    df = df.drop(columns=extra_cols)  # Drop extra columns not in training
    df = df[feature_columns]  # Reorder to match training
    logger.info(f"Columns after alignment: {df.columns.tolist()}")
    
    # Verify numeric columns remain numeric
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"Non-numeric values in {col}: {df[col].tolist()}")
            raise ValueError(f"Column {col} contains non-numeric values after alignment: {df[col].tolist()}")
        logger.info(f"{col} after alignment: {df[col].tolist()}")
    
    # Scale numeric features
    logger.info(f"Scaling numeric columns: {numeric_cols}")
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df

@app.route('/', methods=['GET'])
def home():
    logger.info("Received GET request to / endpoint")
    return jsonify({
        'status': 'success',
        'message': 'Flask API is running. Use POST /predict to make predictions.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received POST request to /predict endpoint")
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if not file.filename.endswith('.csv'):
            logger.error("Uploaded file is not a CSV")
            return jsonify({'status': 'error', 'message': 'Please upload a CSV file'}), 400
        
        file.seek(0)
        content = file.read().decode('utf-8')
        logger.info(f"File content (first 100 chars): {content[:100]}...")
        
        df = pd.read_csv(io.StringIO(content))
        logger.info("CSV parsed successfully")
        logger.info(f"Raw data columns: {df.columns.tolist()}")
        
        X = preprocess_data(df.copy(), feature_columns, scaler)
        logger.info("Data preprocessed successfully")
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        logger.info("Predictions generated successfully")
        
        df['Churn_Prediction'] = ['Yes' if pred == 1 else 'No' for pred in predictions]
        df['Churn_Probability'] = probabilities
        
        return jsonify({
            'status': 'success',
            'predictions': df[['Churn_Prediction', 'Churn_Probability']].to_dict(orient='records')
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = 8000
    try:
        logger.info(f"Starting Flask API on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except OSError as e:
        logger.error(f"Port {port} in use: {str(e)}")
        port = 5000
        logger.info(f"Switching to port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)