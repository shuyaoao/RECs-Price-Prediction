"""XGBoost model for time series forecasting."""
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os
from india_wind_solar import config

def build_model():
    """
    Build and configure XGBoost model.
    
    Returns:
        Configured XGBoost regressor
    """
    
    # Create model
    model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    
    return model

def save_model(model, path):
    """
    Save XGBoost model to disk.
    
    Args:
        model: Trained XGBoost model
        path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Load XGBoost model from disk.
    
    Args:
        path: Path to saved model
        
    Returns:
        Loaded XGBoost model
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model