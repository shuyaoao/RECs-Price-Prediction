"""LightGBM model for time series forecasting."""
import numpy as np
import lightgbm as lgb
import os
import pickle
from india_wind_solar import config

def build_model():
    """
    Build and configure LightGBM model.
    
    Returns:
        Configured LightGBM regressor
    """
    # Create model using parameters from config
    model = lgb.LGBMRegressor(**config.LIGHTGBM_PARAMS)
    
    return model

def save_model(model, path):
    """
    Save LightGBM model to disk using pickle.
    
    Args:
        model: Trained LightGBM model
        path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model in pickle format
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved in pickle format to {path}")

def load_model(path):
    """
    Load LightGBM model from disk using pickle.
    
    Args:
        path: Path to saved model
        
    Returns:
        Loaded LightGBM model
    """
    print(f"Loading model from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    # Load model from pickle
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    return model