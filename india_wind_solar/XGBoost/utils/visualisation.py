"""Visualization utilities for model analysis."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def plot_training_history(model, X_val, y_val, save_path=None):
    """
    Plot training history of XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X_val: Validation features
        y_val: Validation targets
        save_path: Path to save the plot
    """
    results = model.evals_result()
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['validation_0']['rmse'], label='Train')
    if 'validation_1' in results:
        plt.plot(results['validation_1']['rmse'], label='Validation')
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title('Training Performance')
    plt.legend()
    
    # Feature importance
    plt.subplot(1, 2, 2)
    xgb.plot_importance(model, max_num_features=10, height=0.8, ax=plt.gca())
    plt.title('Feature Importance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_predictions(actual, predictions, dates=None, scaler=None, title='Model Predictions', save_path=None):
    """
    Plot actual vs predicted values.
    
    Args:
        actual: Actual values
        predictions: Predicted values
        dates: Date indices for the values
        scaler: Optional scaler for inverse transformation
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # If data is still scaled and scaler is provided, inverse transform
    if scaler is not None:
        actual_rescaled = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
        predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        plt.plot(dates if dates is not None else range(len(actual)), actual_rescaled, label='Actual')
        plt.plot(dates if dates is not None else range(len(predictions)), predictions_rescaled, label='Predicted')
    else:
        plt.plot(dates if dates is not None else range(len(actual)), actual, label='Actual')
        plt.plot(dates if dates is not None else range(len(predictions)), predictions, label='Predicted')
    
    plt.title(title)
    plt.xlabel('Date' if dates is not None else 'Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    if dates is not None:
        plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Predictions plot saved to {save_path}")
    
    plt.show()