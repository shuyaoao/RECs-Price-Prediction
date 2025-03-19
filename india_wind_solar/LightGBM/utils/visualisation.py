"""Visualization utilities for model analysis."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from math import sqrt

def plot_feature_importance(model, save_path=None):
    """
    Plot feature importance for LightGBM model.
    
    Args:
        model: Trained LightGBM model
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        # Get number of features
        n_features = len(model.feature_importances_)
        
        # Create feature names if not provided
        feature_names = [f'Feature {i}' for i in range(n_features)]
        
        # Sort importances
        indices = np.argsort(model.feature_importances_)[::-1]
        
        # Plot
        plt.title('Feature Importance')
        plt.bar(range(n_features), model.feature_importances_[indices])
        plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    else:
        print("Model does not have feature importances attribute")

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

def plot_training_vs_validation_loss(train_losses, val_losses, title='Training vs Validation Loss', save_path=None):
    """
    Plot training loss versus validation loss.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Add annotations for minimum values
    min_train_idx = np.argmin(train_losses)
    min_val_idx = np.argmin(val_losses)
    
    min_train = train_losses[min_train_idx]
    min_val = val_losses[min_val_idx]
    
    plt.annotate(f'Min: {min_train:.6f}', 
                xy=(min_train_idx + 1, min_train),
                xytext=(min_train_idx + 1, min_train * 1.1),
                arrowprops=dict(facecolor='blue', shrink=0.05))
    
    plt.annotate(f'Min: {min_val:.6f}', 
                xy=(min_val_idx + 1, min_val),
                xytext=(min_val_idx + 1, min_val * 1.1),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training vs validation loss plot saved to {save_path}")
    
    plt.show()