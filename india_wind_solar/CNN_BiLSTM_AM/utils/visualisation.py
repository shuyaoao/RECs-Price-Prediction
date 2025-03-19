"""Visualization utilities for model analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_training_history(history, save_path=None):
    """
    Plot training history of a PyTorch model.
    
    Args:
        history: Dictionary containing 'loss', 'val_loss', and 'lr'
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot learning rate
    ax2.plot(history['lr'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_predictions(actual, predictions, dates, scaler=None, title='Model Predictions', save_path=None):
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
        plt.plot(dates, actual_rescaled, label='Actual')
        plt.plot(dates, predictions_rescaled, label='Predicted')
    else:
        plt.plot(dates, actual, label='Actual')
        plt.plot(dates, predictions, label='Predicted')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Predictions plot saved to {save_path}")
    
    plt.show()