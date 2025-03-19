"""Evaluation script for CNN-BiLSTM-AM model."""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from india_wind_solar import config
from india_wind_solar.CNN_BiLSTM_AM.data.dataloader import prepare_data_for_model
from india_wind_solar.CNN_BiLSTM_AM.models.cnn_bilstm_am import CNNBiLSTMAM, AttentionLayer
from india_wind_solar.CNN_BiLSTM_AM.utils.visualisation import plot_predictions

def evaluate():
    """Evaluate the CNN-BiLSTM-AM model on test data."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, train_dates, val_dates, test_dates = prepare_data_for_model()
    
    # Convert test data to PyTorch tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    print(f"Test data shape: {X_test.shape}")
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.MODEL_SAVE_DIR, config.MODEL_NAME)
    model = CNNBiLSTMAM(sequence_length=config.SEQUENCE_LENGTH)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_predictions = test_predictions.cpu().numpy()
    
    # Reshape predictions
    test_predictions = test_predictions.flatten()
    
    # Inverse transform to get original scale
    test_predictions_rescaled = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
    mask = y_test_rescaled != 0
    mape = 100 * np.mean(np.abs((y_test_rescaled[mask] - test_predictions_rescaled[mask]) / y_test_rescaled[mask]))
    
    # Print metrics
    print("\nTest Set Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), config.RESULTS_DIR)
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot predictions
    test_dates_array = np.array(test_dates)[config.SEQUENCE_LENGTH:]
    
    plot_predictions(
        y_test,
        test_predictions,
        test_dates_array,
        scaler=scaler,
        title='CNN-BiLSTM-AM Test Predictions',
        save_path=os.path.join(results_dir, 'test_predictions.png')
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE'],
        'Value': [mse, rmse, mae, mape]
    })
    metrics_df.to_csv(os.path.join(results_dir, 'test_metrics.csv'), index=False)
    print(f"Metrics saved to {os.path.join(results_dir, 'test_metrics.csv')}")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Date': test_dates_array,
        'Actual': y_test_rescaled,
        'Predicted': test_predictions_rescaled
    })
    predictions_df.to_csv(os.path.join(results_dir, 'test_predictions.csv'), index=False)
    print(f"Predictions saved to {os.path.join(results_dir, 'test_predictions.csv')}")
    
    return mse, rmse, mae, mape, predictions_df

if __name__ == "__main__":
    evaluate()