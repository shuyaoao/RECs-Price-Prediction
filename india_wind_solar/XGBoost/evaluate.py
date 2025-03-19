"""Evaluation script for XGBoost model."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from india_wind_solar import config
from india_wind_solar.XGBoost.data.dataloader import prepare_data_for_model
from india_wind_solar.XGBoost.models.xgboost import load_model
from india_wind_solar.XGBoost.utils.visualisation import plot_predictions

def evaluate():
    """Evaluate the XGBoost model on test data."""
    print("Preparing data...")
    _, _, _, _, X_test, y_test, scaler, _, _, test_dates = prepare_data_for_model()
    
    print(f"Test data shape: {X_test.shape}")
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/saved", config.XGBOOST_MODEL_NAME)
    model = load_model(model_path)
    print("Loaded Model")
    
    # Make predictions
    test_predictions = model.predict(X_test)
    
    # Inverse transform to get original scale
    test_predictions_rescaled = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_rescaled, test_predictions_rescaled)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, test_predictions_rescaled)
    r2 = r2_score(y_test_rescaled, test_predictions_rescaled)
    
    # Calculate MAPE
    # Avoid division by zero
    mask = y_test_rescaled != 0
    mape = 100 * np.mean(np.abs((y_test_rescaled[mask] - test_predictions_rescaled[mask]) / y_test_rescaled[mask]))
    
    # Print metrics
    print("\nTest Set Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot predictions
    test_dates_array = np.array(test_dates)[config.SEQUENCE_LENGTH:]
    
    plot_predictions(
        y_test,
        test_predictions,
        dates=test_dates_array,
        scaler=scaler,
        title='XGBoost Test Predictions',
        save_path=os.path.join(results_dir, 'test_predictions.png')
    )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²'],
        'Value': [mse, rmse, mae, mape, r2]
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
    
    return mse, rmse, mae, mape, r2, predictions_df

if __name__ == "__main__":
    evaluate()