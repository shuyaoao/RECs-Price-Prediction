import os
import pandas as pd
import matplotlib.pyplot as plt
from india_wind_solar.ARIMA.model import model
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pickle
from india_wind_solar import config

# Get base directory for absolute paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load test dataset
_, test_data = model.load_data()
print("loaded data")

# Load best trained ARIMA model using absolute path
model_path = os.path.join(base_dir, config.ARIMA_SAVE_DIR, config.ARIMA_PKL)

# Correctly load the ARIMA model
arima_result = ARIMAResults.load(model_path)
forecast_steps = len(test_data)
test_forecast = arima_result.forecast(steps=forecast_steps)

# Convert the forecasted values to a Pandas Series with the correct index
test_forecast = pd.Series(test_forecast, index=test_data.index)

# Calculate test error metrics
test_mse = mean_squared_error(test_data, test_forecast)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_data, test_forecast)
test_mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100

# Print results
print(f"\n📊 Test Metrics:")
print(f" - MSE:  {test_mse:.4f}")
print(f" - RMSE: {test_rmse:.4f}")
print(f" - MAE:  {test_mae:.4f}")
print(f" - MAPE: {test_mape:.2f}%")

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data.values, label="Test Data", color="blue", marker='o')
plt.plot(test_forecast.index, test_forecast.values, label="Test Forecast", 
         color="red", linestyle="dashed", marker='x')
plt.xlabel("Date")
plt.ylabel("Price (USD/MWh)")
plt.title("ARIMA Model: Test Set Predictions vs Actual Values")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Save plot
results_dir = os.path.join(base_dir, config.ARIMA_SAVE_DIR)
os.makedirs(results_dir, exist_ok=True)
plot_path = os.path.join(results_dir, "test_predictions.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n✅ Plot saved as: {plot_path}")
