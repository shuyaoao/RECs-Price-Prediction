import os
import matplotlib.pyplot as plt
from india_wind_solar.ARIMA.model import model
from india_wind_solar import config
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMAResults
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load train & validation data
print("Loading Data")
train_data, _ = model.load_data()
print(train_data.index)


# Ensure stationarity
print("Ensuring Stationarity")
train_series = model.make_stationary(train_data)
print("Ensured Stationarity")

# Select best (p, d, q) order
print("Selecting best order")
best_order = model.select_arima_order(train_series)
print(f"Best ARIMA Order: {best_order}")


# Train ARIMA on training data
model = ARIMA(train_series, order=best_order)
model_fit = model.fit()

# Rolling Forecast Validation
history = list(train_series)
predictions = []

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_pkl_path = os.path.join(base_dir, config.ARIMA_SAVE_DIR, config.ARIMA_PKL)

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(model_pkl_path), exist_ok=True)

# Save the model
with open(model_pkl_path, "wb") as model_file:
    pickle.dump(model_fit, model_file)
print(f"✅ ARIMA model saved as: {model_pkl_path}")
