import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from india_wind_solar import config

def load_data():
    """Load the time series dataset and apply train-test split."""
    try:
        # Read the Excel file
        df = pd.read_excel(config.DATASET_PATH, engine="openpyxl")
        
        # Convert to datetime and set index
        df['Published Date(in GMT)'] = pd.to_datetime(df['Published Date(in GMT)'])
        
        # Filter by market activity type first
        df = df[df["Market Activity"].str.lower() == config.MARKET_ACTIVITY_TYPE.lower()]
        
        # Sort and create time series
        df = df.sort_values('Published Date(in GMT)')
        df = df.set_index('Published Date(in GMT)')
        
        # Resample to daily frequency and forward fill missing values
        price_series = df['Price (USD/MWh)'].resample('D').mean().ffill()
        
        # Split data using date ranges
        train_data = price_series[config.TRAIN_START:config.VALIDATION_END]
        test_data = price_series[config.TEST_START:config.TEST_END]
        
        print(f"Data shapes - Train: {train_data.shape}, Test: {test_data.shape}")
        
        return train_data, test_data
        
    except Exception as e:
        raise Exception(f"Error in load_data: {str(e)}")

def check_stationarity(series):
    """Perform the Augmented Dickey-Fuller (ADF) test for stationarity."""
    result = adfuller(series)
    return result[1]  # p-value

def make_stationary(series):
    """Apply differencing if data is not stationary."""
    p_value = check_stationarity(series)
    if p_value >= 0.05:  # If not stationary, apply differencing
        series = series.diff().dropna()
        print("Data was not stationary, differencing applied.")
    return series

# def select_arima_order(series):
#     """Use Auto-ARIMA to determine the best (p, d, q) order."""
#     model = auto_arima(series, **config.AUTO_ARIMA_PARAMS)
#     return model.order  # Returns best (p, d, q) values

def select_arima_order(series):
    """Use Auto-ARIMA to determine the best (p, d, q) order, enforcing differencing (d ≥ 1)."""
    model = auto_arima(series, d=1, seasonal=False, trace=True, stepwise=True)
    return model.order

def train_arima(series, order):
    """Train ARIMA model using best (p, d, q) order."""
    model = ARIMA(series, order=order)
    result = model.fit()
    return result