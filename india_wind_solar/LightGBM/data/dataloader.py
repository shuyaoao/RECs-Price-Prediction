"""Data loading and preprocessing utilities for LightGBM models."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from india_wind_solar import config

def load_data():
    """Load the time series dataset and apply train-test split."""
    try:
        # Read the Excel file
        df = pd.read_excel(config.DATASET_PATH, engine="openpyxl")
        
        # Convert to datetime and set index
        df['Published Date(in GMT)'] = pd.to_datetime(df['Published Date(in GMT)'])
        
        # Filter by market activity type if specified
        if config.MARKET_ACTIVITY_TYPE:
            df = df[df["Market Activity"].str.lower() == config.MARKET_ACTIVITY_TYPE.lower()]
        
        # Sort and create time series
        df = df.sort_values('Published Date(in GMT)')
        df = df.set_index('Published Date(in GMT)')
        
        # Resample to daily frequency and forward fill missing values
        price_series = df['Price (USD/MWh)'].resample('D').mean().ffill()
        
        # Split data using date ranges
        train_data = price_series[config.TRAIN_START:config.TRAIN_END]
        validation_data = price_series[config.VALIDATION_START:config.VALIDATION_END]
        test_data = price_series[config.TEST_START:config.TEST_END]
        
        print(f"Data shapes - Train: {train_data.shape}, Validation: {validation_data.shape}, Test: {test_data.shape}")
        
        return train_data, validation_data, test_data
        
    except Exception as e:
        raise Exception(f"Error in load_data: {str(e)}")

def create_features(data, seq_length=config.SEQUENCE_LENGTH):
    """
    Create features for LightGBM model.
    For time series forecasting, we use lagged values as features.
    
    Args:
        data: Time series data
        seq_length: Number of previous time steps to use as features
        
    Returns:
        X: Feature matrix
        y: Target values
    """
    X, y = [], []
    data_array = data.values
    
    for i in range(len(data) - seq_length):
        # Use previous values as features
        features = data_array[i:i+seq_length]
        target = data_array[i+seq_length]
        
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)

def prepare_data_for_model():
    """Prepare data for model training, validation, and testing."""
    # Load data
    train_data, validation_data, test_data = load_data()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
    scaled_validation = scaler.transform(validation_data.values.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test_data.values.reshape(-1, 1)).flatten()
    
    # Create sequences
    X_train, y_train = create_features(pd.Series(scaled_train))
    X_validation, y_validation = create_features(pd.Series(scaled_validation))
    X_test, y_test = create_features(pd.Series(scaled_test))
    
    # Reshape X for LightGBM (samples, features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_validation = X_validation.reshape(X_validation.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test, scaler, train_data.index, validation_data.index, test_data.index