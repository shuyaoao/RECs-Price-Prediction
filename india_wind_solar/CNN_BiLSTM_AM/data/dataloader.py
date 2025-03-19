"""Data loading and preprocessing utilities for PyTorch models."""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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

def create_sequences(data, seq_length=config.SEQUENCE_LENGTH):
    """Create sequences for time series prediction."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data_for_model():
    """Prepare data for model training, validation, and testing."""
    # Load data
    train_data, validation_data, test_data = load_data()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
    scaled_validation = scaler.transform(validation_data.values.reshape(-1, 1)).flatten()
    scaled_test = scaler.transform(test_data.values.reshape(-1, 1)).flatten()
    
    # Split scaled data back into train and validation
    train_len = len(train_data)
    scaled_train = scaled_train[:train_len]
    
    # Create sequences
    X_train, y_train = create_sequences(scaled_train)
    X_validation, y_validation = create_sequences(scaled_validation)
    X_test, y_test = create_sequences(scaled_test)
    
    # Reshape for CNN input [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test, scaler, train_data.index, validation_data.index, test_data.index


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_data_loaders(batch_size=config.BATCH_SIZE):
    """Get PyTorch DataLoaders for training, validation, and testing."""
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, train_dates, val_dates, test_dates = prepare_data_for_model()
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler, train_dates, val_dates, test_dates

if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler, train_dates, val_dates, test_dates = get_data_loaders()
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    print(f"Scaler: {scaler}")
    print(f"Train dates: {train_dates[0]} to {train_dates[-1]}")
    print(f"Validation dates: {val_dates[0]} to {val_dates[-1]}")
    print(f"Test dates: {test_dates[0]} to {test_dates[-1]}")
    