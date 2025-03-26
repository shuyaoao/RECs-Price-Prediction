"""Data loading and preprocessing utilities for PyTorch models."""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
# from data_config import india_wind_solar_config
# from model_config import CNN_BiLSTM_AM_config

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN_BiLSTM_AM_DataLoader():
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config

    def load_data(self):
        """Load the time series dataset and apply train-test split."""
        try:
            # Read the Excel file
            df = pd.read_excel(self.data_config.DATASET_PATH, engine="openpyxl")
            
            # Convert to datetime and set index
            df['Published Date(in GMT)'] = pd.to_datetime(df['Published Date(in GMT)'])
            
            # Filter by market activity type if specified
            if self.data_config.MARKET_ACTIVITY_TYPE:
                df = df[df["Market Activity"].str.lower() == self.data_config.MARKET_ACTIVITY_TYPE.lower()]
            
            # Sort and create time series
            df = df.sort_values('Published Date(in GMT)')
            df = df.set_index('Published Date(in GMT)')
            
            # Resample to daily frequency and forward fill missing values
            price_series = df['Price (USD/MWh)'].resample('D').mean().ffill()
            
            # Split data using date ranges
            train_data = price_series[self.data_config.TRAIN_START:self.data_config.TRAIN_END]
            validation_data = price_series[self.data_config.VALIDATION_START:self.data_config.VALIDATION_END]
            test_data = price_series[self.data_config.TEST_START:self.data_config.TEST_END]
            
            print(f"Data shapes - Train: {train_data.shape}, Validation: {validation_data.shape}, Test: {test_data.shape}")
            
            return train_data, validation_data, test_data
            
        except Exception as e:
            raise Exception(f"Error in load_data: {str(e)}")

    def create_sequences(self, data):
        """Create sequences for time series prediction."""
        xs, ys = [], []
        for i in range(len(data) - self.model_config.SEQUENCE_LENGTH):
            x = data[i:i+self.model_config.SEQUENCE_LENGTH]
            y = data[i+self.model_config.SEQUENCE_LENGTH]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def prepare_data_for_model(self):
        """Prepare data for model training, validation, and testing."""
        # Load data
        train_data, validation_data, test_data = self.load_data()
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
        scaled_validation = scaler.transform(validation_data.values.reshape(-1, 1)).flatten()
        scaled_test = scaler.transform(test_data.values.reshape(-1, 1)).flatten()
        
        # Split scaled data back into train and validation
        train_len = len(train_data)
        scaled_train = scaled_train[:train_len]
        
        # Create sequences
        X_train, y_train = self.create_sequences(scaled_train)
        X_validation, y_validation = self.create_sequences(scaled_validation)
        X_test, y_test = self.create_sequences(scaled_test)
        
        # Reshape for CNN input [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_train, y_train, X_validation, y_validation, X_test, y_test, scaler, train_data.index, validation_data.index, test_data.index

    def get_data_loaders(self):
        """Get PyTorch DataLoaders for training, validation, and testing."""
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, train_dates, val_dates, test_dates = self.prepare_data_for_model()
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.model_config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.model_config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.model_config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, test_loader, scaler, train_dates, val_dates, test_dates

"""Data loading and preprocessing utilities for XGBoost models."""
class TreeBased_DataLoader():
    def __init__(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config

    def load_data(self):
        """Load the time series dataset and apply train-test split."""
        try:
            # Read the Excel file
            df = pd.read_excel(self.data_config.DATASET_PATH, engine="openpyxl")
            
            # Convert to datetime and set index
            df['Published Date(in GMT)'] = pd.to_datetime(df['Published Date(in GMT)'])
            
            # Filter by market activity type if specified
            if self.data_config.MARKET_ACTIVITY_TYPE:
                df = df[df["Market Activity"].str.lower() == self.data_config.MARKET_ACTIVITY_TYPE.lower()]
            
            # Sort and create time series
            df = df.sort_values('Published Date(in GMT)')
            df = df.set_index('Published Date(in GMT)')
            
            # Resample to daily frequency and forward fill missing values
            price_series = df['Price (USD/MWh)'].resample('D').mean().ffill()
            
            # Split data using date ranges
            train_data = price_series[self.data_config.TRAIN_START:self.data_config.TRAIN_END]
            validation_data = price_series[self.data_config.VALIDATION_START:self.data_config.VALIDATION_END]
            test_data = price_series[self.data_config.TEST_START:self.data_config.TEST_END]
            
            print(f"Data shapes - Train: {train_data.shape}, Validation: {validation_data.shape}, Test: {test_data.shape}")
            
            return train_data, validation_data, test_data
            
        except Exception as e:
            raise Exception(f"Error in load_data: {str(e)}")

    def create_features(self, data):
        """
        Create features for XGBoost model.
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
        
        for i in range(len(data) - self.model_config.SEQUENCE_LENGTH):
            # Use previous values as features
            features = data_array[i:i+self.model_config.SEQUENCE_LENGTH]
            target = data_array[i+self.model_config.SEQUENCE_LENGTH]
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)

    def prepare_data_for_model(self):
        """Prepare data for model training, validation, and testing."""
        # Load data
        train_data, validation_data, test_data = self.load_data()
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1)).flatten()
        scaled_validation = scaler.transform(validation_data.values.reshape(-1, 1)).flatten()
        scaled_test = scaler.transform(test_data.values.reshape(-1, 1)).flatten()
        
        # Create sequences (for XGBoost, reshape to 2D)
        X_train, y_train = self.create_features(pd.Series(scaled_train))
        X_validation, y_validation = self.create_features(pd.Series(scaled_validation))
        X_test, y_test = self.create_features(pd.Series(scaled_test))
        
        # Reshape X for XGBoost (samples, features)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_validation = X_validation.reshape(X_validation.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        return X_train, y_train, X_validation, y_validation, X_test, y_test, scaler, train_data.index, validation_data.index, test_data.index

if __name__ == "__main__":
    from data_config import india_wind_solar_config
    from model_config import XGBoost_config
    dataloader = TreeBased_DataLoader(india_wind_solar_config, XGBoost_config)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, train_dates, val_dates, test_dates = dataloader.prepare_data_for_model()
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Train dates: {train_dates}")
    print(f"Validation dates: {val_dates}")
    print(f"Test dates: {test_dates}")