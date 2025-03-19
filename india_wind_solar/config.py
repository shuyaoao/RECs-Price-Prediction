from datetime import datetime
import torch

# Dataset path
DATASET_PATH = "/Users/lishuyao/Documents/Redex/RECs-Price-Prediction/data/S&P India Wind_Solar.xlsx"

# Forecast settings
FORECAST_DAYS = 30

# Train-Test Split Dates
TRAIN_START = datetime.strptime("2024-05-06", "%Y-%m-%d")
TRAIN_END = datetime.strptime("2024-12-05", "%Y-%m-%d")
VALIDATION_START = datetime.strptime("2024-12-06", "%Y-%m-%d")
VALIDATION_END = datetime.strptime("2025-01-05", "%Y-%m-%d")
TEST_START = datetime.strptime("2025-01-06", "%Y-%m-%d")
TEST_END = datetime.strptime("2025-02-07", "%Y-%m-%d")

MARKET_ACTIVITY_TYPE = 'offer'

ARIMA_SAVE_DIR = 'results/arima'
ARIMA_PKL = 'best_arima_model.pkl'

# Auto-ARIMA settings
AUTO_ARIMA_PARAMS = {
    "seasonal": False,
    "trace": True,
    "stepwise": True
}

# Model parameters for CNNBiLSTMAM
SEQUENCE_LENGTH = 10  # Number of time steps to look back
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2
PATIENCE = 50  # Early stopping patience

# CNN parameters
CNN_FILTERS = [64, 128]
CNN_KERNEL_SIZES = [3, 5]
CNN_POOL_SIZE = 2

# LSTM parameters
LSTM_UNITS = 128
BIDIRECTIONAL = True

# Attention parameters
ATTENTION_UNITS = 64

# Model save directory
MODEL_SAVE_DIR = "models/saved"
MODEL_NAME = "cnn_bilstm_am_model.h5"

# Results directory
RESULTS_DIR = "results"

# PyTorch specific settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device for PyTorch computations
RANDOM_SEED = 42  # For reproducibility
NUM_WORKERS = 4  # Number of workers for data loading

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'n_estimators': 200,
    'seed': RANDOM_SEED
}

# XGBoost model name
XGBOOST_MODEL_NAME = "xgboost_model.pkl"

# LightGBM parameters
LIGHTGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'n_estimators': 200
}

# LightGBM model name
LIGHTGBM_MODEL_NAME = "lightgbm_model.pkl"