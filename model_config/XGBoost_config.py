import torch

MODEL_ID = 'XGBoost'

# PyTorch specific settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device for PyTorch computations
RANDOM_SEED = 42  # For reproducibility
NUM_WORKERS = 4  # Number of workers for data loading

SEQUENCE_LENGTH = 20
RANDOM_SEED = 42

XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'n_estimators': 300,
    'seed': RANDOM_SEED
}

# XGBoost model name
SAVED_WEIGHTS = "xgboost_model.pkl"