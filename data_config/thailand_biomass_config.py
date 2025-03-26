from datetime import datetime
import torch
import os

# Dataset path
DATASET_PATH = "/Users/lishuyao/Documents/Redex/RECs-Price-Prediction/data/S&P Thailand Biomass.xlsx"

DATA_ID = 'thailand_biomass'

# # Forecast settings
# FORECAST_DAYS = 30

# Train-Test Split Dates
TRAIN_START = datetime.strptime("2024-05-08", "%Y-%m-%d")
TRAIN_END = datetime.strptime("2024-12-05", "%Y-%m-%d")
VALIDATION_START = datetime.strptime("2024-12-06", "%Y-%m-%d")
VALIDATION_END = datetime.strptime("2025-01-05", "%Y-%m-%d")
TEST_START = datetime.strptime("2025-01-06", "%Y-%m-%d")
TEST_END = datetime.strptime("2025-02-07", "%Y-%m-%d")

MARKET_ACTIVITY_TYPE = 'offer'

# Model save directory
MODEL_SAVE_DIR = os.path.join("model_weights", DATA_ID)

PLOTS_DIR = os.path.join("results", DATA_ID)