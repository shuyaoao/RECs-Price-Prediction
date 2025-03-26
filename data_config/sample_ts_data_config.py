from datetime import datetime
import os

# Dataset path
DATASET_PATH = "/Users/lishuyao/Documents/Redex/RECs-Price-Prediction/data/POP.xlsx"

DATA_ID = 'sample_ts_data'

# # Forecast settings
# FORECAST_DAYS = 30

# Train-Test Split Dates
TRAIN_START = datetime.strptime("1952-01-01", "%Y-%m-%d")
TRAIN_END = datetime.strptime("1999-07-01", "%Y-%m-%d")
VALIDATION_START = datetime.strptime("1999-08-01", "%Y-%m-%d")
VALIDATION_END = datetime.strptime("2009-09-01", "%Y-%m-%d")
TEST_START = datetime.strptime("2009-10-01", "%Y-%m-%d")
TEST_END = datetime.strptime("2019-12-01", "%Y-%m-%d")

MARKET_ACTIVITY_TYPE = 'offer'

# Model save directory
MODEL_SAVE_DIR = os.path.join("model_weights", DATA_ID)

# Results directory
PLOTS_DIR = os.path.join("results", DATA_ID)